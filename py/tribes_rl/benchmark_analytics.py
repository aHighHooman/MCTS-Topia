from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import statistics
import subprocess
import time
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from .config import HybridAgentConfig
from .device import require_cuda_device
from .replay import ReplayStore
from .selfplay import run_selfplay


PROFILE_RE = re.compile(r"\[tribes_rl\.profile\] choose_action (?P<body>.*)")
SEARCH_RE = re.compile(r"\[tribes_rl\.search_profile\] mcts(?:_fast_path)? (?P<body>.*)")
WARMUP_RE = re.compile(r"\[tribes_rl\.warmup\] (?P<body>.*)")
FALLBACK_RE = re.compile(r"(invalid[-_ ]action|fallback)", re.IGNORECASE)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _parse_kv(body: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for token in body.split():
        if "=" not in token:
            continue
        key, raw_value = token.split("=", 1)
        value = raw_value.strip().rstrip(",")
        lowered = value.lower()
        if lowered in {"true", "false"}:
            parsed[key] = lowered == "true"
            continue
        try:
            parsed[key] = float(value) if any(ch in value for ch in ".eE") else int(value)
        except ValueError:
            parsed[key] = value
    return parsed


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower))


def _summarize(values: list[float]) -> dict[str, float]:
    return {
        "count": float(len(values)),
        "mean": statistics.fmean(values) if values else 0.0,
        "median": statistics.median(values) if values else 0.0,
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else 0.0,
    }


def _flatten_summary(prefix: str, values: list[float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in _summarize(values).items()}


def _numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        if field not in row:
            continue
        try:
            values.append(float(row[field]))
        except (TypeError, ValueError):
            continue
    return values


def _parse_profile_logs(text: str) -> dict[str, Any]:
    action_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    warmup_rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        profile_match = PROFILE_RE.search(line)
        if profile_match:
            action_rows.append(_parse_kv(profile_match.group("body")))
            continue
        search_match = SEARCH_RE.search(line)
        if search_match:
            search_rows.append(_parse_kv(search_match.group("body")))
            continue
        warmup_match = WARMUP_RE.search(line)
        if warmup_match:
            warmup_rows.append(_parse_kv(warmup_match.group("body")))

    summary: dict[str, Any] = {
        "action_decisions": len(action_rows),
        "search_decisions": len(search_rows),
        "warmup_count": len(warmup_rows),
    }
    for field in ("encode_ms", "policy_ms", "search_ms", "total_ms", "actions"):
        summary.update(_flatten_summary(f"action_{field}", _numeric_values(action_rows, field)))
    for field in ("select_ms", "eval_ms", "expand_ms", "total_inner_ms", "eval_batches", "eval_positions", "eval_cache_hits", "root_actions", "searched_actions"):
        summary.update(_flatten_summary(f"mcts_{field}", _numeric_values(search_rows, field)))
    summary.update(_flatten_summary("warmup_ms", _numeric_values(warmup_rows, "warmup_ms")))
    total_search_ms = float(summary.get("action_search_ms_mean", 0.0))
    total_action_ms = float(summary.get("action_total_ms_mean", 0.0))
    summary["mean_search_share"] = total_search_ms / total_action_ms if total_action_ms > 0 else 0.0
    return summary


def _hardware_snapshot(device: torch.device) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        snapshot.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(index),
                "cuda_capability": f"{props.major}.{props.minor}",
                "cuda_total_memory_gb": props.total_memory / (1024 ** 3),
                "cuda_max_memory_allocated_mb": torch.cuda.max_memory_allocated(index) / (1024 ** 2),
                "cuda_max_memory_reserved_mb": torch.cuda.max_memory_reserved(index) / (1024 ** 2),
            }
        )
    return snapshot


def _bot_command(cfg: HybridAgentConfig, workdir: Path, checkpoint_path: Path) -> list[str]:
    return [
        "python",
        str(workdir / "py" / "hybrid_rl_bot.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--replay-dir",
        str(cfg.replay.replay_dir),
        "--simulations",
        str(cfg.search.num_simulations),
        "--max-depth",
        str(cfg.search.max_depth),
        "--top-k-actions",
        str(cfg.search.top_k_actions),
        "--search-batch-size",
        str(cfg.search.batch_size),
    ]


def _count_replay_steps(replay_dir: Path, prefix: str) -> tuple[int, int]:
    store = ReplayStore(replay_dir, capacity_steps=0, shard_prefix=prefix)
    return len(store.shards()), store.step_count()


def _new_replay_step_counts(replay_dir: Path, prefix: str, before: set[Path]) -> list[int]:
    counts: list[int] = []
    store = ReplayStore(replay_dir, capacity_steps=0, shard_prefix=prefix)
    for shard in store.shards():
        if shard in before:
            continue
        try:
            payload, records = store._load_shard(shard)
        except Exception:
            continue
        counts.append(int(payload.get("step_count", len(records))))
    return counts


def _candidate_matrix(args: argparse.Namespace) -> Iterable[dict[str, int]]:
    for simulations in args.simulations:
        for search_batch_size in args.search_batch_sizes:
            for top_k_actions in args.top_k_actions:
                for max_turns in args.max_turns_capitals:
                    for action_timeout_ms in args.external_action_timeout_ms:
                        yield {
                            "simulations": int(simulations),
                            "search_batch_size": int(search_batch_size),
                            "top_k_actions": int(top_k_actions),
                            "max_turns_capitals": int(max_turns),
                            "external_action_timeout_ms": int(action_timeout_ms),
                        }


def run_candidate(
    base_cfg: HybridAgentConfig,
    candidate: dict[str, int],
    *,
    games: int,
    seed_base: int,
    run_index: int,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    cfg.search.num_simulations = candidate["simulations"]
    cfg.search.batch_size = candidate["search_batch_size"]
    cfg.search.top_k_actions = candidate["top_k_actions"]
    cfg.search.sample_action = False
    cfg.search.dirichlet_epsilon = 0.0
    cfg.selfplay.max_turns_capitals = candidate["max_turns_capitals"]
    cfg.selfplay.external_action_timeout_ms = candidate["external_action_timeout_ms"]
    cfg.selfplay.profile_selfplay = True
    cfg.training.device = str(device)

    run_label = (
        f"analytics_{run_index:04d}_s{cfg.search.num_simulations}_sb{cfg.search.batch_size}"
        f"_k{cfg.search.top_k_actions}_mt{cfg.selfplay.max_turns_capitals}"
        f"_atm{cfg.selfplay.external_action_timeout_ms}"
    )
    cfg.replay.replay_dir = output_dir / "replay" / run_label
    cfg.training.output_dir = output_dir / "artifacts" / run_label
    cfg.training.checkpoint_dir = cfg.training.output_dir / "checkpoints"
    cfg.diagnostics.metrics_csv = cfg.training.output_dir / "metrics.csv"
    cfg.replay.replay_dir.mkdir(parents=True, exist_ok=True)
    cfg.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    workdir = Path(__file__).resolve().parents[2]
    command = _bot_command(cfg, workdir, cfg.training.checkpoint_path)
    tribes = ["Xin Xi", "Imperius"]
    game_seconds: list[float] = []
    replay_steps_per_game: list[int] = []
    returncodes: list[int] = []
    combined_logs: list[str] = []
    started = time.perf_counter()
    start_wall = _utc_now()

    for game_idx in range(games):
        seed = seed_base + run_index * 10_000 + game_idx
        cfg.selfplay.game_seed = seed
        cfg.selfplay.agent_seed = seed
        cfg.selfplay.level_seed = seed
        before_shards = set(ReplayStore(cfg.replay.replay_dir, 0, cfg.replay.shard_prefix).shards())
        game_started = time.perf_counter()
        result = run_selfplay(
            cfg,
            [command, list(command)],
            tribes,
            workdir,
            checkpoint_path=cfg.training.checkpoint_path,
            replay_store=ReplayStore(cfg.replay.replay_dir, cfg.replay.capacity_steps, cfg.replay.shard_prefix),
            device=device,
            progress_label=f"analytics run={run_index} game={game_idx + 1}/{games}",
        )
        game_seconds.append(time.perf_counter() - game_started)
        returncodes.append(int(result.returncode))
        combined_logs.append("\n".join([result.stdout or "", result.stderr or ""]))
        replay_steps_per_game.append(sum(_new_replay_step_counts(cfg.replay.replay_dir, cfg.replay.shard_prefix, before_shards)))
        if result.returncode != 0:
            break

    elapsed_sec = time.perf_counter() - started
    replay_shards, replay_steps = _count_replay_steps(cfg.replay.replay_dir, cfg.replay.shard_prefix)
    profile = _parse_profile_logs("\n".join(combined_logs))
    decisions = int(profile.get("action_decisions", 0))
    configured_simulations = decisions * int(cfg.search.num_simulations)
    profile["configured_simulations_sec"] = configured_simulations / elapsed_sec if elapsed_sec > 0 else 0.0
    profile["configured_simulations_per_action"] = int(cfg.search.num_simulations)
    profile["actions_sec"] = decisions / elapsed_sec if elapsed_sec > 0 else 0.0
    profile["turn_time_proxy_ms"] = (statistics.fmean(game_seconds) * 1000.0 / max(1, cfg.selfplay.max_turns_capitals)) if game_seconds else 0.0

    fallback_count = sum(1 for line in "\n".join(combined_logs).splitlines() if FALLBACK_RE.search(line))
    hardware = _hardware_snapshot(device)
    row = {
        "schema_version": 1,
        "run_index": run_index,
        "start_wall_utc": start_wall,
        "end_wall_utc": _utc_now(),
        "elapsed_sec": elapsed_sec,
        "games_requested": int(games),
        "games_completed": len(game_seconds),
        "returncodes": returncodes,
        "game_seconds_mean": statistics.fmean(game_seconds) if game_seconds else 0.0,
        "game_seconds_median": statistics.median(game_seconds) if game_seconds else 0.0,
        "replay_shards": replay_shards,
        "replay_steps": replay_steps,
        "replay_steps_sec": replay_steps / elapsed_sec if elapsed_sec > 0 else 0.0,
        "mean_replay_steps_game": replay_steps / len(game_seconds) if game_seconds else 0.0,
        "median_replay_steps_game": statistics.median(replay_steps_per_game) if replay_steps_per_game else 0.0,
        "fallback_count": fallback_count,
        "fallback_rate_per_decision": fallback_count / decisions if decisions else 0.0,
        "config": _jsonable(cfg),
        "candidate": dict(candidate),
        "profile": profile,
        "hardware": hardware,
        "bottleneck_hint": _bottleneck_hint(profile, hardware),
    }
    return row


def _bottleneck_hint(profile: dict[str, Any], hardware: dict[str, Any]) -> str:
    policy_ms = float(profile.get("action_policy_ms_mean", 0.0))
    search_ms = float(profile.get("action_search_ms_mean", 0.0))
    encode_ms = float(profile.get("action_encode_ms_mean", 0.0))
    eval_ms = float(profile.get("mcts_eval_ms_mean", 0.0))
    select_ms = float(profile.get("mcts_select_ms_mean", 0.0))
    expand_ms = float(profile.get("mcts_expand_ms_mean", 0.0))
    dominant = max(
        [("encode", encode_ms), ("model_policy", policy_ms), ("search", search_ms), ("mcts_eval", eval_ms), ("mcts_select", select_ms), ("mcts_expand", expand_ms)],
        key=lambda item: item[1],
    )
    if dominant[0] == "mcts_eval" and hardware.get("cuda_available"):
        return "model_evaluation_dominates_mcts; try larger search batches/top-k pruning or inspect GPU utilization"
    if dominant[0] in {"mcts_select", "mcts_expand"}:
        return "native_tree_work_dominates; search depth/action branching is likely CPU-bound"
    if dominant[0] == "encode":
        return "observation_encoding_dominates; Python feature construction is likely a bottleneck"
    if dominant[0] == "model_policy":
        return "root_model_forward_dominates; inspect batch shape/model size/device placement"
    if dominant[0] == "search":
        return "search_wrapper_dominates; compare inner MCTS timings against total search time"
    return "no_clear_single_bottleneck"


def _flat_row(row: dict[str, Any]) -> dict[str, Any]:
    flat = {
        "run_index": row["run_index"],
        "elapsed_sec": row["elapsed_sec"],
        "games_completed": row["games_completed"],
        "game_seconds_mean": row["game_seconds_mean"],
        "replay_steps": row["replay_steps"],
        "replay_steps_sec": row["replay_steps_sec"],
        "fallback_count": row["fallback_count"],
        "bottleneck_hint": row["bottleneck_hint"],
    }
    flat.update(row["candidate"])
    flat.update(row["profile"])
    for key in ("device", "cuda_device_name", "cuda_max_memory_allocated_mb", "cuda_max_memory_reserved_mb", "torch_num_threads"):
        flat[f"hardware_{key}"] = row["hardware"].get(key)
    return flat


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows = [_flat_row(row) for row in rows]
    fieldnames = sorted({key for row in flat_rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model + native MCTS analytics benchmarks.")
    parser.add_argument("--simulations", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--search-batch-sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--top-k-actions", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--max-turns-capitals", type=int, nargs="+", default=[20, 40])
    parser.add_argument("--external-action-timeout-ms", type=int, nargs="+", default=[30_000, 120_000])
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path, default=Path("rl/analytics_benchmarks"))
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--match-timeout-seconds", type=int, default=None)
    parser.add_argument("--device", default=None, help="Override device. Defaults to CUDA via project policy.")
    args = parser.parse_args()

    cfg = HybridAgentConfig()
    if args.match_timeout_seconds is not None:
        cfg.selfplay.timeout_seconds = int(args.match_timeout_seconds)
    device = torch.device(args.device) if args.device else require_cuda_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.jsonl or args.output_dir / f"analytics_benchmark_{int(time.time())}.jsonl"
    csv_path = args.csv or jsonl_path.with_suffix(".csv")
    candidates = list(_candidate_matrix(args))

    rows: list[dict[str, Any]] = []
    with jsonl_path.open("a", encoding="utf-8") as handle:
        for run_index, candidate in enumerate(candidates):
            row = run_candidate(
                cfg,
                candidate,
                games=args.games,
                seed_base=args.seed_base,
                run_index=run_index,
                output_dir=args.output_dir,
                device=device,
            )
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            _write_csv(csv_path, rows)
            profile = row["profile"]
            print(
                "[analytics-bench] "
                f"run={run_index} sims={candidate['simulations']} batch={candidate['search_batch_size']} "
                f"top_k={candidate['top_k_actions']} "
                f"max_turns={candidate['max_turns_capitals']} action_timeout_ms={candidate['external_action_timeout_ms']} "
                f"actions={profile.get('action_decisions', 0)} action_ms={profile.get('action_total_ms_mean', 0.0):.1f} "
                f"search_ms={profile.get('action_search_ms_mean', 0.0):.1f} sims_sec={profile.get('configured_simulations_sec', 0.0):.1f} "
                f"hint={row['bottleneck_hint']} jsonl={jsonl_path} csv={csv_path}",
                flush=True,
            )


if __name__ == "__main__":
    main()
