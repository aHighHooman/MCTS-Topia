from __future__ import annotations

import argparse
import cProfile
import csv
import json
import socket
import subprocess
import re
import statistics
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

PY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PY_ROOT.parent
DEFAULT_SELFPLAY_MCTS_NN_CONFIG = PY_ROOT / "profiling" / "configs" / "selfplay_mcts_nn.json"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from profiling.config import load_config_defaults
from search.config import HybridAgentConfig
from training.replay import ReplayStore
from training.selfplay import run_selfplay
from training.train import (
    _persistent_bridge_command,
    _shutdown_persistent_server,
    _wait_for_port,
)


PROFILE_PATTERN = re.compile(
    r"\[mcts_nn\.profile\]\s+choose_action\s+"
    r"encode_ms=(?P<encode_ms>[-+0-9.]+)\s+"
    r"policy_ms=(?P<policy_ms>[-+0-9.]+)\s+"
    r"search_ms=(?P<search_ms>[-+0-9.]+)\s+"
    r"total_ms=(?P<total_ms>[-+0-9.]+)\s+"
    r"actions=(?P<actions>\d+)\s+"
    r"device=(?P<device>\S+)\s+"
    r"(?:action_budget_sec|turn_budget_remaining_sec)=(?P<action_budget_sec>[-+0-9.]+)"
)
SEARCH_PATTERN = re.compile(
    r"\[mcts_nn\.search_profile\]\s+mcts\s+"
    r"sims=(?P<sims>\d+)\s+"
    r"batch=(?P<batch>\d+)\s+"
    r"(?:paths=(?P<paths>\d+)\s+)?"
    r"(?:nodes=(?P<nodes>\d+)\s+)?"
    r"eval_batches=(?P<eval_batches>\d+)\s+"
    r"eval_positions=(?P<eval_positions>\d+)\s+"
    r"(?:eval_cache_hits=(?P<eval_cache_hits>\d+)\s+)?"
    r"(?:eval_cache_size=(?P<eval_cache_size>\d+)\s+)?"
    r"(?:avg_depth=(?P<avg_depth>[-+0-9.]+)\s+)?"
    r"(?:selected_max_depth=(?P<selected_max_depth>\d+)\s+)?"
    r"(?:root_visit_entropy=(?P<root_visit_entropy>[-+0-9.]+)\s+)?"
    r"(?:top_visit_share=(?P<top_visit_share>[-+0-9.]+)\s+)?"
    r"select_ms=(?P<select_ms>[-+0-9.]+)\s+"
    r"eval_ms=(?P<eval_ms>[-+0-9.]+)\s+"
    r"expand_ms=(?P<expand_ms>[-+0-9.]+)\s+"
    r"total_inner_ms=(?P<total_inner_ms>[-+0-9.]+)"
    r"(?:\s+root_actions=(?P<root_actions>\d+))?"
    r"(?:\s+searched_actions=(?P<searched_actions>\d+))?"
    r"(?:\s+native_unsupported_transition=(?P<native_unsupported_transition>\d+))?"
    r"(?:\s+native_invalid_transition=(?P<native_invalid_transition>\d+))?"
    r"(?:\s+native_approximate_transition=(?P<native_approximate_transition>\d+))?"
)
WARMUP_PATTERN = re.compile(r"\[mcts_nn\.warmup\]\s+warmup_ms=(?P<warmup_ms>[-+0-9.]+)")
FALLBACK_PATTERN = re.compile(r"(invalid[-_ ]action|fallback)", re.IGNORECASE)


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


def _format_seconds(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes, rem = divmod(seconds, 60.0)
    return f"{int(minutes)}m{rem:04.1f}s"


def _format_rate(numerator: float, elapsed_sec: float) -> str:
    if elapsed_sec <= 0.0:
        return "0.0"
    return f"{numerator / elapsed_sec:.1f}"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[index])


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p95": _percentile(values, 95.0),
        "max": float(max(values)),
    }


def _parse_profile(stderr: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float]]:
    action_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    warmup_ms: list[float] = []
    for line in stderr.splitlines():
        match = PROFILE_PATTERN.search(line)
        if match:
            row: dict[str, Any] = {}
            for key, value in match.groupdict().items():
                row[key] = int(value) if key == "actions" else value if key == "device" else float(value)
            action_rows.append(row)
            continue
        match = SEARCH_PATTERN.search(line)
        if match:
            row = {}
            for key, value in match.groupdict().items():
                if value is None:
                    row[key] = 0
                else:
                    row[key] = int(value) if key in {
                        "sims",
                        "batch",
                        "paths",
                        "nodes",
                        "eval_batches",
                        "eval_positions",
                        "eval_cache_hits",
                        "eval_cache_size",
                        "selected_max_depth",
                        "root_actions",
                        "searched_actions",
                        "native_unsupported_transition",
                        "native_invalid_transition",
                        "native_approximate_transition",
                    } else float(value)
            search_rows.append(row)
            continue
        match = WARMUP_PATTERN.search(line)
        if match:
            warmup_ms.append(float(match.group("warmup_ms")))
    return action_rows, search_rows, warmup_ms


def _new_replay_step_count(replay_dir: Path, prefix: str, before: set[Path]) -> int:
    store = ReplayStore(replay_dir, capacity_steps=0, shard_prefix=prefix)
    total = 0
    for shard in store.shards():
        if shard in before:
            continue
        try:
            payload, records = store._load_shard(shard)
        except Exception:
            continue
        total += int(payload.get("step_count", len(records)))
    return total


def _bot_command(args: argparse.Namespace, cfg: HybridAgentConfig, workdir: Path) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "bots" / "hybrid_nn_bot.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--replay-dir",
        str(cfg.replay.replay_dir),
        "--simulations",
        str(cfg.search.num_simulations),
        "--top-k-actions",
        str(cfg.search.top_k_actions),
        "--search-batch-size",
        str(cfg.search.batch_size),
        "--max-game-actions",
        str(cfg.selfplay.max_actions_per_game),
        "--wall-clock-per-action-seconds",
        str(cfg.selfplay.wall_clock_per_action_seconds),
    ]
    if args.deterministic:
        command.append("--deterministic")
    if bool(getattr(cfg.search, "reuse_tree", False)):
        command.append("--reuse-tree")
    return command


def _format_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "(no rows)"
    widths = {
        key: max(len(label), *(len(str(row.get(key, ""))) for row in rows))
        for key, label in columns
    }
    header = "  ".join(label.ljust(widths[key]) for key, label in columns)
    divider = "  ".join("-" * widths[key] for key, _label in columns)
    body = ["  ".join(str(row.get(key, "")).ljust(widths[key]) for key, _label in columns) for row in rows]
    return "\n".join([header, divider, *body])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["name"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _phase_timing_rows(action_rows: list[dict[str, Any]], search_rows: list[dict[str, Any]], total_sec: float) -> list[dict[str, Any]]:
    phases = [
        ("choose_action.encode", [float(row["encode_ms"]) for row in action_rows], len(action_rows)),
        ("choose_action.policy_forward", [float(row["policy_ms"]) for row in action_rows], len(action_rows)),
        ("choose_action.mcts_search", [float(row["search_ms"]) for row in action_rows], len(action_rows)),
        ("choose_action.total", [float(row["total_ms"]) for row in action_rows], len(action_rows)),
        ("mcts.select", [float(row["select_ms"]) for row in search_rows], sum(int(row.get("paths", row["sims"])) for row in search_rows)),
        ("mcts.evaluate", [float(row["eval_ms"]) for row in search_rows], sum(int(row["eval_positions"]) for row in search_rows)),
        ("mcts.expand", [float(row["expand_ms"]) for row in search_rows], sum(int(row.get("paths", row["sims"])) for row in search_rows)),
        ("mcts.inner_total", [float(row["total_inner_ms"]) for row in search_rows], sum(int(row.get("paths", row["sims"])) for row in search_rows)),
    ]
    total_ms = max(1e-9, total_sec * 1000.0)
    rows = []
    for name, values, items in phases:
        cumulative_ms = sum(values)
        calls = len(values)
        rows.append(
            {
                "name": name,
                "calls": calls,
                "items": items,
                "total_ms": f"{cumulative_ms:.3f}",
                "self_ms": f"{cumulative_ms:.3f}",
                "avg_ms": f"{cumulative_ms / max(1, calls):.3f}",
                "pct": f"{cumulative_ms / total_ms * 100.0:.1f}",
            }
        )
    return rows


def _hotspot_rows(
    timing_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    *,
    total_sec: float,
    min_ms: float,
    min_pct: float,
    limit: int,
) -> list[dict[str, Any]]:
    total_ms = max(1e-9, total_sec * 1000.0)
    rows: list[dict[str, Any]] = []
    for row in timing_rows:
        phase_ms = float(row["total_ms"])
        pct = float(row["pct"])
        if phase_ms < min_ms or pct < min_pct:
            continue
        rows.append(
            {
                "source": "phase",
                "name": row["name"],
                "calls": row["calls"],
                "items": row["items"],
                "time_ms": row["total_ms"],
                "self_ms": row["self_ms"],
                "pct": row["pct"],
            }
        )
    for row in profile_rows:
        name = str(row["function"])
        if name.startswith("py\\profiling\\selfplay_mcts_nn.py:") or name.startswith("py/profiling/selfplay_mcts_nn.py:"):
            continue
        function_ms = float(row["cum_ms"])
        pct = function_ms / total_ms * 100.0
        if function_ms < min_ms or pct < min_pct:
            continue
        rows.append(
            {
                "source": "function",
                "name": name,
                "calls": row["calls"],
                "items": "",
                "time_ms": row["cum_ms"],
                "self_ms": row["self_ms"],
                "pct": f"{pct:.1f}",
            }
        )
    rows.sort(key=lambda row: float(row["time_ms"]), reverse=True)
    return rows[:limit]


def _profile_rows(profile_path: Path, *, root: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not profile_path.exists():
        return []
    import pstats

    stats = pstats.Stats(str(profile_path))
    entries = []
    for (filename, line, func_name), stat in stats.stats.items():
        primitive_calls, total_calls, total_time, cumulative_time, _callers = stat
        if total_time <= 0.0 and cumulative_time <= 0.0:
            continue
        try:
            display_file = str(Path(filename).resolve().relative_to(root))
        except (OSError, ValueError):
            display_file = filename
        entries.append(
            {
                "function": f"{display_file}:{line}:{func_name}",
                "primitive_calls": primitive_calls,
                "calls": total_calls,
                "self_ms": total_time * 1000.0,
                "cum_ms": cumulative_time * 1000.0,
                "avg_self_us": total_time * 1_000_000.0 / max(1, primitive_calls),
            }
        )
    entries.sort(key=lambda row: row["cum_ms"], reverse=True)
    rows = [
        {
            "function": row["function"],
            "primitive_calls": row["primitive_calls"],
            "calls": row["calls"],
            "self_ms": f"{row['self_ms']:.3f}",
            "cum_ms": f"{row['cum_ms']:.3f}",
            "avg_self_us": f"{row['avg_self_us']:.3f}",
        }
        for row in entries
    ]
    return rows if limit is None else rows[:limit]


def _find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_profiled_persistent_bot_server(
    cfg: HybridAgentConfig,
    bot_server_script: Path,
    checkpoint_path: Path,
    replay_dir: Path,
    log_dir: Path,
    profile_output: Path,
) -> tuple[subprocess.Popen[str], str, int, Path, Path]:
    host = "127.0.0.1"
    port = _find_free_local_port()
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"persistent_bot_{port}.stdout.log"
    stderr_path = log_dir / f"persistent_bot_{port}.stderr.log"
    command = [
        sys.executable,
        str(bot_server_script),
        "--host",
        host,
        "--port",
        str(port),
        "--checkpoint",
        str(checkpoint_path),
        "--replay-dir",
        str(replay_dir),
        "--simulations",
        str(cfg.search.num_simulations),
        "--top-k-actions",
        str(cfg.search.top_k_actions),
        "--search-batch-size",
        str(cfg.search.batch_size),
        "--max-game-actions",
        str(max(1, int(cfg.selfplay.max_actions_per_game))),
        "--wall-clock-per-action-seconds",
        str(max(0.0, float(cfg.selfplay.wall_clock_per_action_seconds))),
        "--profile-output",
        str(profile_output),
    ]
    if cfg.selfplay.profile_selfplay:
        command.append("--profile-selfplay")
    if bool(getattr(cfg.search, "reuse_tree", False)):
        command.append("--reuse-tree")
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=stdout_handle, stderr=stderr_handle, text=True)
    stdout_handle.close()
    stderr_handle.close()
    try:
        _wait_for_port(host, port, process)
    except Exception:
        _shutdown_persistent_server(host, port, process, timeout_seconds=3.0)
        stdout_tail = "\n".join(stdout_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
        stderr_tail = "\n".join(stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
        raise RuntimeError(f"Persistent bot server failed to start.\nStdout tail:\n{stdout_tail}\nStderr tail:\n{stderr_tail}")
    return process, host, port, stdout_path, stderr_path


def _read_new_text(paths: list[Path], offsets: dict[Path, int]) -> str:
    parts: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        offset = offsets.get(path, 0)
        if offset < len(text):
            parts.append(text[offset:])
            offsets[path] = len(text)
    return "\n".join(parts)


def _configure(args: argparse.Namespace) -> HybridAgentConfig:
    cfg = HybridAgentConfig()
    cfg.search.num_simulations = int(args.simulations)
    cfg.search.batch_size = int(args.search_batch_size)
    cfg.search.top_k_actions = int(args.top_k_actions)
    cfg.search.seed = int(args.seed_base)
    cfg.search.reuse_tree = bool(args.reuse_tree)
    if args.no_dirichlet:
        cfg.search.dirichlet_epsilon = 0.0
    cfg.selfplay.profile_selfplay = True
    cfg.selfplay.persistent_bot = bool(args.persistent_bot)
    cfg.selfplay.run_mode = str(args.run_mode)
    cfg.selfplay.game_mode = str(args.game_mode)
    cfg.selfplay.map_type = str(args.map_type)
    cfg.selfplay.map_size = str(args.map_size)
    cfg.selfplay.max_turns_capitals = int(args.max_turns_capitals)
    cfg.selfplay.max_actions_per_turn = int(args.max_actions_per_turn)
    cfg.selfplay.max_actions_per_game = int(args.max_actions_per_game)
    cfg.selfplay.adjudicate_incomplete_games = bool(getattr(args, "adjudicate_incomplete_games", False))
    if getattr(args, "adjudicator_bot", None):
        cfg.selfplay.adjudicator_bot = str(args.adjudicator_bot)
    cfg.selfplay.adjudication_max_turns_capitals = int(getattr(args, "adjudication_max_turns_capitals", cfg.selfplay.adjudication_max_turns_capitals))
    cfg.selfplay.adjudication_max_actions_per_game = int(getattr(args, "adjudication_max_actions_per_game", cfg.selfplay.adjudication_max_actions_per_game))
    cfg.selfplay.timeout_seconds = int(args.match_timeout_seconds)
    cfg.selfplay.external_action_timeout_ms = int(args.external_action_timeout_ms)
    cfg.selfplay.progress_interval_seconds = int(args.progress_interval_seconds)
    cfg.selfplay.wall_clock_per_action_seconds = float(args.wall_clock_per_action_seconds)
    if args.java_executable:
        cfg.selfplay.java_executable = str(args.java_executable)
    if args.java_classpath:
        cfg.selfplay.java_classpath = str(args.java_classpath)
    if args.java_main_class:
        cfg.selfplay.java_main_class = str(args.java_main_class)
    cfg.replay.replay_dir = args.output_dir / "replay"
    cfg.training.output_dir = args.output_dir
    cfg.training.checkpoint_path = args.checkpoint
    return cfg


def _game_summary(index: int, seed: int, elapsed_sec: float, returncode: int, replay_steps: int, stderr: str) -> dict[str, Any]:
    action_rows, search_rows, warmup_ms = _parse_profile(stderr)
    total_ms = [float(row["total_ms"]) for row in action_rows]
    encode_ms = [float(row["encode_ms"]) for row in action_rows]
    policy_ms = [float(row["policy_ms"]) for row in action_rows]
    search_ms = [float(row["search_ms"]) for row in action_rows]
    action_counts = [float(row["actions"]) for row in action_rows]
    configured_sims = [float(row["sims"]) for row in search_rows]
    paths = [float(row.get("paths", row["sims"])) for row in search_rows]
    nodes = [float(row.get("nodes", 0)) for row in search_rows]
    eval_positions = [float(row["eval_positions"]) for row in search_rows]
    eval_batches = [float(row["eval_batches"]) for row in search_rows]
    eval_cache_hits = [float(row.get("eval_cache_hits", 0)) for row in search_rows]
    eval_cache_sizes = [float(row.get("eval_cache_size", 0)) for row in search_rows]
    avg_depths = [float(row.get("avg_depth", 0.0)) for row in search_rows if float(row.get("paths", row["sims"])) > 0.0]
    max_depths = [int(row.get("selected_max_depth", 0)) for row in search_rows]
    root_entropies = [float(row.get("root_visit_entropy", 0.0)) for row in search_rows]
    top_visit_shares = [float(row.get("top_visit_share", 0.0)) for row in search_rows]
    root_actions = [float(row.get("root_actions", 0)) for row in search_rows]
    searched_actions = [float(row.get("searched_actions", 0)) for row in search_rows]
    native_unsupported = [int(row.get("native_unsupported_transition", 0)) for row in search_rows]
    native_invalid = [int(row.get("native_invalid_transition", 0)) for row in search_rows]
    native_approximate = [int(row.get("native_approximate_transition", 0)) for row in search_rows]
    total_configured_simulations = int(sum(configured_sims))
    total_selected_paths = int(sum(paths))
    total_simulations = int(sum(nodes))
    total_eval_positions = int(sum(eval_positions))
    total_eval_batches = int(sum(eval_batches))
    total_eval_cache_hits = int(sum(eval_cache_hits))
    fallback_count = len([line for line in stderr.splitlines() if FALLBACK_PATTERN.search(line)])
    return {
        "game": index,
        "seed": seed,
        "returncode": returncode,
        "elapsed_sec": elapsed_sec,
        "actions_profiled": len(action_rows),
        "replay_steps": replay_steps,
        "actions_per_sec": len(action_rows) / elapsed_sec if elapsed_sec > 0.0 else 0.0,
        "replay_steps_per_sec": replay_steps / elapsed_sec if elapsed_sec > 0.0 else 0.0,
        "selected_paths": total_selected_paths,
        "selected_paths_per_sec": total_selected_paths / elapsed_sec if elapsed_sec > 0.0 else 0.0,
        "simulations": total_simulations,
        "simulations_per_sec": total_simulations / elapsed_sec if elapsed_sec > 0.0 else 0.0,
        "configured_simulations": total_configured_simulations,
        "mean_legal_actions": statistics.fmean(action_counts) if action_counts else 0.0,
        "encode_mean_ms": _summarize(encode_ms)["mean"],
        "policy_mean_ms": _summarize(policy_ms)["mean"],
        "search_mean_ms": _summarize(search_ms)["mean"],
        "total_mean_ms": _summarize(total_ms)["mean"],
        "total_p95_ms": _summarize(total_ms)["p95"],
        "total_max_ms": _summarize(total_ms)["max"],
        "search_profile_rows": len(search_rows),
        "eval_positions": total_eval_positions,
        "eval_positions_per_sec": sum(eval_positions) / elapsed_sec if elapsed_sec > 0.0 else 0.0,
        "eval_batches": total_eval_batches,
        "avg_eval_batch": total_eval_positions / max(1, total_eval_batches),
        "eval_cache_hits": total_eval_cache_hits,
        "eval_cache_hit_rate": total_eval_cache_hits / max(1, total_eval_cache_hits + total_eval_positions),
        "eval_cache_size_max": int(max(eval_cache_sizes, default=0.0)),
        "avg_depth": statistics.fmean(avg_depths) if avg_depths else 0.0,
        "max_selected_depth": max(max_depths, default=0),
        "root_visit_entropy_mean": statistics.fmean(root_entropies) if root_entropies else 0.0,
        "top_visit_share_mean": statistics.fmean(top_visit_shares) if top_visit_shares else 0.0,
        "root_actions_mean": statistics.fmean(root_actions) if root_actions else 0.0,
        "searched_actions_mean": statistics.fmean(searched_actions) if searched_actions else 0.0,
        "native_unsupported_transition": sum(native_unsupported),
        "native_invalid_transition": sum(native_invalid),
        "native_approximate_transition": sum(native_approximate),
        "warmup_ms": statistics.fmean(warmup_ms) if warmup_ms else 0.0,
        "fallback_count": fallback_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile full Tribes self-play games using the MCTS neural-network bot."
    )
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=12345)
    parser.add_argument("--checkpoint", type=Path, default=Path("py/profiling/profiling_model.pt"))
    parser.add_argument("--device", default=None, help="Device metadata passed to the self-play wrapper. Defaults to cuda if available, otherwise cpu.")
    parser.add_argument("--output-dir", type=Path, default=Path("debug-logs/selfplay-mcts-nn-profile"))
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--tribes", nargs=2, default=["Xin Xi", "Imperius"])
    parser.add_argument("--run-mode", default="PlayLG")
    parser.add_argument("--game-mode", default="Capitals")
    parser.add_argument("--map-type", default="Continents")
    parser.add_argument("--map-size", default="Small")
    parser.add_argument("--simulations", type=int, default=192)
    parser.add_argument("--search-batch-size", type=int, default=64)
    parser.add_argument("--top-k-actions", type=int, default=64)
    parser.add_argument("--max-turns-capitals", type=int, default=20)
    parser.add_argument("--max-actions-per-turn", type=int, default=80)
    parser.add_argument("--max-actions-per-game", type=int, default=128)
    parser.add_argument("--adjudicate-incomplete-games", action="store_true")
    parser.add_argument("--adjudicator-bot", default="py/bots/simple_bot.py")
    parser.add_argument("--adjudication-max-turns-capitals", type=int, default=80)
    parser.add_argument("--adjudication-max-actions-per-game", type=int, default=2048)
    parser.add_argument("--match-timeout-seconds", type=int, default=600)
    parser.add_argument("--external-action-timeout-ms", type=int, default=120000)
    parser.add_argument("--progress-interval-seconds", type=int, default=30)
    parser.add_argument("--wall-clock-per-action-seconds", type=float, default=None)
    parser.add_argument("--wall-clock-per-turn-seconds", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--reuse-tree", action="store_true", help="Experimentally reuse/promote native MCTS subtrees during self-play profiling.")
    parser.add_argument("--persistent-bot", dest="persistent_bot", action="store_true", default=True, help="Use the same persistent shared bot server setup as RL training.")
    parser.add_argument("--no-persistent-bot", dest="persistent_bot", action="store_false", help="Launch one full bot process per player, matching the older non-persistent path.")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-dirichlet", action="store_true")
    parser.add_argument("--java-executable", default=None)
    parser.add_argument("--java-classpath", default=None)
    parser.add_argument("--java-main-class", default=None)
    parser.add_argument("--game-csv", type=Path, default=None)
    parser.add_argument("--action-csv", type=Path, default=None)
    parser.add_argument("--search-csv", type=Path, default=None)
    parser.add_argument("--function-csv", type=Path, default=None)
    parser.add_argument("--profile-stats", type=Path, default=None, help="Raw cProfile .prof output for the persistent bot process.")
    parser.add_argument("--function-limit", type=int, default=40)
    parser.add_argument("--min-hotspot-ms", type=float, default=1.0, help="Hide timing/function rows below this cumulative millisecond threshold.")
    parser.add_argument("--min-hotspot-pct", type=float, default=1.0, help="Hide timing/function rows below this percent-of-run threshold.")
    parser.add_argument("--jsonl", type=Path, default=None)
    args = load_config_defaults(parser, default_config=DEFAULT_SELFPLAY_MCTS_NN_CONFIG)
    if args.wall_clock_per_action_seconds is None:
        args.wall_clock_per_action_seconds = (
            args.wall_clock_per_turn_seconds
            if args.wall_clock_per_turn_seconds is not None
            else 2.0
        )

    workdir = Path(args.workdir).resolve() if args.workdir is not None else PY_ROOT.parent.resolve()
    args.output_dir = Path(args.output_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _configure(args)
    cfg.replay.replay_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    replay_store = ReplayStore(cfg.replay.replay_dir, cfg.replay.capacity_steps, cfg.replay.shard_prefix)

    command = _bot_command(args, cfg, workdir)
    bot_commands = [command, list(command)]
    game_rows: list[dict[str, Any]] = []
    all_action_rows: list[dict[str, Any]] = []
    all_search_rows: list[dict[str, Any]] = []
    profile = cProfile.Profile()
    benchmark_started_at = time.perf_counter()
    persistent_process = None
    persistent_host = "127.0.0.1"
    persistent_port = 0
    persistent_log_paths: list[Path] = []
    persistent_offsets: dict[Path, int] = {}
    bot_profile_path = args.profile_stats or (args.output_dir / "persistent_bot.prof")

    print(
        "Self-play MCTS NN profile: "
        f"games={args.games} node_budget={cfg.search.num_simulations} batch={cfg.search.batch_size} "
        f"map={cfg.selfplay.map_type}/{cfg.selfplay.map_size} max_turns={cfg.selfplay.max_turns_capitals} "
        f"persistent_bot={cfg.selfplay.persistent_bot} checkpoint={args.checkpoint}",
        flush=True,
    )

    try:
        if cfg.selfplay.persistent_bot:
            bot_server_script = workdir / "py" / "training" / "persistent_bot_server.py"
            bot_bridge_script = workdir / "py" / "training" / "persistent_bot_bridge.py"
            persistent_process, persistent_host, persistent_port, stdout_path, stderr_path = _start_profiled_persistent_bot_server(
                cfg,
                bot_server_script,
                args.checkpoint,
                cfg.replay.replay_dir,
                args.output_dir / "persistent_bot_logs",
                bot_profile_path,
            )
            persistent_log_paths = [stdout_path, stderr_path]
            persistent_offsets = {path: len(path.read_text(encoding="utf-8", errors="replace")) for path in persistent_log_paths if path.exists()}
            bridge = _persistent_bridge_command(bot_bridge_script, persistent_host, persistent_port)
            bot_commands = [bridge, list(bridge)]
            print(f"[persistent bot] started host={persistent_host} port={persistent_port}", flush=True)

        profile.enable()
        for index in range(1, max(1, int(args.games)) + 1):
            seed = int(args.seed_base) + index - 1
            cfg.selfplay.game_seed = seed
            cfg.selfplay.agent_seed = seed
            cfg.selfplay.level_seed = seed
            before_shards = set(ReplayStore(cfg.replay.replay_dir, 0, cfg.replay.shard_prefix).shards())
            started_at = time.perf_counter()
            print(f"[game {index}/{args.games}] seed={seed}", flush=True)
            result = run_selfplay(
                cfg,
                bot_commands,
                list(args.tribes),
                workdir,
                checkpoint_path=args.checkpoint,
                replay_store=replay_store,
                device=device,
                progress_label=f"profile-selfplay-game-{index}",
            )
            elapsed_sec = time.perf_counter() - started_at
            profile_text = result.stderr
            if cfg.selfplay.persistent_bot:
                profile_text = "\n".join([profile_text, _read_new_text(persistent_log_paths, persistent_offsets)])
            replay_steps = _new_replay_step_count(cfg.replay.replay_dir, cfg.replay.shard_prefix, before_shards)
            action_rows, search_rows, _warmup_ms = _parse_profile(profile_text)
            for row in action_rows:
                all_action_rows.append({"game": index, "seed": seed, **row})
            for row in search_rows:
                all_search_rows.append({"game": index, "seed": seed, **row})
            row = _game_summary(index, seed, elapsed_sec, int(result.returncode), replay_steps, profile_text)
            game_rows.append(row)
            print(
                "  "
                f"elapsed={_format_seconds(elapsed_sec)} returncode={result.returncode} "
                f"actions={row['actions_profiled']} actions/s={row['actions_per_sec']:.2f} "
                f"paths/s={row['selected_paths_per_sec']:.1f} "
                f"sims/s={row['simulations_per_sec']:.1f} "
                f"eval_pos/s={row['eval_positions_per_sec']:.1f} "
                f"avg_depth={row['avg_depth']:.2f} entropy={row['root_visit_entropy_mean']:.2f} "
                f"avg_ms encode={row['encode_mean_ms']:.1f} policy={row['policy_mean_ms']:.1f} "
                f"search={row['search_mean_ms']:.1f} total={row['total_mean_ms']:.1f} "
                f"p95_total={row['total_p95_ms']:.1f}",
                flush=True,
            )
            if result.returncode != 0:
                print(profile_text[-4000:], file=sys.stderr, flush=True)
                break
        profile.disable()
    finally:
        if profile.getstats():
            try:
                profile.disable()
            except RuntimeError:
                pass
        if persistent_process is not None:
            _shutdown_persistent_server(persistent_host, persistent_port, persistent_process)
            print("[persistent bot] stopped", flush=True)

    total_elapsed = time.perf_counter() - benchmark_started_at
    total_actions = sum(int(row["actions_profiled"]) for row in game_rows)
    total_selected_paths = sum(int(row["selected_paths"]) for row in game_rows)
    total_simulations = sum(int(row["simulations"]) for row in game_rows)
    total_replay_steps = sum(int(row["replay_steps"]) for row in game_rows)
    total_eval_positions = sum(int(row["eval_positions"]) for row in game_rows)
    total_eval_batches = sum(int(row["eval_batches"]) for row in game_rows)
    total_eval_cache_hits = sum(int(row["eval_cache_hits"]) for row in game_rows)
    total_native_unsupported = sum(int(row["native_unsupported_transition"]) for row in game_rows)
    total_native_invalid = sum(int(row["native_invalid_transition"]) for row in game_rows)
    total_native_approximate = sum(int(row["native_approximate_transition"]) for row in game_rows)
    print(
        "\nSelf-play MCTS NN profile: "
        f"games={len(game_rows)} mode=full_selfplay node_budget_per_action={cfg.search.num_simulations} "
        f"batch={cfg.search.batch_size} device={device} checkpoint={args.checkpoint} "
        f"persistent_bot={cfg.selfplay.persistent_bot} "
        f"map={cfg.selfplay.map_type}/{cfg.selfplay.map_size} "
        f"max_turns={cfg.selfplay.max_turns_capitals} total_ms={total_elapsed * 1000.0:.3f}"
    )

    print("\nPer-game throughput")
    display_rows = [
        {
            "game": row["game"],
            "seed": row["seed"],
            "sec": f"{row['elapsed_sec']:.2f}",
            "actions": row["actions_profiled"],
            "actions/s": f"{row['actions_per_sec']:.2f}",
            "sims/s": f"{row['simulations_per_sec']:.1f}",
            "paths/s": f"{row['selected_paths_per_sec']:.1f}",
            "eval_pos/s": f"{row['eval_positions_per_sec']:.1f}",
            "depth": f"{row['avg_depth']:.2f}",
            "max_depth": row["max_selected_depth"],
            "entropy": f"{row['root_visit_entropy_mean']:.2f}",
            "top_visit": f"{row['top_visit_share_mean'] * 100.0:.1f}%",
            "encode_ms": f"{row['encode_mean_ms']:.1f}",
            "policy_ms": f"{row['policy_mean_ms']:.1f}",
            "search_ms": f"{row['search_mean_ms']:.1f}",
            "total_ms": f"{row['total_mean_ms']:.1f}",
            "p95_ms": f"{row['total_p95_ms']:.1f}",
            "fallbacks": row["fallback_count"],
        }
        for row in game_rows
    ]
    print(
        _format_table(
            display_rows,
            [
                ("game", "#"),
                ("seed", "seed"),
                ("sec", "sec"),
                ("actions", "actions"),
                ("actions/s", "actions/s"),
                ("sims/s", "sims/s"),
                ("paths/s", "paths/s"),
                ("eval_pos/s", "eval_pos/s"),
                ("depth", "depth"),
                ("max_depth", "max_d"),
                ("entropy", "entropy"),
                ("top_visit", "top_visit"),
                ("encode_ms", "enc_ms"),
                ("policy_ms", "nn_ms"),
                ("search_ms", "mcts_ms"),
                ("total_ms", "total_ms"),
                ("p95_ms", "p95_ms"),
                ("fallbacks", "fallbacks"),
            ],
        )
    )

    print(
        "\nAggregate self-play work: "
        f"actions={total_actions} "
        f"actions_per_sec={_format_rate(total_actions, total_elapsed)} "
        f"replay_steps={total_replay_steps} "
        f"replay_steps_per_sec={_format_rate(total_replay_steps, total_elapsed)} "
        f"simulations={total_simulations} "
        f"simulations_per_sec={_format_rate(total_simulations, total_elapsed)} "
        f"selected_paths={total_selected_paths} "
        f"selected_paths_per_sec={_format_rate(total_selected_paths, total_elapsed)} "
        f"eval_positions={total_eval_positions} "
        f"eval_positions_per_sec={_format_rate(total_eval_positions, total_elapsed)} "
        f"eval_batches={total_eval_batches} "
        f"avg_eval_batch={total_eval_positions / max(1, total_eval_batches):.2f} "
        f"eval_cache_hits={total_eval_cache_hits} "
        f"eval_cache_hit_rate={total_eval_cache_hits / max(1, total_eval_cache_hits + total_eval_positions):.4f} "
        f"native_unsupported_transition={total_native_unsupported} "
        f"native_invalid_transition={total_native_invalid} "
        f"native_approximate_transition={total_native_approximate}"
    )

    if all_search_rows:
        depth_values = [float(row.get("avg_depth", 0.0)) for row in all_search_rows if int(row.get("paths", row.get("sims", 0))) > 0]
        root_entropy_values = [float(row.get("root_visit_entropy", 0.0)) for row in all_search_rows]
        top_visit_values = [float(row.get("top_visit_share", 0.0)) for row in all_search_rows]
        root_action_values = [float(row.get("root_actions", 0)) for row in all_search_rows]
        searched_action_values = [float(row.get("searched_actions", 0)) for row in all_search_rows]
        eval_batch_values = [
            float(row.get("eval_positions", 0)) / max(1.0, float(row.get("eval_batches", 0)))
            for row in all_search_rows
        ]
        search_shape_rows = [
            {
                "name": "selected_depth",
                "samples": len(depth_values),
                "mean": f"{_summarize(depth_values)['mean']:.3f}",
                "median": f"{_summarize(depth_values)['median']:.3f}",
                "p95": f"{_summarize(depth_values)['p95']:.3f}",
                "max": max((int(row.get("selected_max_depth", 0)) for row in all_search_rows), default=0),
            },
            {
                "name": "root_visit_entropy",
                "samples": len(root_entropy_values),
                "mean": f"{_summarize(root_entropy_values)['mean']:.3f}",
                "median": f"{_summarize(root_entropy_values)['median']:.3f}",
                "p95": f"{_summarize(root_entropy_values)['p95']:.3f}",
                "max": f"{max(root_entropy_values, default=0.0):.3f}",
            },
            {
                "name": "top_visit_share",
                "samples": len(top_visit_values),
                "mean": f"{_summarize(top_visit_values)['mean']:.3f}",
                "median": f"{_summarize(top_visit_values)['median']:.3f}",
                "p95": f"{_summarize(top_visit_values)['p95']:.3f}",
                "max": f"{max(top_visit_values, default=0.0):.3f}",
            },
            {
                "name": "root_actions",
                "samples": len(root_action_values),
                "mean": f"{_summarize(root_action_values)['mean']:.3f}",
                "median": f"{_summarize(root_action_values)['median']:.3f}",
                "p95": f"{_summarize(root_action_values)['p95']:.3f}",
                "max": f"{max(root_action_values, default=0.0):.0f}",
            },
            {
                "name": "searched_actions",
                "samples": len(searched_action_values),
                "mean": f"{_summarize(searched_action_values)['mean']:.3f}",
                "median": f"{_summarize(searched_action_values)['median']:.3f}",
                "p95": f"{_summarize(searched_action_values)['p95']:.3f}",
                "max": f"{max(searched_action_values, default=0.0):.0f}",
            },
            {
                "name": "eval_batch_size",
                "samples": len(eval_batch_values),
                "mean": f"{_summarize(eval_batch_values)['mean']:.3f}",
                "median": f"{_summarize(eval_batch_values)['median']:.3f}",
                "p95": f"{_summarize(eval_batch_values)['p95']:.3f}",
                "max": f"{max(eval_batch_values, default=0.0):.3f}",
            },
        ]
        print("\nSearch shape")
        print(_format_table(search_shape_rows, [("name", "name"), ("samples", "samples"), ("mean", "mean"), ("median", "median"), ("p95", "p95"), ("max", "max")]))

    action_timing_rows = []
    for key, label in [
        ("encode_ms", "encode"),
        ("policy_ms", "policy_forward"),
        ("search_ms", "mcts_search"),
        ("total_ms", "choose_action_total"),
    ]:
        values = [float(row[key]) for row in all_action_rows]
        summary = _summarize(values)
        action_timing_rows.append(
            {
                "name": label,
                "calls": int(summary["count"]),
                "mean_ms": f"{summary['mean']:.3f}",
                "median_ms": f"{summary['median']:.3f}",
                "p95_ms": f"{summary['p95']:.3f}",
                "max_ms": f"{summary['max']:.3f}",
                "pct_action_time": f"{(sum(values) / max(0.001, sum(float(row['total_ms']) for row in all_action_rows)) * 100.0):.1f}",
            }
        )
    print("\nAction timing breakdown")
    print(_format_table(action_timing_rows, [("name", "name"), ("calls", "calls"), ("mean_ms", "mean_ms"), ("median_ms", "median_ms"), ("p95_ms", "p95_ms"), ("max_ms", "max_ms"), ("pct_action_time", "% action")]))

    all_function_rows = _profile_rows(bot_profile_path, root=workdir)
    phase_rows = _phase_timing_rows(all_action_rows, all_search_rows, total_elapsed)
    hotspot_rows = _hotspot_rows(
        phase_rows,
        all_function_rows,
        total_sec=total_elapsed,
        min_ms=max(0.0, float(args.min_hotspot_ms)),
        min_pct=max(0.0, float(args.min_hotspot_pct)),
        limit=max(1, int(args.function_limit)),
    )
    print(
        "\nHotspots "
        f"(>= {float(args.min_hotspot_ms):.3g} ms and >= {float(args.min_hotspot_pct):.3g}% of run)"
    )
    if hotspot_rows:
        print(_format_table(hotspot_rows, [("source", "source"), ("name", "name"), ("calls", "calls"), ("items", "items"), ("time_ms", "time_ms"), ("self_ms", "self_ms"), ("pct", "%")]))
    else:
        print("No phase or function rows crossed the reporting threshold.")

    function_rows = all_function_rows[: max(1, int(args.function_limit))]
    project_function_rows = [
        row for row in all_function_rows
        if str(row.get("function", "")).startswith("py\\") or str(row.get("function", "")).startswith("py/")
    ][: max(1, int(args.function_limit))]

    args.game_csv = args.game_csv or args.output_dir / "selfplay_games.csv"
    args.action_csv = args.action_csv or args.output_dir / "action_timings.csv"
    args.search_csv = args.search_csv or args.output_dir / "search_timings.csv"
    args.function_csv = args.function_csv or args.output_dir / "function_profile.csv"
    args.jsonl = args.jsonl or args.output_dir / "profile_runs.jsonl"
    _write_csv(args.game_csv, game_rows)
    _write_csv(args.action_csv, all_action_rows)
    _write_csv(args.search_csv, all_search_rows)
    _write_csv(args.function_csv, all_function_rows)
    _write_jsonl(
        args.jsonl,
        [
            {
                "schema_version": 1,
                "elapsed_sec": total_elapsed,
                "config": _jsonable(cfg),
                "games": game_rows,
                "timing_summary": action_timing_rows,
                "phase_timing_summary": phase_rows,
                "function_profile_top": function_rows,
                "project_function_profile_top": project_function_rows,
                "profile_stats": str(bot_profile_path),
                "cprofile_total_calls": sum(int(getattr(stat, "callcount", 0)) for stat in profile.getstats()),
            }
        ],
    )
    print(
        "\nCSV: "
        f"games={args.game_csv} "
        f"actions={args.action_csv} "
        f"search={args.search_csv} "
        f"functions={args.function_csv} "
        f"profile={bot_profile_path} "
        f"jsonl={args.jsonl}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
