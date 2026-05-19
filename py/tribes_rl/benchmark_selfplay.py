from __future__ import annotations

import argparse
import json
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
from .device import move_optimizer_state, require_cuda_device
from .model import HybridPolicyValueNet
from .replay import ReplayStore
from .selfplay import run_selfplay
from .train import _load_checkpoint, train_round


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _fallback_summary(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    text = "\n".join([result.stdout or "", result.stderr or ""])
    lines = [line for line in text.splitlines() if FALLBACK_PATTERN.search(line)]
    debug_paths: list[str] = []
    for line in lines:
        marker = "debug="
        if marker in line:
            debug_paths.append(line.split(marker, 1)[1].strip().split()[0])
    return {
        "fallback_count": len(lines),
        "fallback_debug_paths": debug_paths,
        "fallback_log_tail": lines[-5:],
    }


def _bot_command(cfg: HybridAgentConfig, workdir: Path, checkpoint_path: Path, device: torch.device) -> list[str]:
    command = [
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
    return command


def _run_training_probe(cfg: HybridAgentConfig, device: torch.device) -> dict[str, float]:
    replay_store = ReplayStore(cfg.replay.replay_dir, cfg.replay.capacity_steps, cfg.replay.shard_prefix)
    model = HybridPolicyValueNet(cfg.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    _load_checkpoint(model, cfg.training.checkpoint_path, optimizer)
    move_optimizer_state(optimizer, device)
    started_at = time.perf_counter()
    metrics = train_round(cfg, model, optimizer, replay_store, device)
    metrics["train_sec"] = time.perf_counter() - started_at
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def _phase_a_matrix() -> Iterable[tuple[int, int]]:
    for simulations in (16, 32, 64, 128, 256):
        for batch_size in (8, 16, 32, 64, 128):
            yield simulations, batch_size


def run_candidate(
    base_cfg: HybridAgentConfig,
    *,
    simulations: int,
    search_batch_size: int,
    max_turns_capitals: int | None,
    training_batch_size: int | None,
    replay_batch_size: int | None,
    epochs_per_iteration: int | None,
    games: int,
    seed_base: int,
    run_index: int,
    output_dir: Path,
    device: torch.device,
    include_training: bool,
) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    cfg.search.num_simulations = int(simulations)
    cfg.search.batch_size = int(search_batch_size)
    cfg.training.device = str(device)
    if max_turns_capitals is not None:
        cfg.selfplay.max_turns_capitals = int(max_turns_capitals)
    if training_batch_size is not None:
        cfg.training.batch_size = int(training_batch_size)
    if replay_batch_size is not None:
        cfg.training.replay_batch_size = int(replay_batch_size)
    if epochs_per_iteration is not None:
        cfg.training.epochs_per_iteration = int(epochs_per_iteration)

    run_label = (
        f"run_{run_index:04d}_s{cfg.search.num_simulations}_sb{cfg.search.batch_size}"
        f"_mt{cfg.selfplay.max_turns_capitals}_tb{cfg.training.batch_size}"
        f"_rb{cfg.training.replay_batch_size}_e{cfg.training.epochs_per_iteration}_native"
    )
    cfg.replay.replay_dir = output_dir / "replay" / run_label
    cfg.training.output_dir = output_dir / "trainer_artifacts" / run_label
    cfg.training.checkpoint_dir = cfg.training.output_dir / "checkpoints"
    cfg.diagnostics.metrics_csv = cfg.training.output_dir / "metrics.csv"
    cfg.replay.replay_dir.mkdir(parents=True, exist_ok=True)
    cfg.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    workdir = Path(__file__).resolve().parents[2]
    tribes = ["Xin Xi", "Imperius"]
    command = _bot_command(cfg, workdir, cfg.training.checkpoint_path, device)
    started = time.perf_counter()
    start_wall = _utc_now()
    game_seconds: list[float] = []
    replay_steps_per_game: list[int] = []
    fallback_count = 0
    fallback_matches = 0
    fallback_debug_paths: list[str] = []
    returncodes: list[int] = []

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
            progress_label=f"bench run={run_index} game={game_idx + 1}/{games}",
        )
        elapsed = time.perf_counter() - game_started
        returncodes.append(int(result.returncode))
        new_steps = _new_replay_step_counts(cfg.replay.replay_dir, cfg.replay.shard_prefix, before_shards)
        replay_steps_per_game.append(sum(new_steps))
        game_seconds.append(elapsed)
        fallback = _fallback_summary(result)
        fallback_count += int(fallback["fallback_count"])
        if fallback["fallback_count"]:
            fallback_matches += 1
            fallback_debug_paths.extend(fallback["fallback_debug_paths"])
        if result.returncode != 0:
            break

    selfplay_sec = time.perf_counter() - started
    replay_shards, replay_steps = _count_replay_steps(cfg.replay.replay_dir, cfg.replay.shard_prefix)
    completed_games = len(game_seconds)
    chosen_actions = replay_steps
    total_simulations = chosen_actions * int(cfg.search.num_simulations)
    train_metrics = _run_training_probe(cfg, device) if include_training else {}
    seconds_game = selfplay_sec / completed_games if completed_games else 0.0
    replay_steps_sec = replay_steps / selfplay_sec if selfplay_sec > 0 else 0.0

    return {
        "schema_version": 1,
        "run_index": run_index,
        "start_wall_utc": start_wall,
        "end_wall_utc": _utc_now(),
        "selfplay_sec": selfplay_sec,
        "games_requested": int(games),
        "games_completed": completed_games,
        "returncodes": returncodes,
        "seconds_game": seconds_game,
        "games_sec": completed_games / selfplay_sec if selfplay_sec > 0 else 0.0,
        "game_seconds_mean": statistics.fmean(game_seconds) if game_seconds else 0.0,
        "game_seconds_median": statistics.median(game_seconds) if game_seconds else 0.0,
        "total_configured_simulations_per_action": int(cfg.search.num_simulations),
        "estimated_simulations_sec": total_simulations / selfplay_sec if selfplay_sec > 0 else 0.0,
        "total_replay_shards": replay_shards,
        "total_replay_steps": replay_steps,
        "mean_replay_steps_game": replay_steps / completed_games if completed_games else 0.0,
        "median_replay_steps_game": statistics.median(replay_steps_per_game) if replay_steps_per_game else 0.0,
        "mean_chosen_actions_game": chosen_actions / completed_games if completed_games else 0.0,
        "replay_steps_sec": replay_steps_sec,
        "fallback_count": fallback_count,
        "fallback_matches": fallback_matches,
        "fallback_rate_per_decision": fallback_count / chosen_actions if chosen_actions else 0.0,
        "fallback_debug_paths": fallback_debug_paths,
        "train_metrics": train_metrics,
        "config": _jsonable(cfg),
        "seeds": {
            "seed_base": int(seed_base),
            "game_seeds": [seed_base + run_index * 10_000 + game_idx for game_idx in range(completed_games)],
        },
    }


def _build_matrix(args: argparse.Namespace, cfg: HybridAgentConfig) -> list[dict[str, Any]]:
    if args.phase == "a":
        return [
            {"simulations": simulations, "search_batch_size": batch_size}
            for simulations, batch_size in _phase_a_matrix()
        ]
    if args.phase == "b":
        return [
            {"simulations": args.best_simulations, "search_batch_size": args.best_search_batch_size, "max_turns_capitals": max_turns}
            for max_turns in (40, 80, 120)
        ]
    if args.phase == "c":
        return [
            {
                "simulations": args.best_simulations,
                "search_batch_size": args.best_search_batch_size,
                "training_batch_size": train_batch,
                "replay_batch_size": replay_batch,
                "epochs_per_iteration": epochs,
            }
            for train_batch in (32, 64, 128)
            for replay_batch in (128, 256, 512)
            for epochs in (1, 2)
        ]
    return [
        {"simulations": simulations, "search_batch_size": batch_size}
        for simulations in args.simulations
        for batch_size in args.search_batch_sizes
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run structured Tribes RL self-play benchmark sweeps.")
    parser.add_argument("--phase", choices=["a", "b", "c", "custom"], default="custom")
    parser.add_argument("--simulations", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--search-batch-sizes", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--best-simulations", type=int, default=64)
    parser.add_argument("--best-search-batch-size", type=int, default=32)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path, default=Path("rl/benchmarks"))
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--include-training", action="store_true")
    parser.add_argument("--max-turns-capitals", type=int, default=None)
    parser.add_argument("--match-timeout-seconds", type=int, default=None)
    args = parser.parse_args()

    cfg = HybridAgentConfig()
    if args.max_turns_capitals is not None:
        cfg.selfplay.max_turns_capitals = args.max_turns_capitals
    if args.match_timeout_seconds is not None:
        cfg.selfplay.timeout_seconds = args.match_timeout_seconds

    device = require_cuda_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.jsonl or args.output_dir / f"selfplay_benchmark_{int(time.time())}.jsonl"
    matrix = _build_matrix(args, cfg)

    with jsonl_path.open("a", encoding="utf-8") as handle:
        for run_index, candidate in enumerate(matrix):
            row = run_candidate(
                cfg,
                simulations=int(candidate["simulations"]),
                search_batch_size=int(candidate["search_batch_size"]),
                max_turns_capitals=candidate.get("max_turns_capitals"),
                training_batch_size=candidate.get("training_batch_size"),
                replay_batch_size=candidate.get("replay_batch_size"),
                epochs_per_iteration=candidate.get("epochs_per_iteration"),
                games=args.games,
                seed_base=args.seed_base,
                run_index=run_index,
                output_dir=args.output_dir,
                device=device,
                include_training=args.include_training,
            )
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            print(
                "[bench] "
                f"run={run_index} sims={row['config']['search']['num_simulations']} "
                f"search_batch={row['config']['search']['batch_size']} "
                f"max_turns={row['config']['selfplay']['max_turns_capitals']} "
                "backend=native "
                f"games={row['games_completed']} seconds_game={row['seconds_game']:.2f} "
                f"replay_steps_sec={row['replay_steps_sec']:.2f} fallback_count={row['fallback_count']} "
                f"jsonl={jsonl_path}",
                flush=True,
            )


if __name__ == "__main__":
    main()
