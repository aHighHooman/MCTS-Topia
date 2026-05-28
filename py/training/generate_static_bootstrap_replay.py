from __future__ import annotations

import argparse
import copy
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from .config import HybridAgentConfig, rl_path
from .replay import ReplayStore
from .selfplay import run_selfplay
from .train import (
    _append_metrics,
    _archive_replay_shards,
    _format_action_counts,
    _format_seconds,
    _load_records_from_shards,
    _match_end_reason,
    _summarize_replay_shards,
    _write_augmented_iteration_shard,
)


def _parse_tribes(raw: str) -> list[str]:
    tribes = [part.strip() for part in raw.split(",") if part.strip()]
    if len(tribes) < 2:
        raise argparse.ArgumentTypeError("Expected at least two comma-separated tribe names, e.g. 'Xin Xi,Imperius'.")
    return tribes


def _delete_replay_shards(shards: list[Path]) -> int:
    deleted = 0
    for shard in shards:
        try:
            if shard.exists():
                shard.unlink()
                deleted += 1
        except OSError as exc:
            print(f"[replay] warning: failed to delete raw shard {shard}: {exc}", flush=True)
    return deleted


def _static_bot_command(
    bot_script: Path,
    cfg: HybridAgentConfig,
    *,
    checkpoint_path: Path,
    seed: int,
    player_offset: int,
    deterministic: bool,
    static_eval_variant: str | None,
) -> list[str]:
    command = [
        "python",
        str(bot_script),
        "--checkpoint",
        str(checkpoint_path),
        "--replay-dir",
        str(cfg.replay.replay_dir),
        "--static-only-hybrid-nn",
        "--simulations",
        str(int(cfg.search.num_simulations)),
        "--max-depth",
        str(int(cfg.search.max_depth)),
        "--top-k-actions",
        str(int(cfg.search.top_k_actions)),
        "--search-batch-size",
        str(int(cfg.search.batch_size)),
        "--max-actions",
        str(int(cfg.model.max_actions)),
        "--seed",
        str(int(seed) + int(player_offset)),
    ]
    if cfg.selfplay.wall_clock_per_action_seconds is not None:
        command.extend(
            [
                "--wall-clock-per-action-seconds",
                str(max(0.0, float(cfg.selfplay.wall_clock_per_action_seconds))),
            ]
        )
    if deterministic:
        command.append("--deterministic")
    if static_eval_variant:
        command.extend(["--static-eval-variant", static_eval_variant])
    return command


def _ensure_successful_static_selfplay(
    result: subprocess.CompletedProcess[str],
    before_shards: set[Path],
    after_shards: set[Path],
    label: str,
) -> None:
    if result.returncode != 0:
        raise RuntimeError(
            f"Static self-play failed ({label}).\n"
            f"Return code: {result.returncode}\n"
            f"Stdout tail:\n{chr(10).join(result.stdout.splitlines()[-20:])}\n"
            f"Stderr tail:\n{chr(10).join(result.stderr.splitlines()[-20:])}"
        )
    if not (after_shards - before_shards):
        raise RuntimeError(
            f"Static self-play completed but produced no replay shard ({label}).\n"
            "This usually means the self-play runner is not connected to ReplayStore for this bot command.\n"
            f"Stdout tail:\n{chr(10).join(result.stdout.splitlines()[-20:])}\n"
            f"Stderr tail:\n{chr(10).join(result.stderr.splitlines()[-20:])}"
        )


def _append_bootstrap_game_row(path: Path, row: dict[str, Any]) -> None:
    _append_metrics(path, row)


def generate_static_bootstrap_replay(
    cfg: HybridAgentConfig,
    *,
    games: int,
    tribes: list[str],
    workdir: Path,
    static_bot_script: Path,
    checkpoint_path: Path,
    device: torch.device,
    start_seed: int,
    deterministic: bool,
    static_eval_variant: str | None,
    augment_symmetries: bool,
    delete_raw: bool,
    archive_raw: bool,
    augmented_iteration: int,
    max_records_per_augmented_shard: int,
    metrics_csv: Path | None,
) -> None:
    if delete_raw and archive_raw:
        raise ValueError("Use either delete_raw or archive_raw, not both.")
    if delete_raw and not augment_symmetries:
        raise ValueError("delete_raw would delete the only replay copy. Enable augment_symmetries first.")
    if archive_raw and not augment_symmetries:
        raise ValueError("archive_raw is only meaningful when augment_symmetries is enabled.")
    if not static_bot_script.exists():
        raise FileNotFoundError(f"Static bot script not found: {static_bot_script}")

    cfg.replay.replay_dir.mkdir(parents=True, exist_ok=True)
    replay_store = ReplayStore(
        cfg.replay.replay_dir,
        cfg.replay.capacity_steps,
        cfg.replay.shard_prefix,
        cache_records=False,
    )
    before_all_shards = set(replay_store.shards())
    generated_raw_shards: list[Path] = []
    total_elapsed = 0.0
    total_steps = 0
    total_decisive = 0
    total_draws = 0
    total_actions: Counter[str] = Counter()

    print(
        f"[static-bootstrap] start games={games} replay_dir={cfg.replay.replay_dir} "
        f"bot={static_bot_script} sims={cfg.search.num_simulations} "
        f"search_batch={cfg.search.batch_size} augment_symmetries={augment_symmetries} "
        f"delete_raw={delete_raw} archive_raw={archive_raw}",
        flush=True,
    )

    for game_idx in range(games):
        seed = int(start_seed) + game_idx
        label = f"static-bootstrap g{game_idx + 1}/{games} seed={seed}"
        before_game_shards = set(replay_store.shards())
        commands = [
            _static_bot_command(
                static_bot_script,
                cfg,
                checkpoint_path=checkpoint_path,
                seed=seed,
                player_offset=0,
                deterministic=deterministic,
                static_eval_variant=static_eval_variant,
            ),
            _static_bot_command(
                static_bot_script,
                cfg,
                checkpoint_path=checkpoint_path,
                seed=seed,
                player_offset=1,
                deterministic=deterministic,
                static_eval_variant=static_eval_variant,
            ),
        ]
        game_cfg = copy.deepcopy(cfg)
        game_cfg.selfplay.game_seed = seed
        game_cfg.selfplay.agent_seed = seed
        game_cfg.selfplay.level_seed = seed

        print(f"[game {game_idx + 1}/{games}] seed={seed} start shards={len(before_game_shards)}", flush=True)
        started_at = time.monotonic()
        result = run_selfplay(
            game_cfg,
            commands,
            tribes,
            workdir,
            checkpoint_path=checkpoint_path,
            replay_store=ReplayStore(
                cfg.replay.replay_dir,
                cfg.replay.capacity_steps,
                cfg.replay.shard_prefix,
                load_existing=False,
            ),
            device=device,
            progress_label=label,
        )
        elapsed = time.monotonic() - started_at
        total_elapsed += elapsed
        replay_store.refresh()
        after_game_shards = set(replay_store.shards())
        _ensure_successful_static_selfplay(result, before_game_shards, after_game_shards, label)

        new_shards = sorted(after_game_shards - before_game_shards)
        generated_raw_shards.extend(new_shards)

        summary = _summarize_replay_shards(replay_store, new_shards)
        total_steps += int(summary.get("steps", 0) or 0)
        total_decisive += int(summary.get("decisive", 0) or 0)
        total_draws += int(summary.get("draws", 0) or 0)

        actions = summary.get("actions", {})
        if isinstance(actions, dict):
            total_actions.update({str(key): int(value) for key, value in actions.items()})

        print(
            f"[game {game_idx + 1}/{games}] done time={_format_seconds(elapsed)} "
            f"steps={summary.get('steps', 0)} shards=+{len(new_shards)} "
            f"ended={_match_end_reason(result.stdout, result.stderr)} "
            f"outcomes decisive/draw={summary.get('decisive', 0)}/{summary.get('draws', 0)} "
            f"actions {_format_action_counts(summary)}",
            flush=True,
        )

        if metrics_csv is not None:
            _append_bootstrap_game_row(
                metrics_csv,
                {
                    "game": game_idx + 1,
                    "seed": seed,
                    "seconds": elapsed,
                    "steps": int(summary.get("steps", 0) or 0),
                    "shards": len(new_shards),
                    "decisive": int(summary.get("decisive", 0) or 0),
                    "draws": int(summary.get("draws", 0) or 0),
                    "end_reason": _match_end_reason(result.stdout, result.stderr),
                },
            )

    replay_store.refresh()
    generated_raw_shards = sorted(set(generated_raw_shards) - before_all_shards)
    if not generated_raw_shards:
        raise RuntimeError("No raw replay shards were generated.")

    raw_records = len(_load_records_from_shards(replay_store, generated_raw_shards))
    augmented_shards: list[Path] = []
    deleted_raw = 0
    archived_raw = 0

    if augment_symmetries:
        before_aug_shards = set(replay_store.shards())
        _, returned_records, raw_record_count = _write_augmented_iteration_shard(
            replay_store,
            generated_raw_shards,
            cfg,
            int(augmented_iteration),
            max_return_records=0,
            max_records_per_shard=max(1, int(max_records_per_augmented_shard)),
        )
        _ = returned_records

        if delete_raw:
            deleted_raw = _delete_replay_shards(generated_raw_shards)
        elif archive_raw:
            archive_dir = cfg.replay.replay_dir / "raw" / "static_bootstrap"
            archived_raw = len(_archive_replay_shards(generated_raw_shards, archive_dir))

        replay_store.refresh()
        augmented_shards = sorted(
            shard for shard in set(replay_store.shards()) - before_aug_shards
            if shard.exists()
        )
        print(
            f"[replay] augmentation complete raw_shards={len(generated_raw_shards)} "
            f"raw_records={raw_record_count} augmented_shards={len(augmented_shards)} "
            f"deleted_raw_shards={deleted_raw} archived_raw_shards={archived_raw}",
            flush=True,
        )

    kept_raw = sum(1 for shard in generated_raw_shards if shard.exists())
    final_shards = sorted(set(replay_store.shards()) - before_all_shards)
    print(
        f"[static-bootstrap] done games={games} total_time={_format_seconds(total_elapsed)} "
        f"raw_records={raw_records} raw_shards={len(generated_raw_shards)} kept_raw_shards={kept_raw} "
        f"augmented_shards={len(augmented_shards)} final_new_shards={len(final_shards)} "
        f"outcomes decisive/draw={total_decisive}/{total_draws} "
        f"actions {_format_action_counts({'actions': dict(total_actions)})}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate bootstrap replay by running static-only HybridRLBot against itself."
    )
    parser.add_argument("--games", type=int, required=True, help="Number of static-vs-static self-play games to generate.")
    parser.add_argument("--replay-dir", type=Path, default=rl_path("replay_static_bootstrap"))
    parser.add_argument("--shard-prefix", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=rl_path("checkpoints", "static_bootstrap_dummy.pt"))
    parser.add_argument("--static-bot-script", type=Path, default=None)
    parser.add_argument("--tribes", type=_parse_tribes, default=_parse_tribes("Xin Xi,Imperius"))
    parser.add_argument("--device", type=str, default="cpu", help="Device passed through to the self-play runner.")
    parser.add_argument("--start-seed", type=int, default=10_000_000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--static-eval-variant", type=str, default=None)

    parser.add_argument("--simulations", "--mcts-sims", dest="simulations", type=int, default=256)
    parser.add_argument("--max-depth", type=int, default=0)
    parser.add_argument("--top-k-actions", type=int, default=64)
    parser.add_argument("--search-batch-size", type=int, default=64)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--game-mode", type=str, default=None)
    parser.add_argument("--max-turns-capitals", type=int, default=None)
    parser.add_argument("--max-actions-per-turn", type=int, default=None)
    parser.add_argument("--max-actions-per-game", type=int, default=None)
    parser.add_argument("--match-timeout-seconds", type=int, default=None)
    parser.add_argument("--external-action-timeout-ms", type=int, default=None)
    parser.add_argument("--wall-clock-per-action-seconds", type=float, default=None)
    parser.add_argument("--progress-interval-seconds", type=int, default=None)
    parser.add_argument("--adjudicate-incomplete-games", action="store_true")
    parser.add_argument("--adjudicator-bot", type=str, default=None)
    parser.add_argument("--adjudication-max-turns-capitals", type=int, default=None)
    parser.add_argument("--adjudication-max-actions-per-game", type=int, default=None)

    parser.add_argument("--augment-symmetries", action="store_true", help="Write D4 augmented replay shards after generation.")
    parser.add_argument("--delete-raw", action="store_true", help="Delete raw static replay shards after augmented shards are written.")
    parser.add_argument("--archive-raw", action="store_true", help="Move raw static replay shards to replay_dir/raw/static_bootstrap after augmentation.")
    parser.add_argument("--augmented-iteration", type=int, default=0)
    parser.add_argument("--max-records-per-augmented-shard", type=int, default=10_000)
    parser.add_argument("--metrics-csv", type=Path, default=None)
    args = parser.parse_args()

    if args.games <= 0:
        raise SystemExit("--games must be positive.")

    cfg = HybridAgentConfig()
    cfg.replay.replay_dir = args.replay_dir.resolve()
    if args.shard_prefix is not None:
        cfg.replay.shard_prefix = args.shard_prefix

    cfg.search.num_simulations = int(args.simulations)
    cfg.search.max_depth = int(args.max_depth)
    cfg.search.top_k_actions = int(args.top_k_actions)
    cfg.search.batch_size = int(args.search_batch_size)

    if args.max_actions is not None:
        cfg.model.max_actions = int(args.max_actions)
    if args.game_mode is not None:
        cfg.selfplay.game_mode = args.game_mode
    if args.max_turns_capitals is not None:
        cfg.selfplay.max_turns_capitals = int(args.max_turns_capitals)
    if args.max_actions_per_turn is not None:
        cfg.selfplay.max_actions_per_turn = int(args.max_actions_per_turn)
    if args.max_actions_per_game is not None:
        cfg.selfplay.max_actions_per_game = int(args.max_actions_per_game)
    if args.match_timeout_seconds is not None:
        cfg.selfplay.timeout_seconds = int(args.match_timeout_seconds)
    if args.external_action_timeout_ms is not None:
        cfg.selfplay.external_action_timeout_ms = int(args.external_action_timeout_ms)
    if args.wall_clock_per_action_seconds is not None:
        cfg.selfplay.wall_clock_per_action_seconds = float(args.wall_clock_per_action_seconds)
    if args.progress_interval_seconds is not None:
        cfg.selfplay.progress_interval_seconds = int(args.progress_interval_seconds)
    if args.adjudicate_incomplete_games:
        cfg.selfplay.adjudicate_incomplete_games = True
    if args.adjudicator_bot is not None:
        cfg.selfplay.adjudicator_bot = args.adjudicator_bot
    if args.adjudication_max_turns_capitals is not None:
        cfg.selfplay.adjudication_max_turns_capitals = int(args.adjudication_max_turns_capitals)
    if args.adjudication_max_actions_per_game is not None:
        cfg.selfplay.adjudication_max_actions_per_game = int(args.adjudication_max_actions_per_game)

    workdir = Path(__file__).resolve().parents[2]

    # Important: this defaults to the replay-capable hybrid bot, not native_static_mcts_bot.py.
    static_bot_script = args.static_bot_script or (workdir / "py" / "bots" / "hybrid_nn_bot.py")

    checkpoint_path = args.checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.exists():
        # Static-only HybridRLBot does not load this checkpoint, but run_selfplay receives a path.
        torch.save({"kind": "static_bootstrap_placeholder", "created_at": time.time()}, checkpoint_path)

    metrics_csv = args.metrics_csv
    if metrics_csv is None:
        metrics_csv = args.replay_dir / "static_bootstrap_games.csv"

    generate_static_bootstrap_replay(
        cfg,
        games=int(args.games),
        tribes=args.tribes,
        workdir=workdir,
        static_bot_script=static_bot_script,
        checkpoint_path=checkpoint_path,
        device=torch.device(args.device),
        start_seed=int(args.start_seed),
        deterministic=bool(args.deterministic),
        static_eval_variant=args.static_eval_variant,
        augment_symmetries=bool(args.augment_symmetries),
        delete_raw=bool(args.delete_raw),
        archive_raw=bool(args.archive_raw),
        augmented_iteration=int(args.augmented_iteration),
        max_records_per_augmented_shard=int(args.max_records_per_augmented_shard),
        metrics_csv=metrics_csv,
    )


if __name__ == "__main__":
    main()
