from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from training.config import HybridAgentConfig, rl_path
from .java_selfplay import run_selfplay_match
from .replay import ReplayStore


ITERATION_RE = re.compile(r"(?:^|_)iter_(\d+)(?:\.|$)", re.IGNORECASE)


@dataclass(frozen=True)
class Competitor:
    name: str
    checkpoint: Path
    iteration: int | None


def _checkpoint_iteration(path: Path) -> int | None:
    match = ITERATION_RE.search(path.name)
    if match:
        return int(match.group(1))
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        value = payload.get("iteration") if isinstance(payload, dict) else None
        return int(value) if value is not None else None
    except Exception:
        return None


def discover_checkpoints(checkpoint_dir: Path, pattern: str, limit: int | None) -> list[Competitor]:
    paths = sorted(path for path in checkpoint_dir.glob(pattern) if path.is_file())
    competitors = [
        Competitor(path.stem, path, _checkpoint_iteration(path))
        for path in paths
    ]
    competitors.sort(key=lambda item: (item.iteration is None, item.iteration if item.iteration is not None else item.name))
    if limit is not None and limit > 0:
        competitors = competitors[-limit:]
    if len(competitors) < 2:
        raise ValueError(f"Need at least two checkpoints in {checkpoint_dir} matching {pattern!r}.")
    return competitors


def _bot_command(
    workdir: Path,
    checkpoint: Path,
    replay_dir: Path,
    cfg: HybridAgentConfig,
    *,
    deterministic: bool,
) -> list[str]:
    command = [
        "python",
        str(workdir / "py" / "bots" / "hybrid_nn_bot.py"),
        "--checkpoint",
        str(checkpoint),
        "--replay-dir",
        str(replay_dir),
        "--simulations",
        str(cfg.search.num_simulations),
        "--top-k-actions",
        str(cfg.search.top_k_actions),
        "--search-batch-size",
        str(cfg.search.batch_size),
        "--max-game-actions",
        str(cfg.selfplay.max_actions_per_game),
    ]
    if deterministic:
        command.append("--deterministic")
    return command


def _latest_episode_outcome(replay_dir: Path, prefix: str) -> dict[str, Any]:
    store = ReplayStore(replay_dir, capacity_steps=0, shard_prefix=prefix)
    latest: tuple[float, Path] | None = None
    for shard in store.shards():
        stamp = shard.stat().st_mtime
        if latest is None or stamp > latest[0]:
            latest = (stamp, shard)
    if latest is None:
        return {"winner_id": None, "final_scores": None, "ranking": None}
    _, records = store._load_shard(latest[1])
    if not records:
        return {"winner_id": None, "final_scores": None, "ranking": None}
    return records[-1].outcome or {"winner_id": None, "final_scores": None, "ranking": None}


def _score_for_player(outcome: dict[str, Any], player_id: int) -> float:
    winner_id = outcome.get("winner_id")
    if winner_id is None:
        return 0.5
    return 1.0 if int(winner_id) == int(player_id) else 0.0


def _expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (rb - ra) / 400.0))


def _update_elo(elos: dict[str, float], a: str, b: str, score_a: float, k: float) -> None:
    ea = _expected_score(elos[a], elos[b])
    delta = k * (score_a - ea)
    elos[a] += delta
    elos[b] -= delta


def _pairings(count: int, games_per_pair: int, include_seat_swaps: bool) -> Iterable[tuple[int, int, int]]:
    for i in range(count):
        for j in range(i + 1, count):
            for game in range(games_per_pair):
                yield i, j, game
                if include_seat_swaps:
                    yield j, i, game


def _latest_vs_history_pairings(
    competitors: list[Competitor],
    games_per_pair: int,
    include_seat_swaps: bool,
    history_step: int,
) -> Iterable[tuple[int, int, int]]:
    latest_idx = len(competitors) - 1
    latest = competitors[latest_idx]
    if latest.iteration is None:
        raise ValueError("Latest-vs-history mode requires iteration numbers in checkpoint names or payloads.")
    step = max(1, int(history_step))
    history_indexes = [
        idx for idx, competitor in enumerate(competitors[:-1])
        if competitor.iteration is not None and competitor.iteration % step == 0
    ]
    if not history_indexes:
        raise ValueError(f"No historical checkpoints have iterations divisible by {step}.")
    for opponent_idx in reversed(history_indexes):
        for game in range(games_per_pair):
            yield latest_idx, opponent_idx, game
            if include_seat_swaps:
                yield opponent_idx, latest_idx, game


def _write_summary_csv(path: Path, competitors: list[Competitor], elos: dict[str, float], records: list[dict[str, Any]]) -> None:
    rows = []
    for competitor in competitors:
        games = [record for record in records if competitor.name in (record["seat0"], record["seat1"])]
        scores = []
        for record in games:
            if record["seat0"] == competitor.name:
                scores.append(float(record["score0"]))
            else:
                scores.append(1.0 - float(record["score0"]))
        rows.append({
            "name": competitor.name,
            "iteration": "" if competitor.iteration is None else competitor.iteration,
            "checkpoint": str(competitor.checkpoint),
            "elo": round(elos[competitor.name], 2),
            "games": len(games),
            "score": round(sum(scores), 3),
            "score_rate": round(statistics.fmean(scores), 4) if scores else 0.0,
        })
    rows.sort(key=lambda row: float(row["elo"]), reverse=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "iteration", "checkpoint", "elo", "games", "score", "score_rate"])
        writer.writeheader()
        writer.writerows(rows)


def run_tournament(args: argparse.Namespace) -> tuple[Path, Path, dict[str, float]]:
    workdir = Path(__file__).resolve().parents[2]
    competitors = discover_checkpoints(args.checkpoint_dir, args.pattern, args.limit)
    cfg = HybridAgentConfig()
    cfg.training.output_dir = args.output_dir
    cfg.search.num_simulations = args.simulations
    cfg.search.batch_size = args.search_batch_size
    cfg.search.top_k_actions = args.top_k_actions
    cfg.selfplay.max_turns_capitals = args.max_turns_capitals
    cfg.selfplay.max_actions_per_turn = args.max_actions_per_turn
    cfg.selfplay.max_actions_per_game = args.max_actions_per_game
    cfg.selfplay.timeout_seconds = args.match_timeout_seconds
    cfg.selfplay.external_action_timeout_ms = args.external_action_timeout_ms
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.jsonl or args.output_dir / f"checkpoint_tournament_{int(time.time())}.jsonl"
    csv_path = args.csv or args.output_dir / "checkpoint_tournament_summary.csv"
    elos = {competitor.name: float(args.initial_elo) for competitor in competitors}
    records: list[dict[str, Any]] = []
    tribes = ["Xin Xi", "Imperius"]

    with jsonl_path.open("a", encoding="utf-8") as handle:
        pairings = (
            _latest_vs_history_pairings(competitors, args.games_per_pair, args.seat_swaps, args.history_step)
            if args.latest_vs_history
            else _pairings(len(competitors), args.games_per_pair, args.seat_swaps)
        )
        for match_index, (seat0_idx, seat1_idx, game_idx) in enumerate(pairings):
            seat0 = competitors[seat0_idx]
            seat1 = competitors[seat1_idx]
            seed = args.seed_base + match_index
            cfg.selfplay.game_seed = seed
            cfg.selfplay.agent_seed = seed
            cfg.selfplay.level_seed = seed
            match_dir = args.output_dir / "matches" / f"match_{match_index:05d}"
            replay0 = match_dir / "seat0"
            replay1 = match_dir / "seat1"
            replay0.mkdir(parents=True, exist_ok=True)
            replay1.mkdir(parents=True, exist_ok=True)
            commands = [
                _bot_command(workdir, seat0.checkpoint, replay0, cfg, deterministic=args.deterministic),
                _bot_command(workdir, seat1.checkpoint, replay1, cfg, deterministic=args.deterministic),
            ]
            started_at = time.perf_counter()
            result = run_selfplay_match(
                cfg,
                commands,
                tribes,
                workdir,
                progress_label=f"tournament match={match_index} {seat0.name} vs {seat1.name}",
            )
            elapsed = time.perf_counter() - started_at
            outcome = _latest_episode_outcome(replay0, cfg.replay.shard_prefix)
            score0 = _score_for_player(outcome, 0)
            if result.returncode == 0:
                _update_elo(elos, seat0.name, seat1.name, score0, args.k_factor)
            row = {
                "match_index": match_index,
                "game_index": game_idx,
                "seed": seed,
                "seat0": seat0.name,
                "seat1": seat1.name,
                "seat0_iteration": seat0.iteration,
                "seat1_iteration": seat1.iteration,
                "seat0_checkpoint": str(seat0.checkpoint),
                "seat1_checkpoint": str(seat1.checkpoint),
                "returncode": int(result.returncode),
                "elapsed_sec": elapsed,
                "score0": score0,
                "score1": 1.0 - score0,
                "winner_id": outcome.get("winner_id"),
                "final_scores": outcome.get("final_scores"),
                "ranking": outcome.get("ranking"),
                "elo_after": {name: round(value, 3) for name, value in sorted(elos.items())},
            }
            if result.returncode != 0:
                row["stdout_tail"] = "\n".join((result.stdout or "").splitlines()[-20:])
                row["stderr_tail"] = "\n".join((result.stderr or "").splitlines()[-20:])
            records.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"[tournament] match={match_index + 1} seat0={seat0.name} seat1={seat1.name} "
                f"seed={seed} score0={score0:.1f} rc={result.returncode} "
                f"elo0={elos[seat0.name]:.1f} elo1={elos[seat1.name]:.1f}",
                flush=True,
            )
            if result.returncode != 0 and args.stop_on_error:
                break

    _write_summary_csv(csv_path, competitors, elos, records)
    return jsonl_path, csv_path, elos


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a checkpoint-vs-checkpoint self-play Elo tournament.")
    parser.add_argument("--checkpoint-dir", type=Path, default=rl_path("checkpoints"))
    parser.add_argument("--pattern", default="latest_iter_*.pt")
    parser.add_argument("--limit", type=int, default=None, help="Only use the last N discovered checkpoints.")
    parser.add_argument("--games-per-pair", type=int, default=1)
    parser.add_argument("--latest-vs-history", action="store_true", help="Evaluate only the newest checkpoint against older step-aligned checkpoints.")
    parser.add_argument("--history-step", type=int, default=5, help="Iteration multiple used by --latest-vs-history.")
    parser.add_argument("--no-seat-swaps", dest="seat_swaps", action="store_false")
    parser.set_defaults(seat_swaps=True)
    parser.add_argument("--seed-base", type=int, default=900_000)
    parser.add_argument("--initial-elo", type=float, default=1500.0)
    parser.add_argument("--k-factor", type=float, default=32.0)
    parser.add_argument("--output-dir", type=Path, default=rl_path("tournaments"))
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--search-batch-size", type=int, default=32)
    parser.add_argument("--top-k-actions", type=int, default=32)
    parser.add_argument("--max-turns-capitals", type=int, default=80)
    parser.add_argument("--max-actions-per-turn", type=int, default=80)
    parser.add_argument("--max-actions-per-game", type=int, default=512)
    parser.add_argument("--match-timeout-seconds", type=int, default=1800)
    parser.add_argument("--external-action-timeout-ms", type=int, default=120_000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    jsonl_path, csv_path, elos = run_tournament(args)
    print(f"[tournament] wrote jsonl={jsonl_path}", flush=True)
    print(f"[tournament] wrote csv={csv_path}", flush=True)
    for name, elo in sorted(elos.items(), key=lambda item: item[1], reverse=True):
        print(f"[tournament] elo {name} {elo:.1f}", flush=True)


if __name__ == "__main__":
    main()
