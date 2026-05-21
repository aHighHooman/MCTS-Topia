from __future__ import annotations

"""
Static MCTS autoresearch evaluator.

Purpose:
    Evaluate the current static-eval variant against:
      1. the baseline variant, and
      2. the latest accepted/previous variant,

    using a constant wall-clock budget PER ACTION, not fixed simulations and not
    a shared per-turn budget.

Typical usage from repo root after committing an experiment:

    python -m auto.static_autoresearch_eval \
      --iteration static-v2 \
      --candidate-ref HEAD \
      --previous-ref static-v1 \
      --baseline-ref static-v1 \
      --require-clean

This file is intended to live at:

    py/auto/static_autoresearch_eval.py

It assumes your repo uses the existing training/selfplay infrastructure and the
native static MCTS implementation:

    search.native.static_mcts.run_native_static_mcts(..., wall_time_seconds=...)

The script creates temporary git worktrees so different C++ static-eval versions
can coexist safely in the same tournament. Each worktree gets its own
TORCH_EXTENSIONS_DIR to avoid native-extension cache collisions.
"""

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time

import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PY_ROOT.parent
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from search.config import HybridAgentConfig
from training.replay import ReplayStore
from training.selfplay import run_selfplay


LEDGER_PATH = Path.home() / ".cache" / "static_mcts_autoresearch" / "static_eval_elos.csv"
BASELINE_ELO = 1000.0
K_FACTOR = 16.0
BLACKBOX_GAMES_PER_OPPONENT = 8
BLACKBOX_SEAT_SWAPS = True
BLACKBOX_WALL_TIME_SEC = 0.1
BLACKBOX_SEARCH_BATCH_SIZE = 32
BLACKBOX_TOP_K_ACTIONS = 64
BLACKBOX_RUN_MODE = "PlayLG"
BLACKBOX_GAME_MODE = "Capitals"
BLACKBOX_MAP_TYPE = "Drylands"
BLACKBOX_MAP_SIZE = "Tiny"
BLACKBOX_TRIBES = ["Imperius", "Imperius"]
BLACKBOX_MAX_TURNS_CAPITALS = 80
BLACKBOX_MAX_ACTIONS_PER_TURN = 80
BLACKBOX_MAX_ACTIONS_PER_GAME = 512
BLACKBOX_MATCH_TIMEOUT_SECONDS = 1800
BLACKBOX_EXTERNAL_ACTION_TIMEOUT_MS = 120_000
BLACKBOX_MIN_SCORE_VS_PREVIOUS = 0.55
BLACKBOX_MIN_SCORE_VS_BASELINE = 0.45


STATIC_BOT_SOURCE = r'''
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def _action_type(action: dict[str, Any]) -> str:
    payload = action.get("payload") if isinstance(action.get("payload"), dict) else {}
    return str(action.get("type") or action.get("t") or payload.get("type") or payload.get("t") or "").upper()


def _fallback_action_id(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return ""
    for action in actions:
        if _action_type(action) != "END_TURN":
            return str(action.get("id"))
    return str(actions[0].get("id"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-action wall-clock native static MCTS bot wrapper.")
    parser.add_argument("--py-root", type=Path, required=True)
    parser.add_argument("--torch-extensions-dir", type=Path, required=True)
    parser.add_argument("--wall-time-sec", type=float, required=True)
    parser.add_argument("--search-batch-size", type=int, default=32)
    parser.add_argument("--top-k-actions", type=int, default=64)
    parser.add_argument("--max-game-actions", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--static-eval-variant",
        default="",
        help="Optional TRIBES_STATIC_EVAL_VARIANT value, e.g. baseline. Empty uses current/tuned code.",
    )
    args = parser.parse_args()

    py_root = str(args.py_root.resolve())
    if py_root not in sys.path:
        sys.path.insert(0, py_root)

    os.environ["TORCH_EXTENSIONS_DIR"] = str(args.torch_extensions_dir.resolve())
    if args.static_eval_variant:
        os.environ["TRIBES_STATIC_EVAL_VARIANT"] = str(args.static_eval_variant)

    from search.config import HybridAgentConfig
    from nn.encoding import normalize_message
    from search.native.static_mcts import run_native_static_mcts

    try:
        from nn.belief import BeliefTracker
        tracker = BeliefTracker()
    except Exception:
        tracker = None

    cfg = HybridAgentConfig()
    cfg.search.batch_size = int(args.search_batch_size)
    cfg.search.top_k_actions = int(args.top_k_actions)
    cfg.search.num_simulations = 10_000_000  # ignored when wall_time_seconds is provided, but keep large as fallback.
    cfg.selfplay.max_actions_per_game = int(args.max_game_actions)

    if args.deterministic:
        cfg.search.sample_action = False
        cfg.search.dirichlet_epsilon = 0.0
        cfg.search.root_temperature = 1e-6

    action_counter = 0

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"error": "invalid_json"}), flush=True)
            continue

        msg_type = message.get("type")
        if msg_type == "game_over":
            print(json.dumps({"ok": True}), flush=True)
            break

        if msg_type != "action_request":
            print(json.dumps({"error": f"unsupported message type: {msg_type}"}), flush=True)
            continue

        actions = list(message.get("actions", []) or [])
        if not actions:
            print(json.dumps({"actionId": "", "rankedActionIds": []}), flush=True)
            continue

        try:
            normalized = normalize_message(message)
            if tracker is not None:
                normalized = tracker.annotate(normalized)
            payload = {
                "player_id": int(normalized.get("player_id", 0) or 0),
                "observation": normalized["observation"],
                "actions": list(normalized.get("actions", []) or []),
            }

            cfg.search.seed = int(args.seed) + action_counter
            action_counter += 1

            result = run_native_static_mcts(
                payload,
                cfg.search,
                cfg.model,
                wall_time_seconds=max(0.001, float(args.wall_time_sec)),
            )

            ranked = [
                action_id for action_id, _share in sorted(
                    result.visit_distribution.items(),
                    key=lambda item: float(item[1]),
                    reverse=True,
                )
            ]
            selected = str(result.action_id or (ranked[0] if ranked else _fallback_action_id(payload["actions"])))
            if selected not in {str(action.get("id")) for action in payload["actions"]}:
                selected = _fallback_action_id(payload["actions"])

            print(
                json.dumps(
                    {
                        "actionId": selected,
                        "rankedActionIds": ranked,
                        "value": float(result.value),
                        "visitDistribution": result.visit_distribution,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
        except Exception:
            traceback.print_exc(file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


@dataclass(frozen=True)
class Competitor:
    name: str
    ref: str
    rev: str
    worktree: Path
    py_root: Path
    torch_extensions_dir: Path


@dataclass
class MatchRecord:
    match_index: int
    opponent_group: str
    game_index: int
    seed: int
    seat0: str
    seat1: str
    returncode: int
    elapsed_sec: float
    score0: float
    score1: float
    winner_id: int | None
    final_scores: Any
    ranking: Any
    stdout_tail: str = ""
    stderr_tail: str = ""


def _run(
    cmd: list[str],
    *,
    cwd: Path = REPO_ROOT,
    check: bool = True,
    capture: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=capture,
        check=check,
    )


def _git(args: list[str], *, cwd: Path = REPO_ROOT, check: bool = True) -> str:
    result = _run(["git", *args], cwd=cwd, check=check, capture=True)
    return (result.stdout or "").strip()


def _git_rev(ref: str, *, cwd: Path = REPO_ROOT) -> str:
    return _git(["rev-parse", "--verify", ref], cwd=cwd)


def _safe_name(value: str) -> str:
    out = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "variant"


def _remove_worktree(path: Path) -> None:
    if not path.exists():
        return
    try:
        _git(["worktree", "remove", "--force", str(path)], cwd=REPO_ROOT, check=True)
    except Exception:
        shutil.rmtree(path, ignore_errors=True)
        try:
            _git(["worktree", "prune"], cwd=REPO_ROOT, check=False)
        except Exception:
            pass


def _prepare_worktree(name: str, ref: str, output_dir: Path, *, keep_existing: bool) -> Competitor:
    rev = _git_rev(ref)
    safe = _safe_name(name)
    worktree = (output_dir / "worktrees" / safe).resolve()
    torch_ext = (output_dir / "torch_extensions" / safe).resolve()

    if not keep_existing:
        _remove_worktree(worktree)
    if not worktree.exists():
        worktree.parent.mkdir(parents=True, exist_ok=True)
        _git(["worktree", "add", "--force", "--detach", str(worktree), rev], cwd=REPO_ROOT)
    torch_ext.mkdir(parents=True, exist_ok=True)

    return Competitor(
        name=name,
        ref=ref,
        rev=rev[:12],
        worktree=worktree,
        py_root=worktree / "py",
        torch_extensions_dir=torch_ext,
    )


def _write_static_bot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(STATIC_BOT_SOURCE, encoding="utf-8")


def _bot_command(
    bot_script: Path,
    competitor: Competitor,
    replay_dir: Path,
    args: argparse.Namespace,
    *,
    seed: int,
) -> list[str]:
    # replay_dir is currently not consumed by the wrapper. Replay is captured by
    # the central ReplayStore passed into run_selfplay. Keep the variable here so
    # match layout remains clear and the command can be extended later.
    del replay_dir
    command = [
        sys.executable,
        str(bot_script),
        "--py-root",
        str(competitor.py_root),
        "--torch-extensions-dir",
        str(competitor.torch_extensions_dir),
        "--wall-time-sec",
        str(float(args.wall_time_sec)),
        "--search-batch-size",
        str(int(args.search_batch_size)),
        "--top-k-actions",
        str(int(args.top_k_actions)),
        "--max-game-actions",
        str(int(args.max_actions_per_game)),
        "--seed",
        str(seed),
    ]
    if args.deterministic:
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
    delta = k * (float(score_a) - ea)
    elos[a] += delta
    elos[b] -= delta


def _updated_rating(rating: float, opponent_rating: float, score: float, k: float = K_FACTOR) -> float:
    return rating + k * (float(score) - _expected_score(rating, opponent_rating))


def _same_git_ref(left: str, right: str) -> bool:
    return _git_rev(left) == _git_rev(right)


def _read_ledger(path: Path = LEDGER_PATH) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {str(row["iteration"]): dict(row) for row in rows if row.get("iteration")}


def _previous_elo(previous_ref: str, baseline_ref: str, ledger: dict[str, dict[str, str]]) -> float:
    if _same_git_ref(previous_ref, baseline_ref):
        return BASELINE_ELO
    if previous_ref not in ledger:
        raise RuntimeError(
            f"Previous ref {previous_ref!r} is not in the protected Elo ledger. "
            "Bootstrap is only allowed when previous-ref and baseline-ref point to the same commit."
        )
    return float(ledger[previous_ref]["candidate_elo"])


def _append_ledger_row(summary: dict[str, Any], path: Path = LEDGER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "iteration",
        "candidate_ref",
        "candidate_rev",
        "previous_ref",
        "previous_rev",
        "baseline_ref",
        "baseline_rev",
        "candidate_elo",
        "candidate_elo_delta_vs_previous",
        "candidate_elo_delta_from_start",
        "baseline_check_candidate_elo",
        "score_rate_vs_previous",
        "score_rate_vs_baseline",
        "total_seconds",
        "written_at_unix",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: summary.get(name, "") for name in fieldnames})


def _apply_blackbox_settings(args: argparse.Namespace) -> None:
    args.games_per_opponent = BLACKBOX_GAMES_PER_OPPONENT
    args.seat_swaps = BLACKBOX_SEAT_SWAPS
    args.wall_time_sec = BLACKBOX_WALL_TIME_SEC
    args.search_batch_size = BLACKBOX_SEARCH_BATCH_SIZE
    args.top_k_actions = BLACKBOX_TOP_K_ACTIONS
    args.run_mode = BLACKBOX_RUN_MODE
    args.game_mode = BLACKBOX_GAME_MODE
    args.map_type = BLACKBOX_MAP_TYPE
    args.map_size = BLACKBOX_MAP_SIZE
    args.level_file = None
    args.tribes = list(BLACKBOX_TRIBES)
    args.max_turns_capitals = BLACKBOX_MAX_TURNS_CAPITALS
    args.max_actions_per_turn = BLACKBOX_MAX_ACTIONS_PER_TURN
    args.max_actions_per_game = BLACKBOX_MAX_ACTIONS_PER_GAME
    args.match_timeout_seconds = BLACKBOX_MATCH_TIMEOUT_SECONDS
    args.external_action_timeout_ms = BLACKBOX_EXTERNAL_ACTION_TIMEOUT_MS
    args.min_score_vs_previous = BLACKBOX_MIN_SCORE_VS_PREVIOUS
    args.min_score_vs_baseline = BLACKBOX_MIN_SCORE_VS_BASELINE


def _build_cfg(args: argparse.Namespace, output_dir: Path) -> HybridAgentConfig:
    cfg = HybridAgentConfig()
    cfg.training.output_dir = output_dir
    cfg.search.batch_size = int(args.search_batch_size)
    cfg.search.top_k_actions = int(args.top_k_actions)
    cfg.search.num_simulations = 10_000_000
    cfg.selfplay.run_mode = str(args.run_mode)
    cfg.selfplay.game_mode = str(args.game_mode)
    cfg.selfplay.max_turns_capitals = int(args.max_turns_capitals)
    cfg.selfplay.max_actions_per_turn = int(args.max_actions_per_turn)
    cfg.selfplay.max_actions_per_game = int(args.max_actions_per_game)
    cfg.selfplay.timeout_seconds = int(args.match_timeout_seconds)
    cfg.selfplay.external_action_timeout_ms = int(args.external_action_timeout_ms)
    cfg.selfplay.persistent_bot = True
    cfg.selfplay.profile_selfplay = False

    if args.level_file is not None:
        cfg.selfplay.level_file = str(args.level_file)
    if args.java_executable:
        cfg.selfplay.java_executable = str(args.java_executable)
    if args.java_classpath:
        cfg.selfplay.java_classpath = str(args.java_classpath)
    if args.java_main_class:
        cfg.selfplay.java_main_class = str(args.java_main_class)

    # These may not exist in older configs, but setting them is harmless for
    # newer Java wrappers that support generated-map options.
    setattr(cfg.selfplay, "map_type", str(args.map_type))
    setattr(cfg.selfplay, "map_size", str(args.map_size))
    return cfg


def _run_match(
    *,
    match_index: int,
    opponent_group: str,
    game_index: int,
    seed: int,
    seat0: Competitor,
    seat1: Competitor,
    bot_script: Path,
    cfg: HybridAgentConfig,
    args: argparse.Namespace,
    output_dir: Path,
) -> MatchRecord:
    cfg.selfplay.game_seed = int(seed)
    cfg.selfplay.agent_seed = int(seed)
    cfg.selfplay.level_seed = int(seed)

    match_dir = output_dir / "matches" / f"match_{match_index:05d}_{seat0.name}_vs_{seat1.name}"
    replay_dir = match_dir / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        _bot_command(bot_script, seat0, match_dir / "seat0", args, seed=seed * 2 + 0),
        _bot_command(bot_script, seat1, match_dir / "seat1", args, seed=seed * 2 + 1),
    ]
    tribes = list(args.tribes)

    replay_store = ReplayStore(replay_dir, capacity_steps=0, shard_prefix=cfg.replay.shard_prefix, load_existing=False)

    started_at = time.perf_counter()
    stdout_tail = ""
    stderr_tail = ""
    returncode = 0
    try:
        completed = run_selfplay(
            cfg,
            commands,
            tribes,
            REPO_ROOT,
            checkpoint_path=Path("rl/checkpoints/latest.pt"),
            replay_store=replay_store,
            device=torch.device("cpu"),
            progress_label=f"static-eval match={match_index} {seat0.name} vs {seat1.name}",
        )
        returncode = int(completed.returncode)
        if returncode != 0:
            stdout_tail = (completed.stdout or "")[-4000:]
            stderr_tail = (completed.stderr or "")[-4000:]
    except Exception as exc:
        returncode = 1
        stderr_tail = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started_at

    outcome = _latest_episode_outcome(replay_dir, cfg.replay.shard_prefix)
    score0 = _score_for_player(outcome, 0) if returncode == 0 else 0.0

    return MatchRecord(
        match_index=match_index,
        opponent_group=opponent_group,
        game_index=game_index,
        seed=seed,
        seat0=seat0.name,
        seat1=seat1.name,
        returncode=returncode,
        elapsed_sec=elapsed,
        score0=score0,
        score1=1.0 - score0 if returncode == 0 else 0.0,
        winner_id=outcome.get("winner_id"),
        final_scores=outcome.get("final_scores"),
        ranking=outcome.get("ranking"),
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _record_to_dict(record: MatchRecord, elos: dict[str, float]) -> dict[str, Any]:
    row = {
        "match_index": record.match_index,
        "opponent_group": record.opponent_group,
        "game_index": record.game_index,
        "seed": record.seed,
        "seat0": record.seat0,
        "seat1": record.seat1,
        "returncode": record.returncode,
        "elapsed_sec": round(record.elapsed_sec, 3),
        "score0": record.score0,
        "score1": record.score1,
        "winner_id": record.winner_id,
        "final_scores": record.final_scores,
        "ranking": record.ranking,
        "elo_after": {name: round(value, 3) for name, value in sorted(elos.items())},
    }
    if record.returncode != 0:
        row["stdout_tail"] = record.stdout_tail
        row["stderr_tail"] = record.stderr_tail
    return row


def _write_summary_csv(path: Path, competitors: list[Competitor], elos: dict[str, float], records: list[MatchRecord]) -> None:
    rows: list[dict[str, Any]] = []
    for competitor in competitors:
        games = [record for record in records if competitor.name in (record.seat0, record.seat1)]
        scores: list[float] = []
        for record in games:
            if record.returncode != 0:
                continue
            scores.append(float(record.score0) if record.seat0 == competitor.name else float(record.score1))
        rows.append(
            {
                "name": competitor.name,
                "ref": competitor.ref,
                "rev": competitor.rev,
                "elo": round(elos[competitor.name], 2),
                "games": len(scores),
                "score": round(sum(scores), 3),
                "score_rate": round(statistics.fmean(scores), 4) if scores else 0.0,
            }
        )
    rows.sort(key=lambda row: float(row["elo"]), reverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "ref", "rev", "elo", "games", "score", "score_rate"])
        writer.writeheader()
        writer.writerows(rows)


def _candidate_group_scores(records: list[MatchRecord], candidate_name: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[float]] = {}
    for record in records:
        if record.returncode != 0:
            continue
        if candidate_name not in (record.seat0, record.seat1):
            continue
        score = float(record.score0) if record.seat0 == candidate_name else float(record.score1)
        groups.setdefault(record.opponent_group, []).append(score)

    return {
        group: {
            "games": float(len(scores)),
            "score": float(sum(scores)),
            "score_rate": float(statistics.fmean(scores)) if scores else 0.0,
        }
        for group, scores in sorted(groups.items())
    }


def _warmup_competitor(bot_script: Path, competitor: Competitor, args: argparse.Namespace) -> None:
    """Force native-extension build before the match clock/action timeout matters."""
    msg = {
        "type": "action_request",
        "player_id": 0,
        "observation": {
            "active_player_id": 0,
            "tick": 0,
            "can_end_turn": True,
            "board": {
                "size": 4,
                "tiles": [
                    [
                        {
                            "x": x,
                            "y": y,
                            "visible": True,
                            "explored": True,
                            "terrain": "PLAIN",
                            "city_id": -1,
                            "unit_id": 0,
                            "road": False,
                        }
                        for x in range(4)
                    ]
                    for y in range(4)
                ],
            },
            "units": [],
            "cities": [
                {
                    "id": 10,
                    "tribe_id": 0,
                    "x": 1,
                    "y": 1,
                    "level": 1,
                    "population": 0,
                    "population_need": 2,
                    "production": 2,
                    "is_capital": True,
                    "has_walls": False,
                }
            ],
            "tribes": [
                {"id": 0, "stars": 10, "score": 0, "researched_tech_ids": ["ROADS"], "cities": [10], "extra_units": []},
                {"id": 1, "stars": 10, "score": 0, "researched_tech_ids": ["ROADS"], "cities": [], "extra_units": []},
            ],
        },
        "actions": [
            {"id": "end", "type": "END_TURN"},
            {"id": "road", "type": "BUILD_ROAD", "x": 1, "y": 2, "position": {"x": 1, "y": 2}},
        ],
    }
    command = _bot_command(bot_script, competitor, Path("."), args, seed=123)
    completed = subprocess.run(
        command,
        input=json.dumps(msg) + "\n" + json.dumps({"type": "game_over"}) + "\n",
        text=True,
        capture_output=True,
        timeout=max(60, int(args.external_action_timeout_ms // 1000)),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"warmup failed for {competitor.name} ({competitor.ref})\n"
            f"stdout:\n{completed.stdout[-2000:]}\n\nstderr:\n{completed.stderr[-4000:]}"
        )


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.require_clean and _git(["status", "--porcelain"]):
        raise RuntimeError("Working tree is not clean. Commit or stash changes before evaluating variants.")
    if not args.iteration:
        raise RuntimeError("--iteration is required so accepted candidates can be recorded in the protected ledger.")

    bot_script = output_dir / "_static_walltime_bot.py"
    _write_static_bot(bot_script)

    candidate = _prepare_worktree("candidate", args.candidate_ref, output_dir, keep_existing=args.keep_worktrees)
    previous = _prepare_worktree("previous", args.previous_ref, output_dir, keep_existing=args.keep_worktrees)
    baseline = _prepare_worktree("baseline", args.baseline_ref, output_dir, keep_existing=args.keep_worktrees)
    competitors = [candidate, previous, baseline]

    ledger = _read_ledger()
    if args.iteration in ledger:
        raise RuntimeError(f"Iteration {args.iteration!r} is already present in the protected Elo ledger.")
    previous_elo = _previous_elo(args.previous_ref, args.baseline_ref, ledger)
    candidate_start_elo = previous_elo
    candidate_elo = candidate_start_elo
    baseline_check_candidate_elo = candidate_start_elo

    if not args.skip_warmup:
        for competitor in competitors:
            print(f"[warmup] {competitor.name} ref={competitor.ref} rev={competitor.rev}", flush=True)
            _warmup_competitor(bot_script, competitor, args)

    cfg = _build_cfg(args, output_dir)
    jsonl_path = args.jsonl or output_dir / f"static_autoresearch_{int(time.time())}.jsonl"
    csv_path = args.csv or output_dir / "static_autoresearch_summary.csv"

    display_elos = {
        "candidate": candidate_elo,
        "previous": previous_elo,
        "baseline": BASELINE_ELO,
    }
    records: list[MatchRecord] = []
    match_index = 0

    pair_plan = [
        ("vs_previous", candidate, previous),
        ("vs_baseline", candidate, baseline),
    ]

    for group_name, left, right in pair_plan:
        for game_index in range(int(args.games_per_opponent)):
            seed = int(args.seed_base) + (10_000 if group_name == "vs_baseline" else 0) + game_index

            seat_orders = [(left, right)]
            if args.seat_swaps:
                seat_orders.append((right, left))

            for seat0, seat1 in seat_orders:
                record = _run_match(
                    match_index=match_index,
                    opponent_group=group_name,
                    game_index=game_index,
                    seed=seed,
                    seat0=seat0,
                    seat1=seat1,
                    bot_script=bot_script,
                    cfg=cfg,
                    args=args,
                    output_dir=output_dir,
                )
                records.append(record)
                if record.returncode == 0:
                    candidate_score = float(record.score0) if record.seat0 == "candidate" else float(record.score1)
                    if group_name == "vs_previous":
                        candidate_elo = _updated_rating(candidate_elo, previous_elo, candidate_score)
                    elif group_name == "vs_baseline":
                        baseline_check_candidate_elo = _updated_rating(
                            baseline_check_candidate_elo,
                            BASELINE_ELO,
                            candidate_score,
                        )
                    display_elos = {
                        "candidate": candidate_elo,
                        "previous": previous_elo,
                        "baseline": BASELINE_ELO,
                        "baseline_check_candidate": baseline_check_candidate_elo,
                    }

                row = _record_to_dict(record, display_elos)
                _write_jsonl(jsonl_path, [row])
                print(
                    f"[match] {match_index + 1} group={group_name} seed={seed} "
                    f"seat0={seat0.name} seat1={seat1.name} score0={record.score0:.1f} "
                    f"rc={record.returncode} elo_candidate={candidate_elo:.1f}",
                    flush=True,
                )
                match_index += 1

                if record.returncode != 0 and args.stop_on_error:
                    _write_summary_csv(csv_path, competitors, display_elos, records)
                    raise RuntimeError(f"match failed: {row}")

    _write_summary_csv(csv_path, competitors, display_elos, records)

    candidate_groups = _candidate_group_scores(records, "candidate")
    crashes = sum(1 for record in records if record.returncode != 0)
    candidate_elo_delta_vs_previous = float(candidate_elo - previous_elo)
    candidate_elo_delta_from_start = float(candidate_elo - candidate_start_elo)
    prev_rate = candidate_groups.get("vs_previous", {}).get("score_rate", 0.0)
    base_rate = candidate_groups.get("vs_baseline", {}).get("score_rate", 0.0)

    status = "keep"
    reasons: list[str] = []
    if crashes:
        status = "crash"
        reasons.append(f"{crashes} failed match(es)")
    if prev_rate < float(args.min_score_vs_previous):
        status = "discard" if status != "crash" else status
        reasons.append(f"score_rate_vs_previous={prev_rate:.3f} < {float(args.min_score_vs_previous):.3f}")
    if base_rate < float(args.min_score_vs_baseline):
        status = "discard" if status != "crash" else status
        reasons.append(f"score_rate_vs_baseline={base_rate:.3f} < {float(args.min_score_vs_baseline):.3f}")

    summary = {
        "iteration": args.iteration,
        "status": status,
        "reasons": reasons,
        "candidate_ref": candidate.ref,
        "candidate_rev": candidate.rev,
        "previous_ref": previous.ref,
        "previous_rev": previous.rev,
        "baseline_ref": baseline.ref,
        "baseline_rev": baseline.rev,
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "wall_time_sec_per_action": float(args.wall_time_sec),
        "games_per_opponent": int(args.games_per_opponent),
        "seat_swaps": bool(args.seat_swaps),
        "crashes": crashes,
        "candidate_elo": round(candidate_elo, 3),
        "candidate_elo_delta_vs_previous": round(candidate_elo_delta_vs_previous, 3),
        "candidate_elo_delta_from_start": round(candidate_elo_delta_from_start, 3),
        "baseline_check_candidate_elo": round(baseline_check_candidate_elo, 3),
        "score_rate_vs_previous": round(float(prev_rate), 6),
        "score_rate_vs_baseline": round(float(base_rate), 6),
        "ledger_path": str(LEDGER_PATH),
        "ledger_written": False,
        "candidate_group_scores": candidate_groups,
        "elos": {name: round(value, 3) for name, value in sorted(display_elos.items())},
    }

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a candidate native static MCTS variant against both the previous accepted variant "
            "and the original baseline using a constant wall-clock budget per action."
        )
    )
    parser.add_argument("--iteration", required=True, help="Accepted-variant name to record if the candidate is kept, e.g. static-v2.")
    parser.add_argument("--candidate-ref", default="HEAD", help="Git ref for the candidate/current variant. Default: HEAD.")
    parser.add_argument("--previous-ref", default="HEAD~1", help="Git ref for latest accepted/previous variant. Default: HEAD~1.")
    parser.add_argument("--baseline-ref", default="main", help="Git ref for baseline/v1 variant. Default: main.")
    parser.add_argument("--output-dir", type=Path, default=Path("rl/static_autoresearch_eval"))
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--keep-worktrees", action="store_true", help="Reuse existing output worktrees instead of recreating them.")
    parser.add_argument("--require-clean", action="store_true", help="Fail if the main working tree has uncommitted changes.")

    parser.add_argument("--java-executable", default=None)
    parser.add_argument("--java-classpath", default=None)
    parser.add_argument("--java-main-class", default=None)

    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")

    args = parser.parse_args()
    args.seed_base = 900_000
    _apply_blackbox_settings(args)
    started_at = time.perf_counter()
    try:
        summary = run_eval(args)
        summary["total_seconds"] = round(time.perf_counter() - started_at, 3)
        summary["written_at_unix"] = int(time.time())
        if summary["status"] == "keep":
            _append_ledger_row(summary)
            summary["ledger_written"] = True
        print("---", flush=True)
        for key in [
            "status",
            "candidate_elo",
            "candidate_elo_delta_vs_previous",
            "candidate_elo_delta_from_start",
            "baseline_check_candidate_elo",
            "wall_time_sec_per_action",
            "games_per_opponent",
            "crashes",
            "ledger_path",
            "ledger_written",
            "jsonl",
            "csv",
            "total_seconds",
        ]:
            print(f"{key}: {summary[key]}", flush=True)
        print("candidate_group_scores:", json.dumps(summary["candidate_group_scores"], sort_keys=True), flush=True)
        print("elos:", json.dumps(summary["elos"], sort_keys=True), flush=True)
        if summary["reasons"]:
            print("reasons:", "; ".join(summary["reasons"]), flush=True)

        return 0 if summary["status"] == "keep" else 2
    except Exception as exc:
        print("---", flush=True)
        print("status: crash", flush=True)
        print(f"error: {type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
