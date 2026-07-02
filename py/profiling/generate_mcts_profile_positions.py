from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

PY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PY_ROOT.parent
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from profiling.config import load_config_defaults
from search.config import HybridAgentConfig
from training.config import rl_path
from training.replay import ReplayStore
from training.selfplay import run_selfplay

DEFAULT_CONFIG = PY_ROOT / "profiling" / "configs" / "generate_mcts_profile_positions.json"


CAPTURE_BOT_SOURCE = r'''from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def _write_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    handle.flush()


def _grid_value(grid: Any, x: int, y: int, default: Any = None) -> Any:
    if not isinstance(grid, list) or y < 0 or y >= len(grid):
        return default
    row = grid[y]
    if not isinstance(row, list) or x < 0 or x >= len(row):
        return default
    return row[x]


def _ensure_board_tiles(observation: dict[str, Any]) -> dict[str, Any]:
    out = dict(observation)
    board = dict(out.get("board", {}) or {})
    if board.get("tiles"):
        out["board"] = board
        return out
    size = int(board.get("size", 0) or 0)
    if size <= 0:
        for key in ("terrain", "unit", "city", "exp", "resource", "building", "road"):
            grid = board.get(key)
            if isinstance(grid, list) and grid:
                size = len(grid)
                break
    tiles = []
    for y in range(size):
        row = []
        for x in range(size):
            explored = bool(_grid_value(board.get("exp"), x, y, False))
            city_id = int(_grid_value(board.get("city"), x, y, 0) or 0)
            row.append(
                {
                    "x": x,
                    "y": y,
                    "terrain": _grid_value(board.get("terrain"), x, y),
                    "resource": _grid_value(board.get("resource"), x, y),
                    "building": _grid_value(board.get("building"), x, y),
                    "city_id": city_id,
                    "unit_id": int(_grid_value(board.get("unit"), x, y, 0) or 0),
                    "explored": explored,
                    "visible": explored,
                    "road": bool(_grid_value(board.get("road"), x, y, False)),
                    "territory_city_id": city_id,
                }
            )
        tiles.append(row)
    board["tiles"] = tiles
    if size:
        board["size"] = size
    out["board"] = board
    return out


def _ensure_entity_aliases(observation: dict[str, Any]) -> dict[str, Any]:
    out = dict(observation)
    out.setdefault("active_player_id", out.get("active", 0))
    out.setdefault("can_end_turn", bool(out.get("end", False)))
    out.setdefault("leveling_up", bool(out.get("lvlup", False)))
    out.setdefault("ranking", out.get("rank", []))
    units = []
    for raw_unit in out.get("units", []) or []:
        unit = dict(raw_unit)
        unit.setdefault("tribe_id", unit.get("p", -1))
        unit.setdefault("city_id", unit.get("c", 0))
        unit.setdefault("type", unit.get("t"))
        unit.setdefault("current_hp", unit.get("hp", unit.get("hpx", 0)))
        unit.setdefault("current_hp_exact", unit.get("current_hp", unit.get("hp", 0)))
        unit.setdefault("max_hp", unit.get("mhp", unit.get("hp", 0)))
        unit.setdefault("kills", unit.get("k", 0))
        unit.setdefault("is_veteran", bool(unit.get("v", False)))
        unit.setdefault("status", unit.get("s"))
        unit.setdefault("is_hidden", bool(unit.get("h", False)))
        unit.setdefault("hidden_at_turn_start", bool(unit.get("hts", False)))
        unit.setdefault("hidden_enemy_hint", bool(unit.get("heh", False)))
        unit.setdefault("attack", unit.get("atk", 0))
        unit.setdefault("defence", unit.get("def", 0))
        unit.setdefault("movement", unit.get("mov", 0))
        unit.setdefault("range", unit.get("r", 0))
        units.append(unit)
    out["units"] = units
    cities = []
    for raw_city in out.get("cities", []) or []:
        city = dict(raw_city)
        city.setdefault("tribe_id", city.get("p", -1))
        city.setdefault("level", city.get("lvl", 0))
        city.setdefault("population", city.get("pop", 0))
        city.setdefault("population_need", city.get("need", 0))
        city.setdefault("production", city.get("prod", 0))
        city.setdefault("is_capital", bool(city.get("cap", False)))
        city.setdefault("has_walls", bool(city.get("wall", False)))
        city.setdefault("points_worth", city.get("pts", 0))
        city.setdefault("infiltrated", bool(city.get("inf", False)))
        city.setdefault("unit_ids", city.get("units", []))
        city.setdefault("buildings", city.get("b", []))
        cities.append(city)
    out["cities"] = cities
    return out


def _ensure_action_aliases(message: dict[str, Any]) -> dict[str, Any]:
    out = dict(message)
    actions = []
    for raw_action in out.get("actions", []) or []:
        action = dict(raw_action)
        action.setdefault("id", f"A{action.get('i', len(actions))}")
        action.setdefault("type", action.get("t"))
        action.setdefault("tribe_id", action.get("p", 0))
        actions.append(action)
    out["actions"] = actions
    return out


def _payload_from_message(message: dict[str, Any], tracker: Any, normalize_message: Any) -> dict[str, Any]:
    message = _ensure_action_aliases(message)
    normalized = normalize_message(message)
    if "observation" not in normalized:
        observation = normalized.get("obs")
        if not isinstance(observation, dict):
            observation = {key: value for key, value in normalized.items() if key not in {"type", "actions", "player_id"}}
        observation = _ensure_board_tiles(observation)
        observation = _ensure_entity_aliases(observation)
        normalized = normalize_message(
            {
                "player_id": int(normalized.get("player_id", 0) or 0),
                "observation": observation,
                "actions": list(normalized.get("actions", []) or []),
            }
        )
    normalized = tracker.annotate(normalized)
    return {
        "player_id": int(normalized.get("player_id", 0) or 0),
        "observation": normalized["observation"],
        "actions": list(normalized.get("actions", []) or []),
    }


def _search_payload(payload: dict[str, Any]) -> dict[str, Any]:
    searchable = dict(payload)
    actions = []
    for action in payload.get("actions", []) or []:
        action_type = str(action.get("type") or action.get("t") or "").upper()
        if action_type in {"RESEARCH", "RESEARCH_TECH"}:
            continue
        actions.append(action)
    searchable["actions"] = actions or list(payload.get("actions", []) or [])
    return searchable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--py-root", type=Path, required=True)
    parser.add_argument("--native-static-exe", type=Path, required=True)
    parser.add_argument("--player-slot", type=int, required=True)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--wall-clock-per-action-seconds", type=float, default=None)
    parser.add_argument("--top-k-actions", type=int, default=64)
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument("--search-batch-size", type=int, default=64)
    parser.add_argument("--static-eval-variant", choices=("baseline", "experimental", "experimental-2", "experimental-training"), default="baseline")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    py_root = str(args.py_root.resolve())
    if py_root not in sys.path:
        sys.path.insert(0, py_root)

    from nn.belief import BeliefTracker
    from nn.encoding import normalize_message

    command = [
        str(args.native_static_exe),
        "--simulations",
        str(int(args.simulations)),
        "--top-k-actions",
        str(int(args.top_k_actions)),
        "--max-actions",
        str(int(args.max_actions)),
        "--search-batch-size",
        str(int(args.search_batch_size)),
        "--static-eval-variant",
        str(args.static_eval_variant),
        "--seed",
        str(int(args.seed)),
    ]
    if args.wall_clock_per_action_seconds is not None:
        command.extend(["--wall-clock-per-action-seconds", str(max(0.0, float(args.wall_clock_per_action_seconds)))])
    if args.deterministic:
        command.append("--deterministic")

    tracker = BeliefTracker()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    action_index = 0
    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg_type = message.get("type")
            if msg_type == "action_request":
                payload = _payload_from_message(message, tracker, normalize_message)
                _write_jsonl(output, {"action_index": action_index, "player_slot": args.player_slot, "payload": payload})
                action_index += 1
                request = dict(_search_payload(payload))
                request["type"] = "action_request"
                completed = subprocess.run(
                    command,
                    input=json.dumps(request, separators=(",", ":")) + "\n",
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if completed.returncode != 0:
                    print(json.dumps({"error": completed.stderr[-1000:] or completed.stdout[-1000:]}), flush=True)
                    return completed.returncode
                lines = [item for item in completed.stdout.splitlines() if item.strip()]
                print(lines[-1] if lines else json.dumps({"error": "native static executable produced no output"}), flush=True)
                continue
            if msg_type == "game_over":
                _write_jsonl(output, {"type": "game_over", "player_slot": args.player_slot})
                break
            print(json.dumps({"error": f"unsupported message type: {msg_type}"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


@dataclass
class Candidate:
    payload: dict[str, Any]
    game_index: int
    seed: int
    player_slot: int
    action_index: int
    source_path: Path
    metrics: dict[str, Any]
    score: float
    reasons: list[str]


def _payload_actions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    actions = payload.get("actions", []) if isinstance(payload, dict) else []
    return [action for action in actions if isinstance(action, dict)]


def _action_type(action: dict[str, Any]) -> str:
    return str(action.get("type") or "UNKNOWN").upper()


def _count_grid_truthy(grid: Any, key: str | None = None) -> int:
    if not isinstance(grid, list):
        return 0
    count = 0
    for row in grid:
        if not isinstance(row, list):
            continue
        for item in row:
            if key is None:
                count += 1 if item else 0
            elif isinstance(item, dict) and item.get(key):
                count += 1
    return count


def _phase(payload: dict[str, Any]) -> str:
    observation = payload.get("observation", {}) if isinstance(payload, dict) else {}
    tick = int(observation.get("tick", 0) or 0) if isinstance(observation, dict) else 0
    if tick <= 8:
        return "early"
    if tick <= 24:
        return "mid"
    return "late"


def _metrics(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str], float]:
    observation = payload.get("observation", {}) if isinstance(payload, dict) else {}
    board = observation.get("board", {}) if isinstance(observation, dict) else {}
    actions = _payload_actions(payload)
    action_types = Counter(_action_type(action) for action in actions)
    units = observation.get("units", []) if isinstance(observation, dict) else []
    cities = observation.get("cities", []) if isinstance(observation, dict) else []
    units = units if isinstance(units, list) else []
    cities = cities if isinstance(cities, list) else []
    player_id = int(payload.get("player_id", observation.get("active_player_id", -1)) or -1)
    enemy_units = sum(
        1
        for unit in units
        if isinstance(unit, dict) and int(unit.get("tribe_id", unit.get("p", player_id)) or player_id) != player_id
    )
    damaged_units = sum(
        1
        for unit in units
        if isinstance(unit, dict)
        and float(unit.get("current_hp", unit.get("hp", unit.get("hpx", 10))) or 0)
        < float(unit.get("max_hp", unit.get("mhp", 10)) or 10)
    )
    visible_tiles = _count_grid_truthy(board.get("tiles"), "visible") or _count_grid_truthy(board.get("exp"))
    board_size = int(board.get("size", 0) or 0)
    total_tiles = board_size * board_size
    visible_ratio = (visible_tiles / float(total_tiles)) if total_tiles else 0.0
    tactical = sum(action_types.get(name, 0) for name in ("ATTACK", "CAPTURE", "CONVERT"))
    economy = sum(
        count
        for name, count in action_types.items()
        if name.startswith("BUILD") or name in {"PRODUCE_UNIT", "TRAIN_UNIT", "RESEARCH_TECH", "LEVEL_UP_CITY"}
    )
    expansion = sum(action_types.get(name, 0) for name in ("MOVE", "CAPTURE"))
    diversity = len([name for name, count in action_types.items() if count > 0])
    branching = len(actions)
    score = (
        branching
        + diversity * 5.0
        + tactical * 4.0
        + economy * 1.5
        + expansion * 0.5
        + len(units) * 1.2
        + len(cities) * 2.0
        + enemy_units * 6.0
        + damaged_units * 3.0
        + min(20.0, visible_ratio * 20.0)
    )
    reasons = []
    for enabled, reason in (
        (branching > 0, "branching"),
        (tactical > 0, "combat"),
        (economy > 0, "economy"),
        (expansion > 0, "expansion"),
        (enemy_units > 0, "enemy_visible"),
        (damaged_units > 0, "damaged_units"),
        (visible_ratio < 0.6, "partial_information"),
        (diversity >= 4, "action_diversity"),
    ):
        if enabled:
            reasons.append(reason)
    return (
        {
            "phase": _phase(payload),
            "tick": observation.get("tick", ""),
            "actions": branching,
            "action_type_count": diversity,
            "action_types": dict(sorted(action_types.items())),
            "units": len(units),
            "cities": len(cities),
            "enemy_units": enemy_units,
            "damaged_units": damaged_units,
            "visible_tiles": visible_tiles,
            "visible_ratio": round(visible_ratio, 4),
            "tactical_actions": tactical,
            "economy_actions": economy,
            "expansion_actions": expansion,
        },
        reasons,
        score,
    )


def _signature(candidate: Candidate) -> tuple[Any, ...]:
    action_types = candidate.metrics.get("action_types", {})
    top_types = tuple(sorted((str(key), int(value)) for key, value in dict(action_types).items() if int(value) > 0))
    return (
        candidate.metrics.get("phase"),
        candidate.metrics.get("tick"),
        candidate.metrics.get("actions"),
        candidate.metrics.get("units"),
        candidate.metrics.get("cities"),
        candidate.metrics.get("enemy_units"),
        top_types,
    )


def _add_selected(selected: list[Candidate], seen: set[tuple[Any, ...]], candidate: Candidate) -> bool:
    signature = _signature(candidate)
    if signature in seen:
        return False
    selected.append(candidate)
    seen.add(signature)
    return True


def select_positions(candidates: list[Candidate], target_count: int) -> list[Candidate]:
    target_count = max(1, min(int(target_count), len(candidates)))
    ranked = sorted(candidates, key=lambda item: (item.score, item.metrics.get("actions", 0)), reverse=True)
    selected: list[Candidate] = []
    seen: set[tuple[Any, ...]] = set()

    per_phase = max(1, target_count // 3)
    for phase in ("early", "mid", "late"):
        phase_count = 0
        for candidate in [item for item in ranked if item.metrics.get("phase") == phase]:
            if phase_count >= per_phase:
                break
            if _add_selected(selected, seen, candidate):
                phase_count += 1

    specialties = [
        ("highest_branching", lambda item: float(item.metrics.get("actions", 0))),
        ("most_action_diverse", lambda item: float(item.metrics.get("action_type_count", 0))),
        ("combat_contact", lambda item: float(item.metrics.get("tactical_actions", 0) * 10 + item.metrics.get("enemy_units", 0))),
        ("economy_heavy", lambda item: float(item.metrics.get("economy_actions", 0))),
        ("largest_armies", lambda item: float(item.metrics.get("units", 0))),
        ("widest_visibility", lambda item: float(item.metrics.get("visible_tiles", 0))),
        ("partial_information", lambda item: float(1.0 - item.metrics.get("visible_ratio", 1.0))),
    ]
    for reason, key_fn in specialties:
        for candidate in sorted(candidates, key=key_fn, reverse=True)[: max(3, target_count // 4)]:
            if len(selected) >= target_count:
                break
            if _add_selected(selected, seen, candidate) and reason not in candidate.reasons:
                candidate.reasons.append(reason)
        if len(selected) >= target_count:
            break

    for candidate in ranked:
        if len(selected) >= target_count:
            break
        _add_selected(selected, seen, candidate)
    selected_sources = {(item.seed, item.player_slot, item.action_index) for item in selected}
    for candidate in ranked:
        if len(selected) >= target_count:
            break
        source = (candidate.seed, candidate.player_slot, candidate.action_index)
        if source in selected_sources:
            continue
        selected.append(candidate)
        selected_sources.add(source)
    return selected


def _load_candidates(path: Path, *, game_index: int, seed: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            payload = record.get("payload")
            if not isinstance(payload, dict):
                continue
            metrics, reasons, score = _metrics(payload)
            candidates.append(
                Candidate(
                    payload=payload,
                    game_index=game_index,
                    seed=seed,
                    player_slot=int(record.get("player_slot", -1)),
                    action_index=int(record.get("action_index", len(candidates))),
                    source_path=path,
                    metrics=metrics,
                    score=score,
                    reasons=reasons,
                )
            )
    return candidates


def _capture_game(args: argparse.Namespace, *, game_index: int, seed: int) -> list[Candidate]:
    cfg = HybridAgentConfig()
    cfg.selfplay.run_mode = args.run_mode
    cfg.selfplay.game_mode = args.game_mode
    cfg.selfplay.map_type = args.map_type
    cfg.selfplay.map_size = args.map_size
    cfg.selfplay.game_seed = seed
    cfg.selfplay.agent_seed = seed
    cfg.selfplay.level_seed = seed
    cfg.selfplay.max_turns_capitals = args.max_turns
    cfg.selfplay.max_actions_per_turn = args.max_actions_per_turn
    cfg.selfplay.max_actions_per_game = args.max_actions_per_game
    cfg.selfplay.timeout_seconds = args.timeout_sec
    cfg.selfplay.persistent_bot = False
    if args.java_executable:
        cfg.selfplay.java_executable = args.java_executable
    if args.java_classpath:
        cfg.selfplay.java_classpath = args.java_classpath
    if args.java_main_class:
        cfg.selfplay.java_main_class = args.java_main_class

    raw_dir = args.raw_dir / f"{args.run_mode}_{args.map_type}_{args.map_size}_seed{seed}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    capture_bot = raw_dir / "_capture_static_mcts_positions_bot.py"
    capture_bot.write_text(CAPTURE_BOT_SOURCE, encoding="utf-8")
    for path in raw_dir.glob("player*.jsonl"):
        path.unlink(missing_ok=True)

    bot_commands = []
    for player_slot in range(len(args.tribes)):
        command = [
            sys.executable,
            str(capture_bot),
            "--output-jsonl",
            str(raw_dir / f"player{player_slot}.jsonl"),
            "--py-root",
            str(PY_ROOT),
            "--native-static-exe",
            str(args.native_static_exe),
            "--player-slot",
            str(player_slot),
            "--simulations",
            str(args.bot_simulations),
            "--top-k-actions",
            str(args.bot_top_k_actions),
            "--max-actions",
            str(args.bot_max_actions),
            "--search-batch-size",
            str(args.bot_batch_size),
            "--static-eval-variant",
            args.static_eval_variant,
            "--seed",
            str(seed + player_slot),
        ]
        if args.bot_wall_time_sec is not None:
            command.extend(["--wall-clock-per-action-seconds", str(args.bot_wall_time_sec)])
        if args.bot_deterministic:
            command.append("--deterministic")
        bot_commands.append(command)

    result = run_selfplay(
        cfg,
        bot_commands,
        args.tribes,
        args.workdir,
        progress_label=f"profile-position-corpus-{game_index}-seed{seed}",
    )

    candidates: list[Candidate] = []
    for jsonl_path in sorted(raw_dir.glob("player*.jsonl")):
        candidates.extend(_load_candidates(jsonl_path, game_index=game_index, seed=seed))
    if getattr(result, "returncode", 0) not in (0, None):
        print(
            f"  warning: self-play seed={seed} returned {getattr(result, 'returncode', '?')} "
            f"stderr_tail={str(getattr(result, 'stderr', '') or '')[-1000:]}",
            file=sys.stderr,
            flush=True,
        )
    if not candidates:
        stdout_tail = str(getattr(result, "stdout", "") or "")[-2000:]
        stderr_tail = str(getattr(result, "stderr", "") or "")[-4000:]
        raise RuntimeError(f"No positions captured for seed={seed}\nstdout_tail={stdout_tail}\nstderr_tail={stderr_tail}")
    return candidates


def write_profiler_payloads(args: argparse.Namespace, selected: list[Candidate]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_map_type = args.map_type.replace(" ", "_")
    safe_map_size = args.map_size.replace(" ", "_")
    for old_path in args.output_dir.glob(f"{args.run_mode}_{safe_map_type}_{safe_map_size}_seed*.json"):
        old_path.unlink(missing_ok=True)

    rows = []
    for index, candidate in enumerate(selected):
        path = args.output_dir / f"{args.run_mode}_{safe_map_type}_{safe_map_size}_seed{index}.json"
        path.write_text(json.dumps(candidate.payload, indent=2, sort_keys=True), encoding="utf-8")
        rows.append(
            {
                "profiler_seed_index": index,
                "path": str(path),
                "score": round(candidate.score, 3),
                "reasons": candidate.reasons,
                "source_seed": candidate.seed,
                "source_player_slot": candidate.player_slot,
                "source_action_index": candidate.action_index,
                "source_path": str(candidate.source_path),
                **candidate.metrics,
            }
        )

    summary_path = args.output_dir / "mcts_profile_position_selection_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "generated_payloads": len(selected),
                "profiler_config_hint": {
                    "positions": len(selected),
                    "selfplay_seed_start": 0,
                    "captured_payload_dir": str(args.output_dir),
                    "reuse_captured_payloads": True,
                },
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(selected)} profiler payloads to {args.output_dir}")
    print(f"summary={summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate saved MCTS profiler positions from static-MCTS self-play.")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--positions", type=int, default=36)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--run-mode", default="PlayLG")
    parser.add_argument("--game-mode", default="Capitals")
    parser.add_argument("--map-type", default="Drylands")
    parser.add_argument("--map-size", default="Tiny")
    parser.add_argument("--tribes", nargs=2, default=["Xin Xi", "Imperius"])
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-actions-per-turn", type=int, default=80)
    parser.add_argument("--max-actions-per-game", type=int, default=1024)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--bot-simulations", type=int, default=64)
    parser.add_argument("--bot-wall-time-sec", type=float, default=None)
    parser.add_argument("--bot-top-k-actions", type=int, default=64)
    parser.add_argument("--bot-max-actions", type=int, default=512)
    parser.add_argument("--bot-batch-size", type=int, default=64)
    parser.add_argument("--static-eval-variant", choices=("baseline", "experimental", "experimental-2", "experimental-training"), default="baseline")
    parser.add_argument("--native-static-exe", type=Path, default=PROJECT_ROOT / "out" / "native" / "static_mcts_bot.exe")
    parser.add_argument("--bot-deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "debug-logs" / "mcts-profile-payloads")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "debug-logs" / "mcts-profile-position-corpus")
    parser.add_argument("--workdir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--java-executable", default=None)
    parser.add_argument("--java-classpath", default=None)
    parser.add_argument("--java-main-class", default=None)
    return load_config_defaults(parser, default_config=DEFAULT_CONFIG)


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.raw_dir = args.raw_dir.resolve()
    args.workdir = args.workdir.resolve()
    candidates: list[Candidate] = []
    for game_index in range(max(1, args.games)):
        seed = args.seed_start + game_index
        print(f"[capture {game_index + 1}/{args.games}] seed={seed} map={args.map_type}/{args.map_size}", flush=True)
        game_candidates = _capture_game(args, game_index=game_index + 1, seed=seed)
        candidates.extend(game_candidates)
        print(f"  game_positions={len(game_candidates)} total_candidates={len(candidates)}", flush=True)
    selected = select_positions(candidates, args.positions)
    write_profiler_payloads(args, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
