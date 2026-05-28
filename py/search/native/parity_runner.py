from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PY_ROOT = Path(__file__).resolve().parents[2]
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from nn.encoding import normalize_message
from project_paths import game_json_jar, game_src_root
from search.native.cpp_extension import load_native_mcts_extension


REPO_ROOT = Path(__file__).resolve().parents[3]
PARITY_JAVA_ROOT = Path(__file__).resolve().parent / "java"


class ParityFailure(AssertionError):
    pass


def _run_java_oracle(
    fixture: Path,
    player: int | None,
    compile_java: bool,
    depth: int,
    max_states: int,
    max_actions_per_state: int,
) -> dict[str, Any]:
    java_exe = _java_executable()
    javac_exe = _javac_executable()
    src_root = game_src_root()
    oracle_source = PARITY_JAVA_ROOT / "core" / "game" / "NativeParityOracle.java"
    sourcepath = os.pathsep.join([str(src_root), str(PARITY_JAVA_ROOT)])
    if compile_java:
        subprocess.run(
            [
                javac_exe,
                "-cp",
                str(game_json_jar()),
                "-sourcepath",
                sourcepath,
                "-d",
                str(REPO_ROOT / "out"),
                str(oracle_source),
            ],
            cwd=REPO_ROOT,
            check=True,
        )

    classpath = f"{REPO_ROOT / 'out'};{game_json_jar()}"
    command = [
        java_exe,
        "-cp",
        classpath,
        "core.game.NativeParityOracle",
        "--fixture",
        str(fixture),
    ]
    if player is not None:
        command.extend(["--player", str(player)])
    command.extend(
        [
            "--depth",
            str(depth),
            "--max-states",
            str(max_states),
            "--max-actions-per-state",
            str(max_actions_per_state),
        ]
    )
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def _java_executable() -> str:
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        candidate = Path(java_home) / "bin" / ("java.exe" if sys.platform.startswith("win") else "java")
        if candidate.exists():
            return str(candidate)
    javac = shutil.which("javac")
    if javac:
        candidate = Path(javac).with_name("java.exe" if sys.platform.startswith("win") else "java")
        if candidate.exists():
            return str(candidate)
    return "java"


def _javac_executable() -> str:
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        candidate = Path(java_home) / "bin" / ("javac.exe" if sys.platform.startswith("win") else "javac")
        if candidate.exists():
            return str(candidate)
    return shutil.which("javac") or "javac"


def _canonical_state(state: dict[str, Any], player_id: int) -> dict[str, Any]:
    message = dict(state)
    message.setdefault("player_id", player_id)
    normalized = normalize_message(message)
    observation = _canonical_observation(dict(normalized.get("observation", {})))
    return {
        "observation": observation,
        "actions": sorted(
            (_canonical_action(action) for action in normalized.get("actions", [])),
            key=lambda item: json.dumps(item, sort_keys=True),
        ),
        "is_terminal": bool(normalized.get("is_terminal", False)),
        "active_player_id": int(
            normalized.get("active_player_id", observation.get("active_player_id", player_id)) or 0
        ),
        "winner_id": normalized.get("winner_id"),
        "final_scores": normalized.get("final_scores"),
        "ranking": normalized.get("ranking", observation.get("ranking")),
        "normalized_terminal_reward": normalized.get("normalized_terminal_reward"),
    }


def _canonical_observation(observation: dict[str, Any]) -> dict[str, Any]:
    out = dict(observation)
    out["board"] = _canonical_board(dict(out.get("board", {})), list(out.get("cities", [])))
    out["units"] = sorted((_canonical_unit(unit) for unit in out.get("units", [])), key=lambda unit: unit.get("id", 0))
    out["cities"] = sorted((_canonical_city(city) for city in out.get("cities", [])), key=lambda city: city.get("id", 0))
    out["tribes"] = sorted((_canonical_tribe(tribe) for tribe in out.get("tribes", [])), key=lambda tribe: tribe.get("id", 0))
    return out


def _canonical_board(board: dict[str, Any], cities: list[Any]) -> dict[str, Any]:
    city_centers = {
        (int(city.get("x", -1) or -1), int(city.get("y", -1) or -1)): int(city.get("id", 0) or 0)
        for city in cities
        if isinstance(city, dict)
    }
    tiles = []
    for row in board.get("tiles", []):
        out_row = []
        for tile in row:
            if not isinstance(tile, dict):
                out_row.append(tile)
                continue
            x = int(tile.get("x", 0) or 0)
            y = int(tile.get("y", 0) or 0)
            out_row.append(
                {
                    "x": x,
                    "y": y,
                    "visible": bool(tile.get("visible", False)),
                    "explored": bool(tile.get("explored", False)),
                    "terrain": tile.get("terrain"),
                    "resource": tile.get("resource"),
                    "building": tile.get("building"),
                    "road": bool(tile.get("road", False)),
                    "unit_id": int(tile.get("unit_id", 0) or 0),
                    # The compact Java payload's board.city plane is territory/city ownership,
                    # while native regenerated payloads currently approximate some territory.
                    # Compare city centers through the authoritative cities list instead.
                    "city_center_id": city_centers.get((x, y), 0),
                }
            )
        tiles.append(out_row)
    return {"size": int(board.get("size", 0) or 0), "tiles": tiles}


def _canonical_unit(unit: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "tribe_id",
        "city_id",
        "type",
        "x",
        "y",
        "current_hp",
        "max_hp",
        "kills",
        "is_veteran",
        "status",
        "is_hidden",
    )
    return {key: unit.get(key) for key in keys if key in unit}


def _canonical_city(city: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "tribe_id",
        "x",
        "y",
        "level",
        "population",
        "population_need",
        "production",
        "is_capital",
        "has_walls",
        "bound",
        "points_worth",
        "infiltrated",
    )
    out = {key: city.get(key) for key in keys if key in city}
    out["units"] = sorted(int(unit_id) for unit_id in city.get("units", []) or [])
    return out


def _canonical_tribe(tribe: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "stars",
        "score",
        "capital_id",
        "kills",
        "pacifist_count",
        "units_disabled_next_turn",
        "result",
    )
    out = {key: tribe.get(key) for key in keys if key in tribe}
    out["researched_tech_ids"] = sorted(str(value) for value in tribe.get("researched_tech_ids", []) or [])
    out["city_ids"] = sorted(int(value) for value in tribe.get("city_ids", []) or [])
    out["extra_unit_ids"] = sorted(int(value) for value in tribe.get("extra_unit_ids", []) or [])
    return out


def _canonical_action(action: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    action_type = action.get("type")
    if action_type == "RESEARCH":
        action_type = "RESEARCH_TECH"
    if action_type is not None:
        out["type"] = action_type
    for key in ("type", "u", "c", "p", "x", "y", "tu", "tc", "ct", "ut", "bt", "rt", "b", "tech", "tp", "bonus"):
        if key == "type":
            continue
        value = action.get(key)
        if value is None:
            continue
        if key in {"u", "c", "p", "tu", "tc", "tp"} and int(value or 0) == 0:
            continue
        out[key] = value
    return out


def _first_diff(left: Any, right: Any, path: str = "$") -> tuple[str, Any, Any] | None:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool) and not isinstance(right, bool):
        return None if float(left) == float(right) else (path, left, right)
    if type(left) is not type(right):
        return path, left, right
    if isinstance(left, dict):
        left_keys = set(left)
        right_keys = set(right)
        for key in sorted(left_keys | right_keys):
            if key not in left and right.get(key) is None:
                continue
            if key not in right and left.get(key) is None:
                continue
            if key not in left:
                return f"{path}.{key}", None, right[key]
            if key not in right:
                return f"{path}.{key}", left[key], None
            diff = _first_diff(left[key], right[key], f"{path}.{key}")
            if diff is not None:
                return diff
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}.length", len(left), len(right)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            diff = _first_diff(left_item, right_item, f"{path}[{index}]")
            if diff is not None:
                return diff
        return None
    if left != right:
        return path, left, right
    return None


def _dump_trace(trace_dir: Path | None, action_id: str, root: dict[str, Any], java_child: dict[str, Any], cpp_child: dict[str, Any] | None) -> None:
    if trace_dir is None:
        return
    trace_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "root": root,
        "java_child": java_child,
        "cpp_child": cpp_child,
    }
    (trace_dir / f"{action_id}.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_parity(args: argparse.Namespace) -> int:
    oracle = _run_java_oracle(
        args.fixture,
        args.player,
        not args.no_compile_java,
        args.depth,
        args.max_states,
        args.max_actions_per_state,
    )
    if int(oracle.get("protocol_version", -1)) not in {1, 2}:
        raise RuntimeError(f"Unsupported Java oracle protocol version: {oracle.get('protocol_version')}")

    player_id = int(oracle["player_id"])
    nodes = list(oracle.get("nodes") or [{"state_id": "root", "depth": 0, "state": oracle["root"], "children": oracle.get("children", [])}])

    extension = load_native_mcts_extension()
    if extension is None:
        raise RuntimeError("Native MCTS extension is unavailable.")

    failures = 0
    checked = 0
    for node in nodes:
        java_parent_state = dict(node["state"])
        java_parent = normalize_message({"player_id": player_id, **java_parent_state})
        parent_actions = list(java_parent.get("actions", []))
        state_id = str(node.get("state_id", "root"))
        depth = int(node.get("depth", 0) or 0)
        for child in node.get("children", []):
            if args.action_id and str(child.get("action_id", "")) != args.action_id:
                continue
            checked += 1
            failures += _check_child(
                args,
                extension,
                player_id,
                state_id,
                depth,
                java_parent,
                parent_actions,
                child,
            )
            if failures and not args.keep_going:
                return 1

    print(f"PARITY SUMMARY checked={checked} failures={failures} depth={args.depth} states={len(nodes)}")
    return 1 if failures else 0


def _check_child(
    args: argparse.Namespace,
    extension: Any,
    player_id: int,
    state_id: str,
    depth: int,
    java_parent: dict[str, Any],
    parent_actions: list[dict[str, Any]],
    child: dict[str, Any],
) -> int:
        action_id = str(child.get("action_id", ""))
        action_index = int(child.get("action_index", -1))
        action = parent_actions[action_index] if 0 <= action_index < len(parent_actions) else {}
        action_type = str(action.get("type", action.get("t", "")))

        if not bool(child.get("ok", False)):
            print(
                f"PARITY FAIL depth={depth} state={state_id} action={action_id} type={action_type} "
                f"path=$.java_oracle java={child.get('error')} cpp=<not-run>",
                file=sys.stderr,
            )
            return 1

        try:
            cpp_root = copy.deepcopy(java_parent)
            tree = extension.NativeMCTS(
                cpp_root,
                [action_index],
                [1.0],
                0.0,
                bool(java_parent.get("is_terminal", False)),
                int(args.seed),
                int(args.max_actions),
            )
            selection = dict(tree.select_leaf(1, 1.0))
            cpp_payload = dict(selection.get("leaf_payload") or {})
            java_canonical = _canonical_state(dict(child["state"]), player_id)
            cpp_canonical = _canonical_state(cpp_payload, player_id)
            diff = _first_diff(java_canonical, cpp_canonical)
            _dump_trace(args.trace_dir, action_id, cpp_root, dict(child["state"]), cpp_payload)
            if diff is not None:
                path, java_value, cpp_value = diff
                raise ParityFailure(
                    f"PARITY FAIL depth={depth} state={state_id} action={action_id} type={action_type} "
                    f"path={path} java={json.dumps(java_value, sort_keys=True)} "
                    f"cpp={json.dumps(cpp_value, sort_keys=True)}"
                )
            print(f"PARITY OK depth={depth} state={state_id} action={action_id} type={action_type}")
            return 0
        except Exception as exc:
            if not isinstance(exc, ParityFailure):
                _dump_trace(args.trace_dir, action_id, java_parent, dict(child.get("state", {})), None)
                print(
                    f"PARITY FAIL depth={depth} state={state_id} action={action_id} type={action_type} "
                    f"path=$.cpp_exception java=<state> cpp={exc}",
                    file=sys.stderr,
                )
            else:
                print(str(exc), file=sys.stderr)
            return 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diff Java oracle children against native strict C++ transitions.")
    parser.add_argument("--fixture", type=Path, required=True, help="Path to a saved game.json fixture.")
    parser.add_argument("--player", type=int, default=None, help="Observer player id. Defaults to the fixture active player.")
    parser.add_argument("--action-id", default=None, help="Only check one root action id.")
    parser.add_argument("--depth", type=int, default=2, help="Number of plies to check from the fixture root.")
    parser.add_argument("--max-states", type=int, default=24, help="Maximum Java states to expand for deeper parity.")
    parser.add_argument("--max-actions-per-state", type=int, default=8, help="Maximum actions sampled from each Java state.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after the first parity failure.")
    parser.add_argument("--no-compile-java", action="store_true", help="Skip javac before running the Java oracle.")
    parser.add_argument("--seed", type=int, default=7, help="Native tree seed.")
    parser.add_argument("--max-actions", type=int, default=256, help="Native max action cap.")
    parser.add_argument("--trace-dir", type=Path, default=None, help="Optional directory for root/java/cpp JSON traces.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run_parity(parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
