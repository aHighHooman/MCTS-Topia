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

ACTION_TYPES = [
    "BUILD", "BURN_FOREST", "CLEAR_FOREST", "DESTROY", "GROW_FOREST", "LEVEL_UP",
    "GATHER", "SPAWN", "BUILD_ROAD", "BUILD_EMBASSY", "END_TURN", "RESEARCH",
    "PROPOSE_PEACE", "ACCEPT_PEACE", "PROPOSE_TREATY", "ACCEPT_TREATY", "CANCEL_TREATY",
    "ATTACK", "CAPTURE", "CONVERT", "DISBAND", "EXAMINE", "HEAL_OTHERS", "INFILTRATE",
    "MAKE_VETERAN", "MOVE", "RECOVER", "UPGRADE_RAMMER", "UPGRADE_SCOUT", "UPGRADE_BOMBER",
]
UNIT_TYPES = [
    "WARRIOR", "RIDER", "DEFENDER", "SWORDMAN", "ARCHER", "CATAPULT", "KNIGHT",
    "MIND_BENDER", "RAFT", "SCOUT", "BOMBER", "SUPERUNIT", "CLOAK", "DAGGER",
    "RAMMER", "JUGGERNAUT", "DINGHY", "PIRATE",
]
BUILDING_TYPES = [
    "PORT", "MINE", "FORGE", "FARM", "WINDMILL", "MARKET", "LUMBER_HUT", "SAWMILL",
    "TEMPLE", "WATER_TEMPLE", "FOREST_TEMPLE", "MOUNTAIN_TEMPLE", "ALTAR_OF_PEACE",
    "EMPERORS_TOMB", "EYE_OF_GOD", "GATE_OF_POWER", "GRAND_BAZAR", "PARK_OF_FORTUNE",
    "TOWER_OF_WISDOM", "EMBASSY",
]
RESOURCE_TYPES = ["FISH", "FRUIT", "ANIMAL", "STARFISH", "LIGHTHOUSE", "ORE", "CROPS", "RUINS"]
TERRAIN_TYPES = ["PLAIN", "SHALLOW_WATER", "DEEP_WATER", "MOUNTAIN", "VILLAGE", "CITY", "FOREST", "FOG"]
RESULT_TYPES = ["WIN", "LOSS", "INCOMPLETE"]
STATUS_TYPES = ["FRESH", "MOVED", "ATTACKED", "MOVED_AND_ATTACKED", "PUSHED", "FINISHED"]
TRIBE_TYPES = [
    "XIN_XI", "IMPERIUS", "BARDUR", "OUMAJI", "KICKOO", "HOODRICK",
    "LUXIDOOR", "VENGIR", "ZEBASI", "AI_MO", "QUETZALI", "YADAKK",
]
TECH_TYPES = [
    "CLIMBING", "FISHING", "HUNTING", "ORGANIZATION", "RIDING", "ARCHERY",
    "FARMING", "FORESTRY", "FREE_SPIRIT", "MEDITATION", "MINING", "ROADS",
    "RAMMING", "SAILING", "STRATEGY", "AQUATISM", "CHIVALRY", "CONSTRUCTION",
    "DIPLOMACY", "MATHEMATICS", "NAVIGATION", "SMITHERY", "SPIRITUALISM",
    "TRADE", "PHILOSOPHY",
]
LEVEL_UP_BONUSES = ["WORKSHOP", "EXPLORER", "CITY_WALL", "RESOURCES", "POP_GROWTH", "BORDER_GROWTH", "PARK", "SUPERUNIT"]
EXAMINE_BONUSES = ["UNIT", "RESEARCH", "POP_GROWTH", "EXPLORER", "RESOURCES"]
RELATIONSHIP_TYPES = ["WAR", "PEACE", "TREATY"]


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
    oracle = json.loads(completed.stdout)
    _expand_oracle_compact_states(oracle)
    _annotate_native_actor_id_floors(oracle)
    _annotate_native_enemy_explored(oracle)
    return oracle


def _expand_oracle_compact_states(oracle: dict[str, Any]) -> None:
    if isinstance(oracle.get("root"), dict):
        oracle["root"] = _expand_compact_state(oracle["root"])
    for child in oracle.get("children", []) or []:
        if isinstance(child, dict) and isinstance(child.get("state"), dict):
            child["state"] = _expand_compact_state(child["state"])
    for node in oracle.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        if isinstance(node.get("state"), dict):
            node["state"] = _expand_compact_state(node["state"])
        for child in node.get("children", []) or []:
            if isinstance(child, dict) and isinstance(child.get("state"), dict):
                child["state"] = _expand_compact_state(child["state"])


def _state_observation(state: dict[str, Any]) -> dict[str, Any]:
    observation = state.get("observation")
    if isinstance(observation, dict):
        return observation
    observation = state.get("obs")
    if isinstance(observation, dict):
        return observation
    return {}


def _enum_name(names: list[str], value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        index = int(value)
    except (TypeError, ValueError):
        return value
    return names[index] if 0 <= index < len(names) else value


def _matrix_codes(matrix: Any, names: list[str]) -> list[list[Any]]:
    if not isinstance(matrix, list):
        return []
    return [[_enum_name(names, value) for value in row] for row in matrix if isinstance(row, list)]


def _array_value(values: list[Any], index: int, fallback: Any = None) -> Any:
    return values[index] if 0 <= index < len(values) else fallback


def _normalize_compact_board(board: Any) -> dict[str, Any]:
    if not isinstance(board, list):
        return {}
    size = int(_array_value(board, 0, 0) or 0)
    terrain = _matrix_codes(_array_value(board, 1, []), TERRAIN_TYPES)
    resource = _matrix_codes(_array_value(board, 2, []), RESOURCE_TYPES)
    building = _matrix_codes(_array_value(board, 3, []), BUILDING_TYPES)
    city = _array_value(board, 4, []) if isinstance(_array_value(board, 4, []), list) else []
    unit = _array_value(board, 5, []) if isinstance(_array_value(board, 5, []), list) else []
    explored = _array_value(board, 6, []) if isinstance(_array_value(board, 6, []), list) else []
    road = _array_value(board, 7, []) if isinstance(_array_value(board, 7, []), list) else []
    tiles: list[list[dict[str, Any]]] = []
    for y in range(size):
      row: list[dict[str, Any]] = []
      for x in range(size):
        row.append({
            "x": x,
            "y": y,
            "terrain": terrain[y][x] if y < len(terrain) and x < len(terrain[y]) else None,
            "resource": resource[y][x] if y < len(resource) and x < len(resource[y]) else None,
            "building": building[y][x] if y < len(building) and x < len(building[y]) else None,
            "city_id": int(city[y][x] or 0) if y < len(city) and isinstance(city[y], list) and x < len(city[y]) else 0,
            "unit_id": int(unit[y][x] or 0) if y < len(unit) and isinstance(unit[y], list) and x < len(unit[y]) else 0,
            "explored": bool(explored[y][x]) if y < len(explored) and isinstance(explored[y], list) and x < len(explored[y]) else False,
            "visible": bool(explored[y][x]) if y < len(explored) and isinstance(explored[y], list) and x < len(explored[y]) else False,
            "road": bool(road[y][x]) if y < len(road) and isinstance(road[y], list) and x < len(road[y]) else False,
        })
      tiles.append(row)
    return {
        "size": size,
        "terrain": terrain,
        "resource": resource,
        "building": building,
        "city": city,
        "unit": unit,
        "explored": explored,
        "road": road,
        "tiles": tiles,
    }


def _normalize_compact_observation(observation: Any) -> Any:
    if not isinstance(observation, list):
        return observation
    return {
        "tick": _array_value(observation, 0, 0),
        "map_type": _array_value(observation, 1),
        "active_player_id": _array_value(observation, 2, 0),
        "can_end_turn": bool(_array_value(observation, 3, 0)),
        "leveling_up": bool(_array_value(observation, 4, 0)),
        "tribes": [
            {
                "id": _array_value(tribe, 0, 0),
                "type": _enum_name(TRIBE_TYPES, _array_value(tribe, 1)),
                "tribe": _enum_name(TRIBE_TYPES, _array_value(tribe, 1)),
                "stars": _array_value(tribe, 2, 0),
                "score": _array_value(tribe, 3, 0),
                "result": _enum_name(RESULT_TYPES, _array_value(tribe, 4)),
                "capital_id": _array_value(tribe, 5, 0),
                "researched_tech_ids": [_enum_name(TECH_TYPES, tech) for tech in (_array_value(tribe, 6, []) or [])],
            }
            for tribe in (_array_value(observation, 5, []) or [])
            if isinstance(tribe, list)
        ],
        "cities": [
            {
                "id": _array_value(city, 0, 0),
                "tribe_id": _array_value(city, 1, -1),
                "x": _array_value(city, 2, 0),
                "y": _array_value(city, 3, 0),
                "level": _array_value(city, 4, 0),
                "population": _array_value(city, 5, 0),
                "population_need": _array_value(city, 6, 0),
                "production": _array_value(city, 7, 0),
                "is_capital": bool(_array_value(city, 8, 0)),
                "has_walls": bool(_array_value(city, 9, 0)),
                "points_worth": _array_value(city, 10, 0),
                "buildings": [
                    {"type": _enum_name(BUILDING_TYPES, _array_value(building, 0)), "x": _array_value(building, 1, 0), "y": _array_value(building, 2, 0)}
                    for building in (_array_value(city, 11, []) or [])
                    if isinstance(building, list)
                ],
            }
            for city in (_array_value(observation, 6, []) or [])
            if isinstance(city, list)
        ],
        "units": [
            {
                "id": _array_value(unit, 0, 0),
                "tribe_id": _array_value(unit, 1, -1),
                "city_id": _array_value(unit, 2, 0),
                "type": _enum_name(UNIT_TYPES, _array_value(unit, 3)),
                "x": _array_value(unit, 4, 0),
                "y": _array_value(unit, 5, 0),
                "current_hp": _array_value(unit, 6, 0),
                "max_hp": _array_value(unit, 7, 0),
                "kills": _array_value(unit, 8, 0),
                "is_veteran": bool(_array_value(unit, 9, 0)),
                "status": _enum_name(STATUS_TYPES, _array_value(unit, 10)),
                "is_hidden": bool(_array_value(unit, 11, 0)),
                "hidden_enemy_hint": bool(_array_value(unit, 12, 0)),
                "attack": _array_value(unit, 13, 0),
                "defence": _array_value(unit, 14, 0),
                "movement": _array_value(unit, 15, 0),
                "range": _array_value(unit, 16, 0),
                "cost": _array_value(unit, 17, 0),
            }
            for unit in (_array_value(observation, 7, []) or [])
            if isinstance(unit, list)
        ],
        "board": _normalize_compact_board(_array_value(observation, 8, [])),
        "ranking": _array_value(observation, 9, []) or [],
        "relationships": [
            [_enum_name(RELATIONSHIP_TYPES, relationship) for relationship in row]
            for row in (_array_value(observation, 10, []) or [])
            if isinstance(row, list)
        ],
    }


def _normalize_compact_action(action: Any, index: int) -> Any:
    if not isinstance(action, list) or not action:
        return action
    action_type = _enum_name(ACTION_TYPES, _array_value(action, 0))
    if action_type == "GATHER":
        action_type = "RESOURCE_GATHERING"
    elif action_type == "RESEARCH":
        action_type = "RESEARCH_TECH"
    out: dict[str, Any] = {"id": f"A{index}", "i": index, "type": action_type}
    if action_type == "MOVE":
        out.update({"unit_id": _array_value(action, 1, 0), "x": _array_value(action, 2, 0), "y": _array_value(action, 3, 0)})
    elif action_type in {"ATTACK", "CONVERT"}:
        out.update({"unit_id": _array_value(action, 1, 0), "target_unit_id": _array_value(action, 2, 0)})
    elif action_type == "CAPTURE":
        out.update({"unit_id": _array_value(action, 1, 0), "target_city_id": _array_value(action, 2, 0), "capture_type": _enum_name(TERRAIN_TYPES, _array_value(action, 3))})
    elif action_type == "INFILTRATE":
        out.update({"unit_id": _array_value(action, 1, 0), "target_city_id": _array_value(action, 2, 0)})
    elif action_type in {"BUILD", "RESOURCE_GATHERING", "SPAWN", "LEVEL_UP"}:
        out.update({"city_id": _array_value(action, 1, 0), "x": _array_value(action, 2, 0), "y": _array_value(action, 3, 0)})
        if action_type == "BUILD":
            out["building_type"] = _enum_name(BUILDING_TYPES, _array_value(action, 4))
        elif action_type == "RESOURCE_GATHERING":
            out["resource_type"] = _enum_name(RESOURCE_TYPES, _array_value(action, 4))
        elif action_type == "SPAWN":
            out["unit_type"] = _enum_name(UNIT_TYPES, _array_value(action, 4))
        else:
            out["bonus"] = _enum_name(LEVEL_UP_BONUSES, _array_value(action, 4))
    elif action_type in {"BURN_FOREST", "CLEAR_FOREST", "DESTROY", "GROW_FOREST"}:
        out.update({"city_id": _array_value(action, 1, 0), "x": _array_value(action, 2, 0), "y": _array_value(action, 3, 0)})
    elif action_type == "BUILD_ROAD":
        out.update({"tribe_id": _array_value(action, 1, 0), "x": _array_value(action, 2, 0), "y": _array_value(action, 3, 0)})
    elif action_type == "RESEARCH_TECH":
        out.update({"tribe_id": _array_value(action, 1, 0), "tech": _enum_name(TECH_TYPES, _array_value(action, 2))})
        out["technology"] = out["tech"]
    elif action_type in {"BUILD_EMBASSY", "PROPOSE_PEACE", "ACCEPT_PEACE", "PROPOSE_TREATY", "ACCEPT_TREATY", "CANCEL_TREATY"}:
        out.update({"tribe_id": _array_value(action, 1, 0), "target_player_id": _array_value(action, 2, -1)})
    elif action_type == "EXAMINE":
        out.update({"unit_id": _array_value(action, 1, 0), "bonus": _enum_name(EXAMINE_BONUSES, _array_value(action, 2))})
    elif action_type in {"DISBAND", "MAKE_VETERAN", "RECOVER", "HEAL_OTHERS", "UPGRADE_RAMMER", "UPGRADE_SCOUT", "UPGRADE_BOMBER"}:
        out["unit_id"] = _array_value(action, 1, 0)
    elif action_type == "END_TURN":
        out["tribe_id"] = _array_value(action, 1, 0)
    return out


def _expand_compact_state(state: dict[str, Any]) -> dict[str, Any]:
    out = dict(state)
    if isinstance(out.get("observation"), list):
        out["observation"] = _normalize_compact_observation(out["observation"])
    if isinstance(out.get("actions"), list):
        out["actions"] = [_normalize_compact_action(action, i) for i, action in enumerate(out.get("actions", []))]
    return out


def _max_actor_id_in_state(state: dict[str, Any]) -> int:
    observation = _state_observation(state)
    max_id = int(observation.get("_native_actor_id_floor", 0) or 0)
    for city in observation.get("cities", []) or []:
        if isinstance(city, dict):
            max_id = max(max_id, int(city.get("id", 0) or 0))
    for unit in observation.get("units", []) or []:
        if isinstance(unit, dict):
            max_id = max(max_id, int(unit.get("id", 0) or 0))
    return max_id


def _set_native_actor_id_floor(state: dict[str, Any], floor: int) -> None:
    observation = _state_observation(state)
    if observation:
        observation["_native_actor_id_floor"] = max(int(observation.get("_native_actor_id_floor", 0) or 0), int(floor))

def _copy_enemy_explored(memory: dict[int, set[int]]) -> dict[int, set[int]]:
    return {tribe_id: set(codes) for tribe_id, codes in memory.items()}


def _set_native_enemy_explored(state: dict[str, Any], memory: dict[int, set[int]]) -> None:
    observation = _state_observation(state)
    if observation:
        observation["_native_enemy_explored"] = {
            str(tribe_id): sorted(codes) for tribe_id, codes in sorted(memory.items()) if codes
        }


def _unit_reveal_radius(state: dict[str, Any], unit: dict[str, Any], x: int, y: int) -> int:
    observation = _state_observation(state)
    terrain = None
    board = observation.get("board", {})
    tiles = board.get("tiles", []) if isinstance(board, dict) else []
    if 0 <= y < len(tiles) and isinstance(tiles[y], list) and 0 <= x < len(tiles[y]) and isinstance(tiles[y][x], dict):
        terrain = tiles[y][x].get("terrain")
    unit_type = str(unit.get("type", unit.get("t", "")))
    return 2 if terrain == "MOUNTAIN" or unit_type in {"SCOUT", "CLOAK", "DINGHY"} else 1


def _unit_by_id(state: dict[str, Any], unit_id: int) -> dict[str, Any] | None:
    observation = _state_observation(state)
    for unit in observation.get("units", []) or []:
        if isinstance(unit, dict) and int(unit.get("id", 0) or 0) == unit_id:
            return unit
    return None


def _mark_enemy_move_reveal(parent: dict[str, Any], child: dict[str, Any], action: dict[str, Any], memory: dict[int, set[int]]) -> None:
    observation = _state_observation(parent)
    active_player_id = int(
        observation.get("active_player_id", observation.get("active", parent.get("active_player_id", parent.get("active", 0)))) or 0
    )
    root_player_id = int(parent.get("root_player_id", parent.get("player_id", parent.get("p", 0))) or 0)
    action_type = str(action.get("type", action.get("t", "")))
    if active_player_id == root_player_id or action_type not in {"MOVE", "STEP_MOVE"}:
        return
    unit_id = int(action.get("unit_id", action.get("u", 0)) or 0)
    previous_unit = _unit_by_id(parent, unit_id)
    moved_unit = _unit_by_id(child, unit_id)
    if previous_unit is None or int(previous_unit.get("tribe_id", previous_unit.get("p", -1)) or -1) != active_player_id:
        return
    if moved_unit is None:
        moved_unit = dict(previous_unit)
        moved_unit["x"] = int(action.get("x", 0) or 0)
        moved_unit["y"] = int(action.get("y", 0) or 0)
    if int(moved_unit.get("tribe_id", moved_unit.get("p", -1)) or -1) != active_player_id:
        return
    board_size = int((_state_observation(child).get("board", {}) or {}).get("size", 0) or 0)
    if board_size <= 0:
        return
    revealed = memory.setdefault(active_player_id, set())

    for state, unit in ((parent, previous_unit), (child, moved_unit)):
        if unit is None:
            continue
        x = int(unit.get("x", 0) or 0)
        y = int(unit.get("y", 0) or 0)
        radius = _unit_reveal_radius(state, unit, x, y)
        for tx in range(max(0, x - radius), min(board_size, x + radius + 1)):
            for ty in range(max(0, y - radius), min(board_size, y + radius + 1)):
                revealed.add(tx * board_size + ty)


def _annotate_native_enemy_explored(oracle: dict[str, Any]) -> None:
    nodes = list(oracle.get("nodes") or [])
    if not nodes:
        return
    memories: dict[str, dict[int, set[int]]] = {}
    root_state_id = str(nodes[0].get("state_id", oracle.get("root_state_id", "root")))
    memories[root_state_id] = {}

    for node in nodes:
        state = node.get("state", {})
        if not isinstance(state, dict):
            continue
        state_id = str(node.get("state_id", ""))
        parent_memory = _copy_enemy_explored(memories.get(state_id, {}))
        memories[state_id] = parent_memory
        _set_native_enemy_explored(state, parent_memory)
        parent_actions = normalize_message({"player_id": int(oracle["player_id"]), **state}).get("actions", [])
        for child in node.get("children", []) or []:
            if not isinstance(child, dict) or not child.get("ok"):
                continue
            child_state = child.get("state")
            if not isinstance(child_state, dict):
                continue
            action_index = int(child.get("action_index", -1) or -1)
            action = parent_actions[action_index] if 0 <= action_index < len(parent_actions) else {}
            child_memory = _copy_enemy_explored(parent_memory)
            _mark_enemy_move_reveal(state, child_state, action, child_memory)
            child_state_id = str(child_state.get("state_id", ""))
            if child_state_id:
                memories[child_state_id] = child_memory
            _set_native_enemy_explored(child_state, child_memory)


def _annotate_native_actor_id_floors(oracle: dict[str, Any]) -> None:
    nodes = list(oracle.get("nodes") or [])
    if not nodes:
        return

    floors: dict[str, int] = {}
    root_state = nodes[0].get("state", {})
    root_state_id = str(nodes[0].get("state_id", oracle.get("root_state_id", "root")))
    root_floor = _max_actor_id_in_state(root_state)
    floors[root_state_id] = root_floor

    for node in nodes:
        state = node.get("state", {})
        state_id = str(node.get("state_id", ""))
        floor = max(floors.get(state_id, 0), _max_actor_id_in_state(state))
        floors[state_id] = floor
        _set_native_actor_id_floor(state, floor)
        for child in node.get("children", []) or []:
            if not isinstance(child, dict) or not child.get("ok"):
                continue
            child_state = child.get("state")
            if not isinstance(child_state, dict):
                continue
            child_state_id = str(child_state.get("state_id", ""))
            child_floor = max(floor, _max_actor_id_in_state(child_state))
            if child_state_id:
                floors[child_state_id] = max(floors.get(child_state_id, 0), child_floor)
            _set_native_actor_id_floor(child_state, child_floor)


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
    if isinstance(message.get("observation"), dict):
        message["observation"] = dict(message["observation"])
        message["observation"].pop("_native_actor_id_floor", None)
        message["observation"].pop("_native_enemy_explored", None)
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
    keys = (
        "tick",
        "map_type",
        "active_player_id",
        "can_end_turn",
        "leveling_up",
        "ranking",
        "relationships",
    )
    out = {key: observation.get(key) for key in keys if key in observation}
    cities = list(observation.get("cities", []))
    out["board"] = _canonical_board(dict(observation.get("board", {})), cities)
    out["units"] = sorted((_canonical_unit(unit) for unit in observation.get("units", [])), key=lambda unit: unit.get("id", 0))
    out["cities"] = sorted((_canonical_city(city) for city in cities), key=lambda city: city.get("id", 0))
    out["tribes"] = sorted((_canonical_tribe(tribe) for tribe in observation.get("tribes", [])), key=lambda tribe: tribe.get("id", 0))
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
    out["researched_tech_ids"] = sorted(str(value).upper() for value in tribe.get("researched_tech_ids", []) or [])
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

    field_aliases = {
        "unit_id": "u",
        "city_id": "c",
        "tribe_id": "p",
        "target_unit_id": "tu",
        "target_city_id": "tc",
        "capture_type": "ct",
        "unit_type": "ut",
        "building_type": "bt",
        "resource_type": "rt",
        "target_player_id": "tp",
    }
    zero_is_empty = {"unit_id", "city_id", "tribe_id", "target_unit_id", "target_city_id", "target_player_id"}

    for key, alias in field_aliases.items():
        value = action.get(key)
        if value is None:
            value = action.get(alias)
        if value is None:
            continue
        if key in zero_is_empty and int(value or 0) <= 0:
            continue
        out[key] = value

    tech = action.get("tech", action.get("technology"))
    if tech is not None:
        out["tech"] = tech
    for key in ("x", "y", "bonus"):
        value = action.get(key)
        if value is not None:
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
            selection = dict(tree.select_leaf(1.0))
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
