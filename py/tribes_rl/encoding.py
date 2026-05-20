from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np
import torch

from .belief import BELIEF_OPPONENT_SCALAR_DIM, BELIEF_PLANE_CHANNEL_START, BELIEF_PLANE_NAMES
from .config import ModelConfig


TERRAIN_TYPES = [
    "FOG",
    "CITY",
    "VILLAGE",
    "MOUNTAIN",
    "FOREST",
    "PLAIN",
    "SHALLOW_WATER",
    "DEEP_WATER",
    "WATER",
]
RESOURCE_TYPES = ["ANIMAL", "CROPS", "FISH", "FRUIT", "ORE", "RUINS", "STARFISH", "WHALE", "FOREST"]
BUILDING_TYPES = [
    "PORT",
    "DOCK",
    "FARM",
    "MINE",
    "FORGE",
    "WINDMILL",
    "SAWMILL",
    "MARKET",
    "CUSTOMS_HOUSE",
    "LUMBER_HUT",
    "TEMPLE",
    "EMBASSY",
    "MONUMENT",
]
UNIT_TYPES = [
    "WARRIOR",
    "RIDER",
    "ARCHER",
    "DEFENDER",
    "SWORDMAN",
    "SWORDSMAN",
    "KNIGHT",
    "CATAPULT",
    "MIND_BENDER",
    "BOAT",
    "DINGHY",
    "RAFT",
    "SHIP",
    "BATTLESHIP",
    "BOMBER",
    "JUGGERNAUT",
    "SUPERUNIT",
    "CLOAK",
    "DAGGER",
    "PIRATE",
    "RAMMER",
    "SCOUT",
]
ACTION_TYPES = [
    "MOVE",
    "STEP_MOVE",
    "ATTACK",
    "CAPTURE",
    "CONVERT",
    "SPAWN",
    "BUILD",
    "RESOURCE_GATHERING",
    "LEVEL_UP",
    "BUILD_ROAD",
    "BUILD_EMBASSY",
    "RESEARCH_TECH",
    "END_TURN",
    "RECOVER",
    "HEAL_OTHERS",
    "EXAMINE",
    "MAKE_VETERAN",
    "INFILTRATE",
    "DISBAND",
    "BURN_FOREST",
    "CLEAR_FOREST",
    "GROW_FOREST",
    "DESTROY",
    "PROPOSE_PEACE",
    "ACCEPT_PEACE",
    "PROPOSE_TREATY",
    "ACCEPT_TREATY",
    "CANCEL_TREATY",
    "UPGRADE_RAMMER",
    "UPGRADE_SCOUT",
    "UPGRADE_BOMBER",
]
TECH_TYPES = [
    "CLIMBING",
    "FISHING",
    "HUNTING",
    "ORGANIZATION",
    "RIDING",
    "ARCHERY",
    "FARMING",
    "FORESTRY",
    "FREE_SPIRIT",
    "MEDITATION",
    "MINING",
    "ROADS",
    "RAMMING",
    "SAILING",
    "STRATEGY",
    "AQUATISM",
    "CHIVALRY",
    "CONSTRUCTION",
    "DIPLOMACY",
    "MATHEMATICS",
    "NAVIGATION",
    "SMITHERY",
    "SPIRITUALISM",
    "TRADE",
    "PHILOSOPHY",
]
RELATIONSHIP_TYPES = ["NEUTRAL", "PEACE", "WAR", "ALLY", "TEAM", "UNKNOWN"]

TERRAIN_TO_INDEX = {name: idx for idx, name in enumerate(TERRAIN_TYPES)}
RESOURCE_TO_INDEX = {name: idx for idx, name in enumerate(RESOURCE_TYPES)}
BUILDING_TO_INDEX = {name: idx for idx, name in enumerate(BUILDING_TYPES)}
UNIT_TO_INDEX = {name: idx for idx, name in enumerate(UNIT_TYPES)}
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTION_TYPES)}
TECH_TO_INDEX = {name: idx for idx, name in enumerate(TECH_TYPES)}


def _one_hot_table(names: list[str]) -> tuple[dict[str, list[float]], list[float]]:
    zero = [0.0 for _ in names]
    table: dict[str, list[float]] = {}
    for index, name in enumerate(names):
        values = [0.0] * len(names)
        values[index] = 1.0
        table[name] = values
    return table, zero


TERRAIN_ONE_HOT, TERRAIN_ZERO_HOT = _one_hot_table(TERRAIN_TYPES)
RESOURCE_ONE_HOT, RESOURCE_ZERO_HOT = _one_hot_table(RESOURCE_TYPES)
BUILDING_ONE_HOT, BUILDING_ZERO_HOT = _one_hot_table(BUILDING_TYPES)
UNIT_ONE_HOT, UNIT_ZERO_HOT = _one_hot_table(UNIT_TYPES)
ACTION_ONE_HOT, ACTION_ZERO_HOT = _one_hot_table(ACTION_TYPES)
TECH_ONE_HOT, TECH_ZERO_HOT = _one_hot_table(TECH_TYPES)


@dataclass
class EncodedObservation:
    board: torch.Tensor
    unit_features: torch.Tensor
    unit_mask: torch.Tensor
    city_features: torch.Tensor
    city_mask: torch.Tensor
    action_features: torch.Tensor
    action_mask: torch.Tensor
    scalar_features: torch.Tensor
    action_ids: List[str]

    def to(self, device: torch.device | str) -> "EncodedObservation":
        return EncodedObservation(
            self.board.to(device),
            self.unit_features.to(device),
            self.unit_mask.to(device),
            self.city_features.to(device),
            self.city_mask.to(device),
            self.action_features.to(device),
            self.action_mask.to(device),
            self.scalar_features.to(device),
            self.action_ids,
        )


def _norm(value: Any, scale: float) -> float:
    if value is None:
        return 0.0
    try:
        numerator = float(value)
    except (TypeError, ValueError):
        return 0.0
    denominator = scale if scale > 1.0 else 1.0
    return numerator / denominator


def _clamped_norm(value: Any, scale: float) -> float:
    normalized = _norm(value, scale)
    if normalized > 1.0:
        return 1.0
    if normalized < -1.0:
        return -1.0
    return normalized


def _one_hot(name: Any, names: list[str]) -> list[float]:
    if names is TERRAIN_TYPES:
        return TERRAIN_ONE_HOT.get(str(name), TERRAIN_ZERO_HOT)
    elif names is RESOURCE_TYPES:
        return RESOURCE_ONE_HOT.get(str(name), RESOURCE_ZERO_HOT)
    elif names is BUILDING_TYPES:
        return BUILDING_ONE_HOT.get(str(name), BUILDING_ZERO_HOT)
    elif names is UNIT_TYPES:
        return UNIT_ONE_HOT.get(str(name), UNIT_ZERO_HOT)
    elif names is ACTION_TYPES:
        return ACTION_ONE_HOT.get(str(name), ACTION_ZERO_HOT)
    elif names is TECH_TYPES:
        return TECH_ONE_HOT.get(str(name), TECH_ZERO_HOT)
    return []


def _message_is_normalized(message: Dict[str, Any]) -> bool:
    observation = message.get("observation")
    if not isinstance(observation, dict) or "active_player_id" not in observation:
        return False
    board = observation.get("board")
    if not isinstance(board, dict) or "tiles" not in board:
        return False
    actions = message.get("actions", [])
    if not actions:
        return True
    first_action = actions[0] if isinstance(actions, list) else None
    return isinstance(first_action, dict) and "type" in first_action


def _sorted_entities(entities: Iterable[Dict[str, Any]], key: str = "id") -> list[dict[str, Any]]:
    return sorted(entities, key=lambda item: int(item.get(key, -1) or -1))


def _put_feature(row: np.ndarray, values: list[float]) -> None:
    count = len(values)
    if count <= row.shape[0]:
        row[:count] = values
    else:
        row[:] = values[: row.shape[0]]


def _matrix_to_plane(matrix: Any, size: int) -> torch.Tensor:
    if not isinstance(matrix, list) or len(matrix) != size:
        return torch.zeros(size, size, dtype=torch.float32)
    rows: list[list[float]] = []
    for row in matrix:
        if not isinstance(row, list) or len(row) != size:
            return torch.zeros(size, size, dtype=torch.float32)
        values: list[float] = []
        for value in row:
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                values.append(0.0)
        rows.append(values)
    return torch.tensor(rows, dtype=torch.float32)


def _belief_opponent_scalars(belief: Any) -> list[float]:
    if not isinstance(belief, dict):
        return [0.0] * BELIEF_OPPONENT_SCALAR_DIM
    raw = belief.get("opponent_scalars", [])
    if not isinstance(raw, list):
        return [0.0] * BELIEF_OPPONENT_SCALAR_DIM
    out: list[float] = []
    for value in raw[:BELIEF_OPPONENT_SCALAR_DIM]:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            out.append(0.0)
    if len(out) < BELIEF_OPPONENT_SCALAR_DIM:
        out.extend([0.0] * (BELIEF_OPPONENT_SCALAR_DIM - len(out)))
    return out


def _copy_belief_planes(channels: torch.Tensor | np.ndarray, belief: Any, board_size: int) -> None:
    if not isinstance(belief, dict):
        return
    planes = belief.get("planes", {})
    if not isinstance(planes, dict):
        return
    for offset, name in enumerate(BELIEF_PLANE_NAMES):
        channel = BELIEF_PLANE_CHANNEL_START + offset
        if channel >= channels.shape[0]:
            break
        matrix = planes.get(name)
        if isinstance(channels, np.ndarray):
            if isinstance(matrix, np.ndarray):
                if matrix.shape == (board_size, board_size):
                    channels[channel, :, :] = matrix
                continue
            if not isinstance(matrix, list) or len(matrix) != board_size:
                continue
            try:
                plane = np.asarray(matrix, dtype=np.float32)
            except (TypeError, ValueError):
                continue
            if plane.shape == (board_size, board_size):
                channels[channel, :, :] = plane
        else:
            channels[channel] = _matrix_to_plane(matrix, board_size)


def _position_payload(action: dict[str, Any]) -> dict[str, Any]:
    return action.get("destination") or action.get("target_pos") or action.get("position") or {}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _entity_position(entity: dict[str, Any] | None) -> tuple[float, float, bool]:
    if not entity or "x" not in entity or "y" not in entity:
        return 0.0, 0.0, False
    try:
        return float(entity.get("x")), float(entity.get("y")), True
    except (TypeError, ValueError):
        return 0.0, 0.0, False


def _owner_relation_flags(entity: dict[str, Any] | None, my_player_id: int) -> list[float]:
    if not entity:
        return [0.0, 0.0, 0.0]
    owner = _as_int(entity.get("tribe_id"), -1)
    if owner == my_player_id:
        return [1.0, 0.0, 0.0]
    if owner >= 0:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def _hp_fraction(unit: dict[str, Any] | None) -> float:
    if not unit:
        return 0.0
    try:
        current_hp = float(unit.get("current_hp", 0) or 0)
        max_hp = float(unit.get("max_hp", 0) or 0)
    except (TypeError, ValueError):
        return 0.0
    return current_hp / (max_hp if max_hp > 1.0 else 1.0)


def _unit_action_summary(unit: dict[str, Any] | None, my_player_id: int, coord_scale: float) -> list[float]:
    if not unit:
        return [0.0] * (1 + len(UNIT_TYPES) + 3 + 9)
    return (
        [1.0]
        + _one_hot(unit.get("type"), UNIT_TYPES)
        + _owner_relation_flags(unit, my_player_id)
        + [
            _norm(unit.get("x", 0), coord_scale),
            _norm(unit.get("y", 0), coord_scale),
            _clamped_norm(unit.get("current_hp", 0), 40.0),
            _hp_fraction(unit),
            _clamped_norm(unit.get("max_hp", 0), 40.0),
            _clamped_norm(unit.get("kills", 0), 16.0),
            1.0 if unit.get("is_veteran") else 0.0,
            1.0 if unit.get("is_hidden") else 0.0,
            _clamped_norm(unit.get("range", 0), 8.0),
        ]
    )


def _city_action_summary(city: dict[str, Any] | None, my_player_id: int, coord_scale: float) -> list[float]:
    if not city:
        return [0.0] * 12
    return (
        [1.0]
        + _owner_relation_flags(city, my_player_id)
        + [
            _norm(city.get("x", 0), coord_scale),
            _norm(city.get("y", 0), coord_scale),
            _clamped_norm(city.get("level", 0), 10.0),
            _clamped_norm(city.get("population", 0), 32.0),
            _clamped_norm(city.get("population_need", 0), 32.0),
            _clamped_norm(city.get("production", 0), 32.0),
            1.0 if city.get("is_capital") else 0.0,
            1.0 if city.get("has_walls") else 0.0,
        ]
    )


def _tile_at(board: dict[str, Any], x_value: Any, y_value: Any) -> dict[str, Any] | None:
    x = _as_int(x_value, -1)
    y = _as_int(y_value, -1)
    tiles = board.get("tiles", [])
    if y < 0 or y >= len(tiles):
        return None
    row = tiles[y]
    if not isinstance(row, list) or x < 0 or x >= len(row):
        return None
    tile = row[x]
    return tile if isinstance(tile, dict) else None


def _tile_action_summary(tile: dict[str, Any] | None) -> list[float]:
    if not tile:
        return [0.0] * (5 + len(TERRAIN_TYPES) + len(RESOURCE_TYPES) + len(BUILDING_TYPES))
    return (
        [
            1.0,
            1.0 if tile.get("explored") else 0.0,
            1.0 if tile.get("road") else 0.0,
            1.0 if _as_int(tile.get("unit_id")) > 0 else 0.0,
            1.0 if _as_int(tile.get("city_id")) > 0 else 0.0,
        ]
        + _one_hot(tile.get("terrain"), TERRAIN_TYPES)
        + _one_hot(tile.get("resource"), RESOURCE_TYPES)
        + _one_hot(tile.get("building"), BUILDING_TYPES)
    )


def _normalize_board(board: Dict[str, Any]) -> Dict[str, Any]:
    if "tiles" in board:
        out = dict(board)
        size = int(out.get("size", 0) or 0)
        normalized_rows: list[list[dict[str, Any]]] = []
        for y, row in enumerate(out.get("tiles", []) or []):
            normalized_row: list[dict[str, Any]] = []
            for x, raw_tile in enumerate(row if isinstance(row, list) else []):
                tile = dict(raw_tile) if isinstance(raw_tile, dict) else {}
                tile.setdefault("x", x)
                tile.setdefault("y", y)
                tile.setdefault("terrain", None)
                tile.setdefault("resource", None)
                tile.setdefault("building", None)
                tile.setdefault("city_id", tile.get("city", 0))
                tile.setdefault("unit_id", tile.get("unit", 0))
                tile.setdefault("explored", bool(tile.get("visible", False)))
                tile.setdefault("road", False)
                normalized_row.append(tile)
            normalized_rows.append(normalized_row)
        out["tiles"] = normalized_rows
        if not size:
            out["size"] = len(normalized_rows)
        return out
    size = int(board.get("size", 0) or 0)
    terrain = board.get("terrain", []) or []
    resource = board.get("resource", []) or []
    building = board.get("building", []) or []
    city = board.get("city", []) or []
    unit = board.get("unit", []) or []
    explored = board.get("exp", []) or []
    road = board.get("road", []) or []
    tiles: list[list[dict[str, Any]]] = []
    for y in range(size):
        row: list[dict[str, Any]] = []
        for x in range(size):
            is_explored = bool(explored[y][x]) if y < len(explored) and x < len(explored[y]) else False
            row.append(
                {
                    "x": x,
                    "y": y,
                    "terrain": terrain[y][x] if y < len(terrain) and x < len(terrain[y]) else None,
                    "resource": resource[y][x] if y < len(resource) and x < len(resource[y]) else None,
                    "building": building[y][x] if y < len(building) and x < len(building[y]) else None,
                    "city_id": city[y][x] if y < len(city) and x < len(city[y]) else 0,
                    "unit_id": unit[y][x] if y < len(unit) and x < len(unit[y]) else 0,
                    "explored": is_explored,
                    "road": bool(road[y][x]) if y < len(road) and x < len(road[y]) else False,
                }
            )
        tiles.append(row)
    out = dict(board)
    out["tiles"] = tiles
    return out


def _normalize_unit(unit: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(unit)
    out.setdefault("tribe_id", out.get("p", -1))
    out.setdefault("city_id", out.get("c", 0))
    out.setdefault("type", out.get("t"))
    out.setdefault("current_hp", out.get("hp", 0))
    out.setdefault("current_hp_exact", out.get("hpx", out.get("current_hp", 0)))
    out.setdefault("max_hp", out.get("mhp", 0))
    out.setdefault("kills", out.get("k", 0))
    out.setdefault("is_veteran", out.get("v", False))
    out.setdefault("status", out.get("s"))
    out.setdefault("is_hidden", out.get("h", False))
    out.setdefault("hidden_at_turn_start", out.get("hts", False))
    out.setdefault("hidden_enemy_hint", out.get("hint", False))
    out.setdefault("attack", out.get("atk", 0))
    out.setdefault("defence", out.get("def", 0))
    out.setdefault("movement", out.get("mov", 0))
    out.setdefault("range", out.get("r", 0))
    return out


def _normalize_city(city: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(city)
    out.setdefault("tribe_id", out.get("p", -1))
    out.setdefault("level", out.get("lvl", 0))
    out.setdefault("population", out.get("pop", 0))
    out.setdefault("population_need", out.get("need", 0))
    out.setdefault("production", out.get("prod", 0))
    out.setdefault("is_capital", out.get("cap", False))
    out.setdefault("has_walls", out.get("wall", False))
    out.setdefault("bound", out.get("bound", 0))
    out.setdefault("points_worth", out.get("pts", 0))
    out.setdefault("infiltrated", out.get("inf", False))
    out.setdefault("unit_ids", out.get("units", []))
    out.setdefault("buildings", [{"type": building.get("t"), **building} for building in out.get("b", []) or []])
    return out


def _normalize_tribe(tribe: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(tribe)
    out.setdefault("researched_tech_ids", out.get("tech", []))
    out.setdefault("capital_id", out.get("cap", 0))
    out.setdefault("city_ids", out.get("cities", []))
    out.setdefault("extra_unit_ids", out.get("extra", []))
    out.setdefault("connected_city_ids", out.get("conn", []))
    out.setdefault("met_tribe_ids", out.get("met", []))
    out.setdefault("known_capital_tribe_ids", out.get("known_caps", []))
    out.setdefault("discovered_lighthouses", out.get("lights", []))
    out.setdefault("pacifist_count", out.get("pacifist", 0))
    out.setdefault("units_disabled_next_turn", out.get("disabled", False))
    out.setdefault("monuments", out.get("mon", {}))
    return out


def _normalize_action(action: Dict[str, Any]) -> Dict[str, Any]:
    if "type" in action:
        return action
    out = dict(action)
    index = int(out.get("i", 0) or 0)
    out.setdefault("id", f"A{index}")
    out.setdefault("type", out.get("t"))
    out.setdefault("unit_id", out.get("u", 0))
    out.setdefault("city_id", out.get("c", 0))
    out.setdefault("tribe_id", out.get("p", 0))
    out.setdefault("target_unit_id", out.get("tu", 0))
    out.setdefault("target_city_id", out.get("tc", 0))
    out.setdefault("target_player_id", out.get("tp", 0))
    out.setdefault("unit_type", out.get("ut"))
    out.setdefault("building_type", out.get("bt"))
    out.setdefault("resource_type", out.get("rt"))
    if "x" in out and "y" in out:
        pos = {"x": out.get("x"), "y": out.get("y")}
        if out.get("type") == "MOVE":
            out.setdefault("destination", pos)
        elif out.get("type") == "BUILD_ROAD":
            out.setdefault("position", pos)
        else:
            out.setdefault("target_pos", pos)
    return out


def normalize_message(message: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(message)
    if "observation" not in out and "obs" in out:
        out["observation"] = out["obs"]
    if "forward_model" not in out and "fm" in out:
        fm = out.get("fm") or {}
        out["forward_model"] = {"enabled": True, "root_state_id": fm.get("root", "root"), "commands": fm.get("cmd", [])}
    if "winner_id" not in out and "winner" in out:
        out["winner_id"] = out.get("winner")
    if "final_scores" not in out and "scores" in out:
        out["final_scores"] = out.get("scores")
    if "ranking" not in out and "rank" in out:
        out["ranking"] = out.get("rank")
    if "normalized_terminal_reward" not in out and "term" in out:
        out["normalized_terminal_reward"] = out.get("term")
    if "is_terminal" not in out and "terminal" in out:
        out["is_terminal"] = out.get("terminal")
    if "observation" in out:
        observation = dict(out["observation"])
        observation.setdefault("active_player_id", observation.get("active", out.get("player_id", 0)))
        observation.setdefault("can_end_turn", observation.get("end", False))
        observation.setdefault("leveling_up", observation.get("lvlup", False))
        observation.setdefault("ranking", observation.get("rank", []))
        observation["board"] = _normalize_board(dict(observation.get("board", {})))
        observation["units"] = [_normalize_unit(dict(unit)) for unit in observation.get("units", []) or []]
        observation["cities"] = [_normalize_city(dict(city)) for city in observation.get("cities", []) or []]
        observation["tribes"] = [_normalize_tribe(dict(tribe)) for tribe in observation.get("tribes", []) or []]
        out["observation"] = observation
    out["actions"] = [_normalize_action(dict(action)) for action in out.get("actions", []) or []]
    return out


def encode_observation(message: Dict[str, Any], model_cfg: ModelConfig) -> EncodedObservation:
    if not _message_is_normalized(message):
        message = normalize_message(message)
    observation = message["observation"]
    board = observation["board"]
    board_size = int(board["size"])
    channels = np.zeros((model_cfg.board_channels, board_size, board_size), dtype=np.float32)
    my_player_id = int(message["player_id"])

    terrain_offset = 10
    resource_offset = terrain_offset + len(TERRAIN_TYPES)
    building_offset = resource_offset + len(RESOURCE_TYPES)
    board_coord_scale = float(board_size - 1) if board_size > 1 else 1.0

    for row in board.get("tiles", []):
        for tile in row:
            x = int(tile.get("x", 0) or 0)
            y = int(tile.get("y", 0) or 0)
            if not (0 <= x < board_size and 0 <= y < board_size):
                continue
            channels[0, y, x] = 1.0
            channels[2, y, x] = 1.0 if tile.get("explored") else 0.0
            channels[3, y, x] = float(x) / board_coord_scale
            channels[4, y, x] = float(y) / board_coord_scale
            channels[5, y, x] = 1.0 if int(tile.get("unit_id", 0) or 0) > 0 else 0.0
            channels[6, y, x] = 1.0 if int(tile.get("city_id", 0) or 0) > 0 else 0.0
            channels[7, y, x] = 1.0 if tile.get("road") else 0.0
            terrain_idx = TERRAIN_TO_INDEX.get(str(tile.get("terrain")), -1)
            if terrain_idx >= 0:
                channels[terrain_offset + terrain_idx, y, x] = 1.0
            resource_idx = RESOURCE_TO_INDEX.get(str(tile.get("resource")), -1)
            if resource_idx >= 0:
                channels[resource_offset + resource_idx, y, x] = 1.0
            building_idx = BUILDING_TO_INDEX.get(str(tile.get("building")), -1)
            if building_idx >= 0:
                channels[building_offset + building_idx, y, x] = 1.0

    _copy_belief_planes(channels, observation.get("belief"), board_size)

    units = _sorted_entities(observation.get("units", []))
    unit_features = np.zeros((model_cfg.max_units, model_cfg.entity_feature_dim), dtype=np.float32)
    unit_mask = np.zeros(model_cfg.max_units, dtype=np.bool_)
    for idx, unit in enumerate(units[: model_cfg.max_units]):
        unit_mask[idx] = True
        owner = int(unit.get("tribe_id", -1) or -1)
        base = [
            _norm(unit.get("id", 0), 2048.0),
            _norm(unit.get("x", 0), board_size - 1),
            _norm(unit.get("y", 0), board_size - 1),
            1.0 if owner == my_player_id else -1.0,
            _norm(owner, 16.0),
            _clamped_norm(unit.get("current_hp", 0), 40.0),
            _clamped_norm(unit.get("max_hp", 0), 40.0),
            _clamped_norm(unit.get("atk", 0), 16.0),
            _clamped_norm(unit.get("def", 0), 16.0),
            _clamped_norm(unit.get("mov", 0), 8.0),
            _clamped_norm(unit.get("range", 0), 8.0),
            _clamped_norm(unit.get("cost", 0), 32.0),
            _clamped_norm(unit.get("kills", 0), 16.0),
            1.0 if unit.get("is_veteran") else 0.0,
            1.0 if unit.get("is_hidden") else 0.0,
            1.0 if unit.get("hidden_enemy_hint") else 0.0,
            1.0 if str(unit.get("status")) == "FRESH" else 0.0,
            1.0 if str(unit.get("status")) == "EXHAUSTED" else 0.0,
        ]
        _put_feature(unit_features[idx], base + _one_hot(unit.get("type"), UNIT_TYPES))

    cities = _sorted_entities(observation.get("cities", []))
    city_features = np.zeros((model_cfg.max_cities, model_cfg.entity_feature_dim), dtype=np.float32)
    city_mask = np.zeros(model_cfg.max_cities, dtype=np.bool_)
    for idx, city in enumerate(cities[: model_cfg.max_cities]):
        city_mask[idx] = True
        owner = int(city.get("tribe_id", -1) or -1)
        buildings = city.get("buildings", []) or []
        building_counts = [0.0] * len(BUILDING_TYPES)
        for building in buildings:
            building_idx = BUILDING_TO_INDEX.get(str((building or {}).get("type")), -1)
            if building_idx >= 0:
                building_counts[building_idx] += 1.0
        base = [
            _norm(city.get("id", 0), 1024.0),
            _norm(city.get("x", 0), board_size - 1),
            _norm(city.get("y", 0), board_size - 1),
            1.0 if owner == my_player_id else -1.0,
            _norm(owner, 16.0),
            _clamped_norm(city.get("level", 0), 10.0),
            _clamped_norm(city.get("population", 0), 32.0),
            _clamped_norm(city.get("population_need", 0), 32.0),
            _clamped_norm(city.get("production", 0), 32.0),
            _clamped_norm(city.get("points_worth", 0), 512.0),
            1.0 if city.get("is_capital") else 0.0,
            1.0 if city.get("has_walls") else 0.0,
            _clamped_norm(len(buildings), 16.0),
        ]
        _put_feature(city_features[idx], base + [min(1.0, c) for c in building_counts])

    actions = list(message.get("actions", []))
    action_features = np.zeros((model_cfg.max_actions, model_cfg.action_feature_dim), dtype=np.float32)
    action_mask = np.zeros(model_cfg.max_actions, dtype=np.bool_)
    action_ids: List[str] = []
    unit_by_id = {_as_int(unit.get("id")): unit for unit in units}
    city_by_id = {_as_int(city.get("id")): city for city in cities}
    coord_scale = max(1.0, float(board_size - 1))
    empty_unit_summary = _unit_action_summary(None, my_player_id, coord_scale)
    empty_city_summary = _city_action_summary(None, my_player_id, coord_scale)
    empty_tile_summary = _tile_action_summary(None)
    unit_action_summary_by_id = {
        unit_id: _unit_action_summary(unit, my_player_id, coord_scale)
        for unit_id, unit in unit_by_id.items()
        if unit_id > 0
    }
    city_action_summary_by_id = {
        city_id: _city_action_summary(city, my_player_id, coord_scale)
        for city_id, city in city_by_id.items()
        if city_id > 0
    }
    tile_action_summary_by_pos: dict[tuple[int, int], list[float]] = {}
    typed_feature_by_key: dict[tuple[Any, Any, Any, Any], list[float]] = {}
    for idx, action in enumerate(actions[: model_cfg.max_actions]):
        action_mask[idx] = True
        action_ids.append(str(action["id"]))
        pos = _position_payload(action)
        # Action ids and entity ids are lookup/alignment handles only. Learned
        # action features use semantic entity/tile summaries so the network can
        # understand what an action touches without depending on arbitrary ids.
        source_unit_id = _as_int(action.get("unit_id"))
        source_city_id = _as_int(action.get("city_id"))
        source_unit = unit_by_id.get(source_unit_id) if source_unit_id > 0 else None
        source_city = city_by_id.get(source_city_id) if source_city_id > 0 else None
        source_x, source_y, has_source_position = _entity_position(source_unit or source_city)
        target_unit_id = _as_int(action.get("target_unit_id"))
        target_city_id = _as_int(action.get("target_city_id"))
        target_unit = unit_by_id.get(target_unit_id)
        target_city = city_by_id.get(target_city_id)
        target_x, target_y, has_target_position = _entity_position(pos if isinstance(pos, dict) else None)
        if not has_target_position:
            target_x, target_y, has_target_position = _entity_position(target_unit or target_city)
        target_tile_summary = empty_tile_summary
        if isinstance(pos, dict) and pos:
            pos_x = _as_int(pos.get("x"), -1)
            pos_y = _as_int(pos.get("y"), -1)
            if pos_x >= 0 and pos_y >= 0:
                pos_key = (pos_x, pos_y)
                target_tile_summary = tile_action_summary_by_pos.get(pos_key)
                if target_tile_summary is None:
                    target_tile_summary = _tile_action_summary(_tile_at(board, pos_x, pos_y))
                    tile_action_summary_by_pos[pos_key] = target_tile_summary
        delta_x = target_x - source_x if has_source_position and has_target_position else 0.0
        delta_y = target_y - source_y if has_source_position and has_target_position else 0.0
        abs_delta_x = abs(delta_x)
        abs_delta_y = abs(delta_y)
        spatial = [
            _norm(source_x, coord_scale) if has_source_position else 0.0,
            _norm(source_y, coord_scale) if has_source_position else 0.0,
            _norm(target_x, coord_scale) if has_target_position else 0.0,
            _norm(target_y, coord_scale) if has_target_position else 0.0,
            _clamped_norm(delta_x, coord_scale),
            _clamped_norm(delta_y, coord_scale),
            _norm(abs_delta_x, coord_scale),
            _norm(abs_delta_y, coord_scale),
            _norm(abs_delta_x + abs_delta_y, 2.0 * coord_scale),
            _norm(max(abs_delta_x, abs_delta_y), coord_scale),
        ]
        intent = [
            1.0 if action.get("destination") else 0.0,
            1.0 if action.get("target_pos") else 0.0,
            1.0 if action.get("position") else 0.0,
            1.0 if str(action.get("type")) == "END_TURN" else 0.0,
            1.0 if str(action.get("type")) in ("ATTACK", "CAPTURE", "CONVERT") else 0.0,
        ]
        semantic = (
            unit_action_summary_by_id.get(source_unit_id, empty_unit_summary)
            + unit_action_summary_by_id.get(target_unit_id, empty_unit_summary)
            + city_action_summary_by_id.get(source_city_id, empty_city_summary)
            + city_action_summary_by_id.get(target_city_id, empty_city_summary)
            + target_tile_summary
        )
        typed_key = (
            action.get("type"),
            action.get("unit_type"),
            action.get("building_type"),
            action.get("resource_type"),
            action.get("tech"),
        )
        typed = typed_feature_by_key.get(typed_key)
        if typed is None:
            typed = (
                _one_hot(typed_key[0], ACTION_TYPES)
                + _one_hot(typed_key[1], UNIT_TYPES)
                + _one_hot(typed_key[2], BUILDING_TYPES)
                + _one_hot(typed_key[3], RESOURCE_TYPES)
                + _one_hot(typed_key[4], TECH_TYPES)
            )
            typed_feature_by_key[typed_key] = typed
        _put_feature(action_features[idx], spatial + intent + semantic + typed)

    tribes = observation.get("tribes", [])
    my_tribe = next((tribe for tribe in tribes if int(tribe.get("id", -1) or -1) == my_player_id), {})
    scores = [float(tribe.get("score", 0) or 0) for tribe in tribes]
    my_score = float(my_tribe.get("score", 0) or 0)
    max_other_score = max([score for tribe, score in zip(tribes, scores) if int(tribe.get("id", -1) or -1) != my_player_id] or [0.0])
    explored_tiles = sum(1 for row in board.get("tiles", []) for tile in row if tile.get("explored"))
    scalar_values = [
        _norm(observation.get("tick", 0), 256.0),
        _norm(my_player_id, 16.0),
        _norm(observation.get("active_player_id", 0), 16.0),
        _norm(my_tribe.get("stars", 0), 128.0),
        _norm(my_score, 10000.0),
        _norm(my_score - max_other_score, 10000.0),
        _norm(len(my_tribe.get("researched_tech_ids", []) or []), 64.0),
        _norm(len(my_tribe.get("cities", []) or []), 64.0),
        1.0 if observation.get("can_end_turn") else 0.0,
        1.0 if observation.get("leveling_up") else 0.0,
        _norm(len(actions), model_cfg.max_actions),
        _norm(len(units), model_cfg.max_units),
        _norm(len(cities), model_cfg.max_cities),
        _norm(explored_tiles, board_size * board_size),
        _norm(len(tribes), 16.0),
        _norm(board_size, 30.0),
        _norm(board_size * board_size, 900.0),
        _norm(board_size, max(1.0, float(model_cfg.board_size))),
    ]
    scalar = np.zeros(model_cfg.scalar_dim, dtype=np.float32)
    _put_feature(scalar, scalar_values + _belief_opponent_scalars(observation.get("belief")))

    return EncodedObservation(
        board=torch.from_numpy(channels).unsqueeze(0),
        unit_features=torch.from_numpy(unit_features).unsqueeze(0),
        unit_mask=torch.from_numpy(unit_mask).unsqueeze(0),
        city_features=torch.from_numpy(city_features).unsqueeze(0),
        city_mask=torch.from_numpy(city_mask).unsqueeze(0),
        action_features=torch.from_numpy(action_features).unsqueeze(0),
        action_mask=torch.from_numpy(action_mask).unsqueeze(0),
        scalar_features=torch.from_numpy(scalar).unsqueeze(0),
        action_ids=action_ids,
    )
