from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np
import torch

from .belief import BUILDING_TECH_EVIDENCE, TECH_PREREQUISITES, UNIT_TECH_EVIDENCE
from search.config import ModelConfig


TERRAIN_TYPES = [
    "PLAIN",
    "SHALLOW_WATER",
    "DEEP_WATER",
    "MOUNTAIN",
    "VILLAGE",
    "CITY",
    "FOREST",
    "FOG",
]
RESOURCE_TYPES = ["FISH", "FRUIT", "ANIMAL", "STARFISH", "LIGHTHOUSE", "ORE", "CROPS", "RUINS"]
BUILDING_TYPES = [
    "PORT",
    "MINE",
    "FORGE",
    "FARM",
    "WINDMILL",
    "MARKET",
    "LUMBER_HUT",
    "SAWMILL",
    "TEMPLE",
    "WATER_TEMPLE",
    "FOREST_TEMPLE",
    "MOUNTAIN_TEMPLE",
    "ALTAR_OF_PEACE",
    "EMPERORS_TOMB",
    "EYE_OF_GOD",
    "GATE_OF_POWER",
    "GRAND_BAZAR",
    "PARK_OF_FORTUNE",
    "TOWER_OF_WISDOM",
    "EMBASSY",
]
UNIT_TYPES = [
    "WARRIOR",
    "RIDER",
    "DEFENDER",
    "SWORDMAN",
    "ARCHER",
    "CATAPULT",
    "KNIGHT",
    "MIND_BENDER",
    "RAFT",
    "SCOUT",
    "BOMBER",
    "SUPERUNIT",
    "CLOAK",
    "DAGGER",
    "RAMMER",
    "JUGGERNAUT",
    "DINGHY",
    "PIRATE",
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
ACTION_TYPE_ALIASES = {
    "GATHER": "RESOURCE_GATHERING",
    "RESEARCH": "RESEARCH_TECH",
}
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
RELATIONSHIP_TYPES = ["WAR", "PEACE", "TREATY", "UNKNOWN"]
UNIT_STATUS_TYPES = ["FRESH", "MOVED", "ATTACKED", "MOVED_AND_ATTACKED", "PUSHED", "FINISHED"]
LEVEL_UP_BONUS_TYPES = [
    "WORKSHOP",
    "EXPLORER",
    "CITY_WALL",
    "RESOURCES",
    "POP_GROWTH",
    "BORDER_GROWTH",
    "PARK",
    "SUPERUNIT",
    "RESEARCH",
    "UNIT",
]

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
RELATIONSHIP_TO_INDEX = {name: idx for idx, name in enumerate(RELATIONSHIP_TYPES)}
UNIT_STATUS_TO_INDEX = {name: idx for idx, name in enumerate(UNIT_STATUS_TYPES)}
LEVEL_UP_BONUS_TO_INDEX = {name: idx for idx, name in enumerate(LEVEL_UP_BONUS_TYPES)}
_BOARD_COORD_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}

BOARD_SCHEMA = (
    "valid",
    "reserved_hidden_authoritative",
    "explored",
    "x",
    "y",
    "tile_unit_id_present",
    "tile_city_id_present",
    "road",
    "reserved_8",
    "reserved_9",
    *(f"terrain:{name}" for name in TERRAIN_TYPES),
    *(f"resource:{name}" for name in RESOURCE_TYPES),
    *(f"building:{name}" for name in BUILDING_TYPES),
    "reserved_41",
    "reserved_42",
    "reserved_43",
    "reserved_44",
    "reserved_45",
    "reserved_46",
    "reserved_47",
    "reserved_48",
    "reserved_49",
    "reserved_50",
    "reserved_51",
    "reserved_52",
    "reserved_53",
    "reserved_54",
    "reserved_55",
    "reserved_56",
    "reserved_57",
    "reserved_58",
    "reserved_59",
    "reserved_60",
    "visible",
    "visible_unit_owner:own",
    "visible_unit_owner:enemy",
    "visible_unit_owner:neutral",
    "visible_city_owner:own",
    "visible_city_owner:enemy",
    "visible_city_owner:neutral",
    *(f"visible_unit_status:{name}" for name in UNIT_STATUS_TYPES),
    "visible_unit_hp_fraction",
    "visible_city_level",
    "visible_territory_owner_signed",
)
BOARD_FEATURE_INDEX = {name: idx for idx, name in enumerate(BOARD_SCHEMA)}

UNIT_FEATURE_SCHEMA = (
    "x",
    "y",
    "owner_signed",
    "current_hp",
    "current_hp_exact",
    "hp_fraction",
    "max_hp",
    "attack",
    "defence",
    "movement",
    "range",
    "cost",
    "kills",
    "is_veteran",
    "is_hidden",
    "hidden_at_turn_start",
    "hidden_enemy_hint",
    *(f"status:{name}" for name in UNIT_STATUS_TYPES),
    *(f"type:{name}" for name in UNIT_TYPES),
)
UNIT_FEATURE_INDEX = {name: idx for idx, name in enumerate(UNIT_FEATURE_SCHEMA)}

CITY_FEATURE_SCHEMA = (
    "x",
    "y",
    "owner_signed",
    "level",
    "population",
    "population_need",
    "production",
    "bound",
    "points_worth",
    "is_capital",
    "has_walls",
    "infiltrated",
    "unit_count",
    "building_count",
    *(f"building_count:{name}" for name in BUILDING_TYPES),
)
CITY_FEATURE_INDEX = {name: idx for idx, name in enumerate(CITY_FEATURE_SCHEMA)}

ACTION_SPATIAL_SCHEMA = (
    "source_x",
    "source_y",
    "target_x",
    "target_y",
    "delta_x",
    "delta_y",
    "abs_delta_x",
    "abs_delta_y",
    "manhattan_delta",
    "chebyshev_delta",
)
ACTION_INTENT_SCHEMA = (
    "has_destination",
    "has_target_pos",
    "has_position",
    "is_end_turn",
    "is_aggressive",
)
UNIT_ACTION_SUMMARY_SCHEMA = (
    "present",
    *(f"type:{name}" for name in UNIT_TYPES),
    "owner:own",
    "owner:enemy",
    "owner:neutral",
    "x",
    "y",
    "current_hp",
    "hp_fraction",
    "max_hp",
    "kills",
    "is_veteran",
    "is_hidden",
    "range",
    *(f"status:{name}" for name in UNIT_STATUS_TYPES),
)
CITY_ACTION_SUMMARY_SCHEMA = (
    "present",
    "owner:own",
    "owner:enemy",
    "owner:neutral",
    "x",
    "y",
    "level",
    "population",
    "population_need",
    "production",
    "is_capital",
    "has_walls",
)
TILE_ACTION_SUMMARY_SCHEMA = (
    "present",
    "explored",
    "road",
    "unit_present",
    "city_present",
    *(f"terrain:{name}" for name in TERRAIN_TYPES),
    *(f"resource:{name}" for name in RESOURCE_TYPES),
    *(f"building:{name}" for name in BUILDING_TYPES),
)
ACTION_NATIVE_CONTEXT_SCHEMA = (
    *(f"capture_type:{name}" for name in TERRAIN_TYPES),
    *(f"target_relationship:{name}" for name in RELATIONSHIP_TYPES),
    "target_player_is_self",
    "pending:propose_peace",
    "pending:accept_peace",
    "pending:propose_treaty",
    "pending:accept_treaty",
    "pending:cancel_treaty",
)
ACTION_TYPED_SCHEMA = (
    *(f"action_type:{name}" for name in ACTION_TYPES),
    *(f"unit_type:{name}" for name in UNIT_TYPES),
    *(f"building_type:{name}" for name in BUILDING_TYPES),
    *(f"resource_type:{name}" for name in RESOURCE_TYPES),
    *(f"tech:{name}" for name in TECH_TYPES),
    *(f"level_up_bonus:{name}" for name in LEVEL_UP_BONUS_TYPES),
)
ACTION_FEATURE_SCHEMA = (
    *ACTION_SPATIAL_SCHEMA,
    *ACTION_INTENT_SCHEMA,
    *(f"source_unit:{name}" for name in UNIT_ACTION_SUMMARY_SCHEMA),
    *(f"target_unit:{name}" for name in UNIT_ACTION_SUMMARY_SCHEMA),
    *(f"source_city:{name}" for name in CITY_ACTION_SUMMARY_SCHEMA),
    *(f"target_city:{name}" for name in CITY_ACTION_SUMMARY_SCHEMA),
    *(f"source_tile:{name}" for name in TILE_ACTION_SUMMARY_SCHEMA),
    *(f"target_tile:{name}" for name in TILE_ACTION_SUMMARY_SCHEMA),
    *ACTION_NATIVE_CONTEXT_SCHEMA,
    *ACTION_TYPED_SCHEMA,
)
ACTION_FEATURE_INDEX = {name: idx for idx, name in enumerate(ACTION_FEATURE_SCHEMA)}
ACTION_NATIVE_CONTEXT_START = ACTION_FEATURE_INDEX["capture_type:PLAIN"]

SCALAR_BASE_SCHEMA = (
    "tick",
    "is_active_player",
    "own_stars",
    "own_score",
    "own_score_margin",
    "own_tech_count",
    "own_cities_compact_count",
    "own_city_count",
    "own_extra_unit_count",
    "own_connected_city_count",
    "own_met_tribe_count",
    "own_known_capital_tribe_count",
    "own_lighthouse_count",
    "own_kills",
    "own_pacifist_count",
    "own_units_disabled_next_turn",
    "can_end_turn",
    "leveling_up",
    "legal_action_count",
    "visible_unit_count",
    "own_visible_unit_count",
    "enemy_visible_unit_count",
    "visible_city_count",
    "own_visible_city_count",
    "enemy_visible_city_count",
    "explored_fraction",
    "visible_fraction",
    "tribe_count",
    "board_size",
    "board_area",
    "board_size_vs_config",
)
SCALAR_MY_TECH_SCHEMA = tuple(f"own_tech:{name}" for name in TECH_TYPES)
SCALAR_MONUMENT_SCHEMA = ("monuments_available", "monuments_built", "monuments_unavailable")
SCALAR_RELATIONSHIP_SCHEMA = tuple(f"relationship_count:{name}" for name in RELATIONSHIP_TYPES) + (
    "pending_incoming_diplomacy",
    "pending_outgoing_diplomacy",
)
SCALAR_OPPONENT_TECH_EVIDENCE_SCHEMA = tuple(f"known_opponent_tech_evidence:{name}" for name in TECH_TYPES) + (
    "known_opponent_tech_evidence_count",
    "known_opponent_military_tech",
    "known_opponent_economy_tech",
    "known_opponent_naval_tech",
    "known_opponent_strategy_diplomacy_tech",
)
SCALAR_FEATURE_SCHEMA = (
    *SCALAR_BASE_SCHEMA,
    *SCALAR_MY_TECH_SCHEMA,
    *SCALAR_MONUMENT_SCHEMA,
    *SCALAR_RELATIONSHIP_SCHEMA,
    *SCALAR_OPPONENT_TECH_EVIDENCE_SCHEMA,
)
SCALAR_FEATURE_INDEX = {name: idx for idx, name in enumerate(SCALAR_FEATURE_SCHEMA)}
SCALAR_MY_TECH_START = SCALAR_FEATURE_INDEX["own_tech:CLIMBING"]
SCALAR_OPPONENT_TECH_EVIDENCE_START = SCALAR_FEATURE_INDEX["known_opponent_tech_evidence:CLIMBING"]


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
        unit_features = self.unit_features
        unit_mask = self.unit_mask
        city_features = self.city_features
        city_mask = self.city_mask
        action_features = self.action_features
        action_mask = self.action_mask
        if not unit_mask.is_cuda:
            max_units = int(unit_mask.sum(dim=1).max().item()) if unit_mask.numel() else 0
            unit_features = unit_features[:, :max_units]
            unit_mask = unit_mask[:, :max_units]
        if not city_mask.is_cuda:
            max_cities = int(city_mask.sum(dim=1).max().item()) if city_mask.numel() else 0
            city_features = city_features[:, :max_cities]
            city_mask = city_mask[:, :max_cities]
        if not action_mask.is_cuda:
            max_actions = int(action_mask.sum(dim=1).max().item()) if action_mask.numel() else 0
            if action_features.shape[1] > 0:
                max_actions = max(1, max_actions)
            action_features = action_features[:, :max_actions]
            action_mask = action_mask[:, :max_actions]
        return EncodedObservation(
            self.board.to(device),
            unit_features.to(device),
            unit_mask.to(device),
            city_features.to(device),
            city_mask.to(device),
            action_features.to(device),
            action_mask.to(device),
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


def _sorted_entities(entities: Iterable[Dict[str, Any]], key: str = "id") -> list[dict[str, Any]]:
    return sorted(entities, key=lambda item: _as_int(item.get(key), -1))


def _put_feature(row: np.ndarray, values: list[float]) -> None:
    count = len(values)
    if count <= row.shape[0]:
        row[:count] = values
    else:
        row[:] = values[: row.shape[0]]


def _board_coordinate_planes(board_size: int) -> tuple[np.ndarray, np.ndarray]:
    cached = _BOARD_COORD_CACHE.get(board_size)
    if cached is not None:
        return cached
    coord_scale = float(board_size - 1) if board_size > 1 else 1.0
    coords = np.arange(board_size, dtype=np.float32) / coord_scale
    x_plane = np.broadcast_to(coords.reshape(1, board_size), (board_size, board_size)).copy()
    y_plane = np.broadcast_to(coords.reshape(board_size, 1), (board_size, board_size)).copy()
    cached = (x_plane, y_plane)
    _BOARD_COORD_CACHE[board_size] = cached
    return cached


def _position_payload(action: dict[str, Any]) -> dict[str, Any]:
    return action.get("destination") or action.get("target_pos") or action.get("position") or {}


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
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


def _owner_bucket(owner: int, my_player_id: int) -> int:
    if owner == my_player_id:
        return 0
    if owner >= 0:
        return 1
    return 2


def _relationship_name(value: Any) -> str:
    name = str(value or "").upper()
    return name if name in RELATIONSHIP_TO_INDEX else "UNKNOWN"


def _unit_status_name(value: Any) -> str:
    return str(value or "").upper()


def _unit_status_flags(value: Any) -> list[float]:
    values = [0.0] * len(UNIT_STATUS_TYPES)
    status = _unit_status_name(value)
    status_idx = UNIT_STATUS_TO_INDEX.get(status, -1)
    if status_idx >= 0:
        values[status_idx] = 1.0
    return values


def _level_up_bonus_flags(value: Any) -> list[float]:
    values = [0.0] * len(LEVEL_UP_BONUS_TYPES)
    bonus_idx = LEVEL_UP_BONUS_TO_INDEX.get(str(value or "").upper(), -1)
    if bonus_idx >= 0:
        values[bonus_idx] = 1.0
    return values


def _relationship_between(observation: dict[str, Any], source_id: int, target_id: int) -> str:
    rows = observation.get("rel") or observation.get("visible_relationships") or observation.get("relationships")
    if not isinstance(rows, list) or source_id < 0 or target_id < 0 or source_id >= len(rows):
        return "UNKNOWN"
    row = rows[source_id]
    if not isinstance(row, list) or target_id >= len(row):
        return "UNKNOWN"
    return _relationship_name(row[target_id])


def _relationship_one_hot(observation: dict[str, Any], source_id: int, target_id: int) -> list[float]:
    values = [0.0] * len(RELATIONSHIP_TYPES)
    values[RELATIONSHIP_TO_INDEX[_relationship_between(observation, source_id, target_id)]] = 1.0
    return values


def _researched_tech_flags(tribe: dict[str, Any]) -> list[float]:
    flags = [0.0] * len(TECH_TYPES)
    for tech in tribe.get("researched_tech_ids", []) or []:
        tech_idx = TECH_TO_INDEX.get(str(tech), -1)
        if tech_idx >= 0:
            flags[tech_idx] = 1.0
    return flags


def _tech_name(value: Any) -> str:
    return str(value or "").strip().upper().replace(" ", "_")


def _add_known_tech_evidence(flags: dict[str, set[str]], owner: int, tech: str, source: str) -> None:
    name = _tech_name(tech)
    if not name or owner < 0:
        return
    flags.setdefault(name, set()).add(source)
    for prerequisite in TECH_PREREQUISITES.get(name, ()):
        flags.setdefault(prerequisite, set()).add(f"prerequisite:{name}")


def _known_opponent_tech_evidence(
    player_id: int,
    units: list[dict[str, Any]],
    cities: list[dict[str, Any]],
) -> dict[str, set[str]]:
    evidence: dict[str, set[str]] = {}
    for unit in units:
        owner = _as_int(unit.get("tribe_id"), -1)
        if owner == player_id or owner < 0:
            continue
        tech = UNIT_TECH_EVIDENCE.get(_tech_name(unit.get("type")))
        if tech:
            _add_known_tech_evidence(evidence, owner, tech, f"visible_unit:{_tech_name(unit.get('type'))}")
    for city in cities:
        owner = _as_int(city.get("tribe_id"), -1)
        if owner == player_id or owner < 0:
            continue
        for building in city.get("buildings", []) or []:
            if not isinstance(building, dict):
                continue
            tech = BUILDING_TECH_EVIDENCE.get(_tech_name(building.get("type")))
            if tech:
                _add_known_tech_evidence(evidence, owner, tech, f"visible_building:{_tech_name(building.get('type'))}")
    return evidence


def _known_opponent_tech_evidence_features(evidence: dict[str, set[str]]) -> list[float]:
    known = set(evidence)
    return (
        [1.0 if tech in known else 0.0 for tech in TECH_TYPES]
        + [
            _norm(len(known), 64.0),
            1.0 if known & {"ARCHERY", "STRATEGY", "SMITHERY", "CHIVALRY", "MATHEMATICS", "PHILOSOPHY"} else 0.0,
            1.0 if known & {"MINING", "FARMING", "FORESTRY", "TRADE", "FISHING"} else 0.0,
            1.0 if known & {"SAILING", "NAVIGATION", "AQUATISM"} else 0.0,
            1.0 if known & {"STRATEGY", "DIPLOMACY"} else 0.0,
        ]
    )


def _monument_status_counts(tribe: dict[str, Any]) -> list[float]:
    counts = {"AVAILABLE": 0.0, "BUILT": 0.0, "UNAVAILABLE": 0.0}
    monuments = tribe.get("monuments", {}) or {}
    if isinstance(monuments, dict):
        for status in monuments.values():
            key = str(status or "").upper()
            if key in counts:
                counts[key] += 1.0
    return [
        _clamped_norm(counts["AVAILABLE"], 16.0),
        _clamped_norm(counts["BUILT"], 16.0),
        _clamped_norm(counts["UNAVAILABLE"], 16.0),
    ]


def _pending_offer_action_flags(action: dict[str, Any]) -> list[float]:
    action_type = str(action.get("type"))
    return [
        1.0 if action_type == "PROPOSE_PEACE" else 0.0,
        1.0 if action_type == "ACCEPT_PEACE" else 0.0,
        1.0 if action_type == "PROPOSE_TREATY" else 0.0,
        1.0 if action_type == "ACCEPT_TREATY" else 0.0,
        1.0 if action_type == "CANCEL_TREATY" else 0.0,
    ]


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
        return [0.0] * (1 + len(UNIT_TYPES) + 3 + 9 + len(UNIT_STATUS_TYPES))
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
        + _unit_status_flags(unit.get("status"))
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
            tile.setdefault("visible", bool(tile.get("explored", False)))
            tile.setdefault("road", False)
            tile.setdefault("territory_city_id", tile.get("territory", tile.get("territory_city", tile.get("city_id", 0))))
            normalized_row.append(tile)
        normalized_rows.append(normalized_row)
    out["tiles"] = normalized_rows
    if not size:
        out["size"] = len(normalized_rows)
    return out


def _normalize_unit(unit: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(unit)
    out.setdefault("tribe_id", -1)
    out.setdefault("city_id", 0)
    out.setdefault("type", None)
    out.setdefault("current_hp", 0)
    out.setdefault("current_hp_exact", out.get("current_hp", 0))
    out.setdefault("max_hp", 0)
    out.setdefault("kills", 0)
    out.setdefault("is_veteran", False)
    out.setdefault("status", None)
    out.setdefault("is_hidden", False)
    out.setdefault("hidden_at_turn_start", False)
    out.setdefault("hidden_enemy_hint", False)
    out.setdefault("attack", 0)
    out.setdefault("defence", 0)
    out.setdefault("movement", 0)
    out.setdefault("range", 0)
    return out


def _normalize_city(city: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(city)
    out.setdefault("tribe_id", -1)
    out.setdefault("level", 0)
    out.setdefault("population", 0)
    out.setdefault("population_need", 0)
    out.setdefault("production", 0)
    out.setdefault("is_capital", False)
    out.setdefault("has_walls", False)
    out.setdefault("bound", 0)
    out.setdefault("points_worth", 0)
    out.setdefault("infiltrated", False)
    out.setdefault("unit_ids", [])
    out.setdefault("buildings", [])
    return out


def _normalize_tribe(tribe: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(tribe)
    out.setdefault("researched_tech_ids", [])
    out.setdefault("capital_id", 0)
    out.setdefault("city_ids", [])
    out.setdefault("extra_unit_ids", [])
    out.setdefault("connected_city_ids", [])
    out.setdefault("met_tribe_ids", [])
    out.setdefault("known_capital_tribe_ids", [])
    out.setdefault("discovered_lighthouses", [])
    out.setdefault("pacifist_count", 0)
    out.setdefault("units_disabled_next_turn", False)
    out.setdefault("monuments", {})
    return out


def _normalize_action(action: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(action)
    out.setdefault("id", "A0")
    out.setdefault("type", None)
    if out.get("type") in ACTION_TYPE_ALIASES:
        out["type"] = ACTION_TYPE_ALIASES[str(out["type"])]
    out.setdefault("unit_id", 0)
    out.setdefault("city_id", 0)
    out.setdefault("tribe_id", 0)
    out.setdefault("target_unit_id", 0)
    out.setdefault("target_city_id", 0)
    out.setdefault("target_player_id", 0)
    out.setdefault("unit_type", None)
    out.setdefault("building_type", None)
    out.setdefault("resource_type", None)
    out.setdefault("capture_type", None)
    out.setdefault("bonus", None)
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
    if "observation" in out:
        observation = dict(out["observation"])
        observation.setdefault("active_player_id", out.get("player_id", 0))
        observation.setdefault("can_end_turn", False)
        observation.setdefault("leveling_up", False)
        observation.setdefault("ranking", [])
        observation["board"] = _normalize_board(dict(observation.get("board", {})))
        observation["units"] = [_normalize_unit(dict(unit)) for unit in observation.get("units", []) or []]
        observation["cities"] = [_normalize_city(dict(city)) for city in observation.get("cities", []) or []]
        observation["tribes"] = [_normalize_tribe(dict(tribe)) for tribe in observation.get("tribes", []) or []]
        out["observation"] = observation
    out["actions"] = [_normalize_action(dict(action)) for action in out.get("actions", []) or []]
    return out


def encode_observation(message: Dict[str, Any], model_cfg: ModelConfig, *, compact: bool = False) -> EncodedObservation:
    message = normalize_message(message)
    observation = message["observation"]
    board = observation["board"]
    board_size = int(board["size"])
    channels = np.zeros((model_cfg.board_channels, board_size, board_size), dtype=np.float32)
    my_player_id = int(message["player_id"])

    terrain_offset = 10
    resource_offset = terrain_offset + len(TERRAIN_TYPES)
    building_offset = resource_offset + len(RESOURCE_TYPES)
    channels[3], channels[4] = _board_coordinate_planes(board_size)

    units_raw = observation.get("units", []) or []
    cities_raw = observation.get("cities", []) or []
    unit_by_id_for_board = {_as_int(unit.get("id")): unit for unit in units_raw}
    city_by_id_for_board = {_as_int(city.get("id")): city for city in cities_raw}
    explored_tiles = 0
    visible_tiles = 0
    for row in board.get("tiles", []):
        for tile in row:
            x = int(tile.get("x", 0) or 0)
            y = int(tile.get("y", 0) or 0)
            if not (0 <= x < board_size and 0 <= y < board_size):
                continue
            channels[0, y, x] = 1.0
            explored = bool(tile.get("explored"))
            visible = bool(tile.get("visible"))
            if explored:
                explored_tiles += 1
                channels[2, y, x] = 1.0
            if visible:
                visible_tiles += 1
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
            if visible:
                channels[BOARD_FEATURE_INDEX["visible"], y, x] = 1.0
            unit_id = _as_int(tile.get("unit_id"))
            unit = unit_by_id_for_board.get(unit_id)
            if unit:
                owner_bucket = _owner_bucket(_as_int(unit.get("tribe_id"), -1), my_player_id)
                channels[BOARD_FEATURE_INDEX["visible_unit_owner:own"] + owner_bucket, y, x] = 1.0
                channels[BOARD_FEATURE_INDEX["visible_unit_hp_fraction"], y, x] = _hp_fraction(unit)
                status = _unit_status_name(unit.get("status"))
                status_idx = UNIT_STATUS_TO_INDEX.get(status, -1)
                if status_idx >= 0:
                    channels[BOARD_FEATURE_INDEX["visible_unit_status:FRESH"] + status_idx, y, x] = 1.0
            city_id = _as_int(tile.get("city_id"))
            city = city_by_id_for_board.get(city_id)
            if city:
                owner_bucket = _owner_bucket(_as_int(city.get("tribe_id"), -1), my_player_id)
                channels[BOARD_FEATURE_INDEX["visible_city_owner:own"] + owner_bucket, y, x] = 1.0
                channels[BOARD_FEATURE_INDEX["visible_city_level"], y, x] = _clamped_norm(city.get("level", 0), 10.0)
            territory_city_id = _as_int(tile.get("territory_city_id"), city_id)
            territory_city = city_by_id_for_board.get(territory_city_id)
            if territory_city:
                channels[BOARD_FEATURE_INDEX["visible_territory_owner_signed"], y, x] = (
                    1.0 if _as_int(territory_city.get("tribe_id"), -1) == my_player_id else -1.0
                )

    units = _sorted_entities(units_raw)
    unit_feature_dim = int(getattr(model_cfg, "unit_feature_dim", getattr(model_cfg, "entity_feature_dim", len(UNIT_FEATURE_SCHEMA))))
    city_feature_dim = int(getattr(model_cfg, "city_feature_dim", getattr(model_cfg, "entity_feature_dim", len(CITY_FEATURE_SCHEMA))))
    unit_limit = min(len(units), model_cfg.max_units) if compact else model_cfg.max_units
    unit_features = np.zeros((unit_limit, unit_feature_dim), dtype=np.float32)
    unit_mask = np.zeros(unit_limit, dtype=np.bool_)
    own_unit_count = 0
    enemy_unit_count = 0
    for idx, unit in enumerate(units[:unit_limit]):
        unit_mask[idx] = True
        owner = _as_int(unit.get("tribe_id"), -1)
        if owner == my_player_id:
            own_unit_count += 1
        elif owner >= 0:
            enemy_unit_count += 1
        base = [
            _norm(unit.get("x", 0), board_size - 1),
            _norm(unit.get("y", 0), board_size - 1),
            1.0 if owner == my_player_id else -1.0,
            _clamped_norm(unit.get("current_hp", 0), 40.0),
            _clamped_norm(unit.get("current_hp_exact", unit.get("current_hp", 0)), 40.0),
            _hp_fraction(unit),
            _clamped_norm(unit.get("max_hp", 0), 40.0),
            _clamped_norm(unit.get("attack", unit.get("atk", 0)), 16.0),
            _clamped_norm(unit.get("defence", unit.get("def", 0)), 16.0),
            _clamped_norm(unit.get("movement", unit.get("mov", 0)), 8.0),
            _clamped_norm(unit.get("range", 0), 8.0),
            _clamped_norm(unit.get("cost", 0), 32.0),
            _clamped_norm(unit.get("kills", 0), 16.0),
            1.0 if unit.get("is_veteran") else 0.0,
            1.0 if unit.get("is_hidden") else 0.0,
            1.0 if unit.get("hidden_at_turn_start") else 0.0,
            1.0 if unit.get("hidden_enemy_hint") else 0.0,
        ]
        _put_feature(unit_features[idx], base + _unit_status_flags(unit.get("status")) + _one_hot(unit.get("type"), UNIT_TYPES))

    cities = _sorted_entities(cities_raw)
    city_limit = min(len(cities), model_cfg.max_cities) if compact else model_cfg.max_cities
    city_features = np.zeros((city_limit, city_feature_dim), dtype=np.float32)
    city_mask = np.zeros(city_limit, dtype=np.bool_)
    own_city_count = 0
    enemy_city_count = 0
    for idx, city in enumerate(cities[:city_limit]):
        city_mask[idx] = True
        owner = _as_int(city.get("tribe_id"), -1)
        if owner == my_player_id:
            own_city_count += 1
        elif owner >= 0:
            enemy_city_count += 1
        buildings = city.get("buildings", []) or []
        building_counts = [0.0] * len(BUILDING_TYPES)
        for building in buildings:
            building_idx = BUILDING_TO_INDEX.get(str((building or {}).get("type")), -1)
            if building_idx >= 0:
                building_counts[building_idx] += 1.0
        base = [
            _norm(city.get("x", 0), board_size - 1),
            _norm(city.get("y", 0), board_size - 1),
            1.0 if owner == my_player_id else -1.0,
            _clamped_norm(city.get("level", 0), 10.0),
            _clamped_norm(city.get("population", 0), 32.0),
            _clamped_norm(city.get("population_need", 0), 32.0),
            _clamped_norm(city.get("production", 0), 32.0),
            _clamped_norm(city.get("bound", 0), 64.0),
            _clamped_norm(city.get("points_worth", 0), 512.0),
            1.0 if city.get("is_capital") else 0.0,
            1.0 if city.get("has_walls") else 0.0,
            1.0 if city.get("infiltrated") else 0.0,
            _clamped_norm(len(city.get("unit_ids", []) or []), 64.0),
            _clamped_norm(len(buildings), 16.0),
        ]
        _put_feature(city_features[idx], base + [min(1.0, c) for c in building_counts])

    actions = list(message.get("actions", []))
    action_limit = min(len(actions), model_cfg.max_actions) if compact else model_cfg.max_actions
    if compact and model_cfg.max_actions > 0:
        action_limit = max(1, action_limit)
    action_features = np.zeros((action_limit, model_cfg.action_feature_dim), dtype=np.float32)
    action_mask = np.zeros(action_limit, dtype=np.bool_)
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
    relationship_feature_by_pair: dict[tuple[int, int], list[float]] = {}
    pending_incoming = 0.0
    pending_outgoing_targets: set[int] = set()
    for idx, action in enumerate(actions[: min(len(actions), action_limit)]):
        action_mask[idx] = True
        action_ids.append(str(action["id"]))
        action_type = str(action.get("type"))
        action_target_player_id = _as_int(action.get("target_player_id"), -1)
        if action_type.startswith("ACCEPT_") and action_target_player_id == my_player_id:
            pending_incoming = 1.0
        elif action_type.startswith("PROPOSE_") and action_target_player_id >= 0:
            pending_outgoing_targets.add(action_target_player_id)
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
        source_tile_summary = empty_tile_summary
        if has_source_position:
            source_key = (_as_int(source_x, -1), _as_int(source_y, -1))
            if source_key[0] >= 0 and source_key[1] >= 0:
                source_tile_summary = tile_action_summary_by_pos.get(source_key)
                if source_tile_summary is None:
                    source_tile_summary = _tile_action_summary(_tile_at(board, source_key[0], source_key[1]))
                    tile_action_summary_by_pos[source_key] = source_tile_summary
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
        elif has_target_position:
            pos_key = (_as_int(target_x, -1), _as_int(target_y, -1))
            if pos_key[0] >= 0 and pos_key[1] >= 0:
                target_tile_summary = tile_action_summary_by_pos.get(pos_key)
                if target_tile_summary is None:
                    target_tile_summary = _tile_action_summary(_tile_at(board, pos_key[0], pos_key[1]))
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
            1.0 if action_type == "END_TURN" else 0.0,
            1.0 if action_type in ("ATTACK", "CAPTURE", "CONVERT") else 0.0,
        ]
        semantic = (
            unit_action_summary_by_id.get(source_unit_id, empty_unit_summary)
            + unit_action_summary_by_id.get(target_unit_id, empty_unit_summary)
            + city_action_summary_by_id.get(source_city_id, empty_city_summary)
            + city_action_summary_by_id.get(target_city_id, empty_city_summary)
            + source_tile_summary
            + target_tile_summary
        )
        target_player_id = action_target_player_id
        action_actor_id = _as_int(action.get("tribe_id"), _as_int(observation.get("active_player_id"), my_player_id))
        relationship_key = (action_actor_id, target_player_id)
        relationship = relationship_feature_by_pair.get(relationship_key)
        if relationship is None:
            relationship = _relationship_one_hot(observation, action_actor_id, target_player_id)
            relationship_feature_by_pair[relationship_key] = relationship
        native_context = (
            _one_hot(action.get("capture_type"), TERRAIN_TYPES)
            + relationship
            + [
                1.0 if target_player_id == my_player_id else 0.0,
            ]
            + _pending_offer_action_flags(action)
        )
        typed_key = (
            action.get("type"),
            action.get("unit_type"),
            action.get("building_type"),
            action.get("resource_type"),
            action.get("tech"),
            action.get("bonus"),
        )
        typed = typed_feature_by_key.get(typed_key)
        if typed is None:
            typed = (
                _one_hot(typed_key[0], ACTION_TYPES)
                + _one_hot(typed_key[1], UNIT_TYPES)
                + _one_hot(typed_key[2], BUILDING_TYPES)
                + _one_hot(typed_key[3], RESOURCE_TYPES)
                + _one_hot(typed_key[4], TECH_TYPES)
                + _level_up_bonus_flags(typed_key[5])
            )
            typed_feature_by_key[typed_key] = typed
        _put_feature(action_features[idx], spatial + intent + semantic + native_context + typed)

    tribes = observation.get("tribes", [])
    my_tribe = next((tribe for tribe in tribes if _as_int(tribe.get("id"), -1) == my_player_id), {})
    scores = [float(tribe.get("score", 0) or 0) for tribe in tribes]
    my_score = float(my_tribe.get("score", 0) or 0)
    max_other_score = max([score for tribe, score in zip(tribes, scores) if _as_int(tribe.get("id"), -1) != my_player_id] or [0.0])
    direct_relationship_counts = [0.0] * len(RELATIONSHIP_TYPES)
    pending_outgoing = 0.0
    for tribe in tribes:
        other_id = _as_int(tribe.get("id"), -1)
        if other_id == my_player_id or other_id < 0:
            continue
        direct_relationship_counts[RELATIONSHIP_TO_INDEX[_relationship_between(observation, my_player_id, other_id)]] += 1.0
        if other_id in pending_outgoing_targets:
            pending_outgoing = 1.0
    scalar_values = [
        _norm(observation.get("tick", 0), 256.0),
        1.0 if int(observation.get("active_player_id", 0) or 0) == my_player_id else 0.0,
        _norm(my_tribe.get("stars", 0), 128.0),
        _norm(my_score, 10000.0),
        _norm(my_score - max_other_score, 10000.0),
        _norm(len(my_tribe.get("researched_tech_ids", []) or []), 64.0),
        _norm(len(my_tribe.get("cities", []) or []), 64.0),
        _norm(len(my_tribe.get("city_ids", []) or []), 64.0),
        _norm(len(my_tribe.get("extra_unit_ids", []) or []), 256.0),
        _norm(len(my_tribe.get("connected_city_ids", []) or []), 64.0),
        _norm(len(my_tribe.get("met_tribe_ids", []) or []), 16.0),
        _norm(len(my_tribe.get("known_capital_tribe_ids", []) or []), 16.0),
        _norm(len(my_tribe.get("discovered_lighthouses", []) or []), 16.0),
        _clamped_norm(my_tribe.get("kills", 0), 256.0),
        _clamped_norm(my_tribe.get("pacifist_count", 0), 64.0),
        1.0 if my_tribe.get("units_disabled_next_turn") else 0.0,
        1.0 if observation.get("can_end_turn") else 0.0,
        1.0 if observation.get("leveling_up") else 0.0,
        _norm(len(actions), model_cfg.max_actions),
        _norm(len(units), model_cfg.max_units),
        _norm(own_unit_count, model_cfg.max_units),
        _norm(enemy_unit_count, model_cfg.max_units),
        _norm(len(cities), model_cfg.max_cities),
        _norm(own_city_count, model_cfg.max_cities),
        _norm(enemy_city_count, model_cfg.max_cities),
        _norm(explored_tiles, board_size * board_size),
        _norm(visible_tiles, board_size * board_size),
        _norm(len(tribes), 16.0),
        _norm(board_size, 30.0),
        _norm(board_size * board_size, 900.0),
        _norm(board_size, max(1.0, float(model_cfg.board_size))),
    ]
    scalar = np.zeros(model_cfg.scalar_dim, dtype=np.float32)
    relationship_scalars = [_norm(count, 16.0) for count in direct_relationship_counts] + [pending_incoming, pending_outgoing]
    opponent_tech_evidence = _known_opponent_tech_evidence(my_player_id, units, cities)
    _put_feature(
        scalar,
        scalar_values
        + _researched_tech_flags(my_tribe)
        + _monument_status_counts(my_tribe)
        + relationship_scalars
        + _known_opponent_tech_evidence_features(opponent_tech_evidence),
    )

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
