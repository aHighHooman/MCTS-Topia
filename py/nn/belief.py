from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping

import numpy as np

BELIEF_VERSION = 1
BELIEF_PLANE_CHANNEL_START = 41
BELIEF_OPPONENT_SCALAR_DIM = 70
MAX_BELIEF_OPPONENTS = 7

BELIEF_PLANE_NAMES = (
    "own_unit_presence",
    "enemy_unit_presence",
    "own_city_center",
    "enemy_city_center",
    "own_territory",
    "enemy_territory",
    "unexplored",
    "frontier_unexplored",
    "deep_unexplored",
    "possible_enemy_capital",
    "hidden_cloak_hint_source",
    "possible_hidden_cloak",
    "known_enemy_threat",
    "possible_hidden_cloak_threat",
    "known_enemy_city_zone",
    "last_seen_enemy_unit",
    "possible_enemy_unit_position",
    "possible_enemy_melee_threat",
    "possible_enemy_ranged_threat",
    "enemy_belief_age",
)

UNIT_TECH_EVIDENCE = {
    "RIDER": "RIDING",
    "ARCHER": "ARCHERY",
    "DEFENDER": "STRATEGY",
    "SWORDMAN": "SMITHERY",
    "SWORDSMAN": "SMITHERY",
    "KNIGHT": "CHIVALRY",
    "CATAPULT": "MATHEMATICS",
    "MIND_BENDER": "PHILOSOPHY",
    "CLOAK": "DIPLOMACY",
    "DAGGER": "DIPLOMACY",
    "SCOUT": "RIDING",
    "DINGHY": "SAILING",
    "RAFT": "SAILING",
    "BOAT": "SAILING",
    "SHIP": "NAVIGATION",
    "BATTLESHIP": "NAVIGATION",
    "BOMBER": "AQUATISM",
    "JUGGERNAUT": "AQUATISM",
    "PIRATE": "NAVIGATION",
    "RAMMER": "NAVIGATION",
}

BUILDING_TECH_EVIDENCE = {
    "PORT": "FISHING",
    "DOCK": "FISHING",
    "MINE": "MINING",
    "FORGE": "MINING",
    "FARM": "FARMING",
    "WINDMILL": "FARMING",
    "LUMBER_HUT": "FORESTRY",
    "SAWMILL": "FORESTRY",
    "MARKET": "TRADE",
    "CUSTOMS_HOUSE": "TRADE",
    "EMBASSY": "DIPLOMACY",
}

# Mirrors Types.TECHNOLOGY parent links in src/core/Types.java.
# Direct evidence comes from observed objects; prerequisite evidence is
# deterministic inference from the game tech tree, not a learned guess.
TECH_PREREQUISITES = {
    "ARCHERY": ("HUNTING",),
    "FARMING": ("ORGANIZATION",),
    "FORESTRY": ("HUNTING",),
    "FREE_SPIRIT": ("RIDING",),
    "MEDITATION": ("CLIMBING",),
    "MINING": ("CLIMBING",),
    "ROADS": ("RIDING",),
    "RAMMING": ("FISHING",),
    "SAILING": ("FISHING",),
    "STRATEGY": ("ORGANIZATION",),
    "AQUATISM": ("RAMMING", "FISHING"),
    "CHIVALRY": ("FREE_SPIRIT", "RIDING"),
    "CONSTRUCTION": ("FARMING", "ORGANIZATION"),
    "DIPLOMACY": ("STRATEGY", "ORGANIZATION"),
    "MATHEMATICS": ("FORESTRY", "HUNTING"),
    "NAVIGATION": ("SAILING", "FISHING"),
    "SMITHERY": ("MINING", "CLIMBING"),
    "SPIRITUALISM": ("ARCHERY", "HUNTING"),
    "TRADE": ("ROADS", "RIDING"),
    "PHILOSOPHY": ("MEDITATION", "CLIMBING"),
}

MILITARY_TECHS = {"ARCHERY", "STRATEGY", "SMITHERY", "CHIVALRY", "MATHEMATICS", "PHILOSOPHY"}
ECONOMY_TECHS = {"MINING", "FARMING", "FORESTRY", "TRADE", "FISHING"}
NAVAL_TECHS = {"SAILING", "NAVIGATION", "AQUATISM"}
STRATEGY_DIPLOMACY_TECHS = {"STRATEGY", "DIPLOMACY"}


def _empty_planes(size: int) -> dict[str, list[list[float]]]:
    return {name: [[0.0 for _ in range(size)] for _ in range(size)] for name in BELIEF_PLANE_NAMES}


def _empty_numpy_planes(size: int) -> dict[str, np.ndarray]:
    stack = np.zeros((len(BELIEF_PLANE_NAMES), size, size), dtype=np.float32)
    return {name: stack[index] for index, name in enumerate(BELIEF_PLANE_NAMES)}


def _set(planes: dict[str, Any], name: str, x: Any, y: Any, value: float = 1.0) -> None:
    plane = planes[name]
    size = len(plane)
    try:
        ix = int(x)
        iy = int(y)
    except (TypeError, ValueError):
        return
    if 0 <= ix < size and 0 <= iy < size:
        row = plane[iy]
        fvalue = float(value)
        if fvalue > row[ix]:
            row[ix] = fvalue


def _neighbors(x: int, y: int, size: int) -> Iterable[tuple[int, int]]:
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            ny = y + dy
            if 0 <= nx < size and 0 <= ny < size:
                yield nx, ny


def _distance(a_x: int, a_y: int, b_x: int, b_y: int) -> int:
    return max(abs(a_x - b_x), abs(a_y - b_y))


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _norm(value: Any, scale: float) -> float:
    try:
        return float(value) / max(1.0, float(scale))
    except (TypeError, ValueError):
        return 0.0


def _tech_name(value: Any) -> str:
    return str(value or "").strip().upper().replace(" ", "_")


@dataclass
class LastSeenEnemyUnit:
    owner: int
    unit_type: str
    x: int
    y: int
    current_hp: int
    max_hp: int
    attack_range: int
    movement: int
    last_seen_tick: int

    def copy(self) -> "LastSeenEnemyUnit":
        return LastSeenEnemyUnit(**self.__dict__)


@dataclass
class BeliefSnapshot:
    player_id: int
    tech_evidence: dict[int, set[str]]
    direct_tech_evidence: dict[int, set[str]]
    inferred_tech_evidence: dict[int, set[str]]
    observed_capitals: set[int]
    last_seen_enemy_units: dict[int, LastSeenEnemyUnit]

    def annotate_without_update(self, message: Mapping[str, Any]) -> dict[str, Any]:
        from .encoding import normalize_message

        normalized = normalize_message(dict(message))
        normalized["observation"]["belief"] = _build_belief(
            normalized,
            tech_evidence=self.tech_evidence,
            direct_tech_evidence=self.direct_tech_evidence,
            inferred_tech_evidence=self.inferred_tech_evidence,
            observed_capitals=set(self.observed_capitals),
            last_seen_enemy_units=self.last_seen_enemy_units,
            update=False,
            numpy_planes=True,
        )
        return normalized


class BeliefTracker:
    def __init__(self) -> None:
        self.tech_evidence: dict[int, set[str]] = {}
        self.direct_tech_evidence: dict[int, set[str]] = {}
        self.inferred_tech_evidence: dict[int, set[str]] = {}
        self.observed_capitals: set[int] = set()
        self.last_seen_enemy_units: dict[int, LastSeenEnemyUnit] = {}
        self.player_id: int | None = None

    def reset(self) -> None:
        self.tech_evidence.clear()
        self.direct_tech_evidence.clear()
        self.inferred_tech_evidence.clear()
        self.observed_capitals.clear()
        self.last_seen_enemy_units.clear()
        self.player_id = None

    def snapshot(self, player_id: int | None = None) -> BeliefSnapshot:
        return BeliefSnapshot(
            player_id=int(self.player_id if player_id is None and self.player_id is not None else (player_id or 0)),
            tech_evidence={tribe_id: set(techs) for tribe_id, techs in self.tech_evidence.items()},
            direct_tech_evidence={tribe_id: set(techs) for tribe_id, techs in self.direct_tech_evidence.items()},
            inferred_tech_evidence={tribe_id: set(techs) for tribe_id, techs in self.inferred_tech_evidence.items()},
            observed_capitals=set(self.observed_capitals),
            last_seen_enemy_units={unit_id: unit.copy() for unit_id, unit in self.last_seen_enemy_units.items()},
        )

    def annotate(self, message: Mapping[str, Any]) -> dict[str, Any]:
        from .encoding import normalize_message

        normalized = normalize_message(copy.deepcopy(dict(message)))
        self.player_id = int(normalized.get("player_id", 0) or 0)
        normalized["observation"]["belief"] = _build_belief(
            normalized,
            tech_evidence=self.tech_evidence,
            direct_tech_evidence=self.direct_tech_evidence,
            inferred_tech_evidence=self.inferred_tech_evidence,
            observed_capitals=self.observed_capitals,
            last_seen_enemy_units=self.last_seen_enemy_units,
            update=True,
        )
        return normalized


def _build_belief(
    message: dict[str, Any],
    *,
    tech_evidence: dict[int, set[str]],
    direct_tech_evidence: dict[int, set[str]],
    inferred_tech_evidence: dict[int, set[str]],
    observed_capitals: set[int],
    last_seen_enemy_units: dict[int, LastSeenEnemyUnit],
    update: bool,
    numpy_planes: bool = False,
) -> dict[str, Any]:
    observation = message.get("observation", {})
    board = observation.get("board", {})
    size = int(board.get("size", 0) or 0)
    planes = _empty_numpy_planes(size) if numpy_planes else _empty_planes(size)
    use_numpy = numpy_planes
    player_id = int(message.get("player_id", observation.get("active_player_id", 0)) or 0)
    units = list(observation.get("units", []) or [])
    cities = list(observation.get("cities", []) or [])
    tribes = list(observation.get("tribes", []) or [])
    unit_info = [
        (
            unit,
            _as_int(unit.get("id"), -1),
            _as_int(unit.get("tribe_id"), -1),
            _as_int(unit.get("x"), -1),
            _as_int(unit.get("y"), -1),
        )
        for unit in units
    ]
    city_info = [
        (
            city,
            _as_int(city.get("id"), -1),
            _as_int(city.get("tribe_id"), -1),
            _as_int(city.get("x"), -1),
            _as_int(city.get("y"), -1),
        )
        for city in cities
    ]
    city_owner = {city_id: owner for _city, city_id, owner, _x, _y in city_info}
    unit_owner_by_id = {unit_id: owner for _unit, unit_id, owner, _x, _y in unit_info}
    own_occupied: set[tuple[int, int]] = {(x, y) for _unit, _unit_id, owner, x, y in unit_info if owner == player_id}

    if update:
        _update_evidence(
            tech_evidence,
            direct_tech_evidence,
            inferred_tech_evidence,
            observed_capitals,
            player_id,
            units,
            cities,
        )
        _update_last_seen_units(last_seen_enemy_units, player_id, units, _as_int(observation.get("tick"), 0))

    own_unit_presence = planes["own_unit_presence"]
    enemy_unit_presence = planes["enemy_unit_presence"]
    own_city_center = planes["own_city_center"]
    enemy_city_center = planes["enemy_city_center"]
    own_territory = planes["own_territory"]
    enemy_territory = planes["enemy_territory"]
    unexplored_plane = planes["unexplored"]
    frontier_unexplored = planes["frontier_unexplored"]
    deep_unexplored = planes["deep_unexplored"]
    possible_enemy_capital = planes["possible_enemy_capital"]
    hidden_cloak_hint_source = planes["hidden_cloak_hint_source"]
    possible_hidden_cloak = planes["possible_hidden_cloak"]
    possible_hidden_cloak_threat = planes["possible_hidden_cloak_threat"]
    known_enemy_city_zone = planes["known_enemy_city_zone"]

    for unit, _unit_id, owner, x, y in unit_info:
        if owner == player_id:
            if use_numpy and 0 <= x < size and 0 <= y < size:
                own_unit_presence[y, x] = 1.0
            else:
                _set(planes, "own_unit_presence", x, y)
            if unit.get("hidden_enemy_hint") or unit.get("hint"):
                if use_numpy and 0 <= x < size and 0 <= y < size:
                    hidden_cloak_hint_source[y, x] = 1.0
                else:
                    _set(planes, "hidden_cloak_hint_source", x, y)
                for nx, ny in _neighbors(x, y, size):
                    if (nx, ny) not in own_occupied:
                        if use_numpy:
                            if possible_hidden_cloak[ny, nx] < 0.5:
                                possible_hidden_cloak[ny, nx] = 0.5
                        else:
                            possible_hidden_cloak[ny][nx] = max(possible_hidden_cloak[ny][nx], 0.5)
        elif owner >= 0:
            if use_numpy and 0 <= x < size and 0 <= y < size:
                enemy_unit_presence[y, x] = 1.0
            else:
                _set(planes, "enemy_unit_presence", x, y)
            _mark_known_threat(planes, unit, size)

    for city, _city_id, owner, x, y in city_info:
        if owner == player_id:
            if use_numpy and 0 <= x < size and 0 <= y < size:
                own_city_center[y, x] = 1.0
            else:
                _set(planes, "own_city_center", x, y)
        elif owner >= 0:
            if use_numpy and 0 <= x < size and 0 <= y < size:
                enemy_city_center[y, x] = 1.0
                known_enemy_city_zone[y, x] = 1.0
            else:
                _set(planes, "enemy_city_center", x, y)
                _set(planes, "known_enemy_city_zone", x, y)
            if city.get("is_capital"):
                observed_capitals.add(owner)
                if use_numpy and 0 <= x < size and 0 <= y < size:
                    possible_enemy_capital[y, x] = 1.0
                else:
                    _set(planes, "possible_enemy_capital", x, y, 1.0)

    unexplored_tiles: list[tuple[int, int]] = []
    explored_grid = np.zeros((size, size), dtype=np.bool_) if use_numpy else [[False for _ in range(size)] for _ in range(size)]
    for y, row in enumerate(board.get("tiles", []) or []):
        if not isinstance(row, list):
            continue
        for x, tile in enumerate(row[:size]):
            if not isinstance(tile, dict):
                continue
            tx = _as_int(tile.get("x"), x)
            ty = _as_int(tile.get("y"), y)
            if not (0 <= tx < size and 0 <= ty < size):
                continue
            if tile.get("explored"):
                explored_grid[ty][tx] = True
                city_id = _as_int(tile.get("city_id"), 0)
                owner = city_owner.get(city_id, -1)
                if owner == player_id:
                    if use_numpy:
                        own_territory[ty, tx] = 1.0
                    else:
                        _set(planes, "own_territory", tx, ty)
                elif owner >= 0:
                    if use_numpy:
                        enemy_territory[ty, tx] = 1.0
                        known_enemy_city_zone[ty, tx] = 1.0
                    else:
                        _set(planes, "enemy_territory", tx, ty)
                        _set(planes, "known_enemy_city_zone", tx, ty)
                unit_id = _as_int(tile.get("unit_id"), 0)
                unit_owner = unit_owner_by_id.get(unit_id, -1)
                if unit_owner >= 0:
                    if unit_owner == player_id:
                        if use_numpy:
                            own_unit_presence[ty, tx] = 1.0
                        else:
                            _set(planes, "own_unit_presence", tx, ty)
                    else:
                        if use_numpy:
                            enemy_unit_presence[ty, tx] = 1.0
                        else:
                            _set(planes, "enemy_unit_presence", tx, ty)
            else:
                if use_numpy:
                    unexplored_plane[ty, tx] = 1.0
                else:
                    _set(planes, "unexplored", tx, ty)
                unexplored_tiles.append((tx, ty))

    if use_numpy:
        if unexplored_tiles:
            padded = np.pad(explored_grid, 1, mode="constant", constant_values=False)
            neighbor_explored = (
                padded[:-2, :-2]
                | padded[:-2, 1:-1]
                | padded[:-2, 2:]
                | padded[1:-1, :-2]
                | padded[1:-1, 2:]
                | padded[2:, :-2]
                | padded[2:, 1:-1]
                | padded[2:, 2:]
            )
            unexplored_mask = unexplored_plane > 0.0
            frontier_unexplored[unexplored_mask & neighbor_explored] = 1.0
            deep_unexplored[unexplored_mask & ~neighbor_explored] = 1.0
    else:
        for x, y in unexplored_tiles:
            frontier = False
            for ny in range(max(0, y - 1), min(size, y + 2)):
                for nx in range(max(0, x - 1), min(size, x + 2)):
                    if (nx != x or ny != y) and explored_grid[ny][nx]:
                        frontier = True
                        break
                if frontier:
                    break
            _set(planes, "frontier_unexplored" if frontier else "deep_unexplored", x, y)

    visible_enemy_unit_ids = {
        unit_id
        for _unit, unit_id, owner, _x, _y in unit_info
        if owner >= 0 and owner != player_id
    }
    _mark_possible_enemy_positions(planes, last_seen_enemy_units, visible_enemy_unit_ids, size, _as_int(observation.get("tick"), 0))

    opponent_ids = sorted(_as_int(tribe.get("id"), -1) for tribe in tribes if _as_int(tribe.get("id"), -1) != player_id)
    if any(tribe_id not in observed_capitals for tribe_id in opponent_ids):
        if use_numpy and unexplored_tiles:
            unexplored_mask = unexplored_plane > 0.0
            possible_enemy_capital[unexplored_mask] = np.maximum(possible_enemy_capital[unexplored_mask], np.float32(0.25))
        else:
            for x, y in unexplored_tiles:
                possible_enemy_capital[y][x] = max(possible_enemy_capital[y][x], 0.25)

    if use_numpy:
        cloak_mask = possible_hidden_cloak > 0.0
        if cloak_mask.any():
            padded = np.pad(cloak_mask, 1, mode="constant", constant_values=False)
            threat_mask = (
                padded[:-2, :-2]
                | padded[:-2, 1:-1]
                | padded[:-2, 2:]
                | padded[1:-1, :-2]
                | padded[1:-1, 2:]
                | padded[2:, :-2]
                | padded[2:, 1:-1]
                | padded[2:, 2:]
            )
            possible_hidden_cloak_threat[threat_mask] = np.maximum(possible_hidden_cloak_threat[threat_mask], np.float32(0.35))
    else:
        for y, row in enumerate(possible_hidden_cloak):
            for x, value in enumerate(row):
                if value > 0.0:
                    for nx, ny in _neighbors(x, y, size):
                        possible_hidden_cloak_threat[ny][nx] = max(possible_hidden_cloak_threat[ny][nx], 0.35)

    return {
        "version": BELIEF_VERSION,
        "planes": planes,
        "opponent_scalars": _opponent_scalars(player_id, tribes, units, cities, tech_evidence, observed_capitals),
    }


def _max_square(plane: Any, x: int, y: int, radius: int, value: float, size: int) -> None:
    if radius < 0 or not (0 <= x < size and 0 <= y < size):
        return
    y0 = max(0, y - radius)
    y1 = min(size, y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(size, x + radius + 1)
    if isinstance(plane, np.ndarray):
        np.maximum(plane[y0:y1, x0:x1], np.float32(value), out=plane[y0:y1, x0:x1])
        return
    for row_index in range(y0, y1):
        row = plane[row_index]
        for col_index in range(x0, x1):
            if value > row[col_index]:
                row[col_index] = value


def _mark_known_threat(planes: dict[str, Any], unit: Mapping[str, Any], size: int) -> None:
    ux = _as_int(unit.get("x"), -1)
    uy = _as_int(unit.get("y"), -1)
    attack_range = max(1, _as_int(unit.get("range"), 1))
    movement = max(0, _as_int(unit.get("movement"), 0))
    threat_plane = planes["known_enemy_threat"]
    if isinstance(threat_plane, np.ndarray):
        if movement > 0:
            _max_square(threat_plane, ux, uy, attack_range + movement, 0.35, size)
        _max_square(threat_plane, ux, uy, attack_range, 1.0, size)
        return
    for y in range(size):
        for x in range(size):
            dist = _distance(ux, uy, x, y)
            if dist <= attack_range:
                threat_plane[y][x] = max(threat_plane[y][x], 1.0)
            elif dist <= attack_range + movement:
                threat_plane[y][x] = max(threat_plane[y][x], 0.35)


def _decay_confidence(age: int) -> float:
    """Simple deterministic decay for stale sightings: now=1.0, 1 turn=0.7, 2 turns=0.4, older=0.2."""
    if age <= 0:
        return 1.0
    if age == 1:
        return 0.7
    if age == 2:
        return 0.4
    return 0.2


def _add_tech_evidence(
    tech_evidence: dict[int, set[str]],
    direct_tech_evidence: dict[int, set[str]],
    inferred_tech_evidence: dict[int, set[str]],
    owner: int,
    tech: str,
) -> None:
    direct = _tech_name(tech)
    if not direct:
        return
    # Direct evidence is something visible in the observation: a unit/building
    # that could not exist without the tech.
    direct_tech_evidence.setdefault(owner, set()).add(direct)
    tech_evidence.setdefault(owner, set()).add(direct)
    # Prerequisites are deterministic tech-tree implications.
    for prerequisite in TECH_PREREQUISITES.get(direct, ()):
        inferred_tech_evidence.setdefault(owner, set()).add(prerequisite)
        tech_evidence.setdefault(owner, set()).add(prerequisite)


def _update_last_seen_units(
    last_seen_enemy_units: dict[int, LastSeenEnemyUnit],
    player_id: int,
    units: list[dict[str, Any]],
    tick: int,
) -> None:
    for unit in units:
        owner = _as_int(unit.get("tribe_id"), -1)
        unit_id = _as_int(unit.get("id"), -1)
        if owner == player_id or owner < 0 or unit_id < 0:
            continue
        last_seen_enemy_units[unit_id] = LastSeenEnemyUnit(
            owner=owner,
            unit_type=_tech_name(unit.get("type")),
            x=_as_int(unit.get("x"), -1),
            y=_as_int(unit.get("y"), -1),
            current_hp=_as_int(unit.get("current_hp"), 0),
            max_hp=_as_int(unit.get("max_hp"), 0),
            attack_range=max(1, _as_int(unit.get("range"), 1)),
            movement=max(1, _as_int(unit.get("movement"), 1)),
            last_seen_tick=tick,
        )


def _mark_possible_enemy_positions(
    planes: dict[str, Any],
    last_seen_enemy_units: dict[int, LastSeenEnemyUnit],
    visible_enemy_unit_ids: set[int],
    size: int,
    tick: int,
) -> None:
    possible_position = planes["possible_enemy_unit_position"]
    belief_age = planes["enemy_belief_age"]
    melee_threat = planes["possible_enemy_melee_threat"]
    ranged_threat = planes["possible_enemy_ranged_threat"]
    use_numpy = isinstance(possible_position, np.ndarray)
    for unit_id, unit in last_seen_enemy_units.items():
        if unit_id in visible_enemy_unit_ids:
            _set(planes, "last_seen_enemy_unit", unit.x, unit.y, 1.0)
            continue
        age = max(0, tick - unit.last_seen_tick)
        confidence = _decay_confidence(age)
        radius = max(0, unit.movement * max(1, age))
        _set(planes, "last_seen_enemy_unit", unit.x, unit.y, confidence)
        if use_numpy:
            _max_square(possible_position, unit.x, unit.y, radius, confidence, size)
            _max_square(belief_age, unit.x, unit.y, radius, confidence, size)
            target_plane = ranged_threat if unit.attack_range > 1 else melee_threat
            _max_square(target_plane, unit.x, unit.y, radius + unit.attack_range, confidence * 0.7, size)
            continue
        for y in range(max(0, unit.y - radius), min(size, unit.y + radius + 1)):
            for x in range(max(0, unit.x - radius), min(size, unit.x + radius + 1)):
                if _distance(unit.x, unit.y, x, y) > radius:
                    continue
                possible_position[y][x] = max(possible_position[y][x], confidence)
                belief_age[y][x] = max(belief_age[y][x], confidence)
        threat_radius = radius + unit.attack_range
        target = ranged_threat if unit.attack_range > 1 else melee_threat
        for y in range(max(0, unit.y - threat_radius), min(size, unit.y + threat_radius + 1)):
            for x in range(max(0, unit.x - threat_radius), min(size, unit.x + threat_radius + 1)):
                if _distance(unit.x, unit.y, x, y) <= threat_radius:
                    target[y][x] = max(target[y][x], confidence * 0.7)


def _update_evidence(
    tech_evidence: dict[int, set[str]],
    direct_tech_evidence: dict[int, set[str]],
    inferred_tech_evidence: dict[int, set[str]],
    observed_capitals: set[int],
    player_id: int,
    units: list[dict[str, Any]],
    cities: list[dict[str, Any]],
) -> None:
    for unit in units:
        owner = _as_int(unit.get("tribe_id"), -1)
        if owner == player_id or owner < 0:
            continue
        tech = UNIT_TECH_EVIDENCE.get(_tech_name(unit.get("type")))
        if tech:
            _add_tech_evidence(tech_evidence, direct_tech_evidence, inferred_tech_evidence, owner, tech)
    for city in cities:
        owner = _as_int(city.get("tribe_id"), -1)
        if owner == player_id or owner < 0:
            continue
        if city.get("is_capital"):
            observed_capitals.add(owner)
        for building in city.get("buildings", []) or []:
            if not isinstance(building, dict):
                continue
            tech = BUILDING_TECH_EVIDENCE.get(_tech_name(building.get("type")))
            if tech:
                _add_tech_evidence(tech_evidence, direct_tech_evidence, inferred_tech_evidence, owner, tech)


def _opponent_scalars(
    player_id: int,
    tribes: list[dict[str, Any]],
    units: list[dict[str, Any]],
    cities: list[dict[str, Any]],
    tech_evidence: dict[int, set[str]],
    observed_capitals: set[int],
) -> list[float]:
    tribe_by_id = {_as_int(tribe.get("id"), -1): tribe for tribe in tribes}
    my_score = float(tribe_by_id.get(player_id, {}).get("score", 0) or 0)
    out: list[float] = []
    for tribe_id in sorted(tid for tid in tribe_by_id if tid != player_id)[:MAX_BELIEF_OPPONENTS]:
        tribe = tribe_by_id[tribe_id]
        score = float(tribe.get("score", 0) or 0)
        visible_cities = [city for city in cities if _as_int(city.get("tribe_id"), -1) == tribe_id]
        visible_units = [unit for unit in units if _as_int(unit.get("tribe_id"), -1) == tribe_id]
        techs = set(tech_evidence.get(tribe_id, set()))
        out.extend(
            [
                _norm(score, 10000.0),
                _norm(my_score - score, 10000.0),
                _norm(len(visible_cities), 64.0),
                _norm(len(visible_units), 256.0),
                1.0 if tribe_id in observed_capitals else 0.0,
                _norm(len(techs), 64.0),
                1.0 if techs & MILITARY_TECHS else 0.0,
                1.0 if techs & ECONOMY_TECHS else 0.0,
                1.0 if techs & NAVAL_TECHS else 0.0,
                1.0 if techs & STRATEGY_DIPLOMACY_TECHS else 0.0,
            ]
        )
    if len(out) < BELIEF_OPPONENT_SCALAR_DIM:
        out.extend([0.0] * (BELIEF_OPPONENT_SCALAR_DIM - len(out)))
    return out[:BELIEF_OPPONENT_SCALAR_DIM]
