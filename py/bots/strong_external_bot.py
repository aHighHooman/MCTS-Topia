import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# This bot is intentionally external-protocol friendly: it does NOT assume access to a
# forward model. Instead it uses richer tactical evaluation, threat maps, strategic modes,
# and turn-local memory to outperform the baseline greedy scorer on many maps.

RNG = random.Random(17)

UNIT_BASE_VALUE = {
    "WARRIOR": 3.0,
    "RIDER": 3.8,
    "ARCHER": 4.1,
    "DEFENDER": 3.6,
    "SWORDMAN": 5.0,
    "KNIGHT": 6.5,
    "CATAPULT": 5.8,
    "MIND_BENDER": 4.8,
    "BOAT": 2.0,
    "SHIP": 3.5,
    "BATTLESHIP": 6.4,
    "SUPERUNIT": 7.5,
}

SPAWN_PREFS_SAFE = {
    "RIDER": 4.4,
    "ARCHER": 4.1,
    "CATAPULT": 3.9,
    "KNIGHT": 4.8,
    "SWORDMAN": 4.0,
    "DEFENDER": 3.4,
    "WARRIOR": 3.2,
    "MIND_BENDER": 2.3,
}

SPAWN_PREFS_DEFENSE = {
    "DEFENDER": 5.2,
    "WARRIOR": 4.4,
    "ARCHER": 4.1,
    "SWORDMAN": 4.7,
    "KNIGHT": 3.0,
    "RIDER": 2.8,
    "CATAPULT": 3.4,
    "MIND_BENDER": 1.8,
}

TECH_RESOURCE_BONUS = {
    "ORGANIZATION": {"FRUIT": 2.0, "CROPS": 0.5},
    "FARMING": {"CROPS": 2.0},
    "CLIMBING": {"MOUNTAIN": 1.2, "ORE": 0.3},
    "MINING": {"ORE": 2.0},
    "HUNTING": {"ANIMAL": 2.0, "FOREST": 0.2},
    "FORESTRY": {"FOREST": 1.4},
    "ARCHERY": {"ANIMAL": 0.6, "FOREST": 0.5},
    "FISHING": {"FISH": 2.0},
    "SAILING": {"WATER": 1.2, "PORT": 0.4},
    "AQUATISM": {"WATER": 0.8},
    "RIDING": {"LAND": 0.7},
    "ROADS": {"CITY": 0.8, "LAND": 0.4},
    "CHIVALRY": {"LAND": 0.8},
    "STRATEGY": {"CITY": 0.8},
    "CONSTRUCTION": {"CITY": 0.6},
    "TRADE": {"WATER": 0.5, "CITY": 0.4},
    "SMITHERY": {"ORE": 0.7},
    "NAVIGATION": {"WATER": 0.8, "STARFISH": 2.3},
    "MATHEMATICS": {"FOREST": 0.4},
}

GOOD_LEVEL_UPS = {"BORDER_GROWTH", "WORKSHOP", "RESOURCES", "SUPERUNIT", "CITY_WALL"}
CITY_DEFENSIVE_LEVEL_UPS = {"CITY_WALL", "SUPERUNIT", "BORDER_GROWTH"}


@dataclass
class BotMemory:
    tick: int = -1
    plan_mode: str = "grow"
    request_count: int = 0
    turn_actions: List[str] = field(default_factory=list)

    def reset_for_tick(self, tick: int, plan_mode: str) -> None:
        if tick != self.tick:
            self.tick = tick
            self.plan_mode = plan_mode
            self.request_count = 0
            self.turn_actions.clear()
        else:
            self.plan_mode = plan_mode


MEMORY = BotMemory()


def chebyshev(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    return max(abs(int(a["x"]) - int(b["x"])), abs(int(a["y"]) - int(b["y"])))


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def pos_key(x: int, y: int) -> Tuple[int, int]:
    return int(x), int(y)


class ObservationView:
    def __init__(self, message: Dict[str, Any]):
        self.message = message
        self.player_id = int(message["player_id"])
        self.observation = message["observation"]
        self.tick = int(self.observation["tick"])
        self.game_mode = str(self.observation.get("game_mode", "CAPITALS"))
        self.active_player_id = int(self.observation.get("active_player_id", self.player_id))
        self.tribes = {int(tribe["id"]): tribe for tribe in self.observation["tribes"]}
        self.cities = {int(city["id"]): city for city in self.observation["cities"]}
        self.units = {int(unit["id"]): unit for unit in self.observation["units"]}
        self.relationships = self.observation.get("visible_relationships", [])
        self.board_size = int(self.observation["board"]["size"])
        self.tiles: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.visible_tiles: List[Dict[str, Any]] = []
        self.explored_tiles: List[Dict[str, Any]] = []
        self.villages: List[Dict[str, Any]] = []
        self.ruins: List[Dict[str, Any]] = []
        self.enemy_cities: List[Dict[str, Any]] = []
        self.enemy_units: List[Dict[str, Any]] = []
        self.my_units: List[Dict[str, Any]] = []
        self.my_cities: List[Dict[str, Any]] = []
        self.known_capitals = set(self.my_tribe().get("known_capital_tribes", []))

        for row in self.observation["board"]["tiles"]:
            for tile in row:
                key = pos_key(tile["x"], tile["y"])
                self.tiles[key] = tile
                if tile["visible"]:
                    self.visible_tiles.append(tile)
                if tile["explored"]:
                    self.explored_tiles.append(tile)
                if tile["visible"] and tile["terrain"] == "VILLAGE":
                    self.villages.append(tile)
                if tile["visible"] and tile.get("resource") == "RUINS":
                    self.ruins.append(tile)

        for city in self.cities.values():
            if int(city["tribe_id"]) == self.player_id:
                self.my_cities.append(city)
            else:
                self.enemy_cities.append(city)

        for unit in self.units.values():
            if int(unit["tribe_id"]) == self.player_id:
                self.my_units.append(unit)
            else:
                self.enemy_units.append(unit)

        self.enemy_reach = self._compute_reach_map(self.enemy_units)
        self.friend_reach = self._compute_reach_map(self.my_units)
        self.city_threat_cache: Dict[int, float] = {}

    def my_tribe(self) -> Dict[str, Any]:
        return self.tribes[self.player_id]

    def unit(self, unit_id: Optional[int]) -> Optional[Dict[str, Any]]:
        return self.units.get(int(unit_id)) if unit_id else None

    def city(self, city_id: Optional[int]) -> Optional[Dict[str, Any]]:
        return self.cities.get(int(city_id)) if city_id else None

    def tile(self, x: int, y: int) -> Optional[Dict[str, Any]]:
        return self.tiles.get(pos_key(x, y))

    def unit_position(self, unit: Dict[str, Any]) -> Tuple[int, int]:
        return pos_key(unit["x"], unit["y"])

    def city_position(self, city: Dict[str, Any]) -> Tuple[int, int]:
        return pos_key(city["x"], city["y"])

    def my_capital(self) -> Optional[Dict[str, Any]]:
        capital_id = self.my_tribe().get("capital_id")
        return self.city(capital_id)

    def city_tiles(self, city_id: int) -> List[Dict[str, Any]]:
        return [tile for tile in self.tiles.values() if int(tile["city_id"]) == int(city_id)]

    def enemies_in_city(self, city_id: int) -> int:
        count = 0
        for tile in self.city_tiles(city_id):
            uid = int(tile.get("unit_id") or 0)
            if uid:
                unit = self.unit(uid)
                if unit and int(unit["tribe_id"]) != self.player_id:
                    count += 1
        return count

    def has_unexplored_neighbor(self, pos: Dict[str, Any]) -> bool:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                x = int(pos["x"]) + dx
                y = int(pos["y"]) + dy
                if not self.in_bounds(x, y):
                    continue
                tile = self.tile(x, y)
                if tile and not tile["explored"]:
                    return True
        return False

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.board_size and 0 <= y < self.board_size

    def nearest_target_distance(self, origin: Dict[str, Any], targets: Sequence[Dict[str, Any]]) -> Optional[int]:
        best = None
        for target in targets:
            dist = chebyshev(origin, target)
            if best is None or dist < best:
                best = dist
        return best

    def tile_threat(self, x: int, y: int) -> float:
        return self.enemy_reach.get(pos_key(x, y), 0.0)

    def tile_support(self, x: int, y: int) -> float:
        return self.friend_reach.get(pos_key(x, y), 0.0)

    def city_threat(self, city: Dict[str, Any]) -> float:
        city_id = int(city["id"])
        if city_id in self.city_threat_cache:
            return self.city_threat_cache[city_id]

        cx, cy = self.city_position(city)
        threat = 0.0
        for enemy in self.enemy_units:
            ex, ey = self.unit_position(enemy)
            dist = max(abs(cx - ex), abs(cy - ey))
            strike = int(enemy.get("range", 1)) + int(enemy.get("mov", 1))
            if dist <= strike + 1:
                threat += self.unit_value(enemy) / max(1, dist)
        threat += 2.5 * self.enemies_in_city(city_id)
        self.city_threat_cache[city_id] = threat
        return threat

    def strategic_mode(self) -> str:
        cap = self.my_capital()
        capital_threat = self.city_threat(cap) if cap else 0.0
        endangered = sum(1 for city in self.my_cities if self.city_threat(city) >= 4.0)
        if capital_threat >= 5.5 or endangered >= 2:
            return "defend"
        if self.tick <= 10 and (self.villages or self.ruins):
            return "expand"
        if self.enemy_cities:
            return "press"
        return "grow"

    def unit_value(self, unit: Dict[str, Any]) -> float:
        base = UNIT_BASE_VALUE.get(str(unit.get("type", "WARRIOR")), 3.0)
        hp_ratio = float(unit.get("current_hp", 1)) / max(1.0, float(unit.get("max_hp", 1)))
        veteran = 0.35 if unit.get("is_veteran") else 0.0
        hidden = 0.25 if unit.get("is_hidden") else 0.0
        return base * (0.45 + 0.55 * hp_ratio) + veteran + hidden

    def direct_capture_targets(self) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for tile in self.villages + self.ruins:
            out.append(pos_key(tile["x"], tile["y"]))
        for city in self.enemy_cities:
            out.append(self.city_position(city))
        return out

    def visible_resource_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for tile in self.visible_tiles:
            terrain = tile.get("terrain")
            resource = tile.get("resource")
            building = tile.get("building")
            if terrain:
                counts[str(terrain)] += 1
            if resource:
                counts[str(resource)] += 1
            if building:
                counts[str(building)] += 1
        return counts

    def _compute_reach_map(self, units: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, int], float]:
        reach: Dict[Tuple[int, int], float] = defaultdict(float)
        for unit in units:
            ux, uy = self.unit_position(unit)
            strike = int(unit.get("range", 1)) + int(unit.get("mov", 1))
            value = self.unit_value(unit)
            for x in range(max(0, ux - strike), min(self.board_size, ux + strike + 1)):
                for y in range(max(0, uy - strike), min(self.board_size, uy + strike + 1)):
                    dist = max(abs(x - ux), abs(y - uy))
                    if dist <= strike:
                        reach[(x, y)] += value / max(1.0, dist)
        return reach


def needed_to_level(city: Dict[str, Any]) -> int:
    return int(city["level"]) + 1 - int(city["production"])


def capital_or_frontline_bonus(view: ObservationView, city: Optional[Dict[str, Any]]) -> float:
    if not city:
        return 0.0
    bonus = 0.0
    if city.get("is_capital"):
        bonus += 2.2
    bonus += min(4.0, view.city_threat(city)) * 0.7
    return bonus


def score_research(action: Dict[str, Any], view: ObservationView, mode: str) -> float:
    if mode == "defend":
        return -1.5
    tech = str(action.get("technology"))
    counts = view.visible_resource_counts()
    score = 0.8
    for key, bonus in TECH_RESOURCE_BONUS.get(tech, {}).items():
        score += counts.get(key, 0) * bonus * 0.18
    if tech in {"ORGANIZATION", "CLIMBING", "HUNTING", "FISHING", "RIDING"}:
        score += 0.6
    if tech in {"STRATEGY", "ARCHERY"} and any(view.city_threat(c) >= 3.0 for c in view.my_cities):
        score += 1.0
    if view.game_mode == "SCORE":
        score += 0.4
    return score


def score_level_up(action: Dict[str, Any], view: ObservationView, mode: str) -> float:
    city = view.city(action.get("city_id"))
    bonus = str(action.get("bonus"))
    score = 1.0 if bonus in GOOD_LEVEL_UPS else 0.1
    if mode == "defend" and bonus in CITY_DEFENSIVE_LEVEL_UPS:
        score += 2.2
    score += capital_or_frontline_bonus(view, city)
    if city and needed_to_level(city) <= 1:
        score += 1.4
    return score


def score_resource_gathering(action: Dict[str, Any], view: ObservationView, mode: str) -> float:
    city = view.city(action.get("city_id"))
    if not city:
        return 0.0
    score = clamp(6 - needed_to_level(city), 0, 5)
    score += capital_or_frontline_bonus(view, city) * 0.5
    if mode == "defend" and view.city_threat(city) >= 4.0:
        score += 0.8
    return score


def score_build(action: Dict[str, Any], view: ObservationView, mode: str) -> float:
    city = view.city(action.get("city_id"))
    if not city:
        return -0.5
    if mode == "defend" and view.city_threat(city) >= 3.5:
        return -1.0
    tribe = view.my_tribe()
    stars = int(tribe.get("stars", 0))
    if stars < 8:
        return -0.4
    building_type = str(action.get("building_type"))
    base = {
        "FARM": 3.4,
        "MINE": 3.2,
        "FORGE": 3.0,
        "WINDMILL": 3.0,
        "MARKET": 3.0,
        "PORT": 2.4,
        "SAWMILL": 2.4,
        "LUMBER_HUT": 2.2,
        "GRAND_BAZAR": 3.5,
        "EMPERORS_TOMB": 3.5,
        "GATE_OF_POWER": 3.5,
        "EYE_OF_GOD": 3.5,
        "PARK_OF_FORTUNE": 3.5,
        "TOWER_OF_WISDOM": 3.5,
        "ALTAR_OF_PEACE": 3.5,
        "TEMPLE": 0.4,
        "WATER_TEMPLE": 0.4,
        "MOUNTAIN_TEMPLE": 0.4,
        "FOREST_TEMPLE": 0.4,
    }.get(building_type, 1.0)
    score = base + clamp(5 - needed_to_level(city), 0, 4)
    if view.game_mode == "SCORE":
        score += 0.4
    return score


def spawn_pref(unit_type: str, city: Dict[str, Any], view: ObservationView, mode: str) -> float:
    threat = view.city_threat(city)
    prefs = SPAWN_PREFS_DEFENSE if threat >= 4.0 or mode == "defend" else SPAWN_PREFS_SAFE
    score = prefs.get(unit_type, 2.8)
    if city.get("is_capital") and unit_type in {"DEFENDER", "WARRIOR", "ARCHER"}:
        score += 0.8
    if mode == "press" and unit_type in {"RIDER", "KNIGHT", "CATAPULT"}:
        score += 0.7
    if mode == "expand" and unit_type in {"RIDER", "WARRIOR"}:
        score += 0.5
    return score


def score_spawn(action: Dict[str, Any], view: ObservationView, mode: str) -> float:
    city = view.city(action.get("city_id"))
    if not city:
        return 0.0
    unit_type = str(action.get("unit_type"))
    score = spawn_pref(unit_type, city, view, mode)
    score += capital_or_frontline_bonus(view, city) * 0.7
    if view.enemies_in_city(int(city["id"])) > 0:
        score += 2.0
    return score


def score_attack(action: Dict[str, Any], view: ObservationView, mode: str) -> float:
    attacker = view.unit(action.get("unit_id"))
    defender = view.unit(action.get("target_unit_id"))
    if not attacker or not defender:
        return -1.0

    att_val = view.unit_value(attacker)
    def_val = view.unit_value(defender)
    ranged = int(attacker.get("range", 1)) > 1
    attacker_hp = float(attacker.get("current_hp", 1))
    defender_hp = float(defender.get("current_hp", 1))
    atk = float(attacker.get("atk", 1))
    deff = float(defender.get("def", 1))
    damage_edge = atk - deff
    kill_likely = attacker_hp >= defender_hp and damage_edge >= 0
    score = 1.0 + 0.75 * def_val

    if kill_likely:
        score += 4.0 + 0.4 * def_val
    elif damage_edge > 0:
        score += 1.8
    else:
        score -= 1.0

    dcity = view.city(defender.get("city_id"))
    if dcity and int(dcity.get("tribe_id")) != view.player_id:
        score += 2.2
        if dcity.get("is_capital"):
            score += 1.8

    for city in view.my_cities:
        if chebyshev(defender, city) <= int(defender.get("range", 1)) + 1:
            score += 1.5
            if city.get("is_capital"):
                score += 1.8
            break

    if ranged:
        enemy_retaliation = view.tile_threat(int(attacker["x"]), int(attacker["y"]))
        if enemy_retaliation > att_val * 1.15 and not kill_likely:
            score -= 1.4
    else:
        retaliation_risk = view.tile_threat(int(defender["x"]), int(defender["y"]))
        if retaliation_risk > att_val * 1.1 and not kill_likely:
            score -= 2.0

    if mode == "defend":
        score += 0.9
    elif mode == "press" and dcity:
        score += 0.6

    return score


def score_convert(action: Dict[str, Any], view: ObservationView, mode: str) -> float:
    target = view.unit(action.get("target_unit_id"))
    if not target:
        return -1.0
    score = 2.0 + view.unit_value(target)
    city = view.city(target.get("city_id"))
    if city and int(city.get("tribe_id")) != view.player_id:
        score += 1.8
    if mode == "defend":
        score += 0.7
    return score


def direct_move_capture_bonus(dest_tile: Optional[Dict[str, Any]], view: ObservationView) -> float:
    if not dest_tile:
        return 0.0
    terrain = dest_tile.get("terrain")
    resource = dest_tile.get("resource")
    city_id = int(dest_tile.get("city_id") or 0)
    bonus = 0.0
    if terrain == "VILLAGE":
        bonus += 6.0
    if resource == "RUINS":
        bonus += 6.2
    if city_id:
        city = view.city(city_id)
        if city and int(city["tribe_id"]) != view.player_id:
            bonus += 5.0 + (1.2 if city.get("is_capital") else 0.0)
    return bonus


def score_move(action: Dict[str, Any], view: ObservationView, mode: str) -> float:
    unit = view.unit(action.get("unit_id"))
    dest = action.get("destination")
    if not unit or not dest:
        return -1.0

    dx, dy = int(dest["x"]), int(dest["y"])
    dest_tile = view.tile(dx, dy)
    current = {"x": unit["x"], "y": unit["y"]}
    current_dist_village = view.nearest_target_distance(current, view.villages + view.ruins)
    dest_dist_village = view.nearest_target_distance(dest, view.villages + view.ruins)
    current_dist_city = view.nearest_target_distance(current, view.enemy_cities)
    dest_dist_city = view.nearest_target_distance(dest, view.enemy_cities)
    score = 0.4

    score += direct_move_capture_bonus(dest_tile, view)

    if dest_dist_village is not None and (current_dist_village is None or dest_dist_village < current_dist_village):
        score += 3.0 if mode in {"expand", "grow"} else 1.4
    if dest_dist_city is not None and (current_dist_city is None or dest_dist_city < current_dist_city):
        score += 2.0 if mode == "press" else 1.0

    if view.has_unexplored_neighbor(dest):
        score += 1.4 if mode == "expand" else 0.7

    cap = view.my_capital()
    if cap:
        current_dist_cap = chebyshev(current, cap)
        dest_dist_cap = chebyshev(dest, cap)
        cap_threat = view.city_threat(cap)
        if mode == "defend" and cap_threat >= 4.0 and dest_dist_cap < current_dist_cap:
            score += 3.6
        if dest_tile and int(dest_tile.get("city_id") or 0) == int(cap["id"]):
            score += 2.3 if mode == "defend" else 0.4

    for city in view.my_cities:
        threat = view.city_threat(city)
        if threat < 4.0:
            continue
        current_dist = chebyshev(current, city)
        new_dist = chebyshev(dest, city)
        if new_dist < current_dist:
            score += 2.2 + 0.25 * min(threat, 5.0)
        if dest_tile and int(dest_tile.get("city_id") or 0) == int(city["id"]):
            score += 1.7

    threat_now = view.tile_threat(int(unit["x"]), int(unit["y"]))
    threat_dest = view.tile_threat(dx, dy)
    support_dest = view.tile_support(dx, dy)
    unit_val = view.unit_value(unit)
    if threat_dest > support_dest + unit_val * 0.45:
        score -= 2.2
    elif threat_dest > threat_now + unit_val * 0.35:
        score -= 1.3
    elif support_dest > threat_dest + 0.8:
        score += 0.7

    city_id = int(dest_tile.get("city_id") or 0) if dest_tile else 0
    if city_id:
        city = view.city(city_id)
        if city and int(city["tribe_id"]) == view.player_id and view.city_threat(city) >= 3.5:
            score += 1.6

    return score


def score_road(action: Dict[str, Any], view: ObservationView, mode: str) -> float:
    stars = int(view.my_tribe().get("stars", 0))
    if stars <= 4:
        return -0.8
    if mode == "defend":
        return 0.6
    return 1.2


def score_recover(action: Dict[str, Any], view: ObservationView, mode: str) -> float:
    unit = view.unit(action.get("unit_id"))
    if not unit:
        return -1.0
    x, y = int(unit["x"]), int(unit["y"])
    danger = view.tile_threat(x, y)
    support = view.tile_support(x, y)
    score = 0.8
    if danger > support:
        score += 1.5
    if float(unit.get("current_hp", 1)) <= 4:
        score += 1.2
    return score


def score_send_stars(action: Dict[str, Any], view: ObservationView, mode: str) -> float:
    stars = int(view.my_tribe().get("stars", 0))
    return 0.3 if stars > 30 and view.game_mode == "SCORE" else -1.5


def score_capture(action: Dict[str, Any], view: ObservationView, mode: str) -> float:
    city = view.city(action.get("target_city_id"))
    score = 8.0
    if city:
        score += 4.0
        if city.get("is_capital"):
            score += 3.0
    return score


def action_score(action: Dict[str, Any], view: ObservationView, mode: str) -> float:
    action_type = str(action["type"])

    if action_type == "CAPTURE":
        return score_capture(action, view, mode)
    if action_type == "EXAMINE":
        return 7.2
    if action_type == "MAKE_VETERAN":
        return 5.0
    if action_type in {"DISBAND", "DESTROY"}:
        return -3.0
    if action_type == "END_TURN":
        return -2.2
    if action_type == "MOVE":
        return score_move(action, view, mode)
    if action_type == "ATTACK":
        return score_attack(action, view, mode)
    if action_type in {"UPGRADE_SHIP", "UPGRADE_BOAT"}:
        return 2.0 if int(view.my_tribe().get("stars", 0)) > 6 else -0.5
    if action_type == "RECOVER":
        return score_recover(action, view, mode)
    if action_type == "HEAL_OTHERS":
        return 2.4 if mode == "defend" else 1.3
    if action_type == "CONVERT":
        return score_convert(action, view, mode)
    if action_type == "BURN_FOREST":
        return 0.4 if int(view.my_tribe().get("stars", 0)) > 5 and mode != "defend" else -0.3
    if action_type == "CLEAR_FOREST":
        return 0.8 if mode != "defend" else -0.3
    if action_type == "GROW_FOREST":
        return 1.1 if mode != "defend" else 0.2
    if action_type == "BUILD":
        return score_build(action, view, mode)
    if action_type == "SPAWN":
        return score_spawn(action, view, mode)
    if action_type == "RESOURCE_GATHERING":
        return score_resource_gathering(action, view, mode)
    if action_type == "LEVEL_UP":
        return score_level_up(action, view, mode)
    if action_type == "BUILD_ROAD":
        return score_road(action, view, mode)
    if action_type == "RESEARCH_TECH":
        return score_research(action, view, mode)
    if action_type == "SEND_STARS":
        return score_send_stars(action, view, mode)
    return 0.0


def tie_break_key(action: Dict[str, Any], score: float) -> Tuple[float, float, float, float]:
    action_type = str(action["type"])
    urgency = {
        "CAPTURE": 6.0,
        "ATTACK": 5.0,
        "SPAWN": 4.5,
        "LEVEL_UP": 4.2,
        "RESOURCE_GATHERING": 4.0,
        "MOVE": 3.5,
        "BUILD": 2.8,
        "RESEARCH_TECH": 2.0,
        "END_TURN": -1.0,
    }.get(action_type, 0.0)
    deterministic = sum(ord(ch) for ch in str(action.get("id", ""))) * 1e-6
    return (score, urgency, deterministic, RNG.random() * 1e-6)


def choose_action(message: Dict[str, Any]) -> Dict[str, Optional[str]]:
    view = ObservationView(message)
    mode = view.strategic_mode()
    MEMORY.reset_for_tick(view.tick, mode)
    MEMORY.request_count += 1

    actions = message.get("actions", [])
    if not actions:
        return {"actionId": None}

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for action in actions:
        score = action_score(action, view, mode)
        scored.append((score, action))

    # Strongly prefer any clearly dominant tactical action.
    scored.sort(key=lambda item: tie_break_key(item[1], item[0]), reverse=True)
    chosen_score, chosen = scored[0]
    MEMORY.turn_actions.append(str(chosen.get("id")))
    return {"actionId": chosen["id"]}


def main() -> None:
    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            message = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        msg_type = message.get("type")
        if msg_type == "action_request":
            response = choose_action(message)
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        elif msg_type == "game_over":
            break


if __name__ == "__main__":
    main()
