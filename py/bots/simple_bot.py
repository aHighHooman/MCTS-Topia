import json
import os
import random
import sys
from pathlib import Path


PY_ROOT = Path(__file__).resolve().parents[1]
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from nn.encoding import normalize_message


RESOURCE_TECH_SCORE = {
    "ORGANIZATION": 5,
    "CLIMBING": 5,
    "MINING": 5,
    "FISHING": 5,
    "HUNTING": 5,
    "NAVIGATION": 5,
    "FARMING": 4,
    "AQUATISM": 4,
    "RIDING": 4,
    "FORESTRY": 4,
    "ARCHERY": 4,
    "SAILING": 4,
    "STRATEGY": 4,
    "CHIVALRY": 4,
}

SPAWN_SCORES_SAFE = {
    "RIDER": 3,
    "ARCHER": 3,
    "CATAPULT": 3,
    "SWORDMAN": 3,
    "KNIGHT": 3,
    "MIND_BENDER": 2,
    "WARRIOR": 3,
    "DEFENDER": 3,
}

SPAWN_SCORES_THREAT = {
    "RIDER": 0,
    "ARCHER": 0,
    "CATAPULT": 0,
    "SWORDMAN": 4,
    "KNIGHT": 4,
    "MIND_BENDER": 0,
    "WARRIOR": 3,
    "DEFENDER": 5,
}

BUILD_BASE_SCORES = {
    "FARM": 4,
    "MINE": 4,
    "FORGE": 4,
    "WINDMILL": 4,
    "MARKET": 4,
    "PORT": 3,
    "SAWMILL": 3,
    "LUMBER_HUT": 3,
    "GRAND_BAZAR": 5,
    "EMPERORS_TOMB": 5,
    "GATE_OF_POWER": 5,
    "EYE_OF_GOD": 5,
    "PARK_OF_FORTUNE": 5,
    "TOWER_OF_WISDOM": 5,
    "ALTAR_OF_PEACE": 5,
    "TEMPLE": 1,
    "WATER_TEMPLE": 1,
    "MOUNTAIN_TEMPLE": 1,
    "FOREST_TEMPLE": 1,
}

GOOD_LEVEL_UPS = {"BORDER_GROWTH", "WORKSHOP", "RESOURCES", "SUPERUNIT"}

def _seed() -> int:
    for name in ("TRIBES_SIMPLE_BOT_SEED", "TRIBES_AGENT_SEED"):
        value = os.environ.get(name)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                pass
    return 13


rng = random.Random(_seed())


def chebyshev(a, b):
    return max(abs(a["x"] - b["x"]), abs(a["y"] - b["y"]))


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


class ObservationView:
    def __init__(self, message):
        self.player_id = message["player_id"]
        self.observation = message["observation"]
        self.tribes = {tribe["id"]: tribe for tribe in self.observation["tribes"]}
        self.cities = {city["id"]: city for city in self.observation["cities"]}
        self.units = {unit["id"]: unit for unit in self.observation["units"]}
        self.relationships = self.observation.get("visible_relationships", [])
        self.board_size = self.observation["board"]["size"]
        self.tiles = {}
        self.visible_tiles = []
        self.explored_tiles = []
        self.villages = []
        self.enemy_cities = []
        self.enemy_units = []

        for row in self.observation["board"]["tiles"]:
            for tile in row:
                key = (tile["x"], tile["y"])
                self.tiles[key] = tile
                tile.setdefault("visible", bool(tile.get("explored")))
                if tile["visible"]:
                    self.visible_tiles.append(tile)
                if tile["explored"]:
                    self.explored_tiles.append(tile)
                if tile["terrain"] == "VILLAGE" and tile["visible"]:
                    self.villages.append(tile)

        for city in self.cities.values():
            if city["tribe_id"] != self.player_id:
                self.enemy_cities.append(city)

        for unit in self.units.values():
            if unit["tribe_id"] != self.player_id:
                self.enemy_units.append(unit)

    def my_tribe(self):
        return self.tribes[self.player_id]

    def unit(self, unit_id):
        return self.units.get(unit_id)

    def city(self, city_id):
        return self.cities.get(city_id)

    def tile(self, x, y):
        return self.tiles.get((x, y))

    def city_tiles(self, city_id):
        return [tile for tile in self.tiles.values() if tile["city_id"] == city_id]

    def enemies_in_city(self, city_id):
        return sum(
            1
            for tile in self.city_tiles(city_id)
            if tile["visible"]
            and tile["unit_id"]
            and self.unit(tile["unit_id"])
            and self.unit(tile["unit_id"])["tribe_id"] != self.player_id
        )

    def stronger_enemy_near(self, unit):
        for enemy in self.enemy_units:
            if chebyshev(unit, enemy) <= enemy["range"] and enemy["atk"] > unit["def"]:
                return enemy
        return None

    def weaker_enemy_near(self, unit):
        for enemy in self.enemy_units:
            if unit["current_hp"] >= enemy["current_hp"] and unit["atk"] > enemy["def"]:
                return enemy
        return None

    def nearest_enemy_city_distance(self, pos):
        best = None
        for city in self.enemy_cities:
            dist = chebyshev(pos, city)
            if best is None or dist < best:
                best = dist
        return best

    def nearest_village_distance(self, pos):
        best = None
        for village in self.villages:
            dist = chebyshev(pos, village)
            if best is None or dist < best:
                best = dist
        return best

    def has_unexplored_neighbor(self, pos):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                x = pos["x"] + dx
                y = pos["y"] + dy
                if x < 0 or y < 0 or x >= self.board_size or y >= self.board_size:
                    continue
                tile = self.tile(x, y)
                if tile and not tile["explored"]:
                    return True
        return False


def eval_research(action):
    return RESOURCE_TECH_SCORE.get(action.get("technology") or action.get("tech"), 3)


def eval_level_up(action):
    return 5 if action.get("bonus") in GOOD_LEVEL_UPS else 0


def needed_to_level(city):
    return city["level"] + 1 - city["production"]


def eval_resource_gathering(action, view):
    city = view.city(action.get("city_id"))
    if not city:
        return 0
    return clamp(5 - needed_to_level(city), 0, 5)


def eval_build(action, view):
    tribe = view.my_tribe()
    city = view.city(action.get("city_id"))
    if not city:
        return 0
    if tribe["stars"] < 8:
        return 0

    base = BUILD_BASE_SCORES.get(action.get("building_type"), 0)
    return clamp(base + 5 - needed_to_level(city), 0, 5)


def eval_spawn(action, view):
    unit_type = action.get("unit_type")
    city_id = action.get("city_id")
    enemies_in_city = view.enemies_in_city(city_id)
    table = SPAWN_SCORES_THREAT if enemies_in_city > 0 else SPAWN_SCORES_SAFE
    return table.get(unit_type, 0)


def eval_attack(action, view):
    attacker = view.unit(action.get("unit_id"))
    defender = view.unit(action.get("target_unit_id"))
    if not attacker or not defender:
        return 0

    ranged = attacker["range"] > 1
    if ranged:
        if defender["atk"] > attacker["def"] and chebyshev(attacker, defender) <= defender["range"]:
            return 0
        if attacker["atk"] > defender["def"] and attacker["current_hp"] >= defender["current_hp"]:
            return 4
        return 2

    if attacker["current_hp"] >= defender["current_hp"]:
        if attacker["atk"] > defender["def"]:
            return 5
        return 1

    if attacker["atk"] > defender["def"]:
        return 1
    return 0


def eval_move(action, view):
    unit = view.unit(action.get("unit_id"))
    dest = action.get("destination")
    if not unit or not dest:
        return 0

    current = {"x": unit["x"], "y": unit["y"]}
    stronger_enemy = view.stronger_enemy_near(unit)
    if stronger_enemy:
        if chebyshev(dest, stronger_enemy) > chebyshev(current, stronger_enemy):
            return 4

    village_dist_before = view.nearest_village_distance(current)
    village_dist_after = view.nearest_village_distance(dest)
    if village_dist_after is not None and (village_dist_before is None or village_dist_after < village_dist_before):
        return 5

    city_dist_before = view.nearest_enemy_city_distance(current)
    city_dist_after = view.nearest_enemy_city_distance(dest)
    if city_dist_after is not None and (city_dist_before is None or city_dist_after < city_dist_before):
        return 4

    weaker_enemy = view.weaker_enemy_near(unit)
    if weaker_enemy and chebyshev(dest, weaker_enemy) < chebyshev(current, weaker_enemy):
        return 3

    if view.has_unexplored_neighbor(dest):
        return 3

    return 0


def eval_road(action, view):
    tribe = view.my_tribe()
    if tribe["stars"] > 4:
        return 3
    return 0


def eval_recover(view, action):
    unit = view.unit(action.get("unit_id"))
    if not unit:
        return 0
    return 1 if view.stronger_enemy_near(unit) else 4


def eval_convert(action, view):
    target = view.unit(action.get("target_unit_id"))
    if not target:
        return 0
    high_value = {"BATTLESHIP", "SUPERUNIT", "SWORDMAN", "KNIGHT", "CATAPULT"}
    medium_value = {"MIND_BENDER", "BOAT", "SHIP", "WARRIOR"}
    if target["type"] in high_value:
        return 5
    if target["type"] in medium_value:
        return 4
    return 2


def score_action(action, view):
    action_type = action["type"]

    if action_type in {"CAPTURE", "EXAMINE", "MAKE_VETERAN"}:
        return 5
    if action_type in {"DISBAND", "DESTROY"}:
        return -2
    if action_type == "END_TURN":
        return -1
    if action_type == "MOVE":
        return eval_move(action, view)
    if action_type == "ATTACK":
        return eval_attack(action, view)
    if action_type in {"UPGRADE_SHIP", "UPGRADE_BOAT"}:
        return 3 if view.my_tribe()["stars"] > 6 else 0
    if action_type == "RECOVER":
        return eval_recover(view, action)
    if action_type == "HEAL_OTHERS":
        return 3
    if action_type == "CONVERT":
        return eval_convert(action, view)
    if action_type == "BURN_FOREST":
        return 1 if view.my_tribe()["stars"] > 5 else 0
    if action_type == "CLEAR_FOREST":
        return 3 if view.my_tribe()["stars"] == 0 else 0
    if action_type == "GROW_FOREST":
        return 3
    if action_type == "BUILD":
        return eval_build(action, view)
    if action_type == "SPAWN":
        return eval_spawn(action, view)
    if action_type == "RESOURCE_GATHERING":
        return eval_resource_gathering(action, view)
    if action_type == "LEVEL_UP":
        return eval_level_up(action)
    if action_type == "BUILD_ROAD":
        return eval_road(action, view)
    if action_type == "RESEARCH_TECH":
        return eval_research(action)
    if action_type == "SEND_STARS":
        return 1 if view.my_tribe()["stars"] > 30 else -1
    return 0


def choose_action(message):
    view = ObservationView(message)
    actions = message.get("actions", [])
    best_score = None
    best_actions = []

    for action in actions:
        score = score_action(action, view)
        if best_score is None or score > best_score:
            best_score = score
            best_actions = [action]
        elif score == best_score:
            best_actions.append(action)

    chosen = rng.choice(best_actions) if best_actions else None
    return {"actionId": chosen["id"] if chosen else None}


def main():
    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        message = json.loads(raw_line)
        if message.get("type") == "action_request":
            message = normalize_message(message)
            response = choose_action(message)
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        elif message.get("type") == "game_over":
            break


if __name__ == "__main__":
    main()
