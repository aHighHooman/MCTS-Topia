#include "native_static_eval.hpp"

#include "native_rules.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace tribes::native {
namespace {

int read_int(const py::handle& object, const char* key, int fallback = 0) {
  if (!py::isinstance<py::dict>(object)) {
    return fallback;
  }
  py::dict dict = py::reinterpret_borrow<py::dict>(object);
  if (!dict.contains(key) || dict[py::str(key)].is_none()) {
    return fallback;
  }
  try {
    return py::cast<int>(dict[py::str(key)]);
  } catch (const py::cast_error&) {
    return fallback;
  }
}

bool read_int_key(const py::handle& object, const char* key, int* out) {
  if (out == nullptr || !py::isinstance<py::dict>(object)) {
    return false;
  }
  py::dict dict = py::reinterpret_borrow<py::dict>(object);
  if (!dict.contains(key) || dict[py::str(key)].is_none()) {
    return false;
  }
  try {
    *out = py::cast<int>(dict[py::str(key)]);
    return true;
  } catch (const py::cast_error&) {
    return false;
  }
}

std::string read_string(const py::handle& object, const char* key) {
  if (!py::isinstance<py::dict>(object)) {
    return "";
  }
  py::dict dict = py::reinterpret_borrow<py::dict>(object);
  if (!dict.contains(key) || dict[py::str(key)].is_none()) {
    return "";
  }
  return py::cast<std::string>(py::str(dict[py::str(key)]));
}

std::string action_type(const NativeAction& action) {
  return action.type.empty() ? read_string(action.payload, "t") : action.type;
}

std::string action_string(const NativeAction& action, const char* primary, const char* fallback = nullptr) {
  std::string value = read_string(action.payload, primary);
  if (value.empty() && fallback != nullptr) {
    value = read_string(action.payload, fallback);
  }
  return value;
}

int action_int(const NativeAction& action, const char* primary, const char* fallback = nullptr) {
  int value = 0;
  if (read_int_key(action.payload, primary, &value)) {
    return value;
  }
  if (fallback != nullptr && read_int_key(action.payload, fallback, &value)) {
    return value;
  }
  return 0;
}

bool action_destination(const NativeAction& action, int* x, int* y) {
  if (x == nullptr || y == nullptr || !py::isinstance<py::dict>(action.payload)) {
    return false;
  }
  py::dict payload = py::reinterpret_borrow<py::dict>(action.payload);
  if (payload.contains("destination") && !payload[py::str("destination")].is_none()) {
    py::handle destination = payload[py::str("destination")];
    int dx = 0;
    int dy = 0;
    if (read_int_key(destination, "x", &dx) && read_int_key(destination, "y", &dy)) {
      *x = dx;
      *y = dy;
      return true;
    }
  }
  int dx = 0;
  int dy = 0;
  if (read_int_key(action.payload, "x", &dx) && read_int_key(action.payload, "y", &dy)) {
    *x = dx;
    *y = dy;
    return true;
  }
  return false;
}

int chebyshev(int ax, int ay, int bx, int by) {
  return std::max(std::abs(ax - bx), std::abs(ay - by));
}

double clamp(double value, double lo, double hi) {
  return std::max(lo, std::min(hi, value));
}

bool use_baseline_eval() {
  const char* value = std::getenv("TRIBES_STATIC_EVAL_VARIANT");
  return value != nullptr && std::string(value) == "baseline";
}

const NativeUnit* unit_by_id(const NativeGameState& state, int id) {
  for (const NativeUnit& unit : state.units) {
    if (unit.id == id) {
      return &unit;
    }
  }
  return nullptr;
}

const NativeCity* city_by_id(const NativeGameState& state, int id) {
  for (const NativeCity& city : state.cities) {
    if (city.id == id) {
      return &city;
    }
  }
  return nullptr;
}

const NativeTribe* tribe_by_id(const NativeGameState& state, int id) {
  for (const NativeTribe& tribe : state.tribes) {
    if (tribe.id == id) {
      return &tribe;
    }
  }
  return nullptr;
}

const NativeTile* tile_at(const NativeGameState& state, int x, int y) {
  for (const NativeTile& tile : state.tiles) {
    if (tile.x == x && tile.y == y) {
      return &tile;
    }
  }
  return nullptr;
}

int unit_attack(const std::string& type) {
  static const std::map<std::string, int> values = {
      {"WARRIOR", 2}, {"RIDER", 2}, {"DEFENDER", 1}, {"SWORDMAN", 3}, {"SWORDSMAN", 3},
      {"ARCHER", 2}, {"CATAPULT", 4}, {"KNIGHT", 3}, {"MIND_BENDER", 0}, {"CLOAK", 0},
      {"DAGGER", 2}, {"RAMMER", 3}, {"SCOUT", 1}, {"BOMBER", 4}, {"SUPERUNIT", 4},
      {"JUGGERNAUT", 4}, {"PIRATE", 3}, {"BATTLESHIP", 4}, {"SHIP", 2}, {"BOAT", 1}};
  auto it = values.find(type);
  return it == values.end() ? 2 : it->second;
}

int unit_defence(const std::string& type) {
  static const std::map<std::string, int> values = {
      {"WARRIOR", 2}, {"RIDER", 1}, {"DEFENDER", 3}, {"SWORDMAN", 3}, {"SWORDSMAN", 3},
      {"ARCHER", 1}, {"CATAPULT", 0}, {"KNIGHT", 1}, {"MIND_BENDER", 1}, {"CLOAK", 0},
      {"DAGGER", 1}, {"RAMMER", 3}, {"SCOUT", 2}, {"BOMBER", 3}, {"SUPERUNIT", 3},
      {"JUGGERNAUT", 4}, {"PIRATE", 2}, {"BATTLESHIP", 4}, {"SHIP", 2}, {"BOAT", 1}};
  auto it = values.find(type);
  return it == values.end() ? 1 : it->second;
}

int unit_range(const std::string& type) {
  static const std::set<std::string> ranged = {
      "ARCHER", "CATAPULT", "MIND_BENDER", "SCOUT", "BOMBER", "BATTLESHIP", "SHIP"};
  return ranged.count(type) ? 2 : 1;
}

int unit_mobility(const std::string& type) {
  static const std::map<std::string, int> values = {
      {"WARRIOR", 1}, {"RIDER", 2}, {"DEFENDER", 1}, {"SWORDMAN", 1}, {"SWORDSMAN", 1},
      {"ARCHER", 1}, {"CATAPULT", 1}, {"KNIGHT", 3}, {"MIND_BENDER", 1}, {"CLOAK", 2},
      {"DAGGER", 1}, {"RAMMER", 1}, {"SCOUT", 2}, {"BOMBER", 2}, {"SUPERUNIT", 1},
      {"JUGGERNAUT", 1}, {"PIRATE", 2}, {"BATTLESHIP", 2}, {"SHIP", 2}, {"BOAT", 2}};
  auto it = values.find(type);
  return it == values.end() ? 1 : it->second;
}

double unit_value(const NativeUnit& unit) {
  static const std::map<std::string, double> values = {
      {"WARRIOR", 2.0}, {"RIDER", 3.0}, {"DEFENDER", 3.0}, {"SWORDMAN", 5.5}, {"SWORDSMAN", 5.5},
      {"ARCHER", 3.5}, {"CATAPULT", 8.0}, {"KNIGHT", 8.0}, {"MIND_BENDER", 5.0}, {"CLOAK", 4.0},
      {"DAGGER", 2.0}, {"RAMMER", 5.0}, {"SCOUT", 5.0}, {"BOMBER", 11.0}, {"SUPERUNIT", 12.0},
      {"JUGGERNAUT", 12.0}, {"PIRATE", 5.0}, {"BATTLESHIP", 12.0}, {"SHIP", 7.0}, {"BOAT", 4.0}};
  auto it = values.find(unit.type);
  const double base = it == values.end() ? 2.5 : it->second;
  const double hp_scale = unit.max_hp > 0 ? clamp(static_cast<double>(unit.current_hp) / unit.max_hp, 0.25, 1.25) : 1.0;
  return base * hp_scale + (unit.veteran ? 1.0 : 0.0);
}

double unit_power(const NativeUnit& unit) {
  const double hp_scale = unit.max_hp > 0 ? clamp(static_cast<double>(unit.current_hp) / unit.max_hp, 0.15, 1.20) : 1.0;
  return (1.4 * unit_attack(unit.type) + 1.1 * unit_defence(unit.type) + 0.45 * unit_range(unit.type) +
          0.30 * unit_mobility(unit.type)) *
             hp_scale +
         (unit.veteran ? 0.8 : 0.0);
}

bool can_threaten(const NativeUnit& attacker, int x, int y) {
  const int threat_reach = unit_mobility(attacker.type) + unit_range(attacker.type);
  return chebyshev(attacker.x, attacker.y, x, y) <= threat_reach;
}

double enemy_attack_pressure_at(const NativeGameState& state, int player_id, int x, int y) {
  double pressure = 0.0;
  for (const NativeUnit& enemy : state.units) {
    if (enemy.tribe_id == player_id || enemy.hidden || enemy.current_hp <= 0) {
      continue;
    }
    const int dist = std::max(1, chebyshev(enemy.x, enemy.y, x, y));
    const int direct_reach = unit_range(enemy.type);
    const int move_attack_reach = unit_mobility(enemy.type) + direct_reach;
    double reach_bonus = 0.12;
    if (dist <= direct_reach) {
      reach_bonus = 1.15;
    } else if (dist <= move_attack_reach) {
      reach_bonus = 0.85;
    } else if (dist <= move_attack_reach + 1) {
      reach_bonus = 0.35;
    }
    pressure += reach_bonus * unit_power(enemy) / static_cast<double>(dist);
  }
  return pressure;
}

double friendly_support_at(const NativeGameState& state, int player_id, int x, int y) {
  double support = 0.0;
  for (const NativeUnit& unit : state.units) {
    if (unit.tribe_id != player_id || unit.hidden || unit.current_hp <= 0) {
      continue;
    }
    const int dist = std::max(1, chebyshev(unit.x, unit.y, x, y));
    support += unit_power(unit) / static_cast<double>(dist);
  }
  return support;
}

bool tile_has_own_city(const NativeGameState& state, int player_id, int x, int y) {
  const NativeTile* tile = tile_at(state, x, y);
  if (tile == nullptr || tile->city_id <= 0) {
    return false;
  }
  const NativeCity* city = city_by_id(state, tile->city_id);
  return city != nullptr && city->tribe_id == player_id;
}

int nearest_friendly_city_distance(const NativeGameState& state, int player_id, int x, int y) {
  int best = -1;
  for (const NativeCity& city : state.cities) {
    if (city.tribe_id == player_id) {
      const int dist = chebyshev(x, y, city.x, city.y);
      if (best < 0 || dist < best) {
        best = dist;
      }
    }
  }
  return best;
}

bool tile_in_player_city(const NativeGameState& state, int player_id, const NativeTile& tile) {
  if (tile.city_id <= 0) {
    return false;
  }
  const NativeCity* city = city_by_id(state, tile.city_id);
  return city != nullptr && city->tribe_id == player_id;
}

int nearest_owned_resource_distance(const NativeGameState& state, int player_id, int x, int y) {
  int best = -1;
  for (const NativeTile& tile : state.tiles) {
    if (tile.visible && !tile.resource.empty() && tile_in_player_city(state, player_id, tile)) {
      const int dist = chebyshev(x, y, tile.x, tile.y);
      if (best < 0 || dist < best) {
        best = dist;
      }
    }
  }
  return best;
}

int tech_tier_score(const std::string& tech) {
  static const std::map<std::string, int> values = {
      {"CLIMBING", 1}, {"FISHING", 1}, {"HUNTING", 1}, {"ORGANIZATION", 1}, {"RIDING", 1},
      {"ARCHERY", 2}, {"FARMING", 2}, {"FORESTRY", 2}, {"FREE_SPIRIT", 2}, {"MEDITATION", 2},
      {"MINING", 2}, {"ROADS", 2}, {"RAMMING", 2}, {"SAILING", 2}, {"STRATEGY", 2},
      {"AQUATISM", 3}, {"CHIVALRY", 3}, {"CONSTRUCTION", 3}, {"DIPLOMACY", 3},
      {"MATHEMATICS", 3}, {"NAVIGATION", 3}, {"SMITHERY", 3}, {"SPIRITUALISM", 3},
      {"TRADE", 3}, {"PHILOSOPHY", 3}};
  auto it = values.find(tech);
  return it == values.end() ? 1 : it->second;
}

int needed_to_level(const NativeCity* city) {
  if (city == nullptr) {
    return 5;
  }
  return std::max(0, city->level + 1 - city->production);
}

bool has_unexplored_neighbor(const NativeGameState& state, int x, int y) {
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0) {
        continue;
      }
      const int nx = x + dx;
      const int ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= state.board_size || ny >= state.board_size) {
        continue;
      }
      const NativeTile* tile = tile_at(state, nx, ny);
      if (tile != nullptr && !tile->explored) {
        return true;
      }
    }
  }
  return false;
}

int nearest_visible_village_distance(const NativeGameState& state, int x, int y) {
  int best = -1;
  for (const NativeTile& tile : state.tiles) {
    if (tile.visible && tile.terrain == "VILLAGE") {
      const int dist = chebyshev(x, y, tile.x, tile.y);
      if (best < 0 || dist < best) {
        best = dist;
      }
    }
  }
  return best;
}

int nearest_visible_ruin_distance(const NativeGameState& state, int x, int y) {
  int best = -1;
  for (const NativeTile& tile : state.tiles) {
    if (tile.visible && tile.resource == "RUINS") {
      const int dist = chebyshev(x, y, tile.x, tile.y);
      if (best < 0 || dist < best) {
        best = dist;
      }
    }
  }
  return best;
}

int nearest_enemy_city_distance(const NativeGameState& state, int player_id, int x, int y) {
  int best = -1;
  for (const NativeCity& city : state.cities) {
    const NativeTile* city_tile = tile_at(state, city.x, city.y);
    if (city.tribe_id != player_id && city_tile != nullptr && city_tile->visible) {
      const int dist = chebyshev(x, y, city.x, city.y);
      if (best < 0 || dist < best) {
        best = dist;
      }
    }
  }
  return best;
}

bool is_trade_node_for_player(const NativeGameState& state, int player_id, const NativeTile& tile) {
  if (tile.road || tile.terrain == "VILLAGE") {
    return true;
  }
  if (tile.terrain == "CITY") {
    const NativeCity* city = city_by_id(state, tile.city_id);
    return city != nullptr && city->tribe_id == player_id;
  }
  return tile.building == "PORT" || tile.building == "DOCK";
}

int adjacent_trade_nodes(const NativeGameState& state, int player_id, int x, int y) {
  int count = 0;
  static const int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
  for (const auto& dir : dirs) {
    const NativeTile* tile = tile_at(state, x + dir[0], y + dir[1]);
    if (tile != nullptr && tile->explored && is_trade_node_for_player(state, player_id, *tile)) {
      ++count;
    }
  }
  return count;
}

const NativeUnit* stronger_enemy_near(const NativeGameState& state, const NativeUnit& unit) {
  for (const NativeUnit& enemy : state.units) {
    if (enemy.tribe_id == unit.tribe_id || enemy.hidden || enemy.current_hp <= 0) {
      continue;
    }
    if (can_threaten(enemy, unit.x, unit.y) && unit_attack(enemy.type) > unit_defence(unit.type)) {
      return &enemy;
    }
  }
  return nullptr;
}

int enemies_in_city(const NativeGameState& state, int city_id, int player_id) {
  int count = 0;
  for (const NativeTile& tile : state.tiles) {
    if (!tile.visible || tile.city_id != city_id || tile.unit_id <= 0) {
      continue;
    }
    const NativeUnit* unit = unit_by_id(state, tile.unit_id);
    if (unit != nullptr && unit->tribe_id != player_id) {
      ++count;
    }
  }
  return count;
}

double research_score_baseline(const NativeAction& action) {
  static const std::map<std::string, double> scores = {
      {"ORGANIZATION", 5.0}, {"CLIMBING", 5.0}, {"MINING", 5.0}, {"FISHING", 5.0},
      {"HUNTING", 5.0}, {"NAVIGATION", 5.0}, {"FARMING", 4.0}, {"AQUATISM", 4.0},
      {"RIDING", 4.0}, {"FORESTRY", 4.0}, {"ARCHERY", 4.0}, {"SAILING", 4.0},
      {"STRATEGY", 4.0}, {"CHIVALRY", 4.0}};
  const std::string tech = action_string(action, "technology", "tech");
  auto it = scores.find(tech);
  return it == scores.end() ? 3.0 : it->second;
}

double research_score_tuned(const NativeAction& action, const NativeGameState& state, int player_id) {
  const std::string tech = action_string(action, "technology", "tech");
  const NativeTribe* tribe = tribe_by_id(state, player_id);
  const int city_count = static_cast<int>(std::count_if(state.cities.begin(), state.cities.end(), [player_id](const NativeCity& city) {
    return city.tribe_id == player_id;
  }));
  const int stars = tribe == nullptr ? 0 : tribe->stars;
  int visible_fish = 0;
  int visible_animals = 0;
  int visible_fruit = 0;
  int visible_ore = 0;
  int visible_forest = 0;
  int visible_water = 0;
  int visible_mountain = 0;
  for (const NativeTile& tile : state.tiles) {
    if (!tile.visible) {
      continue;
    }
    visible_fish += tile.resource == "FISH" || tile.resource == "WHALE" || tile.resource == "STARFISH";
    visible_animals += tile.resource == "ANIMAL";
    visible_fruit += tile.resource == "FRUIT";
    visible_ore += tile.resource == "METAL" || tile.resource == "ORE";
    visible_forest += tile.terrain == "FOREST";
    visible_water += tile.terrain == "WATER" || tile.terrain == "OCEAN";
    visible_mountain += tile.terrain == "MOUNTAIN";
  }
  double score = 1.8 + 0.45 * tech_tier_score(tech);
  if (tech == "ORGANIZATION") score += 2.2 + 0.55 * visible_fruit;
  if (tech == "HUNTING") score += 2.0 + 0.45 * visible_animals;
  if (tech == "FISHING") score += 1.9 + 0.50 * visible_fish + (visible_water > 3 ? 0.7 : 0.0);
  if (tech == "MINING") score += 1.3 + 0.70 * visible_ore + 0.15 * visible_mountain;
  if (tech == "RIDING") score += 1.5 + (city_count >= 2 ? 0.8 : 0.0);
  if (tech == "ROADS") score += 1.3 + (city_count >= 2 ? 1.4 : 0.0);
  if (tech == "FORESTRY") score += 0.9 + 0.18 * visible_forest;
  if (tech == "FARMING") score += 1.0 + 0.25 * visible_fruit;
  if (tech == "ARCHERY" || tech == "STRATEGY") score += 0.7;
  if (tech == "CHIVALRY" || tech == "SMITHERY" || tech == "MATHEMATICS") score += city_count >= 2 ? 1.2 : 0.2;
  if (tech == "SAILING" || tech == "NAVIGATION" || tech == "AQUATISM") score += visible_water > 4 ? 1.5 : -0.2;
  if (stars < 4 + tech_tier_score(tech) * std::max(1, city_count)) {
    score -= 1.1;
  }
  return clamp(score, 0.0, 8.5);
}

double build_score_baseline(const NativeAction& action, const NativeGameState& state, int player_id) {
  static const std::map<std::string, double> scores = {
      {"FARM", 4.0}, {"MINE", 4.0}, {"FORGE", 4.0}, {"WINDMILL", 4.0}, {"MARKET", 4.0},
      {"PORT", 3.0}, {"SAWMILL", 3.0}, {"LUMBER_HUT", 3.0}, {"GRAND_BAZAR", 5.0},
      {"EMPERORS_TOMB", 5.0}, {"GATE_OF_POWER", 5.0}, {"EYE_OF_GOD", 5.0},
      {"PARK_OF_FORTUNE", 5.0}, {"TOWER_OF_WISDOM", 5.0}, {"ALTAR_OF_PEACE", 5.0},
      {"TEMPLE", 1.0}, {"WATER_TEMPLE", 1.0}, {"MOUNTAIN_TEMPLE", 1.0}, {"FOREST_TEMPLE", 1.0}};
  const NativeTribe* tribe = tribe_by_id(state, player_id);
  if (tribe != nullptr && tribe->stars < 8) {
    return 0.0;
  }
  const NativeCity* city = city_by_id(state, action.city_id);
  if (city == nullptr) {
    return 0.0;
  }
  const std::string building = action_string(action, "building_type", "bt");
  auto it = scores.find(building);
  const double base = it == scores.end() ? 0.0 : it->second;
  return clamp(base + 5.0 - needed_to_level(city), 0.0, 5.0);
}

double build_score_tuned(const NativeAction& action, const NativeGameState& state, int player_id) {
  const NativeTribe* tribe = tribe_by_id(state, player_id);
  const int stars = tribe == nullptr ? 0 : tribe->stars;
  const NativeCity* city = city_by_id(state, action.city_id);
  if (city == nullptr) {
    return 0.0;
  }
  const std::string building = action_string(action, "building_type", "bt");
  double base = 0.0;
  if (building == "FARM" || building == "MINE") base = 6.2;
  else if (building == "LUMBER_HUT") base = 4.5;
  else if (building == "SAWMILL" || building == "FORGE" || building == "WINDMILL" || building == "MARKET") base = 5.2;
  else if (building == "PORT") base = 4.1;
  else if (building.find("TEMPLE") != std::string::npos) base = state.tick > 35 ? 2.0 : 0.4;
  else base = 2.7;
  const int need = needed_to_level(city);
  const double level_push = need <= 1 ? 2.4 : (need == 2 ? 1.1 : 0.15);
  const double econ_ok = stars >= 6 ? 0.7 : -0.6;
  return clamp(base + level_push + econ_ok + 0.15 * city->level, 0.0, 9.0);
}

double spawn_score_baseline(const NativeAction& action, const NativeGameState& state, int player_id) {
  static const std::map<std::string, double> safe = {
      {"RIDER", 3.0}, {"ARCHER", 3.0}, {"CATAPULT", 3.0}, {"SWORDMAN", 3.0},
      {"SWORDSMAN", 3.0}, {"KNIGHT", 3.0}, {"MIND_BENDER", 2.0}, {"WARRIOR", 3.0},
      {"DEFENDER", 3.0}};
  static const std::map<std::string, double> threat = {
      {"RIDER", 0.0}, {"ARCHER", 0.0}, {"CATAPULT", 0.0}, {"SWORDMAN", 4.0},
      {"SWORDSMAN", 4.0}, {"KNIGHT", 4.0}, {"MIND_BENDER", 0.0}, {"WARRIOR", 3.0},
      {"DEFENDER", 5.0}};
  const std::string unit = action_string(action, "unit_type", "ut");
  const auto& table = enemies_in_city(state, action.city_id, player_id) > 0 ? threat : safe;
  auto it = table.find(unit);
  return it == table.end() ? 0.0 : it->second;
}

double spawn_score_tuned(const NativeAction& action, const NativeGameState& state, int player_id) {
  const std::string unit = action_string(action, "unit_type", "ut");
  const NativeCity* city = city_by_id(state, action.city_id);
  const double threat = city == nullptr ? 0.0 : enemy_attack_pressure_at(state, player_id, city->x, city->y);
  double score = 0.0;
  if (unit == "WARRIOR") score = 4.2;
  else if (unit == "RIDER") score = 5.1;
  else if (unit == "DEFENDER") score = threat > 4.0 ? 6.0 : 3.8;
  else if (unit == "ARCHER") score = threat > 2.0 ? 4.8 : 4.2;
  else if (unit == "SWORDMAN" || unit == "SWORDSMAN") score = 6.4;
  else if (unit == "KNIGHT") score = 7.0;
  else if (unit == "CATAPULT") score = threat > 5.0 ? 2.2 : 6.0;
  else if (unit == "MIND_BENDER") score = threat > 3.0 ? 2.0 : 4.2;
  else if (unit == "CLOAK") score = 4.4;
  else score = 3.2;
  score += clamp(threat, 0.0, 6.0) * 0.22;
  if (city != nullptr && city->production <= 1) {
    score -= 0.8;
  }
  return clamp(score, 0.0, 9.0);
}

double capture_score_tuned(const NativeAction& action, const NativeGameState& state, int player_id) {
  const std::string capture_type = action_string(action, "capture_type", "ct");
  const NativeUnit* unit = unit_by_id(state, action.unit_id);
  if (capture_type == "VILLAGE") {
    if (unit == nullptr) {
      return 8.5;
    }
    const NativeTile* tile = tile_at(state, unit->x, unit->y);
    if (tile != nullptr && tile->terrain == "VILLAGE") {
      return 10.0;
    }
    return 7.5;
  }
  int target_city_id = action_int(action, "target_city_id", "tc");
  if (target_city_id == 0) {
    target_city_id = action.city_id;
  }
  const NativeCity* city = city_by_id(state, target_city_id);
  if (city == nullptr || city->tribe_id == player_id) {
    return 8.0;
  }
  return clamp(8.2 + 0.5 * city->level + (city->capital ? 2.2 : 0.0), 0.0, 10.0);
}

double attack_score_baseline(const NativeAction& action, const NativeGameState& state) {
  const NativeUnit* attacker = unit_by_id(state, action.unit_id);
  const NativeUnit* defender = unit_by_id(state, action_int(action, "target_unit_id", "tu"));
  if (attacker == nullptr || defender == nullptr) {
    return 0.0;
  }
  const int atk = unit_attack(attacker->type);
  const int def = unit_defence(defender->type);
  const bool ranged = unit_range(attacker->type) > 1;
  if (ranged) {
    if (unit_attack(defender->type) > unit_defence(attacker->type) &&
        chebyshev(attacker->x, attacker->y, defender->x, defender->y) <= unit_range(defender->type)) {
      return 0.0;
    }
    if (atk > def && attacker->current_hp >= defender->current_hp) {
      return 4.0;
    }
    return 2.0;
  }
  if (attacker->current_hp >= defender->current_hp) {
    return atk > def ? 5.0 : 1.0;
  }
  return atk > def ? 1.0 : 0.0;
}

double attack_score_tuned(const NativeAction& action, const NativeGameState& state, int player_id) {
  const NativeUnit* attacker = unit_by_id(state, action.unit_id);
  const NativeUnit* defender = unit_by_id(state, action_int(action, "target_unit_id", "tu"));
  if (attacker == nullptr || defender == nullptr) {
    return 0.0;
  }

  const double target_value = unit_value(*defender);
  const double attacker_value = unit_value(*attacker);
  const bool ranged = unit_range(attacker->type) > 1;
  const double attack_ratio = static_cast<double>(unit_attack(attacker->type)) /
      static_cast<double>(std::max(1, unit_defence(defender->type)));
  const double hp_ratio = static_cast<double>(std::max(1, attacker->current_hp)) /
      static_cast<double>(std::max(1, defender->current_hp));

  // This is still an approximation; the real rules/combat forecast should replace it when available.
  const bool likely_kill = unit_attack(attacker->type) * std::max(1, attacker->current_hp) >=
      unit_defence(defender->type) * std::max(1, defender->current_hp);
  const double expected_damage_value = target_value * clamp(0.30 * attack_ratio + 0.20 * hp_ratio, 0.15, 1.20);
  const double retaliation = ranged ? 0.0 : unit_power(*defender) *
      clamp(static_cast<double>(defender->current_hp) / std::max(1, defender->max_hp), 0.2, 1.0);

  double score = 1.4 + 0.65 * expected_damage_value - 0.16 * attacker_value;
  if (likely_kill) {
    score += 3.2 + 0.40 * target_value;
  }
  if (ranged) {
    score += 0.9;
  } else {
    score -= 0.42 * retaliation;
    if (!likely_kill && attacker->current_hp < defender->current_hp) {
      score -= 1.3;
    }
  }

  const NativeTile* target_tile = tile_at(state, defender->x, defender->y);
  if (target_tile != nullptr && target_tile->city_id > 0) {
    const NativeCity* target_city = city_by_id(state, target_tile->city_id);
    if (target_city != nullptr && target_city->tribe_id != player_id) {
      score += 1.4 + (target_city->capital ? 2.6 : 0.0);
    }
  }

  const double post_attack_danger = enemy_attack_pressure_at(state, player_id, defender->x, defender->y);
  if (!ranged && post_attack_danger > 5.0 && !likely_kill) {
    score -= 1.0;
  }
  return clamp(score, 0.0, 10.0);
}

double move_score_baseline(const NativeAction& action, const NativeGameState& state, int player_id) {
  const NativeUnit* unit = unit_by_id(state, action.unit_id);
  if (unit == nullptr) {
    return 0.0;
  }
  int dx = unit->x;
  int dy = unit->y;
  if (!action_destination(action, &dx, &dy)) {
    return 0.0;
  }
  const NativeUnit* stronger = stronger_enemy_near(state, *unit);
  if (stronger != nullptr &&
      chebyshev(dx, dy, stronger->x, stronger->y) > chebyshev(unit->x, unit->y, stronger->x, stronger->y)) {
    return 4.0;
  }
  const int village_before = nearest_visible_village_distance(state, unit->x, unit->y);
  const int village_after = nearest_visible_village_distance(state, dx, dy);
  if (village_after >= 0 && (village_before < 0 || village_after < village_before)) {
    return 5.0;
  }
  const int city_before = nearest_enemy_city_distance(state, player_id, unit->x, unit->y);
  const int city_after = nearest_enemy_city_distance(state, player_id, dx, dy);
  if (city_after >= 0 && (city_before < 0 || city_after < city_before)) {
    return 4.0;
  }
  for (const NativeUnit& enemy : state.units) {
    if (enemy.tribe_id != player_id && enemy.current_hp > 0 &&
        unit->current_hp >= enemy.current_hp &&
        unit_attack(unit->type) > unit_defence(enemy.type) &&
        chebyshev(dx, dy, enemy.x, enemy.y) < chebyshev(unit->x, unit->y, enemy.x, enemy.y)) {
      return 3.0;
    }
  }
  return has_unexplored_neighbor(state, dx, dy) ? 3.0 : 0.0;
}

double move_score_tuned(const NativeAction& action, const NativeGameState& state, int player_id) {
  const NativeUnit* unit = unit_by_id(state, action.unit_id);
  if (unit == nullptr) {
    return 0.0;
  }
  int dx = unit->x;
  int dy = unit->y;
  if (!action_destination(action, &dx, &dy)) {
    return 0.0;
  }
  double score = 0.0;
  const int village_before = nearest_visible_village_distance(state, unit->x, unit->y);
  const int village_after = nearest_visible_village_distance(state, dx, dy);
  if (village_after == 0) score += 7.5;
  else if (village_after >= 0 && (village_before < 0 || village_after < village_before)) score += 5.4 + 0.75 * (village_before - village_after);

  const int ruin_before = nearest_visible_ruin_distance(state, unit->x, unit->y);
  const int ruin_after = nearest_visible_ruin_distance(state, dx, dy);
  if (ruin_after == 0) score += 8.0;
  else if (ruin_after >= 0 && (ruin_before < 0 || ruin_after < ruin_before)) score += 4.9 + 0.65 * (ruin_before - ruin_after);

  const int city_before = nearest_enemy_city_distance(state, player_id, unit->x, unit->y);
  const int city_after = nearest_enemy_city_distance(state, player_id, dx, dy);
  if (city_after == 0) score += 7.0;
  else if (city_after >= 0 && (city_before < 0 || city_after < city_before)) score += 3.0 + 0.35 * (city_before - city_after);

  // Standing closer to arbitrary visible resources is not itself useful in Polytopia.
  // Only give a tiny bonus near owned resources, where the unit may be defending an economic tile.
  const int resource_before = nearest_owned_resource_distance(state, player_id, unit->x, unit->y);
  const int resource_after = nearest_owned_resource_distance(state, player_id, dx, dy);
  if (resource_after >= 0 && resource_after <= 1 && (resource_before < 0 || resource_after < resource_before)) score += 0.35;
  if (has_unexplored_neighbor(state, dx, dy)) score += 2.0;
  if (tile_has_own_city(state, player_id, dx, dy)) score += unit->current_hp < unit->max_hp ? 1.0 : -0.5;

  for (const NativeUnit& enemy : state.units) {
    if (enemy.tribe_id == player_id || enemy.current_hp <= 0 || enemy.hidden) {
      continue;
    }
    const bool can_fight = unit_attack(unit->type) >= unit_defence(enemy.type) && unit->current_hp >= enemy.current_hp - 2;
    if (can_fight && chebyshev(dx, dy, enemy.x, enemy.y) < chebyshev(unit->x, unit->y, enemy.x, enemy.y)) {
      score += 1.1;
    }
  }

  const double danger = enemy_attack_pressure_at(state, player_id, dx, dy);
  const double support = friendly_support_at(state, player_id, dx, dy);
  score -= clamp(danger - 0.55 * support, 0.0, 8.0) * 0.55;
  const int home_dist = nearest_friendly_city_distance(state, player_id, dx, dy);
  if (home_dist > 0 && home_dist <= 2 && unit->current_hp < unit->max_hp / 2) {
    score += 1.0;
  }
  return clamp(score, 0.0, 10.0);
}

double road_score_tuned(const NativeAction& action, const NativeGameState& state, int player_id) {
  const NativeTribe* tribe = tribe_by_id(state, player_id);
  const int stars = tribe == nullptr ? 0 : tribe->stars;
  int x = 0;
  int y = 0;
  if (!action_destination(action, &x, &y)) {
    return 0.0;
  }
  const NativeTile* tile = tile_at(state, x, y);
  if (tile == nullptr || tile->road || stars < 3) {
    return 0.0;
  }
  const bool water = tile->terrain == "SHALLOW_WATER" || tile->terrain == "WATER" || tile->terrain == "DEEP_WATER";
  const int cost = water ? 5 : 3;
  if (stars < cost) {
    return 0.0;
  }
  double score = -0.4;
  if (tile->terrain == "VILLAGE") {
    score += 1.6;
  }
  if (tile_in_player_city(state, player_id, *tile)) {
    score += 0.9;
  } else if (tile->city_id > 0) {
    score -= 1.0;
  }
  const int links = adjacent_trade_nodes(state, player_id, x, y);
  if (links >= 2) {
    score += 3.2 + 0.7 * static_cast<double>(links - 2);
  } else if (links == 1) {
    score += 0.7;
  } else {
    score -= 1.4;
  }
  if (water) {
    score -= 1.0;
  }
  if (stars <= cost + 2) {
    score -= 0.8;
  }
  return clamp(score, 0.0, 5.2);
}

double action_score_baseline(const NativeAction& action, const NativeGameState& state) {
  const int player_id = state.active_player_id;
  const std::string type = action_type(action);
  if (type == "CAPTURE" || type == "EXAMINE" || type == "MAKE_VETERAN") {
    return 5.0;
  }
  if (type == "DISBAND" || type == "DESTROY") {
    return -2.0;
  }
  if (type == "END_TURN") {
    return -1.0;
  }
  if (type == "MOVE" || type == "STEP_MOVE") {
    return move_score_baseline(action, state, player_id);
  }
  if (type == "ATTACK") {
    return attack_score_baseline(action, state);
  }
  if (type == "UPGRADE_SHIP" || type == "UPGRADE_BOAT" || type == "UPGRADE_RAMMER" ||
      type == "UPGRADE_SCOUT" || type == "UPGRADE_BOMBER") {
    const NativeTribe* tribe = tribe_by_id(state, player_id);
    return tribe != nullptr && tribe->stars > 6 ? 3.0 : 0.0;
  }
  if (type == "RECOVER") {
    const NativeUnit* unit = unit_by_id(state, action.unit_id);
    return unit != nullptr && stronger_enemy_near(state, *unit) != nullptr ? 1.0 : 4.0;
  }
  if (type == "HEAL_OTHERS") {
    return 3.0;
  }
  if (type == "CONVERT") {
    const NativeUnit* target = unit_by_id(state, action_int(action, "target_unit_id", "tu"));
    if (target == nullptr) {
      return 0.0;
    }
    static const std::set<std::string> high = {"BATTLESHIP", "SUPERUNIT", "SWORDMAN", "SWORDSMAN", "KNIGHT", "CATAPULT"};
    static const std::set<std::string> medium = {"MIND_BENDER", "BOAT", "SHIP", "WARRIOR"};
    if (high.count(target->type)) {
      return 5.0;
    }
    return medium.count(target->type) ? 4.0 : 2.0;
  }
  const NativeTribe* tribe = tribe_by_id(state, player_id);
  const int stars = tribe == nullptr ? 0 : tribe->stars;
  if (type == "BURN_FOREST") {
    return stars > 5 ? 1.0 : 0.0;
  }
  if (type == "CLEAR_FOREST") {
    return stars == 0 ? 3.0 : 0.0;
  }
  if (type == "GROW_FOREST") {
    return 3.0;
  }
  if (type == "BUILD") {
    return build_score_baseline(action, state, player_id);
  }
  if (type == "SPAWN") {
    return spawn_score_baseline(action, state, player_id);
  }
  if (type == "RESOURCE_GATHERING") {
    return clamp(5.0 - needed_to_level(city_by_id(state, action.city_id)), 0.0, 5.0);
  }
  if (type == "LEVEL_UP") {
    static const std::set<std::string> good = {"BORDER_GROWTH", "WORKSHOP", "RESOURCES", "SUPERUNIT"};
    return good.count(action_string(action, "bonus", "b")) ? 5.0 : 0.0;
  }
  if (type == "BUILD_ROAD") {
    return stars > 4 ? 3.0 : 0.0;
  }
  if (type == "RESEARCH_TECH") {
    return research_score_baseline(action);
  }
  if (type == "SEND_STARS") {
    return stars > 30 ? 1.0 : -1.0;
  }
  return 0.0;
}

double action_score_tuned(const NativeAction& action, const NativeGameState& state) {
  const int player_id = state.active_player_id;
  const std::string type = action_type(action);
  if (type == "CAPTURE") return capture_score_tuned(action, state, player_id);
  if (type == "EXAMINE") return 8.6;
  if (type == "MAKE_VETERAN") return 8.5;
  if (type == "DISBAND") return -3.5;
  if (type == "DESTROY") return -2.2;
  if (type == "END_TURN") return -1.8;
  if (type == "MOVE" || type == "STEP_MOVE") return move_score_tuned(action, state, player_id);
  if (type == "ATTACK") return attack_score_tuned(action, state, player_id);
  const NativeTribe* tribe = tribe_by_id(state, player_id);
  const int stars = tribe == nullptr ? 0 : tribe->stars;
  if (type == "UPGRADE_SHIP" || type == "UPGRADE_BOAT" || type == "UPGRADE_RAMMER" ||
      type == "UPGRADE_SCOUT" || type == "UPGRADE_BOMBER") {
    return stars >= 7 ? 5.0 : 1.0;
  }
  if (type == "RECOVER") {
    const NativeUnit* unit = unit_by_id(state, action.unit_id);
    if (unit == nullptr) return 0.0;
    const double danger = enemy_attack_pressure_at(state, player_id, unit->x, unit->y);
    const double missing_hp = std::max(0, unit->max_hp - unit->current_hp);
    return clamp(2.4 + 0.28 * missing_hp - 0.25 * danger, 0.0, 7.0);
  }
  if (type == "HEAL_OTHERS") return 5.0;
  if (type == "CONVERT") {
    const NativeUnit* target = unit_by_id(state, action_int(action, "target_unit_id", "tu"));
    return target == nullptr ? 0.0 : clamp(2.5 + 0.75 * unit_value(*target), 0.0, 10.0);
  }
  if (type == "BURN_FOREST") return stars < 4 ? 1.0 : 2.5;
  if (type == "CLEAR_FOREST") return stars <= 1 ? 4.4 : 1.4;
  if (type == "GROW_FOREST") return 2.6;
  if (type == "BUILD") return build_score_tuned(action, state, player_id);
  if (type == "SPAWN") return spawn_score_tuned(action, state, player_id);
  if (type == "RESOURCE_GATHERING") {
    const NativeCity* city = city_by_id(state, action.city_id);
    return city == nullptr ? 2.0 : clamp(6.4 - 0.95 * needed_to_level(city), 1.5, 8.5);
  }
  if (type == "LEVEL_UP") {
    const std::string bonus = action_string(action, "bonus", "b");
    if (bonus == "SUPERUNIT") return 10.0;
    if (bonus == "BORDER_GROWTH") return 7.0;
    if (bonus == "WORKSHOP" || bonus == "RESOURCES") return 8.2;
    if (bonus == "CITY_WALL") return 4.0;
    return 2.2;
  }
  if (type == "BUILD_ROAD") return road_score_tuned(action, state, player_id);
  if (type == "RESEARCH_TECH") return research_score_tuned(action, state, player_id);
  if (type == "BUILD_EMBASSY") return stars >= 5 ? 3.2 : 0.8;
  if (type == "PROPOSE_PEACE" || type == "PROPOSE_TREATY" || type == "ACCEPT_PEACE" || type == "ACCEPT_TREATY") return 1.2;
  if (type == "CANCEL_TREATY") return 0.4;
  if (type == "SEND_STARS") return -1.0;
  return 0.0;
}

std::vector<double> priors_for_state(const NativeGameState& state, const std::vector<NativeAction>& actions) {
  std::vector<double> scores;
  scores.reserve(state.legal_action_indexes.size());
  double min_score = 0.0;
  double max_score = 0.0;
  bool have_score = false;
  for (int action_index : state.legal_action_indexes) {
    const double score = action_index >= 0 && action_index < static_cast<int>(actions.size())
        ? (use_baseline_eval() ? action_score_baseline(actions[action_index], state)
                               : action_score_tuned(actions[action_index], state))
        : 0.0;
    scores.push_back(score);
    min_score = have_score ? std::min(min_score, score) : score;
    max_score = have_score ? std::max(max_score, score) : score;
    have_score = true;
  }
  if (scores.empty()) {
    return scores;
  }
  if (!have_score || std::abs(max_score - min_score) < 1e-9) {
    return std::vector<double>(scores.size(), 1.0 / static_cast<double>(scores.size()));
  }
  constexpr double temperature = 1.5;
  std::vector<double> priors(scores.size(), 0.0);
  double total = 0.0;
  for (size_t i = 0; i < scores.size(); ++i) {
    priors[i] = std::exp((scores[i] - max_score) / temperature);
    total += priors[i];
  }
  if (total <= 0.0 || !std::isfinite(total)) {
    return std::vector<double>(scores.size(), 1.0 / static_cast<double>(scores.size()));
  }
  for (double& prior : priors) {
    prior /= total;
  }
  return priors;
}

double state_value_baseline(const NativeGameState& state) {
  if (state.terminal && state.terminal_value_known) {
    return state.terminal_value;
  }
  const int player_id = state.active_player_id;
  const NativeTribe* me = tribe_by_id(state, player_id);
  double material = 0.0;
  double enemy_material = 0.0;
  int my_cities = 0;
  int enemy_cities = 0;
  int visible_villages = 0;
  double capital_threat = 0.0;
  double city_threat = 0.0;
  double best_enemy_score = 0.0;

  for (const NativeUnit& unit : state.units) {
    if (unit.current_hp <= 0 || unit.hidden) {
      continue;
    }
    if (unit.tribe_id == player_id) {
      material += unit_value(unit);
    } else {
      enemy_material += unit_value(unit);
    }
  }
  for (const NativeCity& city : state.cities) {
    if (city.tribe_id == player_id) {
      ++my_cities;
      double threat = 0.0;
      for (const NativeUnit& enemy : state.units) {
        if (enemy.tribe_id != player_id && enemy.current_hp > 0 && !enemy.hidden) {
          threat += unit_value(enemy) / std::max(1, chebyshev(city.x, city.y, enemy.x, enemy.y));
        }
      }
      city_threat += std::min(5.5, threat);
      if (city.capital) {
        capital_threat += std::min(7.0, threat);
      }
    } else {
      const NativeTile* city_tile = tile_at(state, city.x, city.y);
      if (city_tile != nullptr && city_tile->visible) {
        ++enemy_cities;
      }
    }
  }
  for (const NativeTile& tile : state.tiles) {
    if (tile.visible && tile.terrain == "VILLAGE") {
      ++visible_villages;
    }
  }
  for (const NativeTribe& tribe : state.tribes) {
    if (tribe.id != player_id) {
      best_enemy_score = std::max(best_enemy_score, static_cast<double>(tribe.score));
    }
  }
  const double my_stars = me == nullptr ? 0.0 : static_cast<double>(me->stars);
  const double my_score = me == nullptr ? 0.0 : static_cast<double>(me->score);
  const double raw =
      (material - enemy_material) * 1.8 +
      7.5 * static_cast<double>(my_cities - enemy_cities) +
      0.45 * my_stars +
      0.14 * my_score +
      2.1 * static_cast<double>(visible_villages) +
      0.12 * (my_score - best_enemy_score) -
      1.2 * capital_threat -
      city_threat;
  return std::tanh(raw / 80.0);
}

double state_value_tuned(const NativeGameState& state) {
  if (state.terminal && state.terminal_value_known) {
    return state.terminal_value;
  }
  const int player_id = state.active_player_id;
  const NativeTribe* me = tribe_by_id(state, player_id);
  double my_material = 0.0;
  double enemy_material = 0.0;
  double my_power = 0.0;
  double enemy_power = 0.0;
  double vulnerable_penalty = 0.0;
  double attack_opportunity = 0.0;
  int my_units = 0;
  int enemy_units = 0;
  int wounded_units = 0;

  for (const NativeUnit& unit : state.units) {
    if (unit.current_hp <= 0 || unit.hidden) {
      continue;
    }
    if (unit.tribe_id == player_id) {
      ++my_units;
      my_material += unit_value(unit);
      my_power += unit_power(unit);
      if (unit.max_hp > 0 && unit.current_hp < unit.max_hp / 2) {
        ++wounded_units;
      }
      const double danger = enemy_attack_pressure_at(state, player_id, unit.x, unit.y);
      const double support = friendly_support_at(state, player_id, unit.x, unit.y);
      vulnerable_penalty += clamp(danger - 0.45 * support - unit_power(unit), 0.0, 8.0);
    } else {
      ++enemy_units;
      enemy_material += unit_value(unit);
      enemy_power += unit_power(unit);
    }
  }
  for (const NativeUnit& unit : state.units) {
    if (unit.tribe_id != player_id || unit.current_hp <= 0 || unit.hidden) {
      continue;
    }
    for (const NativeUnit& enemy : state.units) {
      if (enemy.tribe_id == player_id || enemy.current_hp <= 0 || enemy.hidden) {
        continue;
      }
      if (can_threaten(unit, enemy.x, enemy.y)) {
        const bool likely_kill = unit_attack(unit.type) * 2 >= enemy.current_hp || unit.current_hp >= enemy.current_hp;
        attack_opportunity += (likely_kill ? 1.25 : 0.45) * unit_value(enemy);
      }
    }
  }

  int my_cities = 0;
  int enemy_cities = 0;
  int my_capitals = 0;
  int enemy_capitals = 0;
  double my_city_quality = 0.0;
  double enemy_city_quality = 0.0;
  double capital_threat = 0.0;
  double city_threat = 0.0;
  double enemy_city_pressure = 0.0;
  for (const NativeCity& city : state.cities) {
    const double quality = 3.0 + 1.7 * city.level + 1.3 * city.production +
        0.45 * city.population + (city.walls ? 2.0 : 0.0) + (city.capital ? 2.5 : 0.0);
    if (city.tribe_id == player_id) {
      ++my_cities;
      my_capitals += city.capital ? 1 : 0;
      my_city_quality += quality;
      const double threat = enemy_attack_pressure_at(state, player_id, city.x, city.y);
      city_threat += std::min(8.0, threat);
      if (city.capital) {
        capital_threat += std::min(12.0, threat);
      }
    } else {
      const NativeTile* city_tile = tile_at(state, city.x, city.y);
      if (city_tile == nullptr || !city_tile->visible) {
        continue;
      }
      ++enemy_cities;
      enemy_capitals += city.capital ? 1 : 0;
      enemy_city_quality += quality;
      for (const NativeUnit& unit : state.units) {
        if (unit.tribe_id == player_id && unit.current_hp > 0 && !unit.hidden) {
          const int dist = std::max(1, chebyshev(city.x, city.y, unit.x, unit.y));
          enemy_city_pressure += (city.capital ? 1.5 : 1.0) * unit_power(unit) / static_cast<double>(dist);
        }
      }
    }
  }

  int visible_villages = 0;
  double village_control = 0.0;
  int explored = 0;
  int visible_resources = 0;
  int roads = 0;
  for (const NativeTile& tile : state.tiles) {
    explored += tile.explored ? 1 : 0;
    visible_resources += tile.visible && !tile.resource.empty() && tile_in_player_city(state, player_id, tile) ? 1 : 0;
    roads += tile.road && tile_in_player_city(state, player_id, tile) ? 1 : 0;
    if (tile.visible && tile.terrain == "VILLAGE") {
      ++visible_villages;
      int my_best = -1;
      int enemy_best = -1;
      for (const NativeUnit& unit : state.units) {
        if (unit.current_hp <= 0 || unit.hidden) {
          continue;
        }
        const int dist = chebyshev(tile.x, tile.y, unit.x, unit.y);
        if (unit.tribe_id == player_id) {
          my_best = my_best < 0 ? dist : std::min(my_best, dist);
        } else {
          enemy_best = enemy_best < 0 ? dist : std::min(enemy_best, dist);
        }
      }
      if (my_best >= 0) {
        village_control += 4.0 / static_cast<double>(1 + my_best);
        if (enemy_best >= 0 && my_best < enemy_best) {
          village_control += 1.2;
        }
      }
    }
  }

  double best_enemy_score = 0.0;
  double best_enemy_stars = 0.0;
  double my_score = me == nullptr ? 0.0 : static_cast<double>(me->score);
  double my_stars = me == nullptr ? 0.0 : static_cast<double>(me->stars);
  double my_tech = 0.0;
  if (me != nullptr) {
    for (const std::string& tech : me->researched_tech_ids) {
      my_tech += tech_tier_score(tech);
    }
  }
  for (const NativeTribe& tribe : state.tribes) {
    if (tribe.id == player_id) {
      continue;
    }
    best_enemy_score = std::max(best_enemy_score, static_cast<double>(tribe.score));
    best_enemy_stars = std::max(best_enemy_stars, static_cast<double>(tribe.stars));
  }

  const double board_area = static_cast<double>(std::max(1, state.board_size * state.board_size));
  const double exploration = 5.0 * static_cast<double>(explored) / board_area;
  const double raw =
      2.15 * (my_material - enemy_material) +
      1.05 * (my_power - enemy_power) +
      10.5 * static_cast<double>(my_cities - enemy_cities) +
      2.4 * static_cast<double>(my_capitals - enemy_capitals) +
      0.62 * (my_city_quality - enemy_city_quality) +
      0.58 * my_stars - 0.24 * best_enemy_stars +
      0.16 * my_score + 0.10 * (my_score - best_enemy_score) +
      1.25 * my_tech +
      1.55 * static_cast<double>(visible_villages) + 1.4 * village_control +
      0.32 * static_cast<double>(visible_resources) + 0.20 * static_cast<double>(roads) + exploration +
      0.55 * attack_opportunity + 0.55 * enemy_city_pressure -
      1.55 * vulnerable_penalty - 1.7 * capital_threat - 0.85 * city_threat -
      0.75 * static_cast<double>(wounded_units) +
      0.30 * static_cast<double>(my_units - enemy_units);
  return std::tanh(raw / 95.0);
}

py::dict evaluation_to_dict(const NativeRoot& root) {
  py::dict out;
  py::list priors;
  for (double prior : priors_for_state(root.state, root.actions)) {
    priors.append(prior);
  }
  out["priors"] = priors;
  out["value"] = use_baseline_eval() ? state_value_baseline(root.state) : state_value_tuned(root.state);
  return out;
}

}  // namespace

py::dict evaluate_static(const py::dict& payload, int max_actions) {
  return evaluation_to_dict(parse_root_payload(payload, max_actions));
}

py::list evaluate_static_batch(const py::list& payloads, int max_actions) {
  py::list out;
  for (const auto& item : payloads) {
    if (py::isinstance<py::dict>(item)) {
      out.append(evaluate_static(py::reinterpret_borrow<py::dict>(item), max_actions));
    } else {
      py::dict empty;
      empty["priors"] = py::list();
      empty["value"] = 0.0;
      out.append(empty);
    }
  }
  return out;
}

}  // namespace tribes::native
