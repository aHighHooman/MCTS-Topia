#include "native_static_eval.hpp"

#include "native_rules.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <set>
#include <string>
#include <utility>
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

enum class StaticEvalVariant {
  Baseline,
  Experimental,
};

constexpr double kValueTanhDenominatorStars = 163.793103448276;

StaticEvalVariant static_eval_variant() {
  const char* value = std::getenv("TRIBES_STATIC_EVAL_VARIANT");
  if (value == nullptr) {
    return StaticEvalVariant::Baseline;
  }
  const std::string variant(value);
  if (variant == "experimental") {
    return StaticEvalVariant::Experimental;
  }
  return StaticEvalVariant::Baseline;
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
  if (x >= 0 && y >= 0 && x < state.board_size && y < state.board_size) {
    int index = y * state.board_size + x;
    if (index >= 0 && index < static_cast<int>(state.tiles.size())) {
      const NativeTile& tile = state.tiles[index];
      if (tile.x == x && tile.y == y) {
        return &tile;
      }
    }
    index = x * state.board_size + y;
    if (index >= 0 && index < static_cast<int>(state.tiles.size())) {
      const NativeTile& tile = state.tiles[index];
      if (tile.x == x && tile.y == y) {
        return &tile;
      }
    }
  }
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

double unit_attack_value(const NativeUnit& unit) {
  return unit.attack > 0.0 ? unit.attack : static_cast<double>(unit_attack(unit.type));
}

double unit_defence_value(const NativeUnit& unit) {
  return unit.defence > 0.0 ? unit.defence : static_cast<double>(unit_defence(unit.type));
}

int unit_range_value(const NativeUnit& unit) {
  return unit.range > 0 ? unit.range : unit_range(unit.type);
}

int unit_mobility_value(const NativeUnit& unit) {
  return unit.movement > 0 ? unit.movement : unit_mobility(unit.type);
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

double unit_power(const NativeGameState& state, const NativeUnit& unit);

bool can_threaten(const NativeUnit& attacker, int x, int y) {
  const int mobility = unit_mobility_value(attacker);
  const int range = unit_range_value(attacker);
  const int threat_reach = mobility + range;
  return chebyshev(attacker.x, attacker.y, x, y) <= threat_reach;
}

double enemy_attack_pressure_at(const NativeGameState& state, int player_id, int x, int y) {
  double pressure = 0.0;
  for (const NativeUnit& enemy : state.units) {
    if (enemy.tribe_id == player_id || enemy.hidden || enemy.current_hp <= 0) {
      continue;
    }
    const int dist = std::max(1, chebyshev(enemy.x, enemy.y, x, y));
    const int direct_reach = unit_range_value(enemy);
    const int move_attack_reach = unit_mobility_value(enemy) + direct_reach;
    double reach_bonus = 0.12;
    if (dist <= direct_reach) {
      reach_bonus = 1.15;
    } else if (dist <= move_attack_reach) {
      reach_bonus = 0.85;
    } else if (dist <= move_attack_reach + 1) {
      reach_bonus = 0.35;
    }
    pressure += reach_bonus * unit_power(state, enemy) / static_cast<double>(dist);
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
    support += unit_power(state, unit) / static_cast<double>(dist);
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

const NativeUnit* stronger_enemy_near(const NativeGameState& state, const NativeUnit& unit) {
  for (const NativeUnit& enemy : state.units) {
    if (enemy.tribe_id == unit.tribe_id || enemy.hidden || enemy.current_hp <= 0) {
      continue;
    }
    if (can_threaten(enemy, unit.x, unit.y) && unit_attack_value(enemy) > unit_defence_value(unit)) {
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

double research_score_baseline(const NativeAction& action, const NativeGameState& state, int player_id) {
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

bool is_water_terrain(const std::string& terrain) {
  return terrain == "WATER" || terrain == "OCEAN" || terrain == "SHALLOW_WATER" ||
      terrain == "DEEP_WATER";
}

bool is_naval_unit(const std::string& type) {
  return type == "RAFT" || type == "BOAT" || type == "SHIP" || type == "BATTLESHIP" ||
      type == "RAMMER" || type == "SCOUT" || type == "BOMBER" || type == "DINGHY" ||
      type == "PIRATE" || type == "JUGGERNAUT";
}

bool is_city_occupied_by_player(const NativeGameState& state, const NativeCity& city, int player_id) {
  const NativeTile* tile = tile_at(state, city.x, city.y);
  if (tile == nullptr || tile->unit_id <= 0) {
    return false;
  }
  const NativeUnit* unit = unit_by_id(state, tile->unit_id);
  return unit != nullptr && unit->tribe_id == player_id && unit->current_hp > 0;
}

const NativeCity* owned_city_center_at(const NativeGameState& state, int player_id, int x, int y);
const NativeCity* capital_for_player(const NativeGameState& state, int player_id);

int tech_cost_for(const NativeTribe* tribe, const std::string& tech, int city_count) {
  const int tier = tech_tier_score(tech);
  int cost = 4 + tier * std::max(1, city_count);
  if (tribe != nullptr &&
      std::find(tribe->researched_tech_ids.begin(), tribe->researched_tech_ids.end(), "PHILOSOPHY") !=
          tribe->researched_tech_ids.end()) {
    cost = static_cast<int>(std::ceil(static_cast<double>(cost) * (2.0 / 3.0)));
  }
  return cost;
}

std::set<int> city_ids_connected_by_candidate_road(
    const NativeGameState& state,
    int player_id,
    int x,
    int y);
bool candidate_merges_city_connections(
    const NativeGameState& state,
    int player_id,
    int x,
    int y,
    bool* connects_capital);
bool candidate_extends_city_connection(
    const NativeGameState& state,
    int player_id,
    const std::set<int>& connected_city_ids,
    int x,
    int y,
    bool* capital_progress);

bool can_spawn_unit_type_soon(
    const NativeGameState& state,
    int player_id,
    const std::string& unit_type,
    int stars_after_research) {
  const int cost = unit_type == "KNIGHT" || unit_type == "CATAPULT" ? 8 :
      (unit_type == "SWORDMAN" || unit_type == "SWORDSMAN" ? 5 : 3);
  const int soon_cost = std::max(0, cost - 2);
  if (stars_after_research < soon_cost) {
    return false;
  }
  for (const NativeCity& city : state.cities) {
    if (city.tribe_id == player_id && city.production > 0) {
      return true;
    }
  }
  return false;
}

bool has_city_resource_followup(const NativeGameState& state, int player_id, const std::string& tech) {
  for (const NativeCity& city : state.cities) {
    if (city.tribe_id != player_id) {
      continue;
    }
    for (const NativeTile& tile : state.tiles) {
      if (!tile.visible || !tile_in_player_city(state, player_id, tile)) {
        continue;
      }
      if ((tech == "SMITHERY" && (tile.resource == "ORE" || tile.building == "MINE")) ||
          (tech == "MATHEMATICS" && (tile.terrain == "FOREST" || tile.building == "LUMBER_HUT"))) {
        return true;
      }
    }
  }
  return false;
}

bool has_safe_catapult_position(const NativeGameState& state, int player_id) {
  for (const NativeCity& city : state.cities) {
    if (city.tribe_id != player_id || is_city_occupied_by_player(state, city, player_id)) {
      continue;
    }
    if (friendly_support_at(state, player_id, city.x, city.y) >= enemy_attack_pressure_at(state, player_id, city.x, city.y)) {
      return true;
    }
  }
  return false;
}

bool has_knight_chain_targets(const NativeGameState& state, int player_id) {
  int targets = 0;
  for (const NativeUnit& unit : state.units) {
    if (unit.tribe_id != player_id && !unit.hidden && unit.current_hp > 0 &&
        (unit.current_hp <= 7 || unit_range_value(unit) >= 2 || unit.type == "CATAPULT")) {
      ++targets;
    }
  }
  return targets >= 2;
}

bool has_city_connection_road_followup(
    const NativeGameState& state,
    int player_id,
    const std::vector<NativeAction>& actions,
    const std::vector<int>& legal_action_indexes) {
  for (int action_index : legal_action_indexes) {
    if (action_index < 0 || action_index >= static_cast<int>(actions.size()) ||
        action_type(actions[action_index]) != "BUILD_ROAD") {
      continue;
    }
    int x = 0;
    int y = 0;
    if (!action_destination(actions[action_index], &x, &y)) {
      continue;
    }
    const std::set<int> connected = city_ids_connected_by_candidate_road(state, player_id, x, y);
    bool connects_capital = false;
    bool capital_progress = false;
    if (candidate_merges_city_connections(state, player_id, x, y, &connects_capital) ||
        candidate_extends_city_connection(state, player_id, connected, x, y, &capital_progress)) {
      return true;
    }
  }
  return false;
}

bool has_unit_tempo_road_followup(const NativeGameState& state, int player_id) {
  for (const NativeUnit& unit : state.units) {
    if (unit.tribe_id != player_id || unit.current_hp <= 0) {
      continue;
    }
    for (const NativeTile& tile : state.tiles) {
      if (!tile.visible) {
        continue;
      }
      const NativeCity* city = nullptr;
      if (tile.city_id > 0) {
        city = city_by_id(state, tile.city_id);
      }
      const bool enemy_city = city != nullptr && city->tribe_id >= 0 && city->tribe_id != player_id;
      const bool threatened_city = owned_city_center_at(state, player_id, tile.x, tile.y) != nullptr &&
          enemy_attack_pressure_at(state, player_id, tile.x, tile.y) > 1.0;
      const bool target = tile.terrain == "VILLAGE" || tile.resource == "RUINS" || enemy_city || threatened_city;
      if (target && chebyshev(unit.x, unit.y, tile.x, tile.y) >= 2 && chebyshev(unit.x, unit.y, tile.x, tile.y) <= 4) {
        return true;
      }
    }
  }
  return false;
}

double build_score_baseline(const NativeAction& action, const NativeGameState& state, int player_id) {
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

double capture_score_baseline(const NativeAction& action, const NativeGameState& state, int player_id) {
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

struct CombatForecast {
  int attack_damage = 0;
  int retaliation_damage = 0;
  int attacker_hp_after = 0;
  int defender_hp_after = 0;
  bool defender_killed = false;
  bool attacker_killed = false;
  bool retaliates = false;
};

bool stiff_unit(const std::string& type) {
  return type == "RAFT" || type == "BOMBER" || type == "CATAPULT" ||
      type == "MIND_BENDER" || type == "CLOAK" || type == "DINGHY" ||
      type == "JUGGERNAUT";
}

CombatForecast forecast_combat(const NativeUnit& attacker, const NativeUnit& defender) {
  CombatForecast forecast;
  const double attacker_hp = attacker.current_hp_exact > 0.0
      ? attacker.current_hp_exact
      : static_cast<double>(attacker.current_hp);
  const double defender_hp = defender.current_hp_exact > 0.0
      ? defender.current_hp_exact
      : static_cast<double>(defender.current_hp);
  const double attacker_attack = unit_attack_value(attacker);
  const double defender_defence = unit_defence_value(defender);
  const double attack_force = attacker_attack * (attacker_hp / std::max(1, attacker.max_hp));
  const double defence_force = defender_defence * (defender_hp / std::max(1, defender.max_hp));
  const double total_damage = attack_force + defence_force;
  forecast.attack_damage = total_damage <= 0.0
      ? 0
      : static_cast<int>(std::round((attack_force / total_damage) * attacker_attack * 4.5));
  forecast.retaliation_damage = total_damage <= 0.0
      ? 0
      : static_cast<int>(std::round((defence_force / total_damage) * defender_defence * 4.5));
  forecast.defender_hp_after = std::max(0, defender.current_hp - forecast.attack_damage);
  forecast.defender_killed = forecast.defender_hp_after <= 0;
  forecast.attacker_hp_after = attacker.current_hp;
  if (!forecast.defender_killed) {
    const int distance = chebyshev(attacker.x, attacker.y, defender.x, defender.y);
    forecast.retaliates =
        distance <= std::max(1, unit_range_value(defender)) &&
        defender_defence > 0.0 &&
        !stiff_unit(defender.type) &&
        attacker.type != "DAGGER" &&
        attacker.type != "PIRATE" &&
        forecast.retaliation_damage > 0;
    if (forecast.retaliates) {
      forecast.attacker_hp_after = std::max(0, attacker.current_hp - forecast.retaliation_damage);
    }
  }
  forecast.attacker_killed = forecast.attacker_hp_after <= 0;
  return forecast;
}

double attack_score_baseline(const NativeAction& action, const NativeGameState& state, int player_id) {
  const NativeUnit* attacker = unit_by_id(state, action.unit_id);
  const NativeUnit* defender = unit_by_id(state, action_int(action, "target_unit_id", "tu"));
  if (attacker == nullptr || defender == nullptr) {
    return 0.0;
  }

  const CombatForecast forecast = forecast_combat(*attacker, *defender);
  const double target_value = unit_value(*defender);
  const double attacker_value = unit_value(*attacker);
  const bool ranged = unit_range_value(*attacker) > 1;
  const double defender_hp = static_cast<double>(std::max(1, defender->current_hp));
  const double attacker_hp = static_cast<double>(std::max(1, attacker->current_hp));
  const double damage_fraction = clamp(static_cast<double>(forecast.attack_damage) / defender_hp, 0.0, 1.4);
  const double retaliation_fraction = clamp(static_cast<double>(forecast.retaliation_damage) / attacker_hp, 0.0, 1.4);

  double score = 1.1 + 4.3 * damage_fraction + 0.28 * target_value * damage_fraction - 0.10 * attacker_value;
  if (forecast.defender_killed) {
    score += 3.2 + 0.45 * target_value;
  }
  if (forecast.attacker_killed) {
    score -= 5.0 + 0.40 * attacker_value;
  } else if (forecast.retaliates) {
    score -= 3.0 * retaliation_fraction + 0.20 * attacker_value * retaliation_fraction;
  }
  if (ranged && !forecast.retaliates) {
    score += 0.9;
  }

  const NativeTile* target_tile = tile_at(state, defender->x, defender->y);
  if (target_tile != nullptr && target_tile->city_id > 0) {
    const NativeCity* target_city = city_by_id(state, target_tile->city_id);
    if (target_city != nullptr && target_city->tribe_id != player_id) {
      score += 1.4 + (target_city->capital ? 2.6 : 0.0);
    }
  }
  const double post_attack_danger = enemy_attack_pressure_at(state, player_id, defender->x, defender->y);
  if (!ranged && post_attack_danger > 5.0 && !forecast.defender_killed) {
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
    const bool can_fight = unit_attack_value(*unit) >= unit_defence_value(enemy) && unit->current_hp >= enemy.current_hp - 2;
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

const NativeCity* owned_city_center_at(const NativeGameState& state, int player_id, int x, int y) {
  for (const NativeCity& city : state.cities) {
    if (city.tribe_id == player_id && city.x == x && city.y == y) {
      return &city;
    }
  }
  return nullptr;
}

const NativeCity* capital_for_player(const NativeGameState& state, int player_id) {
  for (const NativeCity& city : state.cities) {
    if (city.tribe_id == player_id && city.capital) {
      return &city;
    }
  }
  return nullptr;
}

int owned_city_count_for_player(const NativeGameState& state, int player_id) {
  return static_cast<int>(std::count_if(state.cities.begin(), state.cities.end(), [player_id](const NativeCity& city) {
    return city.tribe_id == player_id;
  }));
}

int enemy_city_count_for_player(const NativeGameState& state, int player_id) {
  return static_cast<int>(std::count_if(state.cities.begin(), state.cities.end(), [player_id](const NativeCity& city) {
    return city.tribe_id >= 0 && city.tribe_id != player_id;
  }));
}

void collect_city_ids_from_road_component(
    const NativeGameState& state,
    int player_id,
    int start_x,
    int start_y,
    std::set<int>* city_ids) {
  if (city_ids == nullptr) {
    return;
  }
  const NativeTile* start = tile_at(state, start_x, start_y);
  if (start == nullptr || !start->road) {
    return;
  }
  std::vector<std::pair<int, int>> stack;
  std::set<std::pair<int, int>> visited;
  stack.push_back({start_x, start_y});
  visited.insert({start_x, start_y});
  static const int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
  while (!stack.empty()) {
    const std::pair<int, int> current = stack.back();
    stack.pop_back();
    const int x = current.first;
    const int y = current.second;
    for (const auto& dir : dirs) {
      const int nx = x + dir[0];
      const int ny = y + dir[1];
      const NativeCity* city = owned_city_center_at(state, player_id, nx, ny);
      if (city != nullptr) {
        city_ids->insert(city->id);
      }
      const NativeTile* tile = tile_at(state, nx, ny);
      if (tile != nullptr && tile->road && !visited.count({nx, ny})) {
        visited.insert({nx, ny});
        stack.push_back({nx, ny});
      }
    }
  }
}

std::set<int> city_ids_connected_by_candidate_road(
    const NativeGameState& state,
    int player_id,
    int x,
    int y) {
  std::set<int> city_ids;
  const NativeCity* candidate_city = owned_city_center_at(state, player_id, x, y);
  if (candidate_city != nullptr) {
    city_ids.insert(candidate_city->id);
  }
  static const int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
  for (const auto& dir : dirs) {
    const int nx = x + dir[0];
    const int ny = y + dir[1];
    const NativeCity* city = owned_city_center_at(state, player_id, nx, ny);
    if (city != nullptr) {
      city_ids.insert(city->id);
    }
    collect_city_ids_from_road_component(state, player_id, nx, ny, &city_ids);
  }
  return city_ids;
}

std::vector<std::set<int>> city_connection_groups_adjacent_to_candidate(
    const NativeGameState& state,
    int player_id,
    int x,
    int y) {
  std::vector<std::set<int>> groups;
  std::set<std::pair<int, int>> visited_roads;
  static const int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

  for (const auto& dir : dirs) {
    const int nx = x + dir[0];
    const int ny = y + dir[1];
    const NativeCity* city = owned_city_center_at(state, player_id, nx, ny);
    if (city != nullptr) {
      groups.push_back({city->id});
    }
    const NativeTile* tile = tile_at(state, nx, ny);
    if (tile == nullptr || !tile->road || visited_roads.count({nx, ny})) {
      continue;
    }

    std::set<int> city_ids;
    std::vector<std::pair<int, int>> stack;
    stack.push_back({nx, ny});
    visited_roads.insert({nx, ny});
    while (!stack.empty()) {
      const std::pair<int, int> current = stack.back();
      stack.pop_back();
      for (const auto& component_dir : dirs) {
        const int cx = current.first + component_dir[0];
        const int cy = current.second + component_dir[1];
        const NativeCity* component_city = owned_city_center_at(state, player_id, cx, cy);
        if (component_city != nullptr) {
          city_ids.insert(component_city->id);
        }
        const NativeTile* component_tile = tile_at(state, cx, cy);
        if (component_tile != nullptr && component_tile->road && !visited_roads.count({cx, cy})) {
          visited_roads.insert({cx, cy});
          stack.push_back({cx, cy});
        }
      }
    }
    if (!city_ids.empty()) {
      groups.push_back(city_ids);
    }
  }
  return groups;
}

bool candidate_merges_city_connections(
    const NativeGameState& state,
    int player_id,
    int x,
    int y,
    bool* connects_capital) {
  if (connects_capital != nullptr) {
    *connects_capital = false;
  }
  const std::vector<std::set<int>> groups = city_connection_groups_adjacent_to_candidate(state, player_id, x, y);
  std::set<int> merged_city_ids;
  bool merged_distinct_groups = false;
  for (const std::set<int>& group : groups) {
    bool adds_new_city = false;
    for (int city_id : group) {
      if (!merged_city_ids.count(city_id)) {
        adds_new_city = true;
        break;
      }
    }
    if (adds_new_city && !merged_city_ids.empty()) {
      merged_distinct_groups = true;
    }
    merged_city_ids.insert(group.begin(), group.end());
  }
  if (!merged_distinct_groups || merged_city_ids.size() < 2) {
    return false;
  }
  const NativeCity* capital = capital_for_player(state, player_id);
  if (connects_capital != nullptr && capital != nullptr && merged_city_ids.count(capital->id) > 0) {
    *connects_capital = true;
  }
  return true;
}

int connected_frontier_distance_to_target(
    const NativeGameState& state,
    int player_id,
    int x,
    int y,
    const NativeCity& target) {
  int best = -1;
  static const int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
  std::vector<std::pair<int, int>> stack;
  std::set<std::pair<int, int>> visited;
  auto observe = [&best, &target](int px, int py) {
    const int dist = chebyshev(px, py, target.x, target.y);
    if (best < 0 || dist < best) {
      best = dist;
    }
  };

  for (const auto& dir : dirs) {
    const int nx = x + dir[0];
    const int ny = y + dir[1];
    const NativeCity* city = owned_city_center_at(state, player_id, nx, ny);
    if (city != nullptr) {
      observe(city->x, city->y);
    }
    const NativeTile* tile = tile_at(state, nx, ny);
    if (tile != nullptr && tile->road && !visited.count({nx, ny})) {
      visited.insert({nx, ny});
      stack.push_back({nx, ny});
    }
  }

  while (!stack.empty()) {
    const std::pair<int, int> current = stack.back();
    stack.pop_back();
    observe(current.first, current.second);
    for (const auto& dir : dirs) {
      const int nx = current.first + dir[0];
      const int ny = current.second + dir[1];
      const NativeCity* city = owned_city_center_at(state, player_id, nx, ny);
      if (city != nullptr) {
        observe(city->x, city->y);
      }
      const NativeTile* tile = tile_at(state, nx, ny);
      if (tile != nullptr && tile->road && !visited.count({nx, ny})) {
        visited.insert({nx, ny});
        stack.push_back({nx, ny});
      }
    }
  }
  return best;
}

const NativeCity* nearest_other_owned_city(
    const NativeGameState& state,
    int player_id,
    const NativeCity& source,
    bool prefer_capital) {
  const NativeCity* best = nullptr;
  int best_dist = 0;
  for (const NativeCity& city : state.cities) {
    if (city.tribe_id != player_id || city.id == source.id) {
      continue;
    }
    if (prefer_capital && !city.capital) {
      continue;
    }
    const int dist = chebyshev(source.x, source.y, city.x, city.y);
    if (best == nullptr || dist < best_dist) {
      best = &city;
      best_dist = dist;
    }
  }
  if (best != nullptr || !prefer_capital) {
    return best;
  }
  return nearest_other_owned_city(state, player_id, source, false);
}

bool candidate_extends_city_connection(
    const NativeGameState& state,
    int player_id,
    const std::set<int>& connected_city_ids,
    int x,
    int y,
    bool* capital_progress) {
  bool progress = false;
  bool capital = false;
  const NativeCity* player_capital = capital_for_player(state, player_id);
  for (int city_id : connected_city_ids) {
    const NativeCity* source = city_by_id(state, city_id);
    if (source == nullptr || source->tribe_id != player_id) {
      continue;
    }
    const bool source_is_capital = player_capital != nullptr && source->id == player_capital->id;
    const NativeCity* target = source_is_capital
        ? nearest_other_owned_city(state, player_id, *source, false)
        : (player_capital != nullptr ? player_capital : nearest_other_owned_city(state, player_id, *source, false));
    if (target == nullptr) {
      continue;
    }
    const int frontier_to_target = connected_frontier_distance_to_target(state, player_id, x, y, *target);
    const int candidate_to_target = chebyshev(x, y, target->x, target->y);
    if (frontier_to_target >= 0 && candidate_to_target < frontier_to_target) {
      progress = true;
      if (source_is_capital || target->capital) {
        capital = true;
      }
    }
  }
  if (capital_progress != nullptr) {
    *capital_progress = capital;
  }
  return progress;
}

bool has_affordable_high_value_research(
    const NativeGameState& state,
    const std::vector<NativeAction>& actions,
    const std::vector<int>& legal_action_indexes,
    int player_id) {
  const NativeTribe* tribe = tribe_by_id(state, player_id);
  const int stars = tribe == nullptr ? 0 : tribe->stars;
  const int city_count = owned_city_count_for_player(state, player_id);
  for (int action_index : legal_action_indexes) {
    if (action_index < 0 || action_index >= static_cast<int>(actions.size())) {
      continue;
    }
    const NativeAction& legal = actions[action_index];
    const std::string type = action_type(legal);
    if (type != "RESEARCH_TECH" && type != "RESEARCH") {
      continue;
    }
    const std::string tech = action_string(legal, "technology", "tech");
    if (stars >= tech_cost_for(tribe, tech, city_count) && research_score_baseline(legal, state, player_id) >= 6.0) {
      return true;
    }
  }
  return false;
}

bool has_near_level_resource_gathering(
    const NativeGameState& state,
    const std::vector<NativeAction>& actions,
    const std::vector<int>& legal_action_indexes) {
  for (int action_index : legal_action_indexes) {
    if (action_index < 0 || action_index >= static_cast<int>(actions.size())) {
      continue;
    }
    const NativeAction& legal = actions[action_index];
    if (action_type(legal) != "RESOURCE_GATHERING") {
      continue;
    }
    const NativeCity* city = city_by_id(state, legal.city_id);
    if (city != nullptr && needed_to_level(city) <= 2) {
      return true;
    }
  }
  return false;
}

bool has_defensive_spending_need(
    const NativeGameState& state,
    const std::vector<NativeAction>& actions,
    const std::vector<int>& legal_action_indexes,
    int player_id) {
  double city_threat = 0.0;
  double city_support = 0.0;
  for (const NativeCity& city : state.cities) {
    if (city.tribe_id != player_id) {
      continue;
    }
    city_threat += enemy_attack_pressure_at(state, player_id, city.x, city.y);
    city_support += friendly_support_at(state, player_id, city.x, city.y);
  }
  bool can_spawn = false;
  for (int action_index : legal_action_indexes) {
    if (action_index < 0 || action_index >= static_cast<int>(actions.size())) {
      continue;
    }
    can_spawn = can_spawn || action_type(actions[action_index]) == "SPAWN";
  }
  return can_spawn && city_threat > city_support + 3.0;
}

double road_bonus_readiness(
    const NativeGameState& state,
    const std::vector<NativeAction>& actions,
    const std::vector<int>& legal_action_indexes,
    int player_id,
    int cost) {
  const NativeTribe* tribe = tribe_by_id(state, player_id);
  const int stars = tribe == nullptr ? 0 : tribe->stars;
  const int my_cities = owned_city_count_for_player(state, player_id);
  const int enemy_cities = enemy_city_count_for_player(state, player_id);
  double readiness = 1.0;
  if (stars <= cost + 1) {
    readiness *= 0.35;
  } else if (stars <= cost + 3) {
    readiness *= 0.65;
  }
  if (enemy_cities > my_cities) {
    readiness *= 0.70;
  }
  if (has_defensive_spending_need(state, actions, legal_action_indexes, player_id)) {
    readiness *= 0.55;
  }
  if (has_affordable_high_value_research(state, actions, legal_action_indexes, player_id)) {
    readiness *= 0.70;
  }
  if (has_near_level_resource_gathering(state, actions, legal_action_indexes)) {
    readiness *= 0.75;
  }
  return clamp(readiness, 0.20, 1.0);
}

double road_score_baseline(
    const NativeAction& action,
    const NativeGameState& state,
    const std::vector<NativeAction>& actions,
    const std::vector<int>& legal_action_indexes,
    int player_id) {
  const NativeTribe* tribe = tribe_by_id(state, player_id);
  const int stars = tribe == nullptr ? 0 : tribe->stars;
  int x = 0;
  int y = 0;
  if (!action_destination(action, &x, &y)) {
    return 0.0;
  }
  const NativeTile* tile = tile_at(state, x, y);
  if (tile == nullptr || tile->road) {
    return 0.0;
  }
  const bool water = is_water_terrain(tile->terrain);
  const int cost = water ? 5 : 3;
  if (stars < cost || owned_city_count_for_player(state, player_id) < 2) {
    return 0.0;
  }

  const std::set<int> connected_city_ids = city_ids_connected_by_candidate_road(state, player_id, x, y);
  bool connects_capital = false;
  const bool connects_multiple_cities =
      candidate_merges_city_connections(state, player_id, x, y, &connects_capital);
  bool capital_progress = false;
  const bool extends_connection = candidate_extends_city_connection(
      state, player_id, connected_city_ids, x, y, &capital_progress);
  const bool shortens_unit_target_path = has_unit_tempo_road_followup(state, player_id);

  double score = 0.0;
  if (connects_multiple_cities && connects_capital) {
    score += 4.2;
  } else if (connects_multiple_cities) {
    score += 3.2;
  } else if (capital_progress) {
    score += 2.1;
  } else if (extends_connection) {
    score += 1.4;
  }

  if (score <= 0.0) {
    return 0.0;
  }

  if (tile_in_player_city(state, player_id, *tile)) {
    score += 0.35;
  }
  if (water) {
    score *= 0.65;
  }

  const double raw_score = score;
  const double readiness = road_bonus_readiness(state, actions, legal_action_indexes, player_id, cost);
  score *= readiness;
  const double final = clamp(score, 0.0, 4.4);
  const char* debug = std::getenv("TRIBES_STATIC_EVAL_DEBUG_ROADS");
  if (debug != nullptr && std::string(debug) == "1") {
    std::cerr << "road_eval"
              << " tile=" << x << "," << y
              << " stars=" << stars
              << " connects_city_component=" << (connects_multiple_cities ? 1 : 0)
              << " connects_capital_network=" << (connects_capital ? 1 : 0)
              << " extends_city_network=" << (extends_connection ? 1 : 0)
              << " shortens_unit_target_path=" << (shortens_unit_target_path ? 1 : 0)
              << " raw_score=" << raw_score
              << " readiness=" << readiness
              << " final_score=" << final
              << "\n";
  }
  return final;
}

double action_score_baseline(
    const NativeAction& action,
    const NativeGameState& state,
    const std::vector<NativeAction>& actions,
    const std::vector<int>& legal_action_indexes) {
  const int player_id = state.active_player_id;
  const std::string type = action_type(action);
  if (type == "CAPTURE") return capture_score_baseline(action, state, player_id);
  if (type == "EXAMINE") return 8.6;
  if (type == "MAKE_VETERAN") return 8.5;
  if (type == "DISBAND") return -3.5;
  if (type == "DESTROY") return -2.2;
  if (type == "END_TURN") return -1.8;
  if (type == "MOVE" || type == "STEP_MOVE") return move_score_baseline(action, state, player_id);
  if (type == "ATTACK") return attack_score_baseline(action, state, player_id);
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
  if (type == "BURN_FOREST") return -2.2;
  if (type == "CLEAR_FOREST") return -2.2;
  if (type == "GROW_FOREST") return 2.6;
  if (type == "BUILD") return build_score_baseline(action, state, player_id);
  if (type == "SPAWN") return spawn_score_baseline(action, state, player_id);
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
  if (type == "BUILD_ROAD") return road_score_baseline(action, state, actions, legal_action_indexes, player_id);
  if (type == "RESEARCH_TECH") return research_score_baseline(action, state, player_id);
  if (type == "BUILD_EMBASSY") return stars >= 5 ? 3.2 : 0.8;
  if (type == "PROPOSE_PEACE" || type == "PROPOSE_TREATY" || type == "ACCEPT_PEACE" || type == "ACCEPT_TREATY") return 1.2;
  if (type == "CANCEL_TREATY") return 0.4;
  if (type == "SEND_STARS") return -1.0;
  return 0.0;
}

double action_score_baseline(const NativeAction& action, const NativeGameState& state) {
  static const std::vector<NativeAction> no_actions;
  static const std::vector<int> no_legal_action_indexes;
  return action_score_baseline(action, state, no_actions, no_legal_action_indexes);
}

std::vector<double> priors_for_state(const NativeGameState& state, const std::vector<NativeAction>& actions) {
  std::vector<double> scores;
  scores.reserve(state.legal_action_indexes.size());
  double min_score = 0.0;
  double max_score = 0.0;
  bool have_score = false;
  for (int action_index : state.legal_action_indexes) {
    double score = 0.0;
    if (action_index >= 0 && action_index < static_cast<int>(actions.size())) {
      score = action_score_baseline(actions[action_index], state, actions, state.legal_action_indexes);
    }
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

double state_raw_value(
    const NativeGameState& state,
    double material_weight,
    double power_weight,
    double capital_count_weight,
    double current_score_weight,
    double score_diff_weight,
    double wounded_units_weight,
    double city_capital_bonus) {
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
      my_power += unit_power(state, unit);
      if (unit.max_hp > 0 && unit.current_hp < unit.max_hp / 2) {
        ++wounded_units;
      }
      const double danger = enemy_attack_pressure_at(state, player_id, unit.x, unit.y);
      const double support = friendly_support_at(state, player_id, unit.x, unit.y);
      vulnerable_penalty += clamp(danger - 0.45 * support - unit_power(state, unit), 0.0, 8.0);
    } else {
      ++enemy_units;
      enemy_material += unit_value(unit);
      enemy_power += unit_power(state, unit);
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
        const bool likely_kill = unit_attack_value(unit) * 2.0 >= static_cast<double>(enemy.current_hp) || unit.current_hp >= enemy.current_hp;
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
        0.45 * city.population + (city.walls ? 2.0 : 0.0) +
        (city.capital ? city_capital_bonus : 0.0);
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
          enemy_city_pressure += (city.capital ? 1.5 : 1.0) * unit_power(state, unit) / static_cast<double>(dist);
        }
      }
    }
  }

  int visible_villages = 0;
  double village_control = 0.0;
  int explored = 0;
  int visible_resources = 0;
  for (const NativeTile& tile : state.tiles) {
    explored += tile.explored ? 1 : 0;
    visible_resources += tile.visible && !tile.resource.empty() && tile_in_player_city(state, player_id, tile) ? 1 : 0;
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
  const double exploration = 8.620689655172 * static_cast<double>(explored) / board_area;
  const double raw =
      material_weight * (my_material - enemy_material) +
      power_weight * (my_power - enemy_power) +
      18.103448275862 * static_cast<double>(my_cities - enemy_cities) +
      capital_count_weight * static_cast<double>(my_capitals - enemy_capitals) +
      1.068965517241 * (my_city_quality - enemy_city_quality) +
      my_stars - 0.413793103448 * best_enemy_stars +
      current_score_weight * my_score + score_diff_weight * (my_score - best_enemy_score) +
      2.155172413793 * my_tech +
      2.672413793103 * static_cast<double>(visible_villages) + 2.413793103448 * village_control +
      0.551724137931 * static_cast<double>(visible_resources) + exploration +
      0.948275862069 * attack_opportunity + 0.948275862069 * enemy_city_pressure -
      2.672413793103 * vulnerable_penalty - 2.931034482759 * capital_threat - 1.465517241379 * city_threat -
      wounded_units_weight * static_cast<double>(wounded_units) +
      0.517241379310 * static_cast<double>(my_units - enemy_units);
  return raw;
}

double state_raw_baseline(const NativeGameState& state) {
  return state_raw_value(
      state,
      3.706896551724,
      1.810344827586,
      4.137931034483,
      0.275862068966,
      0.172413793103,
      -1.293103448276,
      4.310344827586);
}

double state_raw_experimental(const NativeGameState& state) {
  return state_raw_value(state, 0.0, 5.517241379310, 0.0, 0.0, 0.0, 0.0, 0.0);
}

double expected_incoming_damage_next_turn(
    const NativeGameState& state,
    const NativeUnit& target) {
  std::vector<double> damages;
  damages.reserve(4);
  for (const NativeUnit& enemy : state.units) {
    if (enemy.tribe_id == target.tribe_id || enemy.hidden || enemy.current_hp <= 0) {
      continue;
    }
    if (!can_threaten(enemy, target.x, target.y)) {
      continue;
    }
    const CombatForecast forecast = forecast_combat(enemy, target);
    if (forecast.attack_damage > 0) {
      damages.push_back(static_cast<double>(forecast.attack_damage));
    }
  }
  if (damages.empty()) {
    return 0.0;
  }
  return *std::max_element(damages.begin(), damages.end());
}

double unit_power(const NativeGameState& state, const NativeUnit& unit) {
  if (unit.current_hp <= 0 || unit.hidden) {
    return 0.0;
  }
  const double current_hp = unit.current_hp_exact > 0.0
      ? unit.current_hp_exact
      : static_cast<double>(unit.current_hp);
  const double max_hp = static_cast<double>(std::max(1, unit.max_hp));
  const double hp_frac = clamp(current_hp / max_hp, 0.0, 1.25);
  const double reach = static_cast<double>(std::max(1, unit_mobility_value(unit) + unit_range_value(unit)));
  const double attack = unit_attack_value(unit);
  const double defence = unit_defence_value(unit);
  const double incoming = expected_incoming_damage_next_turn(state, unit);
  const double survival_attacks = std::max(1.0, current_hp / std::max(1.0, incoming));

  const double projection_power = reach * attack * hp_frac * survival_attacks;
  const double defensive_anchor = defence * hp_frac;
  return projection_power + defensive_anchor + (unit.veteran ? 0.8 : 0.0);
}

double state_value_baseline(const NativeGameState& state) {
  if (state.terminal && state.terminal_value_known) {
    return state.terminal_value;
  }
  return std::tanh(state_raw_baseline(state) / kValueTanhDenominatorStars);
}


double state_value_experimental(const NativeGameState& state) {
  if (state.terminal && state.terminal_value_known) {
    return state.terminal_value;
  }
  return std::tanh(state_raw_experimental(state) / kValueTanhDenominatorStars);
}

StaticEvaluation static_evaluation_for_state(
    const NativeGameState& state,
    const std::vector<NativeAction>& actions) {
  const StaticEvalVariant variant = static_eval_variant();
  StaticEvaluation evaluation;
  evaluation.priors = priors_for_state(state, actions);
  if (variant == StaticEvalVariant::Experimental) {
    evaluation.value = state_value_experimental(state);
  } else {
    evaluation.value = state_value_baseline(state);
  }
  return evaluation;
}

py::dict evaluation_to_dict(const StaticEvaluation& evaluation) {
  py::dict out;
  py::list priors;
  for (double prior : evaluation.priors) {
    priors.append(prior);
  }
  out["priors"] = priors;
  out["value"] = evaluation.value;
  return out;
}

}  // namespace

StaticEvaluation evaluate_static_state(
    const NativeGameState& state,
    const std::vector<NativeAction>& actions) {
  return static_evaluation_for_state(state, actions);
}

py::dict evaluate_static(const py::dict& payload, int max_actions) {
  NativeRoot root = parse_root_payload(payload, max_actions);
  return evaluation_to_dict(static_evaluation_for_state(root.state, root.actions));
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
