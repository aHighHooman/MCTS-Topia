#include "native_rules.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

namespace tribes::native {
namespace {

int read_int(const py::handle& object, const char* key, int fallback) {
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

std::string read_string_any(const py::handle& object, const char* primary, const char* fallback = nullptr) {
  std::string value = read_string(object, primary);
  if (value.empty() && fallback != nullptr) {
    value = read_string(object, fallback);
  }
  return value;
}

bool read_bool(const py::handle& object, const char* key, bool fallback) {
  if (!py::isinstance<py::dict>(object)) {
    return fallback;
  }
  py::dict dict = py::reinterpret_borrow<py::dict>(object);
  if (!dict.contains(key) || dict[py::str(key)].is_none()) {
    return fallback;
  }
  try {
    return py::cast<bool>(dict[py::str(key)]);
  } catch (const py::cast_error&) {
    return fallback;
  }
}

bool has_key(const py::dict& dict, const char* key) {
  return dict.contains(py::str(key)) && !dict[py::str(key)].is_none();
}

bool has_any_key(const py::dict& dict, std::initializer_list<const char*> keys) {
  for (const char* key : keys) {
    if (has_key(dict, key)) {
      return true;
    }
  }
  return false;
}

void require_key(const py::dict& dict, const char* context, std::initializer_list<const char*> keys) {
  if (has_any_key(dict, keys)) {
    return;
  }
  std::string message = "Native strict payload parse failure: missing ";
  bool first = true;
  for (const char* key : keys) {
    if (!first) {
      message += "/";
    }
    message += key;
    first = false;
  }
  message += " in ";
  message += context;
  throw std::runtime_error(message);
}

std::vector<std::string> read_string_list(const py::handle& object, const char* key) {
  std::vector<std::string> out;
  if (!py::isinstance<py::dict>(object)) {
    return out;
  }
  py::dict dict = py::reinterpret_borrow<py::dict>(object);
  if (!dict.contains(py::str(key)) || !py::isinstance<py::list>(dict[py::str(key)])) {
    return out;
  }
  py::list values = py::reinterpret_borrow<py::list>(dict[py::str(key)]);
  out.reserve(py::len(values));
  for (const auto& value : values) {
    out.push_back(py::cast<std::string>(py::str(value)));
  }
  return out;
}

py::dict shallow_copy_dict(const py::dict& input) {
  py::dict out;
  for (const auto& item : input) {
    out[item.first] = item.second;
  }
  return out;
}

py::list shallow_copy_list(const py::handle& input) {
  py::list out;
  if (!py::isinstance<py::list>(input)) {
    return out;
  }
  py::list in = py::reinterpret_borrow<py::list>(input);
  for (const auto& item : in) {
    if (py::isinstance<py::dict>(item)) {
      out.append(shallow_copy_dict(py::reinterpret_borrow<py::dict>(item)));
    } else if (py::isinstance<py::list>(item)) {
      out.append(shallow_copy_list(item));
    } else {
      out.append(item);
    }
  }
  return out;
}

py::dict deepish_copy_observation(const py::dict& input) {
  py::dict out = shallow_copy_dict(input);
  for (const char* key : {"units", "cities", "tribes"}) {
    if (out.contains(key) && py::isinstance<py::list>(out[py::str(key)])) {
      out[py::str(key)] = shallow_copy_list(out[py::str(key)]);
    }
  }
  if (out.contains("board") && py::isinstance<py::dict>(out["board"])) {
    py::dict board = shallow_copy_dict(py::reinterpret_borrow<py::dict>(out["board"]));
    for (const char* key : {"terrain", "resource", "building", "city", "unit", "exp", "road", "tiles"}) {
      if (board.contains(key) && py::isinstance<py::list>(board[py::str(key)])) {
        board[py::str(key)] = shallow_copy_list(board[py::str(key)]);
      }
    }
    out["board"] = board;
  }
  return out;
}

std::string canonical_action_type(const NativeAction& action) {
  std::string type = action.type.empty() ? read_string(action.payload, "t") : action.type;
  return type;
}

int action_int(const NativeAction& action, const char* primary, const char* fallback = nullptr, int default_value = 0) {
  int value = read_int(action.payload, primary, default_value);
  if (value == default_value && fallback != nullptr) {
    value = read_int(action.payload, fallback, default_value);
  }
  return value;
}

std::string action_string(const NativeAction& action, const char* primary, const char* fallback = nullptr) {
  return read_string_any(action.payload, primary, fallback);
}

NativeTile* tile_at(NativeGameState& state, int x, int y) {
  for (NativeTile& tile : state.tiles) {
    if (tile.x == x && tile.y == y) {
      return &tile;
    }
  }
  return nullptr;
}

NativeUnit* unit_by_id(NativeGameState& state, int id) {
  for (NativeUnit& unit : state.units) {
    if (unit.id == id) {
      return &unit;
    }
  }
  return nullptr;
}

NativeCity* city_by_id(NativeGameState& state, int id) {
  for (NativeCity& city : state.cities) {
    if (city.id == id) {
      return &city;
    }
  }
  return nullptr;
}

NativeTribe* tribe_by_id(NativeGameState& state, int id) {
  for (NativeTribe& tribe : state.tribes) {
    if (tribe.id == id) {
      return &tribe;
    }
  }
  return nullptr;
}

const NativeTribe* tribe_by_id_const(const NativeGameState& state, int id) {
  for (const NativeTribe& tribe : state.tribes) {
    if (tribe.id == id) {
      return &tribe;
    }
  }
  return nullptr;
}

void set_matrix_cell(py::dict board, const char* key, int x, int y, const py::handle& value) {
  if (!board.contains(key) || !py::isinstance<py::list>(board[py::str(key)])) {
    return;
  }
  py::list rows = py::reinterpret_borrow<py::list>(board[py::str(key)]);
  if (y < 0 || y >= static_cast<int>(py::len(rows)) || !py::isinstance<py::list>(rows[y])) {
    return;
  }
  py::list row = py::reinterpret_borrow<py::list>(rows[y]);
  if (x < 0 || x >= static_cast<int>(py::len(row))) {
    return;
  }
  row[x] = value;
}

void set_unit_payload_field(NativeGameState& state, int unit_id, const char* normalized, const char* compact, const py::handle& value) {
  if (!state.observation.contains("units") || !py::isinstance<py::list>(state.observation["units"])) {
    return;
  }
  py::list units = py::reinterpret_borrow<py::list>(state.observation["units"]);
  for (const auto& item : units) {
    if (!py::isinstance<py::dict>(item)) {
      continue;
    }
    py::dict unit = py::reinterpret_borrow<py::dict>(item);
    if (read_int(unit, "id", 0) != unit_id) {
      continue;
    }
    unit[normalized] = value;
    if (compact != nullptr) {
      unit[compact] = value;
    }
    return;
  }
}

void set_city_payload_field(NativeGameState& state, int city_id, const char* normalized, const char* compact, const py::handle& value) {
  if (!state.observation.contains("cities") || !py::isinstance<py::list>(state.observation["cities"])) {
    return;
  }
  py::list cities = py::reinterpret_borrow<py::list>(state.observation["cities"]);
  for (const auto& item : cities) {
    if (!py::isinstance<py::dict>(item)) {
      continue;
    }
    py::dict city = py::reinterpret_borrow<py::dict>(item);
    if (read_int(city, "id", 0) != city_id) {
      continue;
    }
    city[normalized] = value;
    if (compact != nullptr) {
      city[compact] = value;
    }
    return;
  }
}

void set_tribe_payload_field(NativeGameState& state, int tribe_id, const char* key, const py::handle& value) {
  if (!state.observation.contains("tribes") || !py::isinstance<py::list>(state.observation["tribes"])) {
    return;
  }
  py::list tribes = py::reinterpret_borrow<py::list>(state.observation["tribes"]);
  for (const auto& item : tribes) {
    if (!py::isinstance<py::dict>(item)) {
      continue;
    }
    py::dict tribe = py::reinterpret_borrow<py::dict>(item);
    if (read_int(tribe, "id", 0) != tribe_id) {
      continue;
    }
    tribe[key] = value;
    return;
  }
}

void sync_tile_to_payload(NativeGameState& state, const NativeTile& tile) {
  if (!state.observation.contains("board") || !py::isinstance<py::dict>(state.observation["board"])) {
    return;
  }
  py::dict board = py::reinterpret_borrow<py::dict>(state.observation["board"]);
  py::object terrain_value = tile.terrain.empty() ? py::object(py::none()) : py::object(py::str(tile.terrain));
  py::object resource_value = tile.resource.empty() ? py::object(py::none()) : py::object(py::str(tile.resource));
  py::object building_value = tile.building.empty() ? py::object(py::none()) : py::object(py::str(tile.building));
  set_matrix_cell(board, "terrain", tile.x, tile.y, terrain_value);
  set_matrix_cell(board, "resource", tile.x, tile.y, resource_value);
  set_matrix_cell(board, "building", tile.x, tile.y, building_value);
  set_matrix_cell(board, "city", tile.x, tile.y, py::int_(tile.city_id));
  set_matrix_cell(board, "unit", tile.x, tile.y, py::int_(tile.unit_id));
  set_matrix_cell(board, "road", tile.x, tile.y, py::int_(tile.road ? 1 : 0));
  if (board.contains("tiles") && py::isinstance<py::list>(board["tiles"])) {
    py::list rows = py::reinterpret_borrow<py::list>(board["tiles"]);
    if (tile.y >= 0 && tile.y < static_cast<int>(py::len(rows)) && py::isinstance<py::list>(rows[tile.y])) {
      py::list row = py::reinterpret_borrow<py::list>(rows[tile.y]);
      if (tile.x >= 0 && tile.x < static_cast<int>(py::len(row)) && py::isinstance<py::dict>(row[tile.x])) {
        py::dict out = py::reinterpret_borrow<py::dict>(row[tile.x]);
        out["terrain"] = terrain_value;
        out["resource"] = resource_value;
        out["building"] = building_value;
        out["city_id"] = tile.city_id;
        out["unit_id"] = tile.unit_id;
        out["road"] = tile.road;
      }
    }
  }
}

int unit_cost(const std::string& type) {
  static const std::map<std::string, int> costs = {
      {"WARRIOR", 2}, {"RIDER", 3}, {"DEFENDER", 3}, {"SWORDMAN", 5}, {"SWORDSMAN", 5},
      {"ARCHER", 3}, {"CATAPULT", 8}, {"KNIGHT", 8}, {"MIND_BENDER", 5}, {"CLOAK", 8},
      {"DAGGER", 2}, {"RAMMER", 5}, {"SCOUT", 5}, {"BOMBER", 15}, {"SUPERUNIT", 10}};
  auto it = costs.find(type);
  return it == costs.end() ? 0 : it->second;
}

int unit_points(const std::string& type) {
  static const std::map<std::string, int> points = {
      {"WARRIOR", 10}, {"RIDER", 15}, {"DEFENDER", 15}, {"SWORDMAN", 25}, {"SWORDSMAN", 25},
      {"ARCHER", 15}, {"CATAPULT", 40}, {"KNIGHT", 40}, {"MIND_BENDER", 25}, {"CLOAK", 0},
      {"SUPERUNIT", 50}};
  auto it = points.find(type);
  return it == points.end() ? 0 : it->second;
}

int unit_attack(const std::string& type) {
  static const std::map<std::string, int> values = {
      {"WARRIOR", 2}, {"RIDER", 2}, {"DEFENDER", 1}, {"SWORDMAN", 3}, {"SWORDSMAN", 3},
      {"ARCHER", 2}, {"CATAPULT", 4}, {"KNIGHT", 3}, {"MIND_BENDER", 0}, {"CLOAK", 0},
      {"DAGGER", 2}, {"RAMMER", 3}, {"SCOUT", 1}, {"BOMBER", 4}, {"SUPERUNIT", 4},
      {"JUGGERNAUT", 4}, {"PIRATE", 3}};
  auto it = values.find(type);
  return it == values.end() ? 2 : it->second;
}

int unit_defence(const std::string& type) {
  static const std::map<std::string, int> values = {
      {"WARRIOR", 2}, {"RIDER", 1}, {"DEFENDER", 3}, {"SWORDMAN", 3}, {"SWORDSMAN", 3},
      {"ARCHER", 1}, {"CATAPULT", 0}, {"KNIGHT", 1}, {"MIND_BENDER", 1}, {"CLOAK", 0},
      {"DAGGER", 1}, {"RAMMER", 3}, {"SCOUT", 2}, {"BOMBER", 3}, {"SUPERUNIT", 3},
      {"JUGGERNAUT", 4}, {"PIRATE", 2}};
  auto it = values.find(type);
  return it == values.end() ? 1 : it->second;
}

int building_cost(const std::string& type) {
  static const std::map<std::string, int> costs = {
      {"PORT", 7}, {"DOCK", 7}, {"FARM", 5}, {"MINE", 5}, {"FORGE", 5}, {"WINDMILL", 5},
      {"SAWMILL", 5}, {"MARKET", 5}, {"CUSTOMS_HOUSE", 5}, {"LUMBER_HUT", 3}, {"TEMPLE", 20},
      {"WATER_TEMPLE", 20}, {"FOREST_TEMPLE", 15}, {"MOUNTAIN_TEMPLE", 20}, {"EMBASSY", 5}};
  auto it = costs.find(type);
  return it == costs.end() ? 0 : it->second;
}

int resource_cost(const std::string& type) {
  if (type == "FISH" || type == "FRUIT" || type == "ANIMAL") {
    return 2;
  }
  return 0;
}

int resource_bonus(const std::string& type) {
  if (type == "FISH" || type == "FRUIT" || type == "ANIMAL") {
    return 1;
  }
  if (type == "STARFISH" || type == "WHALE") {
    return 10;
  }
  return 0;
}

int tech_tier(const std::string& tech) {
  static const std::map<std::string, int> tiers = {
      {"CLIMBING", 1}, {"FISHING", 1}, {"HUNTING", 1}, {"ORGANIZATION", 1}, {"RIDING", 1},
      {"ARCHERY", 2}, {"FARMING", 2}, {"FORESTRY", 2}, {"FREE_SPIRIT", 2}, {"MEDITATION", 2},
      {"MINING", 2}, {"ROADS", 2}, {"RAMMING", 2}, {"SAILING", 2}, {"STRATEGY", 2},
      {"AQUATISM", 3}, {"CHIVALRY", 3}, {"CONSTRUCTION", 3}, {"DIPLOMACY", 3},
      {"MATHEMATICS", 3}, {"NAVIGATION", 3}, {"SMITHERY", 3}, {"SPIRITUALISM", 3},
      {"TRADE", 3}, {"PHILOSOPHY", 3}};
  auto it = tiers.find(tech);
  return it == tiers.end() ? 1 : it->second;
}

int next_unit_id(const NativeGameState& state) {
  int max_id = 0;
  for (const NativeUnit& unit : state.units) {
    max_id = std::max(max_id, unit.id);
  }
  return max_id + 1;
}

int next_city_id(const NativeGameState& state) {
  int max_id = 0;
  for (const NativeCity& city : state.cities) {
    max_id = std::max(max_id, city.id);
  }
  return max_id + 1;
}

void update_tribe_economy(NativeGameState& state, int tribe_id, int stars_delta, int score_delta) {
  NativeTribe* tribe = tribe_by_id(state, tribe_id);
  if (tribe == nullptr) {
    return;
  }
  tribe->stars += stars_delta;
  tribe->score += score_delta;
  set_tribe_payload_field(state, tribe_id, "stars", py::int_(tribe->stars));
  set_tribe_payload_field(state, tribe_id, "score", py::int_(tribe->score));
}

void append_researched_tech_payload(NativeGameState& state, int tribe_id, const std::string& tech) {
  if (!state.observation.contains("tribes") || !py::isinstance<py::list>(state.observation["tribes"])) {
    return;
  }
  py::list tribes = py::reinterpret_borrow<py::list>(state.observation["tribes"]);
  for (const auto& item : tribes) {
    if (!py::isinstance<py::dict>(item)) {
      continue;
    }
    py::dict tribe = py::reinterpret_borrow<py::dict>(item);
    if (read_int(tribe, "id", 0) != tribe_id) {
      continue;
    }
    const char* key = tribe.contains("researched_tech_ids") ? "researched_tech_ids" : "tech";
    py::list techs = tribe.contains(key) && py::isinstance<py::list>(tribe[key])
        ? py::reinterpret_borrow<py::list>(tribe[key])
        : py::list();
    techs.append(py::str(tech));
    tribe[key] = techs;
    return;
  }
}

void mark_unit_removed(NativeGameState& state, NativeUnit& unit) {
  NativeTile* tile = tile_at(state, unit.x, unit.y);
  if (tile != nullptr && tile->unit_id == unit.id) {
    tile->unit_id = 0;
    sync_tile_to_payload(state, *tile);
  }
  unit.current_hp = 0;
  unit.status = "FINISHED";
  set_unit_payload_field(state, unit.id, "current_hp", "hp", py::int_(0));
  set_unit_payload_field(state, unit.id, "status", "s", py::str("FINISHED"));
}

bool is_loss_result(const NativeGameState& state, int tribe_id) {
  const NativeTribe* tribe = tribe_by_id_const(state, tribe_id);
  return tribe != nullptr && tribe->result == "LOSS";
}

bool uses_capital_objective(const NativeGameState& state) {
  const std::string mode = read_string(state.observation, "mode");
  return mode == "CAPITALS" || mode == "MIGHT";
}

void set_terminal_winner(NativeGameState& state, int winner_id, const std::string& reason) {
  state.terminal = true;
  state.winner_id = winner_id;
  state.terminal_value_known = true;
  state.terminal_value = winner_id == state.root_player_id ? 1.0 : -1.0;
  state.terminal_reason = reason;
  state.legal_action_indexes.clear();
}

[[noreturn]] void throw_transition_error(const std::string& reason) {
  throw std::runtime_error("Native strict forward model parity failure: " + reason);
}

void evaluate_capital_terminal(NativeGameState& state) {
  if (!uses_capital_objective(state)) {
    return;
  }
  std::vector<int> capital_city_ids;
  for (const NativeCity& city : state.cities) {
    if (city.capital) {
      capital_city_ids.push_back(city.id);
    }
  }
  if (capital_city_ids.empty()) {
    return;
  }
  std::set<int> candidate_tribes;
  for (const NativeTribe& tribe : state.tribes) {
    if (tribe.result != "LOSS") {
      candidate_tribes.insert(tribe.id);
    }
  }
  for (const NativeCity& city : state.cities) {
    if (city.tribe_id >= 0 && !is_loss_result(state, city.tribe_id)) {
      candidate_tribes.insert(city.tribe_id);
    }
  }
  for (int tribe_id : candidate_tribes) {
    bool controls_all = true;
    for (int capital_city_id : capital_city_ids) {
      const NativeCity* capital_city = nullptr;
      for (const NativeCity& city : state.cities) {
        if (city.id == capital_city_id) {
          capital_city = &city;
          break;
        }
      }
      if (capital_city == nullptr || capital_city->tribe_id != tribe_id) {
        controls_all = false;
        break;
      }
    }
    if (controls_all) {
      set_terminal_winner(state, tribe_id, "capital_objective:player_" + std::to_string(tribe_id));
      return;
    }
  }
}

bool apply_move(NativeGameState& next, const NativeAction& action) {
  const int unit_id = action_int(action, "unit_id", "u", 0);
  NativeUnit* unit = unit_by_id(next, unit_id);
  if (unit == nullptr) {
    return false;
  }
  const int old_x = unit->x;
  const int old_y = unit->y;
  const int x = action_int(action, "x", nullptr, old_x);
  const int y = action_int(action, "y", nullptr, old_y);
  NativeTile* from = tile_at(next, old_x, old_y);
  NativeTile* to = tile_at(next, x, y);
  if (to == nullptr || (to->unit_id > 0 && to->unit_id != unit_id)) {
    return false;
  }
  if (from != nullptr) {
    from->unit_id = 0;
    sync_tile_to_payload(next, *from);
  }
  to->unit_id = unit_id;
  sync_tile_to_payload(next, *to);
  unit->x = x;
  unit->y = y;
  unit->status = "MOVED";
  set_unit_payload_field(next, unit_id, "x", nullptr, py::int_(x));
  set_unit_payload_field(next, unit_id, "y", nullptr, py::int_(y));
  set_unit_payload_field(next, unit_id, "status", "s", py::str("MOVED"));
  return true;
}

bool apply_capture(NativeGameState& next, const NativeAction& action) {
  const int unit_id = action_int(action, "unit_id", "u", 0);
  const int target_city_id = action_int(action, "target_city_id", "tc", action_int(action, "city_id", "c", 0));
  NativeUnit* unit = unit_by_id(next, unit_id);
  NativeCity* city = city_by_id(next, target_city_id);
  if (unit == nullptr || unit->tribe_id < 0) {
    return false;
  }
  const std::string capture_type = action_string(action, "capture_type", "ct");
  if (city == nullptr && capture_type == "VILLAGE") {
    NativeTile* tile = tile_at(next, unit->x, unit->y);
    if (tile == nullptr || tile->terrain != "VILLAGE") {
      return false;
    }
    NativeCity new_city;
    new_city.id = next_city_id(next);
    new_city.tribe_id = unit->tribe_id;
    new_city.x = unit->x;
    new_city.y = unit->y;
    new_city.level = 1;
    new_city.population = 0;
    new_city.population_need = 2;
    new_city.production = 1;
    new_city.capital = false;
    new_city.walls = false;
    next.cities.push_back(new_city);
    tile->terrain = "CITY";
    tile->city_id = new_city.id;
    sync_tile_to_payload(next, *tile);
    if (next.observation.contains("cities") && py::isinstance<py::list>(next.observation["cities"])) {
      py::dict out;
      out["id"] = new_city.id;
      out["p"] = new_city.tribe_id;
      out["tribe_id"] = new_city.tribe_id;
      out["x"] = new_city.x;
      out["y"] = new_city.y;
      out["level"] = new_city.level;
      out["lvl"] = new_city.level;
      out["population"] = new_city.population;
      out["pop"] = new_city.population;
      out["population_need"] = new_city.population_need;
      out["need"] = new_city.population_need;
      out["production"] = new_city.production;
      out["prod"] = new_city.production;
      out["is_capital"] = false;
      out["cap"] = false;
      out["has_walls"] = false;
      out["wall"] = false;
      py::reinterpret_borrow<py::list>(next.observation["cities"]).append(out);
    }
    unit->status = "FINISHED";
    unit->city_id = new_city.id;
    set_unit_payload_field(next, unit->id, "city_id", "c", py::int_(new_city.id));
    set_unit_payload_field(next, unit->id, "status", "s", py::str("FINISHED"));
    update_tribe_economy(next, unit->tribe_id, 0, 10);
    return true;
  }
  if (city == nullptr) {
    return false;
  }
  if (city->tribe_id == unit->tribe_id) {
    return false;
  }
  city->tribe_id = unit->tribe_id;
  unit->status = "FINISHED";
  set_city_payload_field(next, city->id, "tribe_id", "p", py::int_(city->tribe_id));
  set_unit_payload_field(next, unit->id, "status", "s", py::str("FINISHED"));
  evaluate_capital_terminal(next);
  return true;
}

bool apply_attack(NativeGameState& next, const NativeAction& action) {
  const int attacker_id = action_int(action, "unit_id", "u", 0);
  const int target_id = action_int(action, "target_unit_id", "tu", 0);
  NativeUnit* attacker = unit_by_id(next, attacker_id);
  NativeUnit* target = unit_by_id(next, target_id);
  if (attacker == nullptr || target == nullptr || attacker->tribe_id == target->tribe_id) {
    return false;
  }
  const int attack_damage = std::max(1, unit_attack(attacker->type) * 2);
  target->current_hp = std::max(0, target->current_hp - attack_damage);
  set_unit_payload_field(next, target->id, "current_hp", "hp", py::int_(target->current_hp));
  if (target->current_hp <= 0) {
    mark_unit_removed(next, *target);
    attacker->kills += 1;
    set_unit_payload_field(next, attacker->id, "kills", "k", py::int_(attacker->kills));
  } else {
    const int retaliation = std::max(0, unit_defence(target->type) - 1);
    if (retaliation > 0) {
      attacker->current_hp = std::max(0, attacker->current_hp - retaliation);
      set_unit_payload_field(next, attacker->id, "current_hp", "hp", py::int_(attacker->current_hp));
      if (attacker->current_hp <= 0) {
        mark_unit_removed(next, *attacker);
      }
    }
  }
  if (attacker->current_hp > 0) {
    attacker->status = "FINISHED";
    set_unit_payload_field(next, attacker->id, "status", "s", py::str("FINISHED"));
  }
  return true;
}

bool apply_convert(NativeGameState& next, const NativeAction& action) {
  const int unit_id = action_int(action, "unit_id", "u", 0);
  const int target_id = action_int(action, "target_unit_id", "tu", 0);
  NativeUnit* unit = unit_by_id(next, unit_id);
  NativeUnit* target = unit_by_id(next, target_id);
  if (unit == nullptr || target == nullptr || unit->tribe_id == target->tribe_id) {
    return false;
  }
  target->tribe_id = unit->tribe_id;
  unit->status = "FINISHED";
  set_unit_payload_field(next, target->id, "tribe_id", "p", py::int_(target->tribe_id));
  set_unit_payload_field(next, unit->id, "status", "s", py::str("FINISHED"));
  return true;
}

bool has_tech(const NativeTribe& tribe, const std::string& tech) {
  return std::find(tribe.researched_tech_ids.begin(), tribe.researched_tech_ids.end(), tech) != tribe.researched_tech_ids.end();
}

py::dict position_payload(int x, int y) {
  py::dict pos;
  pos["x"] = x;
  pos["y"] = y;
  return pos;
}

void append_generated_action(
    NativeGameState& state,
    std::vector<NativeAction>& actions,
    int max_actions,
    NativeAction action) {
  if (max_actions >= 0 && static_cast<int>(state.legal_action_indexes.size()) >= max_actions) {
    return;
  }
  action.payload["id"] = action.id;
  action.payload["type"] = action.type;
  if (action.unit_id > 0) {
    action.payload["unit_id"] = action.unit_id;
    action.payload["u"] = action.unit_id;
  }
  if (action.city_id > 0) {
    action.payload["city_id"] = action.city_id;
    action.payload["c"] = action.city_id;
  }
  const int index = static_cast<int>(actions.size());
  actions.push_back(std::move(action));
  state.legal_action_indexes.push_back(index);
}

void append_tile_action(
    NativeGameState& state,
    std::vector<NativeAction>& actions,
    int max_actions,
    const std::string& id_suffix,
    const std::string& type,
    int tribe_id,
    int unit_id,
    int city_id,
    int x,
    int y,
    const char* position_key) {
  NativeAction action;
  action.id = "sim:p" + std::to_string(tribe_id) + ":t" + std::to_string(state.tick) + id_suffix;
  action.type = type;
  action.unit_id = unit_id;
  action.city_id = city_id;
  action.payload = py::dict();
  action.payload["tribe_id"] = tribe_id;
  action.payload["p"] = tribe_id;
  action.payload["x"] = x;
  action.payload["y"] = y;
  action.payload[position_key] = position_payload(x, y);
  append_generated_action(state, actions, max_actions, std::move(action));
}

bool unit_can_act(const NativeUnit& unit, int active_player_id) {
  return unit.tribe_id == active_player_id && unit.status != "MOVED" && unit.status != "FINISHED" && unit.status != "EXHAUSTED";
}

bool passable_move_target(const NativeTile& tile) {
  return tile.explored && tile.unit_id <= 0 && tile.terrain != "DEEP_WATER" && tile.terrain != "WATER";
}

void regenerate_unit_actions(NativeGameState& state, std::vector<NativeAction>& actions, int max_actions, const NativeUnit& unit) {
  if (!unit_can_act(unit, state.active_player_id)) {
    return;
  }
  NativeTile* current_tile = tile_at(state, unit.x, unit.y);
  if (current_tile != nullptr && current_tile->city_id > 0) {
    NativeCity* city = city_by_id(state, current_tile->city_id);
    if (city != nullptr && city->tribe_id != unit.tribe_id) {
      NativeAction capture;
      capture.id = "sim:p" + std::to_string(state.active_player_id) + ":t" + std::to_string(state.tick) +
          ":u" + std::to_string(unit.id) + ":capture:c" + std::to_string(city->id);
      capture.type = "CAPTURE";
      capture.unit_id = unit.id;
      capture.city_id = city->id;
      capture.payload = py::dict();
      capture.payload["tribe_id"] = state.active_player_id;
      capture.payload["p"] = state.active_player_id;
      capture.payload["target_city_id"] = city->id;
      capture.payload["tc"] = city->id;
      append_generated_action(state, actions, max_actions, std::move(capture));
    }
  }
  static const int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
  for (const auto& dir : dirs) {
    const int x = unit.x + dir[0];
    const int y = unit.y + dir[1];
    NativeTile* tile = tile_at(state, x, y);
    if (tile == nullptr) {
      continue;
    }
    if (tile->unit_id > 0) {
      NativeUnit* target = unit_by_id(state, tile->unit_id);
      if (target != nullptr && target->tribe_id != unit.tribe_id) {
        NativeAction attack;
        attack.id = "sim:p" + std::to_string(state.active_player_id) + ":t" + std::to_string(state.tick) +
            ":u" + std::to_string(unit.id) + ":attack:u" + std::to_string(target->id);
        attack.type = "ATTACK";
        attack.unit_id = unit.id;
        attack.payload = py::dict();
        attack.payload["tribe_id"] = state.active_player_id;
        attack.payload["p"] = state.active_player_id;
        attack.payload["target_unit_id"] = target->id;
        attack.payload["tu"] = target->id;
        append_generated_action(state, actions, max_actions, std::move(attack));
      }
      continue;
    }
    if (passable_move_target(*tile)) {
      append_tile_action(
          state,
          actions,
          max_actions,
          ":u" + std::to_string(unit.id) + ":move:" + std::to_string(x) + ":" + std::to_string(y),
          "MOVE",
          state.active_player_id,
          unit.id,
          0,
          x,
          y,
          "destination");
    }
  }
  if (unit.current_hp > 0 && unit.max_hp > 0 && unit.current_hp < unit.max_hp) {
    NativeAction recover;
    recover.id = "sim:p" + std::to_string(state.active_player_id) + ":t" + std::to_string(state.tick) +
        ":u" + std::to_string(unit.id) + ":recover";
    recover.type = "RECOVER";
    recover.unit_id = unit.id;
    recover.payload = py::dict();
    recover.payload["tribe_id"] = state.active_player_id;
    recover.payload["p"] = state.active_player_id;
    append_generated_action(state, actions, max_actions, std::move(recover));
  }
}

void regenerate_city_actions(NativeGameState& state, std::vector<NativeAction>& actions, int max_actions, const NativeCity& city) {
  if (city.tribe_id != state.active_player_id) {
    return;
  }
  NativeTribe* tribe = tribe_by_id(state, city.tribe_id);
  const int stars = tribe == nullptr ? 0 : tribe->stars;
  NativeTile* city_tile = tile_at(state, city.x, city.y);
  if (city_tile != nullptr && city_tile->unit_id <= 0 && stars >= unit_cost("WARRIOR")) {
    NativeAction spawn;
    spawn.id = "sim:p" + std::to_string(state.active_player_id) + ":t" + std::to_string(state.tick) +
        ":c" + std::to_string(city.id) + ":spawn:WARRIOR";
    spawn.type = "SPAWN";
    spawn.city_id = city.id;
    spawn.payload = py::dict();
    spawn.payload["tribe_id"] = state.active_player_id;
    spawn.payload["p"] = state.active_player_id;
    spawn.payload["unit_type"] = "WARRIOR";
    spawn.payload["ut"] = "WARRIOR";
    spawn.payload["x"] = city.x;
    spawn.payload["y"] = city.y;
    spawn.payload["position"] = position_payload(city.x, city.y);
    append_generated_action(state, actions, max_actions, std::move(spawn));
  }

  for (const NativeTile& tile : state.tiles) {
    const int distance = std::abs(tile.x - city.x) + std::abs(tile.y - city.y);
    if (distance > 2 || !tile.explored) {
      continue;
    }
    if (!tile.road && stars >= 3) {
      append_tile_action(
          state,
          actions,
          max_actions,
          ":road:" + std::to_string(tile.x) + ":" + std::to_string(tile.y),
          "BUILD_ROAD",
          state.active_player_id,
          0,
          0,
          tile.x,
          tile.y,
          "position");
    }
    if (!tile.resource.empty() && tile.city_id == city.id) {
      NativeAction gather;
      gather.id = "sim:p" + std::to_string(state.active_player_id) + ":t" + std::to_string(state.tick) +
          ":c" + std::to_string(city.id) + ":resource:" + tile.resource + ":" +
          std::to_string(tile.x) + ":" + std::to_string(tile.y);
      gather.type = "RESOURCE_GATHERING";
      gather.city_id = city.id;
      gather.payload = py::dict();
      gather.payload["tribe_id"] = state.active_player_id;
      gather.payload["p"] = state.active_player_id;
      gather.payload["resource_type"] = tile.resource;
      gather.payload["rt"] = tile.resource;
      gather.payload["x"] = tile.x;
      gather.payload["y"] = tile.y;
      gather.payload["position"] = position_payload(tile.x, tile.y);
      append_generated_action(state, actions, max_actions, std::move(gather));
    }
    if (tile.building.empty() && tile.resource.empty() && tile.terrain == "FOREST" && stars >= building_cost("LUMBER_HUT")) {
      NativeAction build;
      build.id = "sim:p" + std::to_string(state.active_player_id) + ":t" + std::to_string(state.tick) +
          ":c" + std::to_string(city.id) + ":build:LUMBER_HUT:" +
          std::to_string(tile.x) + ":" + std::to_string(tile.y);
      build.type = "BUILD";
      build.city_id = city.id;
      build.payload = py::dict();
      build.payload["tribe_id"] = state.active_player_id;
      build.payload["p"] = state.active_player_id;
      build.payload["building_type"] = "LUMBER_HUT";
      build.payload["bt"] = "LUMBER_HUT";
      build.payload["x"] = tile.x;
      build.payload["y"] = tile.y;
      build.payload["position"] = position_payload(tile.x, tile.y);
      append_generated_action(state, actions, max_actions, std::move(build));
    }
  }
}

void regenerate_research_actions(NativeGameState& state, std::vector<NativeAction>& actions, int max_actions) {
  NativeTribe* tribe = tribe_by_id(state, state.active_player_id);
  if (tribe == nullptr || tribe->stars < 4) {
    return;
  }
  static const std::vector<std::string> techs = {
      "CLIMBING", "FISHING", "HUNTING", "ORGANIZATION", "RIDING"};
  for (const std::string& tech : techs) {
    if (has_tech(*tribe, tech)) {
      continue;
    }
    NativeAction action;
    action.id = "sim:p" + std::to_string(state.active_player_id) + ":t" + std::to_string(state.tick) +
        ":research:" + tech;
    action.type = "RESEARCH_TECH";
    action.payload = py::dict();
    action.payload["tribe_id"] = state.active_player_id;
    action.payload["p"] = state.active_player_id;
    action.payload["tech"] = tech;
    append_generated_action(state, actions, max_actions, std::move(action));
  }
}

void append_end_turn_action(NativeGameState& state, std::vector<NativeAction>& actions, int max_actions) {
  NativeAction action;
  action.id = "sim:p" + std::to_string(state.active_player_id) + ":t" + std::to_string(state.tick) + ":end";
  action.type = "END_TURN";
  action.payload = py::dict();
  action.payload["tribe_id"] = state.active_player_id;
  action.payload["p"] = state.active_player_id;
  append_generated_action(state, actions, max_actions, std::move(action));
}

void regenerate_actions(NativeGameState& state, std::vector<NativeAction>& actions, int max_actions) {
  state.legal_action_indexes.clear();
  for (const NativeUnit& unit : state.units) {
    regenerate_unit_actions(state, actions, max_actions, unit);
  }
  for (const NativeCity& city : state.cities) {
    regenerate_city_actions(state, actions, max_actions, city);
  }
  regenerate_research_actions(state, actions, max_actions);
  append_end_turn_action(state, actions, max_actions);
}

bool tile_visible_for_asset(const NativeGameState& state, int x, int y) {
  for (const NativeTile& tile : state.tiles) {
    if (tile.x == x && tile.y == y) {
      return tile.visible;
    }
  }
  return false;
}

std::vector<int> visible_enemy_tribes(const NativeGameState& state) {
  std::set<int> ids;
  for (const NativeUnit& unit : state.units) {
    if (unit.tribe_id != state.root_player_id && !unit.hidden && tile_visible_for_asset(state, unit.x, unit.y)) {
      ids.insert(unit.tribe_id);
    }
  }
  for (const NativeCity& city : state.cities) {
    if (city.tribe_id != state.root_player_id && tile_visible_for_asset(state, city.x, city.y)) {
      ids.insert(city.tribe_id);
    }
  }
  return std::vector<int>(ids.begin(), ids.end());
}

int choose_next_active_player(const NativeGameState& state) {
  std::vector<int> live_ids;
  live_ids.reserve(state.tribes.size());
  for (const NativeTribe& tribe : state.tribes) {
    if (tribe.id >= 0 && tribe.result != "LOSS") {
      live_ids.push_back(tribe.id);
    }
  }
  if (live_ids.empty()) {
    return state.root_player_id;
  }
  std::sort(live_ids.begin(), live_ids.end());
  live_ids.erase(std::unique(live_ids.begin(), live_ids.end()), live_ids.end());
  for (int id : live_ids) {
    if (id > state.active_player_id) {
      return id;
    }
  }
  return live_ids.front();
}

void set_active_player(NativeGameState& state, int player_id) {
  state.active_player_id = player_id;
  state.observation["active_player_id"] = player_id;
  state.observation["active"] = player_id;
  state.observation["can_end_turn"] = true;
  state.observation["end"] = true;
  state.transition_kind.clear();
}

void refresh_unit_turn_status(NativeGameState& state, int previous_player_id, int next_player_id) {
  for (NativeUnit& unit : state.units) {
    if (unit.tribe_id == previous_player_id && unit.status == "FRESH") {
      unit.current_hp = std::min(unit.max_hp, unit.current_hp + 2);
      unit.status = "FINISHED";
      set_unit_payload_field(state, unit.id, "current_hp", "hp", py::int_(unit.current_hp));
      set_unit_payload_field(state, unit.id, "status", "s", py::str(unit.status));
    }
  }
  for (NativeUnit& unit : state.units) {
    if (unit.tribe_id == next_player_id) {
      unit.status = "FRESH";
      set_unit_payload_field(state, unit.id, "status", "s", py::str(unit.status));
    }
  }
}

void add_visible_city_income(NativeGameState& state, int player_id) {
  int production = 0;
  for (const NativeCity& city : state.cities) {
    if (city.tribe_id == player_id && tile_visible_for_asset(state, city.x, city.y)) {
      production += std::max(0, city.production);
    }
  }
  if (production > 0) {
    update_tribe_economy(state, player_id, production, 0);
  }
}

void apply_end_turn_transition(NativeGameState& next) {
  const int previous_player_id = next.active_player_id;
  next.tick += 1;
  next.observation["tick"] = next.tick;
  const int next_player_id = choose_next_active_player(next);
  if (next_player_id != next.root_player_id) {
    throw_transition_error(
        "END_TURN would require exact opponent-perspective observation parity for active_player_id=" +
        std::to_string(next_player_id) +
        " from root_player_id=" +
        std::to_string(next.root_player_id));
  }
  refresh_unit_turn_status(next, previous_player_id, next_player_id);
  set_active_player(next, next_player_id);
  add_visible_city_income(next, next_player_id);
}

bool apply_recover(NativeGameState& next, const NativeAction& action) {
  const int unit_id = action_int(action, "unit_id", "u", 0);
  NativeUnit* unit = unit_by_id(next, unit_id);
  if (unit == nullptr) {
    return false;
  }
  unit->current_hp = std::min(unit->max_hp, unit->current_hp + 2);
  unit->status = "FINISHED";
  set_unit_payload_field(next, unit_id, "current_hp", "hp", py::int_(unit->current_hp));
  set_unit_payload_field(next, unit_id, "status", "s", py::str("FINISHED"));
  return true;
}

bool apply_spawn(NativeGameState& next, const NativeAction& action) {
  int city_id = action_int(action, "city_id", "c", 0);
  if (city_id == 0) {
    const int x = action_int(action, "x", nullptr, -1);
    const int y = action_int(action, "y", nullptr, -1);
    for (const NativeCity& candidate : next.cities) {
      if (candidate.x == x && candidate.y == y) {
        city_id = candidate.id;
        break;
      }
    }
  }
  NativeCity* city = city_by_id(next, city_id);
  if (city == nullptr) {
    return false;
  }
  NativeTile* tile = tile_at(next, city->x, city->y);
  if (tile == nullptr || tile->unit_id > 0) {
    return false;
  }
  const std::string type = action_string(action, "unit_type", "ut").empty()
      ? "WARRIOR"
      : action_string(action, "unit_type", "ut");
  NativeUnit unit;
  unit.id = next_unit_id(next);
  unit.tribe_id = city->tribe_id;
  unit.city_id = city_id;
  unit.x = city->x;
  unit.y = city->y;
  unit.type = type;
  unit.status = "FRESH";
  unit.current_hp = 10;
  unit.max_hp = 10;
  next.units.push_back(unit);
  tile->unit_id = unit.id;
  sync_tile_to_payload(next, *tile);
  if (next.observation.contains("units") && py::isinstance<py::list>(next.observation["units"])) {
    py::dict out;
    out["id"] = unit.id;
    out["p"] = unit.tribe_id;
    out["tribe_id"] = unit.tribe_id;
    out["c"] = unit.city_id;
    out["city_id"] = unit.city_id;
    out["t"] = unit.type;
    out["type"] = unit.type;
    out["x"] = unit.x;
    out["y"] = unit.y;
    out["hp"] = unit.current_hp;
    out["current_hp"] = unit.current_hp;
    out["mhp"] = unit.max_hp;
    out["max_hp"] = unit.max_hp;
    out["k"] = 0;
    out["kills"] = 0;
    out["v"] = false;
    out["is_veteran"] = false;
    out["s"] = unit.status;
    out["status"] = unit.status;
    out["h"] = false;
    out["is_hidden"] = false;
    py::reinterpret_borrow<py::list>(next.observation["units"]).append(out);
  }
  update_tribe_economy(next, city->tribe_id, -unit_cost(type), unit_points(type));
  return true;
}

bool apply_build_road(NativeGameState& next, const NativeAction& action) {
  const int tribe_id = action_int(action, "tribe_id", "p", next.active_player_id);
  const int x = action_int(action, "x", nullptr, -1);
  const int y = action_int(action, "y", nullptr, -1);
  NativeTile* tile = tile_at(next, x, y);
  if (tile == nullptr) {
    return false;
  }
  tile->road = true;
  sync_tile_to_payload(next, *tile);
  const bool water = tile->terrain == "SHALLOW_WATER" || tile->terrain == "DEEP_WATER";
  update_tribe_economy(next, tribe_id, water ? -5 : -3, 0);
  return true;
}

bool apply_resource_gathering(NativeGameState& next, const NativeAction& action) {
  const int city_id = action_int(action, "city_id", "c", 0);
  NativeCity* city = city_by_id(next, city_id);
  if (city == nullptr) {
    return false;
  }
  const int x = action_int(action, "x", nullptr, -1);
  const int y = action_int(action, "y", nullptr, -1);
  NativeTile* tile = tile_at(next, x, y);
  if (tile == nullptr) {
    return false;
  }
  const std::string resource = action_string(action, "resource_type", "rt").empty()
      ? tile->resource
      : action_string(action, "resource_type", "rt");
  tile->resource.clear();
  sync_tile_to_payload(next, *tile);
  if (resource == "STARFISH" || resource == "WHALE") {
    update_tribe_economy(next, city->tribe_id, resource_bonus(resource), 0);
    return true;
  }
  update_tribe_economy(next, city->tribe_id, -resource_cost(resource), 0);
  city->population += resource_bonus(resource);
  set_city_payload_field(next, city_id, "population", "pop", py::int_(city->population));
  return true;
}

bool apply_research(NativeGameState& next, const NativeAction& action) {
  const int tribe_id = action_int(action, "tribe_id", "p", next.active_player_id);
  const std::string tech = action_string(action, "tech");
  NativeTribe* tribe = tribe_by_id(next, tribe_id);
  if (tribe == nullptr || tech.empty()) {
    return false;
  }
  if (std::find(tribe->researched_tech_ids.begin(), tribe->researched_tech_ids.end(), tech) == tribe->researched_tech_ids.end()) {
    tribe->researched_tech_ids.push_back(tech);
    append_researched_tech_payload(next, tribe_id, tech);
  }
  int city_count = 0;
  for (const NativeCity& city : next.cities) {
    if (city.tribe_id == tribe_id) {
      city_count += 1;
    }
  }
  const int cost = 4 + tech_tier(tech) * std::max(1, city_count);
  update_tribe_economy(next, tribe_id, -cost, tech_tier(tech) * 5);
  return true;
}

bool apply_build(NativeGameState& next, const NativeAction& action) {
  const int city_id = action_int(action, "city_id", "c", 0);
  NativeCity* city = city_by_id(next, city_id);
  if (city == nullptr) {
    return false;
  }
  const int x = action_int(action, "x", nullptr, -1);
  const int y = action_int(action, "y", nullptr, -1);
  NativeTile* tile = tile_at(next, x, y);
  if (tile == nullptr) {
    return false;
  }
  std::string building = action_string(action, "building_type", "bt");
  if (building.empty()) {
    return false;
  }
  tile->building = building;
  sync_tile_to_payload(next, *tile);
  update_tribe_economy(next, city->tribe_id, -building_cost(building), 0);
  city->population += 1;
  set_city_payload_field(next, city_id, "population", "pop", py::int_(city->population));
  return true;
}

bool apply_level_up(NativeGameState& next, const NativeAction& action) {
  const int city_id = action_int(action, "city_id", "c", 0);
  NativeCity* city = city_by_id(next, city_id);
  if (city == nullptr) {
    return false;
  }
  city->level += 1;
  city->population = 0;
  city->population_need = std::max(2, city->population_need + 1);
  city->production += 1;
  set_city_payload_field(next, city_id, "level", "lvl", py::int_(city->level));
  set_city_payload_field(next, city_id, "population", "pop", py::int_(city->population));
  set_city_payload_field(next, city_id, "population_need", "need", py::int_(city->population_need));
  set_city_payload_field(next, city_id, "production", "prod", py::int_(city->production));
  update_tribe_economy(next, city->tribe_id, 0, 5);
  return true;
}

bool apply_clear_or_burn_forest(NativeGameState& next, const NativeAction& action, bool burn) {
  const int tribe_id = action_int(action, "tribe_id", "p", next.active_player_id);
  const int x = action_int(action, "x", nullptr, -1);
  const int y = action_int(action, "y", nullptr, -1);
  NativeTile* tile = tile_at(next, x, y);
  if (tile == nullptr) {
    return false;
  }
  tile->terrain = "PLAIN";
  tile->resource.clear();
  tile->building.clear();
  sync_tile_to_payload(next, *tile);
  update_tribe_economy(next, tribe_id, burn ? 2 : -3, 0);
  return true;
}

bool apply_grow_forest(NativeGameState& next, const NativeAction& action) {
  const int tribe_id = action_int(action, "tribe_id", "p", next.active_player_id);
  const int x = action_int(action, "x", nullptr, -1);
  const int y = action_int(action, "y", nullptr, -1);
  NativeTile* tile = tile_at(next, x, y);
  if (tile == nullptr) {
    return false;
  }
  tile->terrain = "FOREST";
  sync_tile_to_payload(next, *tile);
  update_tribe_economy(next, tribe_id, -5, 0);
  return true;
}

bool apply_destroy(NativeGameState& next, const NativeAction& action) {
  const int x = action_int(action, "x", nullptr, -1);
  const int y = action_int(action, "y", nullptr, -1);
  NativeTile* tile = tile_at(next, x, y);
  if (tile == nullptr) {
    return false;
  }
  tile->building.clear();
  sync_tile_to_payload(next, *tile);
  return true;
}

bool apply_disband(NativeGameState& next, const NativeAction& action) {
  const int unit_id = action_int(action, "unit_id", "u", 0);
  NativeUnit* unit = unit_by_id(next, unit_id);
  if (unit == nullptr) {
    return false;
  }
  const int tribe_id = unit->tribe_id;
  mark_unit_removed(next, *unit);
  update_tribe_economy(next, tribe_id, std::max(0, unit_cost(unit->type) / 2), 0);
  return true;
}

bool apply_make_veteran(NativeGameState& next, const NativeAction& action) {
  const int unit_id = action_int(action, "unit_id", "u", 0);
  NativeUnit* unit = unit_by_id(next, unit_id);
  if (unit == nullptr) {
    return false;
  }
  unit->veteran = true;
  unit->max_hp += 5;
  unit->current_hp = unit->max_hp;
  unit->kills = 0;
  unit->status = "FINISHED";
  set_unit_payload_field(next, unit_id, "is_veteran", "v", py::bool_(true));
  set_unit_payload_field(next, unit_id, "max_hp", "mhp", py::int_(unit->max_hp));
  set_unit_payload_field(next, unit_id, "current_hp", "hp", py::int_(unit->current_hp));
  set_unit_payload_field(next, unit_id, "kills", "k", py::int_(0));
  set_unit_payload_field(next, unit_id, "status", "s", py::str("FINISHED"));
  return true;
}

bool apply_upgrade_unit(NativeGameState& next, const NativeAction& action, const std::string& upgraded_type) {
  const int unit_id = action_int(action, "unit_id", "u", 0);
  NativeUnit* unit = unit_by_id(next, unit_id);
  if (unit == nullptr) {
    return false;
  }

  int cost = 0;
  if (upgraded_type == "RAMMER") cost = 5;
  else if (upgraded_type == "SCOUT") cost = 5;
  else if (upgraded_type == "BOMBER") cost = 15;
  if (cost > 0) {
    update_tribe_economy(next, unit->tribe_id, -cost, 0);
  }

  unit->type = upgraded_type;
  unit->status = "FINISHED";
  set_unit_payload_field(next, unit_id, "type", "t", py::str(unit->type));
  set_unit_payload_field(next, unit_id, "status", "s", py::str("FINISHED"));
  return true;
}

void parse_tiles(NativeGameState& state) {
  if (!state.observation.contains("board") || !py::isinstance<py::dict>(state.observation["board"])) {
    throw std::runtime_error("Native strict payload parse failure: observation.board must be an object.");
  }
  py::dict board = py::reinterpret_borrow<py::dict>(state.observation["board"]);
  require_key(board, "board", {"size"});
  state.board_size = read_int(board, "size", 0);
  if (!board.contains("tiles") || !py::isinstance<py::list>(board["tiles"])) {
    throw std::runtime_error("Native strict payload parse failure: board.tiles must be a list.");
  }
  py::list rows = py::reinterpret_borrow<py::list>(board["tiles"]);
  if (state.board_size <= 0 || static_cast<int>(py::len(rows)) != state.board_size) {
    throw std::runtime_error("Native strict payload parse failure: board size does not match tile row count.");
  }
  state.tiles.reserve(static_cast<size_t>(state.board_size * state.board_size));
  for (const auto& row_handle : rows) {
    if (!py::isinstance<py::list>(row_handle)) {
      throw std::runtime_error("Native strict payload parse failure: board tile row must be a list.");
    }
    py::list row = py::reinterpret_borrow<py::list>(row_handle);
    if (static_cast<int>(py::len(row)) != state.board_size) {
      throw std::runtime_error("Native strict payload parse failure: board tile row length does not match board size.");
    }
    for (const auto& tile_handle : row) {
      if (!py::isinstance<py::dict>(tile_handle)) {
        throw std::runtime_error("Native strict payload parse failure: board tile must be an object.");
      }
      py::dict tile_payload = py::reinterpret_borrow<py::dict>(tile_handle);
      require_key(tile_payload, "board tile", {"x"});
      require_key(tile_payload, "board tile", {"y"});
      require_key(tile_payload, "board tile", {"explored"});
      require_key(tile_payload, "board tile", {"terrain"});
      require_key(tile_payload, "board tile", {"city_id", "city"});
      require_key(tile_payload, "board tile", {"unit_id", "unit"});
      NativeTile tile;
      tile.x = read_int(tile_payload, "x", 0);
      tile.y = read_int(tile_payload, "y", 0);
      tile.explored = read_bool(tile_payload, "explored", false);
      tile.visible = read_bool(tile_payload, "visible", tile.explored);
      tile.road = read_bool(tile_payload, "road", false);
      tile.terrain = read_string(tile_payload, "terrain");
      tile.resource = read_string(tile_payload, "resource");
      tile.building = read_string(tile_payload, "building");
      tile.city_id = read_int(tile_payload, "city_id", 0);
      tile.unit_id = read_int(tile_payload, "unit_id", 0);
      state.tiles.push_back(tile);
    }
  }
}

void parse_units(NativeGameState& state) {
  if (!state.observation.contains("units") || !py::isinstance<py::list>(state.observation["units"])) {
    throw std::runtime_error("Native strict payload parse failure: observation.units must be a list.");
  }
  py::list units = py::reinterpret_borrow<py::list>(state.observation["units"]);
  state.units.reserve(py::len(units));
  for (const auto& unit_handle : units) {
    if (!py::isinstance<py::dict>(unit_handle)) {
      throw std::runtime_error("Native strict payload parse failure: unit entry must be an object.");
    }
    py::dict payload = py::reinterpret_borrow<py::dict>(unit_handle);
    require_key(payload, "unit", {"id"});
    require_key(payload, "unit", {"tribe_id", "p"});
    require_key(payload, "unit", {"city_id", "c"});
    require_key(payload, "unit", {"type", "t"});
    require_key(payload, "unit", {"x"});
    require_key(payload, "unit", {"y"});
    require_key(payload, "unit", {"current_hp", "hp"});
    require_key(payload, "unit", {"max_hp", "mhp"});
    require_key(payload, "unit", {"status", "s"});
    NativeUnit unit;
    unit.id = read_int(payload, "id", 0);
    unit.tribe_id = read_int(payload, "tribe_id", read_int(payload, "p", -1));
    unit.city_id = read_int(payload, "city_id", read_int(payload, "c", 0));
    unit.x = read_int(payload, "x", 0);
    unit.y = read_int(payload, "y", 0);
    unit.current_hp = read_int(payload, "current_hp", read_int(payload, "hp", 0));
    unit.max_hp = read_int(payload, "max_hp", read_int(payload, "mhp", 0));
    unit.kills = read_int(payload, "kills", read_int(payload, "k", 0));
    unit.veteran = read_bool(payload, "is_veteran", read_bool(payload, "v", false));
    unit.hidden = read_bool(payload, "is_hidden", read_bool(payload, "h", false));
    unit.type = read_string(payload, "type");
    if (unit.type.empty()) {
      unit.type = read_string(payload, "t");
    }
    unit.status = read_string(payload, "status");
    if (unit.status.empty()) {
      unit.status = read_string(payload, "s");
    }
    state.units.push_back(unit);
  }
}

void parse_cities(NativeGameState& state) {
  if (!state.observation.contains("cities") || !py::isinstance<py::list>(state.observation["cities"])) {
    throw std::runtime_error("Native strict payload parse failure: observation.cities must be a list.");
  }
  py::list cities = py::reinterpret_borrow<py::list>(state.observation["cities"]);
  state.cities.reserve(py::len(cities));
  for (const auto& city_handle : cities) {
    if (!py::isinstance<py::dict>(city_handle)) {
      throw std::runtime_error("Native strict payload parse failure: city entry must be an object.");
    }
    py::dict payload = py::reinterpret_borrow<py::dict>(city_handle);
    require_key(payload, "city", {"id"});
    require_key(payload, "city", {"tribe_id", "p"});
    require_key(payload, "city", {"x"});
    require_key(payload, "city", {"y"});
    require_key(payload, "city", {"level", "lvl"});
    require_key(payload, "city", {"population", "pop"});
    require_key(payload, "city", {"population_need", "need"});
    require_key(payload, "city", {"production", "prod"});
    NativeCity city;
    city.id = read_int(payload, "id", 0);
    city.tribe_id = read_int(payload, "tribe_id", read_int(payload, "p", -1));
    city.x = read_int(payload, "x", 0);
    city.y = read_int(payload, "y", 0);
    city.level = read_int(payload, "level", read_int(payload, "lvl", 0));
    city.population = read_int(payload, "population", read_int(payload, "pop", 0));
    city.population_need = read_int(payload, "population_need", read_int(payload, "need", 0));
    city.production = read_int(payload, "production", read_int(payload, "prod", 0));
    city.capital = read_bool(payload, "is_capital", read_bool(payload, "cap", false));
    city.walls = read_bool(payload, "has_walls", read_bool(payload, "wall", false));
    state.cities.push_back(city);
  }
}

void parse_tribes(NativeGameState& state) {
  if (!state.observation.contains("tribes") || !py::isinstance<py::list>(state.observation["tribes"])) {
    throw std::runtime_error("Native strict payload parse failure: observation.tribes must be a list.");
  }
  py::list tribes = py::reinterpret_borrow<py::list>(state.observation["tribes"]);
  state.tribes.reserve(py::len(tribes));
  for (const auto& tribe_handle : tribes) {
    if (!py::isinstance<py::dict>(tribe_handle)) {
      throw std::runtime_error("Native strict payload parse failure: tribe entry must be an object.");
    }
    py::dict payload = py::reinterpret_borrow<py::dict>(tribe_handle);
    require_key(payload, "tribe", {"id"});
    require_key(payload, "tribe", {"stars"});
    require_key(payload, "tribe", {"score"});
    NativeTribe tribe;
    tribe.id = read_int(payload, "id", 0);
    tribe.stars = read_int(payload, "stars", 0);
    tribe.score = read_int(payload, "score", 0);
    tribe.capital_id = read_int(payload, "capital_id", read_int(payload, "cap", 0));
    tribe.result = read_string(payload, "result");
    if (tribe.result.empty()) {
      tribe.result = read_string(payload, "res");
    }
    tribe.researched_tech_ids = read_string_list(payload, "researched_tech_ids");
    if (tribe.researched_tech_ids.empty()) {
      tribe.researched_tech_ids = read_string_list(payload, "tech");
    }
    state.tribes.push_back(tribe);
  }
}

void parse_observation_state(NativeGameState& state) {
  parse_tiles(state);
  parse_units(state);
  parse_cities(state);
  parse_tribes(state);
}

}  // namespace

NativeRoot parse_root_payload(const py::dict& payload, int max_actions) {
  if (!payload.contains("observation")) {
    throw std::invalid_argument("Native root payload must contain an observation.");
  }
  NativeRoot root;
  root.state.root_player_id = read_int(payload, "root_player_id", read_int(payload, "player_id", 0));
  root.state.observation = shallow_copy_dict(py::cast<py::dict>(payload["observation"]));
  root.state.active_player_id = read_int(root.state.observation, "active_player_id", root.state.root_player_id);
  root.state.tick = read_int(root.state.observation, "tick", 0);
  root.state.terminal = read_bool(payload, "is_terminal", false) || read_bool(payload, "terminal", false);
  root.state.winner_id = read_int(payload, "winner_id", read_int(payload, "winner", -1));
  if (payload.contains("normalized_terminal_reward") || payload.contains("term")) {
    root.state.terminal_value_known = true;
    try {
      root.state.terminal_value = payload.contains("normalized_terminal_reward")
          ? py::cast<double>(payload["normalized_terminal_reward"])
          : py::cast<double>(payload["term"]);
    } catch (const py::cast_error&) {
      root.state.terminal_value_known = false;
      root.state.terminal_value = 0.0;
    }
  }
  parse_observation_state(root.state);
  if (root.state.terminal) {
    root.state.terminal_reason = "root_terminal";
  }

  py::list action_list = payload.contains("actions") && py::isinstance<py::list>(payload["actions"])
      ? py::reinterpret_borrow<py::list>(payload["actions"])
      : py::list();
  const int action_count = max_actions < 0
      ? static_cast<int>(py::len(action_list))
      : std::min<int>(py::len(action_list), std::max(0, max_actions));
  root.actions.reserve(action_count);
  root.state.legal_action_indexes.reserve(action_count);
  for (int i = 0; i < action_count; ++i) {
    if (!py::isinstance<py::dict>(action_list[i])) {
      throw std::runtime_error("Native strict payload parse failure: action entry must be an object.");
    }
    py::dict action_payload = shallow_copy_dict(py::cast<py::dict>(action_list[i]));
    require_key(action_payload, "action", {"type", "t"});
    NativeAction action;
    action.id = read_string(action_payload, "id");
    if (action.id.empty()) {
      int serialized_index = read_int(action_payload, "i", static_cast<int>(root.actions.size()));
      action.id = "A" + std::to_string(serialized_index);
      action_payload["id"] = action.id;
    }
    action.type = read_string(action_payload, "type");
    if (action.type.empty()) {
      action.type = read_string(action_payload, "t");
      if (!action.type.empty()) {
        action_payload["type"] = action.type;
      }
    }
    action.unit_id = read_int(action_payload, "unit_id", 0);
    if (action.unit_id == 0) {
      action.unit_id = read_int(action_payload, "u", 0);
      if (action.unit_id != 0) {
        action_payload["unit_id"] = action.unit_id;
      }
    }
    action.city_id = read_int(action_payload, "city_id", 0);
    if (action.city_id == 0) {
      action.city_id = read_int(action_payload, "c", 0);
      if (action.city_id != 0) {
        action_payload["city_id"] = action.city_id;
      }
    }
    action.payload = action_payload;
    root.state.legal_action_indexes.push_back(static_cast<int>(root.actions.size()));
    root.actions.push_back(action);
  }
  root.state.terminal = root.state.terminal || root.state.legal_action_indexes.empty();
  if (root.state.terminal && root.state.terminal_reason.empty()) {
    root.state.terminal_reason = "no_legal_actions";
  }
  return root;
}

NativeGameState apply_action_strict(
    const NativeGameState& state,
    std::vector<NativeAction>& actions,
    int global_action_index,
    int max_actions) {
  NativeGameState next = state;
  next.observation = deepish_copy_observation(state.observation);
  next.terminal_reason.clear();
  next.transition_kind.clear();
  next.terminal_value_known = false;
  next.terminal_value = 0.0;
  next.winner_id = -1;

  if (global_action_index < 0 || global_action_index >= static_cast<int>(actions.size())) {
    throw_transition_error("invalid_action_index:" + std::to_string(global_action_index));
  }

  const NativeAction& applied = actions[global_action_index];
  const std::string type = canonical_action_type(applied);
  if (type == "END_TURN") {
    apply_end_turn_transition(next);
    evaluate_capital_terminal(next);
    if (next.terminal) {
      return next;
    }
    regenerate_actions(next, actions, max_actions);
    next.terminal = next.legal_action_indexes.empty();
    if (next.terminal) {
      next.terminal_reason = "no_regenerated_actions";
    }
    return next;
  }

  bool applied_ok = false;
  if (type == "MOVE" || type == "STEP_MOVE") {
    applied_ok = apply_move(next, applied);
  } else if (type == "ATTACK") {
    applied_ok = apply_attack(next, applied);
  } else if (type == "CONVERT") {
    applied_ok = apply_convert(next, applied);
  } else if (type == "RECOVER") {
    applied_ok = apply_recover(next, applied);
  } else if (type == "SPAWN") {
    applied_ok = apply_spawn(next, applied);
  } else if (type == "CAPTURE") {
    applied_ok = apply_capture(next, applied);
  } else if (type == "BUILD_ROAD") {
    applied_ok = apply_build_road(next, applied);
  } else if (type == "RESOURCE_GATHERING") {
    applied_ok = apply_resource_gathering(next, applied);
  } else if (type == "RESEARCH_TECH") {
    applied_ok = apply_research(next, applied);
  } else if (type == "BUILD") {
    applied_ok = apply_build(next, applied);
  } else if (type == "LEVEL_UP") {
    applied_ok = apply_level_up(next, applied);
  } else if (type == "BURN_FOREST") {
    applied_ok = apply_clear_or_burn_forest(next, applied, true);
  } else if (type == "CLEAR_FOREST") {
    applied_ok = apply_clear_or_burn_forest(next, applied, false);
  } else if (type == "GROW_FOREST") {
    applied_ok = apply_grow_forest(next, applied);
  } else if (type == "DESTROY") {
    applied_ok = apply_destroy(next, applied);
  } else if (type == "DISBAND") {
    applied_ok = apply_disband(next, applied);
  } else if (type == "MAKE_VETERAN") {
    applied_ok = apply_make_veteran(next, applied);
  } else if (type == "UPGRADE_RAMMER") {
    applied_ok = apply_upgrade_unit(next, applied, "RAMMER");
  } else if (type == "UPGRADE_SCOUT") {
    applied_ok = apply_upgrade_unit(next, applied, "SCOUT");
  } else if (type == "UPGRADE_BOMBER") {
    applied_ok = apply_upgrade_unit(next, applied, "BOMBER");
  }

  if (!applied_ok) {
    throw_transition_error("unsupported_or_failed_transition:" + type);
  }

  if (next.terminal) {
    return next;
  }
  regenerate_actions(next, actions, max_actions);
  next.terminal = next.legal_action_indexes.empty();
  if (next.terminal) {
    next.terminal_reason = "no_regenerated_actions";
  }
  return next;
}

py::dict serialize_evaluation_payload(
    const NativeGameState& state,
    const std::vector<NativeAction>& actions) {
  py::dict payload;
  payload["player_id"] = state.active_player_id;
  payload["root_player_id"] = state.root_player_id;
  payload["active_player_id"] = state.active_player_id;
  payload["observation"] = state.observation;
  py::list legal_actions;
  for (int index : state.legal_action_indexes) {
    if (index >= 0 && index < static_cast<int>(actions.size())) {
      legal_actions.append(actions[index].payload);
    }
  }
  payload["actions"] = legal_actions;
  payload["is_terminal"] = state.terminal;
  payload["native_terminal_reason"] = state.terminal_reason;
  if (!state.transition_kind.empty()) {
    payload["native_transition_kind"] = state.transition_kind;
  }
  if (state.winner_id >= 0) {
    payload["winner_id"] = state.winner_id;
  }
  if (state.terminal_value_known) {
    payload["normalized_terminal_reward"] = state.terminal_value;
    payload["terminal_value"] = state.terminal_value;
  }
  return payload;
}

double value_to_root_perspective(double active_player_value, int root_player_id, int leaf_active_player_id) {
  return root_player_id == leaf_active_player_id ? active_player_value : -active_player_value;
}

}  // namespace tribes::native
