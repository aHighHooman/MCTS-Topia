#include "native_rules.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

namespace tribes::native {
namespace {
void sync_observation_ranking(NativeGameState& state);

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

double read_double(const py::handle& object, const char* key, double fallback) {
  if (!py::isinstance<py::dict>(object)) {
    return fallback;
  }
  py::dict dict = py::reinterpret_borrow<py::dict>(object);
  if (!dict.contains(key) || dict[py::str(key)].is_none()) {
    return fallback;
  }
  try {
    return py::cast<double>(dict[py::str(key)]);
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

void require_present_key(const py::dict& dict, const char* context, std::initializer_list<const char*> keys) {
  for (const char* key : keys) {
    if (dict.contains(py::str(key))) {
      return;
    }
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

std::vector<int> read_int_list(const py::handle& object, const char* key) {
  std::vector<int> out;
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
    out.push_back(py::cast<int>(value));
  }
  return out;
}

std::map<std::string, std::string> read_string_map(const py::handle& object, const char* key) {
  std::map<std::string, std::string> out;
  if (!py::isinstance<py::dict>(object)) {
    return out;
  }
  py::dict dict = py::reinterpret_borrow<py::dict>(object);
  if (!dict.contains(py::str(key)) || !py::isinstance<py::dict>(dict[py::str(key)])) {
    return out;
  }
  py::dict values = py::reinterpret_borrow<py::dict>(dict[py::str(key)]);
  for (const auto& item : values) {
    out[py::cast<std::string>(py::str(item.first))] = py::cast<std::string>(py::str(item.second));
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
  if (x >= 0 && y >= 0 && x < state.board_size && y < state.board_size) {
    int index = y * state.board_size + x;
    if (index >= 0 && index < static_cast<int>(state.tiles.size())) {
      NativeTile& tile = state.tiles[index];
      if (tile.x == x && tile.y == y) {
        return &tile;
      }
    }
    index = x * state.board_size + y;
    if (index >= 0 && index < static_cast<int>(state.tiles.size())) {
      NativeTile& tile = state.tiles[index];
      if (tile.x == x && tile.y == y) {
        return &tile;
      }
    }
  }
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
  if (state.observation.contains("cities") && py::isinstance<py::list>(state.observation["cities"])) {
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
      break;
    }
  }
  if (state.observation.contains("capitals") && py::isinstance<py::list>(state.observation["capitals"])) {
    py::list capitals = py::reinterpret_borrow<py::list>(state.observation["capitals"]);
    for (const auto& item : capitals) {
      if (!py::isinstance<py::dict>(item)) {
        continue;
      }
      py::dict city = py::reinterpret_borrow<py::dict>(item);
      if (read_int(city, "id", 0) != city_id) {
        continue;
      }
      if (compact != nullptr && city.contains(compact)) {
        city[compact] = value;
      } else if (compact == nullptr && city.contains(normalized)) {
        city[normalized] = value;
      }
      return;
    }
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
    py::handle actual_value = value;
    if (tribe_id != state.root_player_id) {
      if (strcmp(key, "stars") == 0 || strcmp(key, "star") == 0) {
        actual_value = py::int_(0);
      } else if (strcmp(key, "kills") == 0) {
        actual_value = py::int_(0);
      } else if (strcmp(key, "pacifist_count") == 0 || strcmp(key, "pacifist") == 0) {
        actual_value = py::int_(0);
      }
    }
    tribe[key] = actual_value;
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
  const int visible_city_id = tile.city_id > 0 ? tile.city_id : (tile.explored ? -1 : 0);
  set_matrix_cell(board, "city", tile.x, tile.y, py::int_(visible_city_id));
  set_matrix_cell(board, "unit", tile.x, tile.y, py::int_(tile.unit_id));
  set_matrix_cell(board, "road", tile.x, tile.y, py::int_(tile.road ? 1 : 0));
  set_matrix_cell(board, "exp", tile.x, tile.y, py::int_(tile.explored ? 1 : 0));
  if (board.contains("tiles") && py::isinstance<py::list>(board["tiles"])) {
    py::list rows = py::reinterpret_borrow<py::list>(board["tiles"]);
    if (tile.y >= 0 && tile.y < static_cast<int>(py::len(rows)) && py::isinstance<py::list>(rows[tile.y])) {
      py::list row = py::reinterpret_borrow<py::list>(rows[tile.y]);
      if (tile.x >= 0 && tile.x < static_cast<int>(py::len(row)) && py::isinstance<py::dict>(row[tile.x])) {
        py::dict out = py::reinterpret_borrow<py::dict>(row[tile.x]);
        out["terrain"] = terrain_value;
        out["resource"] = resource_value;
        out["building"] = building_value;
        out["city_id"] = visible_city_id;
        out["unit_id"] = tile.unit_id;
        out["road"] = tile.road;
        out["explored"] = tile.explored;
        if (out.contains("visible")) {
          out.attr("pop")("visible");
        }
      }
    }
  }
}

void remove_payload_int_from_list(py::dict owner, const char* key, int value) {
  if (!owner.contains(key) || !py::isinstance<py::list>(owner[py::str(key)])) {
    return;
  }
  py::list input = py::reinterpret_borrow<py::list>(owner[py::str(key)]);
  py::list output;
  for (const auto& item : input) {
    try {
      if (py::cast<int>(item) == value) {
        continue;
      }
    } catch (const py::cast_error&) {
    }
    output.append(item);
  }
  owner[key] = output;
}

void remove_unit_ownership_payload(NativeGameState& state, const NativeUnit& unit) {
  if (state.observation.contains("units") && py::isinstance<py::list>(state.observation["units"])) {
    py::list input = py::reinterpret_borrow<py::list>(state.observation["units"]);
    py::list output;
    for (const auto& item : input) {
      if (py::isinstance<py::dict>(item) &&
          read_int(py::reinterpret_borrow<py::dict>(item), "id", -1) == unit.id) {
        continue;
      }
      output.append(item);
    }
    state.observation["units"] = output;
  }
  if (state.observation.contains("tribes") && py::isinstance<py::list>(state.observation["tribes"])) {
    py::list tribes = py::reinterpret_borrow<py::list>(state.observation["tribes"]);
    for (const auto& item : tribes) {
      if (!py::isinstance<py::dict>(item)) {
        continue;
      }
      py::dict tribe = py::reinterpret_borrow<py::dict>(item);
      if (read_int(tribe, "id", -1) == unit.tribe_id) {
        remove_payload_int_from_list(tribe, "extra", unit.id);
        remove_payload_int_from_list(tribe, "extra_unit_ids", unit.id);
      }
    }
  }
  if (state.observation.contains("cities") && py::isinstance<py::list>(state.observation["cities"])) {
    py::list cities = py::reinterpret_borrow<py::list>(state.observation["cities"]);
    for (const auto& item : cities) {
      if (!py::isinstance<py::dict>(item)) {
        continue;
      }
      py::dict city = py::reinterpret_borrow<py::dict>(item);
      remove_payload_int_from_list(city, "units", unit.id);
      remove_payload_int_from_list(city, "unit_ids", unit.id);
    }
  }
}

void remove_unit_from_owner_lists_payload(NativeGameState& state, const NativeUnit& unit) {
  if (state.observation.contains("tribes") && py::isinstance<py::list>(state.observation["tribes"])) {
    py::list tribes = py::reinterpret_borrow<py::list>(state.observation["tribes"]);
    for (const auto& item : tribes) {
      if (!py::isinstance<py::dict>(item)) {
        continue;
      }
      py::dict tribe = py::reinterpret_borrow<py::dict>(item);
      if (read_int(tribe, "id", -1) == unit.tribe_id) {
        remove_payload_int_from_list(tribe, "extra", unit.id);
        remove_payload_int_from_list(tribe, "extra_unit_ids", unit.id);
      }
    }
  }
  if (state.observation.contains("cities") && py::isinstance<py::list>(state.observation["cities"])) {
    py::list cities = py::reinterpret_borrow<py::list>(state.observation["cities"]);
    for (const auto& item : cities) {
      if (!py::isinstance<py::dict>(item)) {
        continue;
      }
      py::dict city = py::reinterpret_borrow<py::dict>(item);
      remove_payload_int_from_list(city, "units", unit.id);
      remove_payload_int_from_list(city, "unit_ids", unit.id);
    }
  }
}

void append_payload_int_to_list(py::dict owner, const char* key, int value) {
  py::list values = owner.contains(key) && py::isinstance<py::list>(owner[py::str(key)])
      ? py::reinterpret_borrow<py::list>(owner[py::str(key)])
      : py::list();
  values.append(value);
  owner[key] = values;
}

void append_extra_unit_payload(NativeGameState& state, int tribe_id, int unit_id) {
  if (!state.observation.contains("tribes") || !py::isinstance<py::list>(state.observation["tribes"])) {
    return;
  }
  py::list tribes = py::reinterpret_borrow<py::list>(state.observation["tribes"]);
  for (const auto& item : tribes) {
    if (!py::isinstance<py::dict>(item)) {
      continue;
    }
    py::dict tribe = py::reinterpret_borrow<py::dict>(item);
    if (read_int(tribe, "id", -1) != tribe_id) {
      continue;
    }
    append_payload_int_to_list(tribe, "extra", unit_id);
    if (tribe.contains("extra_unit_ids")) {
      py::list copy;
      py::list extra = py::reinterpret_borrow<py::list>(tribe["extra"]);
      for (const auto& value : extra) {
        copy.append(value);
      }
      tribe["extra_unit_ids"] = copy;
    }
    return;
  }
}

void move_unit_payload_before_first_owner(NativeGameState& state, int unit_id, int owner_id) {
  if (!state.observation.contains("units") || !py::isinstance<py::list>(state.observation["units"])) {
    return;
  }
  py::list input = py::reinterpret_borrow<py::list>(state.observation["units"]);
  py::object moved = py::none();
  py::list without;
  for (const auto& item : input) {
    if (py::isinstance<py::dict>(item) &&
        read_int(py::reinterpret_borrow<py::dict>(item), "id", -1) == unit_id) {
      moved = py::reinterpret_borrow<py::object>(item);
      continue;
    }
    without.append(item);
  }
  if (moved.is_none()) {
    return;
  }
  py::list output;
  bool inserted = false;
  for (const auto& item : without) {
    if (!inserted && py::isinstance<py::dict>(item) &&
        read_int(py::reinterpret_borrow<py::dict>(item), "tribe_id",
                 read_int(py::reinterpret_borrow<py::dict>(item), "p", -1)) == owner_id) {
      output.append(moved);
      inserted = true;
    }
    output.append(item);
  }
  if (!inserted) {
    output.append(moved);
  }
  state.observation["units"] = output;
}

void move_unit_state_before_first_owner(NativeGameState& state, int unit_id, int owner_id) {
  auto moved_it = std::find_if(state.units.begin(), state.units.end(), [&](const NativeUnit& unit) {
    return unit.id == unit_id;
  });
  if (moved_it == state.units.end()) {
    return;
  }
  NativeUnit moved = *moved_it;
  state.units.erase(moved_it);
  auto insert_it = std::find_if(state.units.begin(), state.units.end(), [&](const NativeUnit& unit) {
    return unit.tribe_id == owner_id;
  });
  state.units.insert(insert_it, moved);
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
      {"DAGGER", 10}, {"RAMMER", 0}, {"SCOUT", 40}, {"BOMBER", 0}, {"SUPERUNIT", 50},
      {"PIRATE", 0}};
  auto it = points.find(type);
  return it == points.end() ? 0 : it->second;
}

double unit_attack(const std::string& type) {
  static const std::map<std::string, double> values = {
      {"WARRIOR", 2}, {"RIDER", 2}, {"DEFENDER", 1}, {"SWORDMAN", 3}, {"SWORDSMAN", 3},
      {"ARCHER", 2}, {"CATAPULT", 4}, {"KNIGHT", 3.5}, {"MIND_BENDER", 0}, {"CLOAK", 2},
      {"DAGGER", 2}, {"RAMMER", 3}, {"SCOUT", 2}, {"BOMBER", 3}, {"SUPERUNIT", 4},
      {"JUGGERNAUT", 4}, {"DINGHY", 2}, {"PIRATE", 2}};
  auto it = values.find(type);
  return it == values.end() ? 2 : it->second;
}

int unit_defence(const std::string& type) {
  static const std::map<std::string, int> values = {
      {"WARRIOR", 2}, {"RIDER", 1}, {"DEFENDER", 3}, {"SWORDMAN", 3}, {"SWORDSMAN", 3},
      {"ARCHER", 1}, {"CATAPULT", 0}, {"KNIGHT", 1}, {"MIND_BENDER", 1}, {"CLOAK", 0},
      {"DAGGER", 1}, {"RAMMER", 3}, {"SCOUT", 1}, {"BOMBER", 2}, {"SUPERUNIT", 3},
      {"JUGGERNAUT", 4}, {"PIRATE", 2}};
  auto it = values.find(type);
  return it == values.end() ? 1 : it->second;
}

int unit_max_hp(const std::string& type) {
  static const std::map<std::string, int> values = {
      {"DEFENDER", 15}, {"SWORDMAN", 15}, {"SWORDSMAN", 15}, {"SUPERUNIT", 40}};
  auto it = values.find(type);
  return it == values.end() ? 10 : it->second;
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
  for (const NativeCity& city : state.cities) {
    max_id = std::max(max_id, city.id);
  }
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
  for (const NativeUnit& unit : state.units) {
    max_id = std::max(max_id, unit.id);
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

bool has_tech(const NativeTribe& tribe, const std::string& tech);
void set_relationship(NativeGameState& state, int from_tribe, int to_tribe, const std::string& relationship);
std::string attacked_status_after(const NativeUnit& unit);
bool city_payload_contains(const NativeGameState& state, int city_id);
bool water_unit_type(const std::string& type);

std::string unit_required_tech(const std::string& type) {
  static const std::map<std::string, std::string> requirements = {
      {"RIDER", "RIDING"}, {"DEFENDER", "STRATEGY"}, {"SWORDMAN", "SMITHERY"},
      {"SWORDSMAN", "SMITHERY"}, {"ARCHER", "ARCHERY"}, {"CATAPULT", "MATHEMATICS"},
      {"KNIGHT", "CHIVALRY"}, {"MIND_BENDER", "PHILOSOPHY"}, {"CLOAK", "DIPLOMACY"}};
  auto it = requirements.find(type);
  return it == requirements.end() ? "" : it->second;
}

bool unit_spawnable(const std::string& type) {
  return type == "WARRIOR" || type == "RIDER" || type == "DEFENDER" || type == "SWORDMAN" ||
      type == "ARCHER" || type == "CATAPULT" || type == "KNIGHT" || type == "MIND_BENDER" ||
      type == "CLOAK";
}

bool unit_unlocked(const NativeTribe& tribe, const std::string& type) {
  const std::string required = unit_required_tech(type);
  return required.empty() || has_tech(tribe, required);
}

int tech_order(const std::string& tech);
std::string tech_id(const std::string& tech);

void append_researched_tech_payload(NativeGameState& state, int tribe_id, const std::string& tech) {
  if (tribe_id != state.root_player_id) {
    return;
  }
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
    NativeTribe* native_tribe = tribe_by_id(state, tribe_id);
    std::vector<std::string> sorted = native_tribe == nullptr
        ? std::vector<std::string>{tech}
        : native_tribe->researched_tech_ids;
    std::sort(sorted.begin(), sorted.end(), [](const std::string& left, const std::string& right) {
      return tech_order(left) < tech_order(right);
    });
    py::list techs;
    for (const std::string& value : sorted) {
      techs.append(py::str(tech_id(value)));
    }
    if (tribe.contains("researched_tech_ids")) {
      tribe["researched_tech_ids"] = techs;
    }
    if (tribe.contains("tech")) {
      py::list compact_techs;
      for (const std::string& value : sorted) {
        compact_techs.append(py::str(tech_id(value)));
      }
      tribe["tech"] = compact_techs;
    }
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

void ingest_cities_from_board(NativeGameState& state) {
  std::set<int> known_ids;
  for (const NativeCity& city : state.cities) {
    known_ids.insert(city.id);
  }
  for (const NativeTile& tile : state.tiles) {
    if (tile.city_id <= 0 || known_ids.count(tile.city_id) > 0) {
      continue;
    }
    NativeCity city;
    city.id = tile.city_id;
    city.x = tile.x;
    city.y = tile.y;
    for (const NativeTile& candidate : state.tiles) {
      if (candidate.city_id == city.id &&
          (candidate.terrain == "CITY" || candidate.terrain == "VILLAGE")) {
        city.x = candidate.x;
        city.y = candidate.y;
        break;
      }
    }
    city.level = 1;
    city.population = 0;
    city.population_need = 2;
    city.production = 2;
    city.tribe_id = -1;
    for (const NativeTribe& tribe : state.tribes) {
      if (tribe.capital_id == city.id) {
        city.tribe_id = tribe.id;
        city.capital = true;
        break;
      }
    }
    state.cities.push_back(city);
    known_ids.insert(city.id);
  }
}

void refine_city_center(NativeCity& city, const NativeGameState& state) {
  int best_score = -1;
  for (const NativeTile& tile : state.tiles) {
    if (tile.city_id != city.id) {
      continue;
    }
    int score = tile.x + tile.y;
    if (tile.terrain == "CITY" || tile.terrain == "VILLAGE") {
      score += 1000;
    }
    if (score > best_score) {
      best_score = score;
      city.x = tile.x;
      city.y = tile.y;
    }
  }
}

void merge_capital_cities(NativeGameState& state) {
  if (!state.observation.contains("capitals") || !py::isinstance<py::list>(state.observation["capitals"])) {
    return;
  }
  py::list capitals = py::reinterpret_borrow<py::list>(state.observation["capitals"]);
  for (const auto& capital_handle : capitals) {
    if (!py::isinstance<py::dict>(capital_handle)) {
      continue;
    }
    py::dict payload = py::reinterpret_borrow<py::dict>(capital_handle);
    const int city_id = read_int(payload, "id", 0);
    if (city_id <= 0) {
      continue;
    }
    NativeCity* existing = city_by_id(state, city_id);
    if (existing != nullptr) {
      existing->x = read_int(payload, "x", existing->x);
      existing->y = read_int(payload, "y", existing->y);
      existing->tribe_id = read_int(payload, "tribe_id", read_int(payload, "p", existing->tribe_id));
      existing->level = read_int(payload, "level", read_int(payload, "lvl", existing->level));
      existing->population = read_int(payload, "population", read_int(payload, "pop", existing->population));
      existing->population_need = read_int(payload, "population_need", read_int(payload, "need", existing->population_need));
      existing->production = read_int(payload, "production", read_int(payload, "prod", existing->production));
      existing->capital = true;
      continue;
    }
    NativeCity city;
    city.id = city_id;
    city.tribe_id = read_int(payload, "tribe_id", read_int(payload, "p", -1));
    city.x = read_int(payload, "x", 0);
    city.y = read_int(payload, "y", 0);
    city.level = read_int(payload, "level", read_int(payload, "lvl", 1));
    city.population = read_int(payload, "population", read_int(payload, "pop", 0));
    city.population_need = read_int(payload, "population_need", read_int(payload, "need", 2));
    city.production = read_int(payload, "production", read_int(payload, "prod", 2));
    city.capital = true;
    state.cities.push_back(city);
  }
}

void assign_city_owners_from_capitals(NativeGameState& state) {
  for (NativeCity& city : state.cities) {
    if (city.tribe_id >= 0) {
      continue;
    }
    for (const NativeTribe& tribe : state.tribes) {
      if (tribe.capital_id == city.id) {
        city.tribe_id = tribe.id;
        city.capital = true;
        break;
      }
    }
  }
}

void sync_tribe_city_ids(NativeGameState& state) {
  for (NativeTribe& tribe : state.tribes) {
    if (!tribe.city_ids.empty()) {
      continue;
    }
    for (const NativeCity& city : state.cities) {
      if (city.tribe_id == tribe.id) {
        tribe.city_ids.push_back(city.id);
      }
    }
  }
}

void capture_capital_city_ids(NativeGameState& state) {
  if (!state.capital_city_ids.empty()) {
    return;
  }
  for (const NativeCity& city : state.cities) {
    if (city.capital) {
      state.capital_city_ids.push_back(city.id);
    }
  }
}

void evaluate_capital_terminal(NativeGameState& state) {
  if (!uses_capital_objective(state) || state.capital_city_ids.empty()) {
    return;
  }
  int live_tribes = 0;
  for (const NativeTribe& tribe : state.tribes) {
    if (tribe.result != "LOSS") {
      live_tribes += 1;
    }
  }
  if (live_tribes > 1 && static_cast<int>(state.capital_city_ids.size()) < live_tribes) {
    return;
  }
  for (int capital_city_id : state.capital_city_ids) {
    if (city_by_id(const_cast<NativeGameState&>(state), capital_city_id) == nullptr) {
      return;
    }
  }
  std::set<int> candidate_tribes;
  for (const NativeTribe& tribe : state.tribes) {
    if (tribe.result != "LOSS") {
      candidate_tribes.insert(tribe.id);
    }
  }
  for (int tribe_id : candidate_tribes) {
    bool controls_all = true;
    for (int capital_city_id : state.capital_city_ids) {
      const NativeCity* capital_city = city_by_id(const_cast<NativeGameState&>(state), capital_city_id);
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
  bool disembarked = false;
  if (water_unit_type(unit->type) &&
      (to->terrain == "PLAIN" || to->terrain == "FOREST" || to->terrain == "MOUNTAIN" ||
       to->terrain == "CITY" || to->terrain == "VILLAGE")) {
    const int old_unit_id = unit->id;
    const int new_unit_id = next_unit_id(next);
    NativeUnit old_unit_ref = *unit;
    old_unit_ref.id = old_unit_id;
    unit->id = new_unit_id;
    unit->type = "WARRIOR";
    unit->attack = unit_attack(unit->type);
    unit->defence = unit_defence(unit->type);
    unit->movement = 1;
    unit->range = 1;
    unit->cost = unit_cost(unit->type);
    to->unit_id = new_unit_id;
    sync_tile_to_payload(next, *to);
    set_unit_payload_field(next, old_unit_id, "id", nullptr, py::int_(new_unit_id));
    set_unit_payload_field(next, new_unit_id, "type", "t", py::str(unit->type));
    set_unit_payload_field(next, new_unit_id, "attack", "atk", py::float_(unit->attack));
    set_unit_payload_field(next, new_unit_id, "defence", "def", py::float_(unit->defence));
    set_unit_payload_field(next, new_unit_id, "movement", "mov", py::int_(unit->movement));
    set_unit_payload_field(next, new_unit_id, "range", "r", py::int_(unit->range));
    set_unit_payload_field(next, new_unit_id, "cost", nullptr, py::int_(unit->cost));
    set_unit_payload_field(next, new_unit_id, "max_hp", "mhp", py::int_(unit->max_hp));
    set_unit_payload_field(next, new_unit_id, "current_hp", "hp", py::int_(unit->current_hp));
    set_unit_payload_field(next, new_unit_id, "current_hp_exact", "hpx", py::float_(unit->current_hp_exact));
    remove_unit_from_owner_lists_payload(next, old_unit_ref);
    append_extra_unit_payload(next, unit->tribe_id, new_unit_id);
    disembarked = true;
  }
  if (disembarked) {
    unit->status = "FINISHED";
  } else if (unit->type == "MIND_BENDER" || unit->type == "CATAPULT" || unit->type == "DEFENDER" ||
      unit->type == "RAFT" || unit->type == "BOMBER" || unit->type == "JUGGERNAUT" ||
      unit->type == "SUPERUNIT") {
    unit->status = "FINISHED";
  } else if (unit->type == "RIDER" && (unit->status == "ATTACKED" || unit->status == "MOVED_AND_ATTACKED")) {
    unit->status = unit->status == "ATTACKED" ? "MOVED_AND_ATTACKED" : "FINISHED";
  } else {
    unit->status = "MOVED";
  }
  const int payload_unit_id = unit->id;
  set_unit_payload_field(next, payload_unit_id, "x", nullptr, py::int_(x));
  set_unit_payload_field(next, payload_unit_id, "y", nullptr, py::int_(y));
  set_unit_payload_field(next, payload_unit_id, "status", "s", py::str(unit->status));
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
    new_city.bound = 1;
    new_city.points_worth = 120;
    new_city.capital = false;
    new_city.walls = false;
    next.cities.push_back(new_city);
    for (NativeTile& candidate : next.tiles) {
      if (candidate.city_id <= 0 &&
          std::max(std::abs(candidate.x - new_city.x), std::abs(candidate.y - new_city.y)) <= 1) {
        candidate.city_id = new_city.id;
      }
    }
    NativeTribe* tribe = tribe_by_id(next, unit->tribe_id);
    if (tribe != nullptr) {
      tribe->city_ids.push_back(new_city.id);
      if (next.observation.contains("tribes") && py::isinstance<py::list>(next.observation["tribes"])) {
        py::list tribes = py::reinterpret_borrow<py::list>(next.observation["tribes"]);
        for (const auto& item : tribes) {
          if (py::isinstance<py::dict>(item)) {
            py::dict tribe_payload = py::reinterpret_borrow<py::dict>(item);
            if (read_int(tribe_payload, "id", -1) == unit->tribe_id) {
              append_payload_int_to_list(tribe_payload, "cities", new_city.id);
              break;
            }
          }
        }
      }
    }
    tile->terrain = "CITY";
    tile->city_id = new_city.id;
    tile->road = true;
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
      out["bound"] = new_city.bound;
      out["points_worth"] = new_city.points_worth;
      out["pts"] = new_city.points_worth;
      out["buildings"] = py::list();
      out["b"] = py::list();
      out["units"] = py::list();
      out["unit_ids"] = py::list();
      out["is_capital"] = false;
      out["cap"] = false;
      out["has_walls"] = false;
      out["wall"] = false;
      out["inf"] = false;
      py::reinterpret_borrow<py::list>(next.observation["cities"]).append(out);
    }
    unit->status = "FINISHED";
    set_unit_payload_field(next, unit->id, "status", "s", py::str("FINISHED"));
    update_tribe_economy(next, unit->tribe_id, 0, 220);
    return true;
  }
  if (city == nullptr) {
    return false;
  }
  if (city->tribe_id == unit->tribe_id) {
    return false;
  }
  const int old_city_owner = city->tribe_id;
  city->tribe_id = unit->tribe_id;
  unit->status = "FINISHED";
  set_relationship(next, unit->tribe_id, old_city_owner, "WAR");
  set_relationship(next, old_city_owner, unit->tribe_id, "WAR");
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
  set_relationship(next, attacker->tribe_id, target->tribe_id, "WAR");
  set_relationship(next, target->tribe_id, attacker->tribe_id, "WAR");
  const double attacker_hp = attacker->current_hp_exact > 0.0 ? attacker->current_hp_exact : static_cast<double>(attacker->current_hp);
  const double target_hp = target->current_hp_exact > 0.0 ? target->current_hp_exact : static_cast<double>(target->current_hp);
  const double attacker_attack = attacker->attack > 0.0 ? attacker->attack : unit_attack(attacker->type);
  const double target_defence = target->defence > 0.0 ? target->defence : unit_defence(target->type);
  const double attack_force = attacker_attack * (attacker_hp / std::max(1, attacker->max_hp));
  const double defence_force = target_defence * (target_hp / std::max(1, target->max_hp));
  const double total_damage = attack_force + defence_force;
  const int attack_damage = total_damage <= 0.0
      ? 0
      : static_cast<int>(std::round((attack_force / total_damage) * attacker_attack * 4.5));
  const int defence_damage = total_damage <= 0.0
      ? 0
      : static_cast<int>(std::round((defence_force / total_damage) * target_defence * 4.5));
  target->current_hp = std::max(0, target->current_hp - attack_damage);
  target->current_hp_exact = static_cast<double>(target->current_hp);
  set_unit_payload_field(next, target->id, "current_hp", "hp", py::int_(target->current_hp));
  set_unit_payload_field(next, target->id, "current_hp_exact", "hpx", py::float_(target->current_hp_exact));
  if (target->current_hp <= 0) {
    remove_unit_ownership_payload(next, *target);
    mark_unit_removed(next, *target);
    attacker->kills += 1;
    set_unit_payload_field(next, attacker->id, "kills", "k", py::int_(attacker->kills));
  } else {
    const int distance = std::max(std::abs(attacker->x - target->x), std::abs(attacker->y - target->y));
    const bool stiff = target->type == "RAFT" || target->type == "BOMBER" || target->type == "CATAPULT" ||
        target->type == "MIND_BENDER" || target->type == "CLOAK" || target->type == "DINGHY" ||
        target->type == "JUGGERNAUT";
    if (distance <= std::max(1, target->range) && target_defence > 0.0 && !stiff &&
        attacker->type != "DAGGER" && attacker->type != "PIRATE" && defence_damage > 0) {
      attacker->current_hp = std::max(0, attacker->current_hp - defence_damage);
      attacker->current_hp_exact = static_cast<double>(attacker->current_hp);
      set_unit_payload_field(next, attacker->id, "current_hp", "hp", py::int_(attacker->current_hp));
      set_unit_payload_field(next, attacker->id, "current_hp_exact", "hpx", py::float_(attacker->current_hp_exact));
      if (attacker->current_hp <= 0) {
        update_tribe_economy(next, attacker->tribe_id, 0, -unit_points(attacker->type));
        remove_unit_ownership_payload(next, *attacker);
        mark_unit_removed(next, *attacker);
      }
    }
  }
  if (attacker->current_hp > 0) {
    attacker->status = attacked_status_after(*attacker);
    set_unit_payload_field(next, attacker->id, "status", "s", py::str(attacker->status));
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
  const int old_owner = target->tribe_id;
  remove_unit_from_owner_lists_payload(next, *target);
  target->tribe_id = unit->tribe_id;
  unit->status = "FINISHED";
  target->status = "FINISHED";
  target->city_id = -1;
  append_extra_unit_payload(next, unit->tribe_id, target->id);
  update_tribe_economy(next, unit->tribe_id, 0, target->type == "WARRIOR" ? 15 : unit_points(target->type));
  set_relationship(next, unit->tribe_id, old_owner, "WAR");
  set_relationship(next, old_owner, unit->tribe_id, "WAR");
  set_unit_payload_field(next, target->id, "tribe_id", "p", py::int_(target->tribe_id));
  set_unit_payload_field(next, target->id, "city_id", "c", py::int_(-1));
  set_unit_payload_field(next, target->id, "status", "s", py::str("FINISHED"));
  set_unit_payload_field(next, unit->id, "status", "s", py::str("FINISHED"));
  move_unit_payload_before_first_owner(next, target->id, old_owner);
  move_unit_state_before_first_owner(next, target->id, old_owner);
  return true;
}

bool has_tech(const NativeTribe& tribe, const std::string& tech) {
  auto normalize = [](std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
      return static_cast<char>(std::toupper(c));
    });
    return value;
  };
  const std::string target = normalize(tech);
  for (const std::string& researched : tribe.researched_tech_ids) {
    if (normalize(researched) == target) {
      return true;
    }
  }
  return false;
}

int tech_order(const std::string& tech) {
  static const std::vector<std::string> ordered = {
      "CLIMBING", "FISHING", "HUNTING", "ORGANIZATION", "RIDING",
      "ARCHERY", "FARMING", "FORESTRY", "FREE_SPIRIT", "MEDITATION",
      "MINING", "ROADS", "RAMMING", "SAILING", "STRATEGY",
      "AQUATISM", "CHIVALRY", "CONSTRUCTION", "DIPLOMACY", "MATHEMATICS",
      "NAVIGATION", "SMITHERY", "SPIRITUALISM", "TRADE", "PHILOSOPHY"};
  std::string normalized = tech;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  for (int i = 0; i < static_cast<int>(ordered.size()); ++i) {
    if (ordered[i] == normalized) {
      return i;
    }
  }
  return static_cast<int>(ordered.size());
}

std::string tech_id(const std::string& tech) {
  std::string out = tech;
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

int tech_cost_for(const NativeTribe& tribe, const std::string& tech) {
  int tier = tech_tier(tech);
  int num_cities = static_cast<int>(tribe.city_ids.size());
  if (num_cities <= 0) {
    num_cities = 1;
  }
  int cost = 4 + tier * num_cities;
  if (has_tech(tribe, "PHILOSOPHY")) {
    cost = static_cast<int>(std::ceil(static_cast<double>(cost) * (1.0 - (1.0 / 3.0))));
  }
  return cost;
}

std::string parent_tech(const std::string& tech) {
  static const std::map<std::string, std::string> parents = {
      {"ARCHERY", "HUNTING"}, {"FARMING", "ORGANIZATION"}, {"FORESTRY", "HUNTING"},
      {"FREE_SPIRIT", "RIDING"}, {"MEDITATION", "CLIMBING"}, {"MINING", "CLIMBING"},
      {"ROADS", "RIDING"}, {"RAMMING", "FISHING"}, {"SAILING", "FISHING"},
      {"STRATEGY", "ORGANIZATION"}, {"AQUATISM", "RAMMING"}, {"CHIVALRY", "FREE_SPIRIT"},
      {"CONSTRUCTION", "FARMING"}, {"DIPLOMACY", "STRATEGY"}, {"MATHEMATICS", "FORESTRY"},
      {"NAVIGATION", "SAILING"}, {"SMITHERY", "MINING"}, {"SPIRITUALISM", "ARCHERY"},
      {"TRADE", "ROADS"}, {"PHILOSOPHY", "MEDITATION"}};
  auto it = parents.find(tech);
  return it == parents.end() ? "" : it->second;
}

bool has_researched_child(const NativeTribe& tribe, const std::string& tech) {
  static const std::map<std::string, std::vector<std::string>> children = {
      {"HUNTING", {"ARCHERY", "FORESTRY"}}, {"ORGANIZATION", {"FARMING", "STRATEGY"}},
      {"RIDING", {"FREE_SPIRIT", "ROADS"}}, {"CLIMBING", {"MEDITATION", "MINING"}},
      {"FISHING", {"RAMMING", "SAILING"}}, {"RAMMING", {"AQUATISM"}},
      {"FREE_SPIRIT", {"CHIVALRY"}}, {"FARMING", {"CONSTRUCTION"}},
      {"STRATEGY", {"DIPLOMACY"}}, {"FORESTRY", {"MATHEMATICS"}},
      {"SAILING", {"NAVIGATION"}}, {"MINING", {"SMITHERY"}},
      {"ARCHERY", {"SPIRITUALISM"}}, {"ROADS", {"TRADE"}}, {"MEDITATION", {"PHILOSOPHY"}}};
  auto it = children.find(tech);
  if (it == children.end()) {
    return false;
  }
  for (const std::string& child : it->second) {
    if (has_tech(tribe, child)) {
      return true;
    }
  }
  return false;
}

bool can_research(const NativeTribe& tribe, const std::string& tech) {
  if (has_tech(tribe, tech) || tribe.stars < tech_cost_for(tribe, tech)) {
    return false;
  }
  const std::string parent = parent_tech(tech);
  return parent.empty() || has_tech(tribe, parent) || has_researched_child(tribe, tech);
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
  static const std::set<std::string> tribe_payload_actions = {
      "RESEARCH_TECH", "BUILD_ROAD", "BUILD_EMBASSY", "PROPOSE_PEACE", "ACCEPT_PEACE",
      "PROPOSE_TREATY", "ACCEPT_TREATY", "CANCEL_TREATY", "SEND_STARS", "END_TURN"};
  if (tribe_payload_actions.count(action.type) == 0) {
    if (action.payload.contains("p")) {
      action.payload.attr("pop")("p");
    }
    if (action.payload.contains("tribe_id")) {
      action.payload.attr("pop")("tribe_id");
    }
  }
  const int index = static_cast<int>(actions.size());
  action.payload["i"] = index;
  action.payload["t"] = action.type;
  action.payload["type"] = action.type;
  action.payload["id"] = action.id.empty() ? ("A" + std::to_string(index)) : action.id;
  if (action.unit_id > 0) {
    action.payload["unit_id"] = action.unit_id;
    action.payload["u"] = action.unit_id;
  }
  if (action.city_id > 0) {
    action.payload["city_id"] = action.city_id;
    action.payload["c"] = action.city_id;
  }
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

bool unit_is_fresh(const NativeUnit& unit) {
  return unit.status == "FRESH";
}

bool unit_can_move(const NativeUnit& unit) {
  return unit.status == "FRESH" || unit.status == "ATTACKED" || unit.status == "MOVED_AND_ATTACKED";
}

bool unit_can_attack(const NativeUnit& unit) {
  if (unit.status == "FRESH") {
    return true;
  }
  return unit.status == "MOVED" &&
      (unit.type == "WARRIOR" || unit.type == "SWORDMAN" || unit.type == "SWORDSMAN" ||
       unit.type == "ARCHER" || unit.type == "RIDER" || unit.type == "KNIGHT" ||
       unit.type == "CLOAK" || unit.type == "DAGGER" || unit.type == "RAMMER" ||
       unit.type == "SCOUT" || unit.type == "PIRATE" || unit.type == "DINGHY");
}

bool passable_move_target(const NativeGameState& state, const NativeTile& tile) {
  const bool explored_for_actor = tile.explored || state.active_player_id != state.root_player_id;
  return explored_for_actor && tile.unit_id <= 0 && tile.terrain != "DEEP_WATER" && tile.terrain != "WATER";
}

bool water_unit_type(const std::string& type) {
  return type == "RAFT" || type == "RAMMER" || type == "SCOUT" || type == "BOMBER" ||
      type == "JUGGERNAUT" || type == "DINGHY" || type == "PIRATE";
}

std::string attacked_status_after(const NativeUnit& unit) {
  if (unit.type == "RIDER") {
    if (unit.status == "FRESH") return "ATTACKED";
    if (unit.status == "MOVED") return "MOVED_AND_ATTACKED";
  }
  return "FINISHED";
}


bool heal_others_target_exists(const NativeGameState& state, const NativeUnit& healer) {
  if (healer.type != "MIND_BENDER" || !unit_can_attack(healer)) {
    return false;
  }
  const int radius = healer.range;
  for (int x = healer.x - radius; x <= healer.x + radius; ++x) {
    for (int y = healer.y - radius; y <= healer.y + radius; ++y) {
      if ((x == healer.x && y == healer.y) || x < 0 || y < 0 ||
          x >= state.board_size || y >= state.board_size) {
        continue;
      }
      const NativeTile* tile = tile_at(const_cast<NativeGameState&>(state), x, y);
      if (tile == nullptr || tile->unit_id <= 0) {
        continue;
      }
      const NativeUnit* target = unit_by_id(const_cast<NativeGameState&>(state), tile->unit_id);
      if (target != nullptr && target->tribe_id == healer.tribe_id &&
          target->current_hp < target->max_hp) {
        return true;
      }
    }
  }
  return false;
}

bool apply_heal_others(NativeGameState& next, const NativeAction& action) {
  const int unit_id = action_int(action, "unit_id", "u", 0);
  NativeUnit* healer = unit_by_id(next, unit_id);
  if (healer == nullptr || healer->type != "MIND_BENDER" || !unit_can_attack(*healer)) {
    return false;
  }

  constexpr int kMindBenderHeal = 4;
  const int radius = healer->range;
  bool healed_any = false;

  for (int x = healer->x - radius; x <= healer->x + radius; ++x) {
    for (int y = healer->y - radius; y <= healer->y + radius; ++y) {
      if ((x == healer->x && y == healer->y) || x < 0 || y < 0 ||
          x >= next.board_size || y >= next.board_size) {
        continue;
      }
      NativeTile* tile = tile_at(next, x, y);
      if (tile == nullptr || tile->unit_id <= 0) {
        continue;
      }
      NativeUnit* target = unit_by_id(next, tile->unit_id);
      if (target == nullptr || target->tribe_id != healer->tribe_id ||
          target->current_hp >= target->max_hp) {
        continue;
      }
      target->current_hp_exact = std::min<double>(
          static_cast<double>(target->max_hp),
          target->current_hp_exact + static_cast<double>(kMindBenderHeal));
      target->current_hp = static_cast<int>(target->current_hp_exact);
      set_unit_payload_field(next, target->id, "current_hp", "hp", py::int_(target->current_hp));
      set_unit_payload_field(next, target->id, "current_hp_exact", "hpx", py::float_(target->current_hp_exact));
      healed_any = true;
    }
  }

  if (!healed_any) {
    return false;
  }

  healer = unit_by_id(next, unit_id);
  if (healer == nullptr) {
    return false;
  }
  healer->status = attacked_status_after(*healer);
  set_unit_payload_field(next, healer->id, "status", "s", py::str(healer->status));
  return true;
}

void sync_all_tiles_to_payload(NativeGameState& state) {
  for (const NativeTile& tile : state.tiles) {
    sync_tile_to_payload(state, tile);
  }
}

void append_visible_city_payload(NativeGameState& state, const NativeCity& city) {
  if (city_payload_contains(state, city.id) ||
      !state.observation.contains("cities") || !py::isinstance<py::list>(state.observation["cities"])) {
    return;
  }
  py::dict out;
  out["id"] = city.id;
  out["p"] = city.tribe_id;
  out["tribe_id"] = city.tribe_id;
  out["x"] = city.x;
  out["y"] = city.y;
  out["level"] = city.level;
  out["lvl"] = city.level;
  out["population"] = city.population;
  out["pop"] = city.population;
  out["population_need"] = city.population_need;
  out["need"] = city.population_need;
  out["production"] = city.production;
  out["prod"] = city.production;
  out["bound"] = city.bound > 0 ? city.bound : 1;
  const int visible_points = city.points_worth > 0 ? city.points_worth : (city.capital ? 180 : 0);
  out["points_worth"] = visible_points;
  out["pts"] = visible_points;
  out["b"] = py::list();
  out["buildings"] = py::list();
  out["units"] = py::list();
  out["unit_ids"] = py::list();
  out["cap"] = city.capital;
  out["is_capital"] = city.capital;
  out["wall"] = city.walls;
  out["has_walls"] = city.walls;
  out["inf"] = city.infiltrated;
  py::reinterpret_borrow<py::list>(state.observation["cities"]).append(out);
  if (state.observation.contains("tribes") && py::isinstance<py::list>(state.observation["tribes"])) {
    py::list tribes = py::reinterpret_borrow<py::list>(state.observation["tribes"]);
    for (const auto& item : tribes) {
      if (py::isinstance<py::dict>(item)) {
        py::dict tribe = py::reinterpret_borrow<py::dict>(item);
        if (read_int(tribe, "id", -1) == city.tribe_id) {
          append_payload_int_to_list(tribe, "cities", city.id);
          break;
        }
      }
    }
  }
}

void append_visible_unit_payload(NativeGameState& state, const NativeUnit& unit) {
  if (!state.observation.contains("units") || !py::isinstance<py::list>(state.observation["units"])) {
    return;
  }
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
  out["hpx"] = unit.current_hp_exact;
  out["current_hp_exact"] = unit.current_hp_exact;
  out["mhp"] = unit.max_hp;
  out["max_hp"] = unit.max_hp;
  out["k"] = unit.kills;
  out["kills"] = unit.kills;
  out["v"] = unit.veteran;
  out["is_veteran"] = unit.veteran;
  out["s"] = unit.status;
  out["status"] = unit.status;
  out["h"] = unit.hidden;
  out["is_hidden"] = unit.hidden;
  out["atk"] = unit.attack;
  out["attack"] = unit.attack;
  out["def"] = unit.defence;
  out["defence"] = unit.defence;
  out["mov"] = unit.movement;
  out["movement"] = unit.movement;
  out["r"] = unit.range;
  out["range"] = unit.range;
  out["cost"] = unit.cost;
  py::reinterpret_borrow<py::list>(state.observation["units"]).append(out);
}

int reveal_from_current_assets(NativeGameState& state) {
  int newly_explored = 0;
  auto reveal_square = [&](int cx, int cy, int radius) {
    for (NativeTile& tile : state.tiles) {
      if (std::max(std::abs(tile.x - cx), std::abs(tile.y - cy)) <= radius) {
        if (!tile.explored) {
          newly_explored += 1;
        }
        tile.explored = true;
        tile.visible = true;
        if (tile.terrain.empty()) {
          tile.terrain = "PLAIN";
        }
        if (tile.city_id <= 0) {
          for (const NativeCity& city : state.cities) {
            if (std::max(std::abs(tile.x - city.x), std::abs(tile.y - city.y)) <= 1) {
              tile.city_id = city.id;
              if (tile.x == city.x && tile.y == city.y) {
                tile.terrain = "CITY";
                tile.road = true;
                append_visible_city_payload(state, city);
              }
              break;
            }
          }
        }
      }
    }
  };
  for (const NativeCity& city : state.cities) {
    if (city.tribe_id == state.root_player_id) {
      reveal_square(city.x, city.y, 1);
    }
  }
  for (const NativeUnit& unit : state.units) {
    if (unit.tribe_id == state.root_player_id && unit.current_hp != 0) {
      reveal_square(unit.x, unit.y, unit.type == "SCOUT" || unit.type == "CLOAK" ? 2 : 1);
    }
  }
  return newly_explored;
}

void sync_observation_turn_flags(NativeGameState& state);
std::string relationship_between(const NativeGameState& state, int from_tribe, int to_tribe);
bool has_pending_offer(const NativeGameState& state, int from_tribe, int to_tribe);
void set_pending_offer(NativeGameState& state, int from_tribe, int to_tribe, const std::string& offer_type);
void clear_pending_offer(NativeGameState& state, int from_tribe, int to_tribe);
void set_relationship(NativeGameState& state, int from_tribe, int to_tribe, const std::string& relationship);
bool city_can_level_up(const NativeCity& city);
bool city_can_add_unit(const NativeCity& city, const NativeGameState& state);
std::vector<NativeTile*> city_territory_tiles(NativeGameState& state, const NativeCity& city);
bool tile_in_city_territory(const NativeTile& tile, const NativeCity& city);
NativeCity* capital_city_for_tribe(NativeGameState& state, int tribe_id);
bool is_action_unlocked(const NativeTribe& tribe, const std::string& action_type);
bool is_building_unlocked(const NativeTribe& tribe, const std::string& building);
bool building_terrain_ok(const std::string& building, const NativeTile& tile);
bool building_feasible_at(const NativeGameState& state, const NativeTribe& tribe, const NativeCity& city, const NativeTile& tile, const std::string& building);
bool city_has_building_at(const NativeCity& city, int x, int y);
bool resource_tech_ok(const NativeTribe& tribe, const std::string& resource);
int road_cost_at(const NativeTile& tile);
bool can_build_road_at(const NativeGameState& state, int tribe_id, const NativeTile& tile);
bool tribe_met(const NativeTribe& tribe, int other_id);
bool can_build_embassy(const NativeGameState& state, int owner_id, int host_id);
bool city_payload_contains(const NativeGameState& state, int city_id);
int tribe_starting_stars(const NativeTribe& tribe);
void ensure_tribe_init_tech(NativeTribe& tribe);
void regenerate_tribe_actions(NativeGameState& state, std::vector<NativeAction>& actions, int max_actions);
void regenerate_city_actions(NativeGameState& state, std::vector<NativeAction>& actions, int max_actions, const NativeCity& city);

int infiltrate_spawn_priority(const NativeCity& city, const NativeTile& tile, const NativeTribe& tribe) {
  if (tile.x == city.x && tile.y == city.y) {
    return 0;
  }
  const bool gets_defence_bonus =
      (tile.terrain == "FOREST" && has_tech(tribe, "ARCHERY")) ||
      (tile.terrain == "MOUNTAIN" && has_tech(tribe, "CLIMBING")) ||
      ((tile.terrain == "SHALLOW_WATER" || tile.terrain == "DEEP_WATER") && has_tech(tribe, "AQUATISM"));
  return gets_defence_bonus ? 1 : 2;
}

std::string infiltrate_spawn_type(const NativeTile& tile) {
  if (tile.terrain == "SHALLOW_WATER" || tile.terrain == "DEEP_WATER") {
    return "PIRATE";
  }
  return "DAGGER";
}

bool apply_infiltrate(NativeGameState& next, const NativeAction& action) {
  const int unit_id = action_int(action, "unit_id", "u", 0);
  const int target_city_id = action_int(action, "target_city_id", "tc", action_int(action, "city_id", "c", 0));
  NativeUnit* cloak = unit_by_id(next, unit_id);
  NativeCity* target_city = city_by_id(next, target_city_id);
  if (cloak == nullptr || target_city == nullptr ||
      (cloak->type != "CLOAK" && cloak->type != "DINGHY") || !unit_can_attack(*cloak) ||
      target_city->infiltrated || target_city->tribe_id == cloak->tribe_id ||
      relationship_between(next, cloak->tribe_id, target_city->tribe_id) == "TREATY") {
    return false;
  }

  const int distance = std::max(std::abs(cloak->x - target_city->x), std::abs(cloak->y - target_city->y));
  if (distance > std::max(1, cloak->range)) {
    return false;
  }

  NativeTile* city_tile = tile_at(next, target_city->x, target_city->y);
  if (city_tile == nullptr) {
    return false;
  }
  NativeUnit* city_occupant = city_tile->unit_id > 0 ? unit_by_id(next, city_tile->unit_id) : nullptr;
  if (city_occupant != nullptr && city_occupant->tribe_id != target_city->tribe_id) {
    return false;
  }

  const int attacker_id = cloak->tribe_id;
  const int defender_id = target_city->tribe_id;
  const double cloak_attack = cloak->attack > 0.0 ? cloak->attack : unit_attack(cloak->type);
  set_relationship(next, attacker_id, defender_id, "WAR");
  set_relationship(next, defender_id, attacker_id, "WAR");

  if (city_occupant != nullptr) {
    city_occupant->current_hp = std::max(0, city_occupant->current_hp - static_cast<int>(cloak_attack));
    city_occupant->current_hp_exact = static_cast<double>(city_occupant->current_hp);
    set_unit_payload_field(next, city_occupant->id, "current_hp", "hp", py::int_(city_occupant->current_hp));
    set_unit_payload_field(next, city_occupant->id, "current_hp_exact", "hpx", py::float_(city_occupant->current_hp_exact));
    if (city_occupant->current_hp <= 0) {
      remove_unit_ownership_payload(next, *city_occupant);
      mark_unit_removed(next, *city_occupant);
      city_tile->unit_id = 0;
      sync_tile_to_payload(next, *city_tile);
    }
  }

  update_tribe_economy(next, attacker_id, target_city->production, 0);
  target_city->infiltrated = true;
  set_city_payload_field(next, target_city->id, "infiltrated", "inf", py::bool_(true));

  NativeTribe* attacker = tribe_by_id(next, attacker_id);
  if (attacker == nullptr) {
    return false;
  }
  std::vector<NativeTile*> spawn_tiles = city_territory_tiles(next, *target_city);
  std::sort(spawn_tiles.begin(), spawn_tiles.end(), [&](const NativeTile* left, const NativeTile* right) {
    const int left_priority = infiltrate_spawn_priority(*target_city, *left, *attacker);
    const int right_priority = infiltrate_spawn_priority(*target_city, *right, *attacker);
    if (left_priority != right_priority) {
      return left_priority < right_priority;
    }
    if (left->x != right->x) {
      return left->x < right->x;
    }
    return left->y < right->y;
  });

  int spawned = 0;
  const int spawn_limit = std::min(5, std::max(0, target_city->level));
  for (NativeTile* tile : spawn_tiles) {
    if (spawned >= spawn_limit) {
      break;
    }
    if (tile == nullptr || tile->unit_id > 0 || tile->terrain.empty()) {
      continue;
    }
    NativeUnit dagger;
    dagger.id = next_unit_id(next);
    dagger.tribe_id = attacker_id;
    dagger.city_id = target_city->id;
    dagger.x = tile->x;
    dagger.y = tile->y;
    dagger.type = infiltrate_spawn_type(*tile);
    dagger.current_hp = unit_max_hp(dagger.type);
    dagger.current_hp_exact = static_cast<double>(dagger.current_hp);
    dagger.max_hp = dagger.current_hp;
    dagger.kills = 0;
    dagger.veteran = false;
    dagger.hidden = false;
    dagger.attack = unit_attack(dagger.type);
    dagger.defence = unit_defence(dagger.type);
    dagger.movement = dagger.type == "PIRATE" ? 2 : 1;
    dagger.range = 1;
    dagger.cost = unit_cost(dagger.type);
    dagger.status = "FINISHED";
    next.units.push_back(dagger);
    target_city->unit_ids.push_back(dagger.id);
    tile->unit_id = dagger.id;
    sync_tile_to_payload(next, *tile);
    append_visible_unit_payload(next, dagger);
    py::list unit_ids;
    for (int existing_unit_id : target_city->unit_ids) {
      unit_ids.append(existing_unit_id);
    }
    set_city_payload_field(next, target_city->id, "units", nullptr, unit_ids);
    set_city_payload_field(next, target_city->id, "unit_ids", nullptr, unit_ids);
    update_tribe_economy(next, attacker_id, 0, unit_points(dagger.type));
    spawned++;
  }

  NativeUnit* spent_cloak = unit_by_id(next, unit_id);
  if (spent_cloak != nullptr) {
    remove_unit_ownership_payload(next, *spent_cloak);
    mark_unit_removed(next, *spent_cloak);
  }
  return true;
}

void regenerate_unit_actions(NativeGameState& state, std::vector<NativeAction>& actions, int max_actions, const NativeUnit& unit) {
  if (unit.tribe_id != state.active_player_id) {
    return;
  }
  if (unit.current_hp == 0) {
    return;
  }
  NativeTribe* tribe = tribe_by_id(state, state.active_player_id);
  if (tribe == nullptr) {
    return;
  }
  if (unit.type == "RAFT") {
    if (has_tech(*tribe, "RAMMING") && tribe->stars >= unit_cost("RAMMER")) {
      NativeAction upgrade;
      upgrade.type = "UPGRADE_RAMMER";
      upgrade.unit_id = unit.id;
      upgrade.payload = py::dict();
      append_generated_action(state, actions, max_actions, std::move(upgrade));
    }
    if (has_tech(*tribe, "SAILING") && tribe->stars >= unit_cost("SCOUT")) {
      NativeAction upgrade;
      upgrade.type = "UPGRADE_SCOUT";
      upgrade.unit_id = unit.id;
      upgrade.payload = py::dict();
      append_generated_action(state, actions, max_actions, std::move(upgrade));
    }
    if (has_tech(*tribe, "NAVIGATION") && tribe->stars >= unit_cost("BOMBER")) {
      NativeAction upgrade;
      upgrade.type = "UPGRADE_BOMBER";
      upgrade.unit_id = unit.id;
      upgrade.payload = py::dict();
      append_generated_action(state, actions, max_actions, std::move(upgrade));
    }
  } else if (unit.type == "SCOUT" && has_tech(*tribe, "NAVIGATION") && tribe->stars >= unit_cost("BOMBER")) {
    NativeAction upgrade;
    upgrade.type = "UPGRADE_BOMBER";
    upgrade.unit_id = unit.id;
    upgrade.payload = py::dict();
    append_generated_action(state, actions, max_actions, std::move(upgrade));
  }
  if (unit.status == "FINISHED") {
    return;
  }
  NativeTile* current_tile = tile_at(state, unit.x, unit.y);

  if (unit_is_fresh(unit) && current_tile != nullptr && current_tile->resource == "RUINS" &&
      !tribe->city_ids.empty()) {
    NativeAction examine;
    examine.id = "sim:p" + std::to_string(state.active_player_id) + ":t" + std::to_string(state.tick) +
        ":u" + std::to_string(unit.id) + ":examine";
    examine.type = "EXAMINE";
    examine.unit_id = unit.id;
    examine.payload = py::dict();
    examine.payload["tribe_id"] = state.active_player_id;
    examine.payload["p"] = state.active_player_id;
    append_generated_action(state, actions, max_actions, std::move(examine));
  }

  if (unit_can_attack(unit) && (unit.type == "CLOAK" || unit.type == "DINGHY")) {
    const int radius = std::max(1, unit.range);
    for (NativeCity& city : state.cities) {
      if (city.tribe_id == unit.tribe_id || city.infiltrated ||
          relationship_between(state, unit.tribe_id, city.tribe_id) == "TREATY") {
        continue;
      }
      if (std::max(std::abs(unit.x - city.x), std::abs(unit.y - city.y)) > radius) {
        continue;
      }
      NativeTile* city_tile = tile_at(state, city.x, city.y);
      NativeUnit* occupant = city_tile != nullptr && city_tile->unit_id > 0 ? unit_by_id(state, city_tile->unit_id) : nullptr;
      if (occupant != nullptr && occupant->tribe_id != city.tribe_id) {
        continue;
      }
      NativeAction infiltrate;
      infiltrate.id = "sim:p" + std::to_string(state.active_player_id) + ":t" + std::to_string(state.tick) +
          ":u" + std::to_string(unit.id) + ":infiltrate:c" + std::to_string(city.id);
      infiltrate.type = "INFILTRATE";
      infiltrate.unit_id = unit.id;
      infiltrate.city_id = city.id;
      infiltrate.payload = py::dict();
      infiltrate.payload["tribe_id"] = state.active_player_id;
      infiltrate.payload["p"] = state.active_player_id;
      infiltrate.payload["target_city_id"] = city.id;
      infiltrate.payload["tc"] = city.id;
      append_generated_action(state, actions, max_actions, std::move(infiltrate));
    }
  }

  if (unit_can_attack(unit) && unit.attack > 0.0 && unit.type != "MIND_BENDER" && unit.type != "CLOAK" && unit.type != "DINGHY") {
    const int radius = std::max(1, unit.range);
    for (int x = std::max(0, unit.x - radius); x <= std::min(state.board_size - 1, unit.x + radius); ++x) {
      for (int y = std::max(0, unit.y - radius); y <= std::min(state.board_size - 1, unit.y + radius); ++y) {
        if (x == unit.x && y == unit.y) {
          continue;
        }
        NativeTile* tile = tile_at(state, x, y);
        if (tile == nullptr || tile->unit_id <= 0) {
          continue;
        }
        NativeUnit* target = unit_by_id(state, tile->unit_id);
        if (target == nullptr || target->tribe_id == unit.tribe_id ||
            relationship_between(state, unit.tribe_id, target->tribe_id) == "TREATY") {
          continue;
        }
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
    }
  }

  if (unit_is_fresh(unit) && current_tile != nullptr && (current_tile->terrain == "VILLAGE" || current_tile->terrain == "CITY")) {
    NativeCity* city = current_tile->city_id > 0 ? city_by_id(state, current_tile->city_id) : nullptr;
    if (current_tile->terrain == "VILLAGE" || (city != nullptr && city->tribe_id != unit.tribe_id &&
        relationship_between(state, unit.tribe_id, city->tribe_id) != "TREATY")) {
      NativeAction capture;
      capture.id = "sim:p" + std::to_string(state.active_player_id) + ":t" + std::to_string(state.tick) +
          ":u" + std::to_string(unit.id) + ":capture";
      capture.type = "CAPTURE";
      capture.unit_id = unit.id;
      capture.city_id = city == nullptr ? 0 : city->id;
      capture.payload = py::dict();
      capture.payload["tribe_id"] = state.active_player_id;
      capture.payload["p"] = state.active_player_id;
      capture.payload["target_city_id"] = city == nullptr ? -1 : city->id;
      capture.payload["tc"] = city == nullptr ? -1 : city->id;
      capture.payload["capture_type"] = current_tile->terrain;
      capture.payload["ct"] = current_tile->terrain;
      append_generated_action(state, actions, max_actions, std::move(capture));
    }
  }

  if (unit_can_attack(unit) && unit.type == "MIND_BENDER") {
    const int radius = std::max(1, unit.range);
    for (int x = std::max(0, unit.x - radius); x <= std::min(state.board_size - 1, unit.x + radius); ++x) {
      for (int y = std::max(0, unit.y - radius); y <= std::min(state.board_size - 1, unit.y + radius); ++y) {
        if (x == unit.x && y == unit.y) {
          continue;
        }
        NativeTile* tile = tile_at(state, x, y);
        if (tile == nullptr || tile->unit_id <= 0) {
          continue;
        }
        NativeUnit* target = unit_by_id(state, tile->unit_id);
        if (target == nullptr || target->tribe_id == unit.tribe_id ||
            relationship_between(state, unit.tribe_id, target->tribe_id) == "TREATY") {
          continue;
        }
        NativeAction convert;
        convert.id = "sim:p" + std::to_string(state.active_player_id) + ":t" + std::to_string(state.tick) +
            ":u" + std::to_string(unit.id) + ":convert:u" + std::to_string(target->id);
        convert.type = "CONVERT";
        convert.unit_id = unit.id;
        convert.payload = py::dict();
        convert.payload["tribe_id"] = state.active_player_id;
        convert.payload["p"] = state.active_player_id;
        convert.payload["target_unit_id"] = target->id;
        convert.payload["tu"] = target->id;
        append_generated_action(state, actions, max_actions, std::move(convert));
      }
    }
  }

  if (unit.type == "MIND_BENDER" && heal_others_target_exists(state, unit)) {
    NativeAction heal;
    heal.id = "sim:p" + std::to_string(state.active_player_id) + ":t" + std::to_string(state.tick) +
        ":u" + std::to_string(unit.id) + ":heal_others";
    heal.type = "HEAL_OTHERS";
    heal.unit_id = unit.id;
    heal.payload = py::dict();
    heal.payload["tribe_id"] = state.active_player_id;
    heal.payload["p"] = state.active_player_id;
    append_generated_action(state, actions, max_actions, std::move(heal));
  }

  if (unit_is_fresh(unit) && has_tech(*tribe, "FREE_SPIRIT")) {
    NativeAction disband;
    disband.type = "DISBAND";
    disband.unit_id = unit.id;
    disband.payload = py::dict();
    append_generated_action(state, actions, max_actions, std::move(disband));
  }

  if (unit.kills >= 3 && !unit.veteran && !water_unit_type(unit.type) &&
      unit.type != "SUPERUNIT" && unit.type != "CLOAK" && unit.type != "DAGGER") {
    NativeAction veteran;
    veteran.type = "MAKE_VETERAN";
    veteran.unit_id = unit.id;
    veteran.payload = py::dict();
    append_generated_action(state, actions, max_actions, std::move(veteran));
  }

  if (unit_can_move(unit)) {
    std::vector<std::pair<int, int>> java_priority_order;
    if (state.active_player_id == state.root_player_id) {
      static const int move_order[8][2] = {{-1, -1}, {1, 1}, {1, 0}, {1, -1}, {0, 1}, {0, -1}, {-1, 1}, {-1, 0}};
      for (const auto& delta : move_order) {
        const int x = unit.x + delta[0];
        const int y = unit.y + delta[1];
        if (x < 0 || y < 0 || x >= state.board_size || y >= state.board_size) {
          continue;
        }
        NativeTile* tile = tile_at(state, x, y);
        if (tile != nullptr && passable_move_target(state, *tile)) {
          java_priority_order.emplace_back(x, y);
        }
      }
    } else {
      std::vector<std::pair<int, int>> move_targets;
      for (int x = std::max(0, unit.x - 1); x <= std::min(state.board_size - 1, unit.x + 1); ++x) {
        for (int y = std::max(0, unit.y - 1); y <= std::min(state.board_size - 1, unit.y + 1); ++y) {
          if (x == unit.x && y == unit.y) {
            continue;
          }
          NativeTile* tile = tile_at(state, x, y);
          if (tile != nullptr && passable_move_target(state, *tile)) {
            move_targets.emplace_back(x, y);
          }
        }
      }
      if (!move_targets.empty()) {
        java_priority_order.push_back(move_targets.front());
        for (auto it = move_targets.rbegin(); it != move_targets.rend(); ++it) {
          if (*it != move_targets.front()) {
            java_priority_order.push_back(*it);
          }
        }
      }
    }
    if (!java_priority_order.empty()) {
      for (const auto& target : java_priority_order) {
        append_tile_action(
            state,
            actions,
            max_actions,
            ":u" + std::to_string(unit.id) + ":move:" + std::to_string(target.first) + ":" + std::to_string(target.second),
            "MOVE",
            state.active_player_id,
            unit.id,
            0,
            target.first,
            target.second,
            "destination");
      }
    }
  }

  if (unit_is_fresh(unit) && unit.current_hp > 0 && unit.max_hp > 0 && unit.current_hp < unit.max_hp) {
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

std::vector<std::string> level_up_bonuses_for(int level) {
  if (level == 1) {
    return {"WORKSHOP", "EXPLORER"};
  }
  if (level == 2) {
    return {"CITY_WALL", "RESOURCES"};
  }
  if (level == 3) {
    return {"POP_GROWTH", "BORDER_GROWTH"};
  }
  return {"PARK", "SUPERUNIT"};
}

void regenerate_city_actions(NativeGameState& state, std::vector<NativeAction>& actions, int max_actions, const NativeCity& city) {
  if (city.tribe_id != state.active_player_id) {
    return;
  }
  NativeTribe* tribe = tribe_by_id(state, city.tribe_id);
  if (tribe == nullptr) {
    return;
  }
  const int stars = tribe->stars;
  std::vector<NativeTile*> tiles = city_territory_tiles(state, city);

  static const std::vector<std::string> building_order = {
      "PORT", "MINE", "FORGE", "FARM", "WINDMILL", "MARKET", "LUMBER_HUT", "SAWMILL",
      "TEMPLE", "WATER_TEMPLE", "FOREST_TEMPLE", "MOUNTAIN_TEMPLE", "ALTAR_OF_PEACE",
      "EMPERORS_TOMB", "EYE_OF_GOD", "GATE_OF_POWER", "GRAND_BAZAR", "PARK_OF_FORTUNE",
      "TOWER_OF_WISDOM"};
  for (NativeTile* tile : tiles) {
    if (!tile->building.empty()) {
      continue;
    }
    for (const std::string& building : building_order) {
      if (stars < building_cost(building) || !building_feasible_at(state, *tribe, city, *tile, building)) {
        continue;
      }
      NativeAction build;
      build.type = "BUILD";
      build.city_id = city.id;
      build.payload = py::dict();
      build.payload["p"] = state.active_player_id;
      build.payload["x"] = tile->x;
      build.payload["y"] = tile->y;
      build.payload["bt"] = building;
      build.payload["building_type"] = building;
      append_generated_action(state, actions, max_actions, std::move(build));
    }
  }

  for (NativeTile* tile : tiles) {
    if (tile->terrain == "FOREST" && tile_in_city_territory(*tile, city) &&
        is_action_unlocked(*tribe, "BURN_FOREST") && stars >= 5) {
      NativeAction burn;
      burn.type = "BURN_FOREST";
      burn.city_id = city.id;
      burn.payload = py::dict();
      burn.payload["p"] = state.active_player_id;
      burn.payload["x"] = tile->x;
      burn.payload["y"] = tile->y;
      append_generated_action(state, actions, max_actions, std::move(burn));
    }
  }
  for (NativeTile* tile : tiles) {
    if (tile->terrain == "FOREST" && tile_in_city_territory(*tile, city) &&
        is_action_unlocked(*tribe, "CLEAR_FOREST")) {
      NativeAction clear;
      clear.type = "CLEAR_FOREST";
      clear.city_id = city.id;
      clear.payload = py::dict();
      clear.payload["p"] = state.active_player_id;
      clear.payload["x"] = tile->x;
      clear.payload["y"] = tile->y;
      append_generated_action(state, actions, max_actions, std::move(clear));
    }
  }
  for (NativeTile* tile : tiles) {
    if (!tile->building.empty() && tile_in_city_territory(*tile, city) &&
        city_has_building_at(city, tile->x, tile->y) &&
        is_action_unlocked(*tribe, "DESTROY")) {
      NativeAction destroy;
      destroy.type = "DESTROY";
      destroy.city_id = city.id;
      destroy.payload = py::dict();
      destroy.payload["p"] = state.active_player_id;
      destroy.payload["x"] = tile->x;
      destroy.payload["y"] = tile->y;
      append_generated_action(state, actions, max_actions, std::move(destroy));
    }
  }
  for (NativeTile* tile : tiles) {
    if (tile->terrain == "PLAIN" &&
        tile_in_city_territory(*tile, city) && is_action_unlocked(*tribe, "GROW_FOREST") && stars >= 5) {
      NativeAction grow;
      grow.type = "GROW_FOREST";
      grow.city_id = city.id;
      grow.payload = py::dict();
      grow.payload["p"] = state.active_player_id;
      grow.payload["x"] = tile->x;
      grow.payload["y"] = tile->y;
      append_generated_action(state, actions, max_actions, std::move(grow));
    }
  }
  for (NativeTile* tile : tiles) {
    if (!tile->resource.empty() && tile_in_city_territory(*tile, city) &&
        resource_tech_ok(*tribe, tile->resource) && stars >= resource_cost(tile->resource)) {
      NativeAction gather;
      gather.type = "RESOURCE_GATHERING";
      gather.city_id = city.id;
      gather.payload = py::dict();
      gather.payload["p"] = state.active_player_id;
      gather.payload["x"] = tile->x;
      gather.payload["y"] = tile->y;
      gather.payload["rt"] = tile->resource;
      gather.payload["resource_type"] = tile->resource;
      append_generated_action(state, actions, max_actions, std::move(gather));
    }
  }

  if (city_can_add_unit(city, state)) {
    static const std::vector<std::string> unit_order = {
        "WARRIOR", "RIDER", "DEFENDER", "SWORDMAN", "ARCHER", "CATAPULT", "KNIGHT", "MIND_BENDER",
        "RAFT", "SCOUT", "BOMBER", "SUPERUNIT", "CLOAK", "DAGGER", "RAMMER", "JUGGERNAUT", "DINGHY", "PIRATE"};
    for (const std::string& unit_type : unit_order) {
      if (!unit_spawnable(unit_type) || stars < unit_cost(unit_type) || !unit_unlocked(*tribe, unit_type)) {
        continue;
      }
      NativeAction spawn;
      spawn.type = "SPAWN";
      spawn.city_id = city.id;
      spawn.payload = py::dict();
      spawn.payload["x"] = city.x;
      spawn.payload["y"] = city.y;
      spawn.payload["ut"] = unit_type;
      spawn.payload["unit_type"] = unit_type;
      append_generated_action(state, actions, max_actions, std::move(spawn));
    }
  }
}

void regenerate_tribe_actions(NativeGameState& state, std::vector<NativeAction>& actions, int max_actions) {
  NativeTribe* tribe = tribe_by_id(state, state.active_player_id);
  if (tribe == nullptr) {
    return;
  }
  const int stars = tribe->stars;
  const int tribe_count = static_cast<int>(state.tribes.size());

  if (has_tech(*tribe, "ROADS") && stars >= 3) {
    std::vector<NativeTile*> road_tiles;
    for (const NativeCity& city : state.cities) {
      if (city.tribe_id != state.active_player_id) {
        continue;
      }
      for (NativeTile& tile : state.tiles) {
        if (tile.city_id != city.id || !tile.explored ||
            !can_build_road_at(state, state.active_player_id, tile) ||
            stars < road_cost_at(tile)) {
          continue;
        }
        road_tiles.push_back(&tile);
      }
    }
    std::sort(road_tiles.begin(), road_tiles.end(), [](const NativeTile* left, const NativeTile* right) {
      if (left->x != right->x) {
        return left->x < right->x;
      }
      return left->y < right->y;
    });
    std::set<std::pair<int, int>> seen;
    for (const NativeTile* tile : road_tiles) {
      if (!seen.emplace(tile->x, tile->y).second) {
        continue;
      }
      NativeAction road;
      road.type = "BUILD_ROAD";
      road.payload = py::dict();
      road.payload["p"] = state.active_player_id;
      road.payload["x"] = tile->x;
      road.payload["y"] = tile->y;
      append_generated_action(state, actions, max_actions, std::move(road));
    }
  }

  for (int target = 0; target < tribe_count; ++target) {
    if (target == state.active_player_id) {
      continue;
    }
    if (can_build_embassy(state, state.active_player_id, target)) {
      NativeAction embassy;
      embassy.type = "BUILD_EMBASSY";
      embassy.payload = py::dict();
      embassy.payload["p"] = state.active_player_id;
      embassy.payload["tp"] = target;
      embassy.payload["target_player_id"] = target;
      append_generated_action(state, actions, max_actions, std::move(embassy));
    }
  }

  static const std::vector<std::string> techs = {
      "CLIMBING", "FISHING", "HUNTING", "ORGANIZATION", "RIDING",
      "ARCHERY", "FARMING", "FORESTRY", "FREE_SPIRIT", "MEDITATION",
      "MINING", "ROADS", "RAMMING", "SAILING", "STRATEGY",
      "AQUATISM", "CHIVALRY", "CONSTRUCTION", "DIPLOMACY", "MATHEMATICS",
      "NAVIGATION", "SMITHERY", "SPIRITUALISM", "TRADE", "PHILOSOPHY"};
  for (const std::string& tech : techs) {
    if (!can_research(*tribe, tech)) {
      continue;
    }
    NativeAction research;
    research.type = "RESEARCH_TECH";
    research.payload = py::dict();
    research.payload["p"] = state.active_player_id;
    research.payload["tech"] = tech;
    append_generated_action(state, actions, max_actions, std::move(research));
  }

  for (int target = 0; target < tribe_count; ++target) {
    if (target == state.active_player_id) {
      continue;
    }
    if (is_action_unlocked(*tribe, "PROPOSE_PEACE") &&
        relationship_between(state, state.active_player_id, target) == "WAR" &&
        !has_pending_offer(state, state.active_player_id, target) &&
        !has_pending_offer(state, target, state.active_player_id)) {
      NativeAction propose;
      propose.type = "PROPOSE_PEACE";
      propose.payload = py::dict();
      propose.payload["p"] = state.active_player_id;
      propose.payload["tp"] = target;
      append_generated_action(state, actions, max_actions, std::move(propose));
    }
    if (has_pending_offer(state, target, state.active_player_id) &&
        state.pending_offer_types[static_cast<size_t>(target)][static_cast<size_t>(state.active_player_id)] == "PEACE") {
      NativeAction accept;
      accept.type = "ACCEPT_PEACE";
      accept.payload = py::dict();
      accept.payload["p"] = state.active_player_id;
      accept.payload["tp"] = target;
      append_generated_action(state, actions, max_actions, std::move(accept));
    }
    if (is_action_unlocked(*tribe, "PROPOSE_TREATY") &&
        relationship_between(state, state.active_player_id, target) == "PEACE" &&
        !has_pending_offer(state, state.active_player_id, target) &&
        !has_pending_offer(state, target, state.active_player_id)) {
      NativeAction propose;
      propose.type = "PROPOSE_TREATY";
      propose.payload = py::dict();
      propose.payload["p"] = state.active_player_id;
      propose.payload["tp"] = target;
      append_generated_action(state, actions, max_actions, std::move(propose));
    }
    if (has_pending_offer(state, target, state.active_player_id) &&
        state.pending_offer_types[static_cast<size_t>(target)][static_cast<size_t>(state.active_player_id)] == "TREATY") {
      NativeAction accept;
      accept.type = "ACCEPT_TREATY";
      accept.payload = py::dict();
      accept.payload["p"] = state.active_player_id;
      accept.payload["tp"] = target;
      append_generated_action(state, actions, max_actions, std::move(accept));
    }
    if (relationship_between(state, state.active_player_id, target) == "TREATY") {
      NativeAction cancel;
      cancel.type = "CANCEL_TREATY";
      cancel.payload = py::dict();
      cancel.payload["p"] = state.active_player_id;
      cancel.payload["tp"] = target;
      append_generated_action(state, actions, max_actions, std::move(cancel));
    }
  }

  if (state.can_end_turn) {
    NativeAction end_turn;
    end_turn.id = "sim:p" + std::to_string(state.active_player_id) + ":t" + std::to_string(state.tick) + ":end";
    end_turn.type = "END_TURN";
    end_turn.payload = py::dict();
    end_turn.payload["p"] = state.active_player_id;
    end_turn.payload["tribe_id"] = state.active_player_id;
    append_generated_action(state, actions, max_actions, std::move(end_turn));
  }
}

void regenerate_actions(NativeGameState& state, std::vector<NativeAction>& actions, int max_actions) {
  state.legal_action_indexes.clear();
  if (state.terminal) {
    return;
  }

  NativeTribe* active_tribe = tribe_by_id(state, state.active_player_id);
  if (active_tribe == nullptr) {
    sync_observation_turn_flags(state);
    return;
  }

  for (int city_id : active_tribe->city_ids) {
    const NativeCity* city = nullptr;
    for (const NativeCity& candidate : state.cities) {
      if (candidate.id == city_id) {
        city = &candidate;
        break;
      }
    }
    if (city == nullptr) {
      continue;
    }
    if (city_can_level_up(*city)) {
      state.leveling_up = true;
      state.can_end_turn = false;
      for (const std::string& bonus : level_up_bonuses_for(city->level)) {
        NativeAction level_up;
        level_up.type = "LEVEL_UP";
        level_up.city_id = city->id;
        level_up.payload = py::dict();
        level_up.payload["p"] = state.active_player_id;
        level_up.payload["x"] = city->x;
        level_up.payload["y"] = city->y;
        level_up.payload["b"] = bonus;
        level_up.payload["bonus"] = bonus;
        append_generated_action(state, actions, max_actions, std::move(level_up));
      }
      sync_observation_turn_flags(state);
      return;
    }
  }

  state.leveling_up = false;
  state.can_end_turn = true;
  regenerate_tribe_actions(state, actions, max_actions);
  for (const NativeCity& city : state.cities) {
    if (city.tribe_id == state.active_player_id) {
      regenerate_city_actions(state, actions, max_actions, city);
    }
  }
  for (const NativeUnit& unit : state.units) {
    regenerate_unit_actions(state, actions, max_actions, unit);
  }
  sync_observation_turn_flags(state);
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

void apply_end_turn_for_tribe(NativeGameState& state, int tribe_id) {
  for (NativeUnit& unit : state.units) {
    if (unit.tribe_id != tribe_id || unit.status != "FRESH" || unit.current_hp <= 0 ||
        unit.max_hp <= 0 || unit.current_hp >= unit.max_hp) {
      continue;
    }
    int recover_hp = 2;
    NativeTile* tile = tile_at(state, unit.x, unit.y);
    NativeCity* territory = tile != nullptr && tile->city_id > 0 ? city_by_id(state, tile->city_id) : nullptr;
    if (territory != nullptr && territory->tribe_id == unit.tribe_id) {
      recover_hp += 2;
    }
    unit.current_hp_exact = std::min<double>(unit.max_hp, unit.current_hp_exact + recover_hp);
    unit.current_hp = static_cast<int>(unit.current_hp_exact);
    unit.status = "FINISHED";
    set_unit_payload_field(state, unit.id, "current_hp", "hp", py::int_(unit.current_hp));
    set_unit_payload_field(state, unit.id, "current_hp_exact", "hpx", py::float_(unit.current_hp_exact));
    set_unit_payload_field(state, unit.id, "status", "s", py::str(unit.status));
  }
}

void apply_init_turn(NativeGameState& state, int tribe_id) {
  NativeTribe* tribe = tribe_by_id(state, tribe_id);
  if (tribe == nullptr) {
    return;
  }

  int production = 0;
  for (NativeCity& city : state.cities) {
    if (city.tribe_id != tribe_id) {
      continue;
    }
    bool produces = !city.infiltrated;
    NativeTile* city_tile = tile_at(state, city.x, city.y);
    if (produces && city_tile != nullptr && city_tile->unit_id > 0) {
      const NativeUnit* occupant = unit_by_id(state, city_tile->unit_id);
      if (occupant != nullptr && occupant->tribe_id != tribe_id) {
        produces = false;
      }
    }
    if (produces) {
      production += std::max(0, city.production);
    }
    city.infiltrated = false;
    set_city_payload_field(state, city.id, "infiltrated", "inf", py::bool_(false));
  }

  if (state.tick == 0) {
    tribe->stars = tribe_starting_stars(*tribe);
  } else {
    tribe->stars += std::max(0, production);
  }
  set_tribe_payload_field(state, tribe_id, "stars", py::int_(tribe->stars));

  for (NativeUnit& unit : state.units) {
    if (unit.tribe_id != tribe_id) {
      continue;
    }
    if (unit.status == "PUSHED") {
      unit.status = "MOVED";
    } else {
      unit.status = "FRESH";
    }
    set_unit_payload_field(state, unit.id, "status", "s", py::str(unit.status));
  }

  tribe->units_disabled_next_turn = false;
  ensure_tribe_init_tech(*tribe);
  state.can_end_turn = true;
  state.leveling_up = false;
}

void apply_end_turn_transition(NativeGameState& next) {
  const int previous_player_id = next.active_player_id;
  apply_end_turn_for_tribe(next, previous_player_id);
  const int next_player_id = choose_next_active_player(next);
  apply_init_turn(next, next_player_id);
  set_active_player(next, next_player_id);
  sync_observation_ranking(next);
  sync_observation_turn_flags(next);
}

void append_city_payload_unit(NativeGameState& state, int city_id, int unit_id);
void append_city_payload_building(NativeGameState& state, int city_id, const NativeBuilding& building);
void ensure_city_payload_visible(NativeGameState& state, const NativeCity& native_city);
void append_tribe_payload_city(NativeGameState& state, int tribe_id, int city_id);

bool apply_propose_peace(NativeGameState& next, const NativeAction& action) {
  const int target = action_int(action, "target_player_id", "tp", -1);
  if (target < 0) {
    return false;
  }
  set_pending_offer(next, next.active_player_id, target, "PEACE");
  return true;
}

bool apply_accept_peace(NativeGameState& next, const NativeAction& action) {
  const int target = action_int(action, "target_player_id", "tp", -1);
  if (target < 0 || !has_pending_offer(next, target, next.active_player_id)) {
    return false;
  }
  set_relationship(next, target, next.active_player_id, "PEACE");
  set_relationship(next, next.active_player_id, target, "PEACE");
  clear_pending_offer(next, target, next.active_player_id);
  clear_pending_offer(next, next.active_player_id, target);
  return true;
}

bool apply_propose_treaty(NativeGameState& next, const NativeAction& action) {
  const int target = action_int(action, "target_player_id", "tp", -1);
  if (target < 0) {
    return false;
  }
  set_pending_offer(next, next.active_player_id, target, "TREATY");
  return true;
}

bool apply_accept_treaty(NativeGameState& next, const NativeAction& action) {
  const int target = action_int(action, "target_player_id", "tp", -1);
  if (target < 0 || !has_pending_offer(next, target, next.active_player_id)) {
    return false;
  }
  set_relationship(next, target, next.active_player_id, "TREATY");
  set_relationship(next, next.active_player_id, target, "TREATY");
  clear_pending_offer(next, target, next.active_player_id);
  clear_pending_offer(next, next.active_player_id, target);
  return true;
}

bool apply_cancel_treaty(NativeGameState& next, const NativeAction& action) {
  const int target = action_int(action, "target_player_id", "tp", -1);
  if (target < 0) {
    return false;
  }
  set_relationship(next, target, next.active_player_id, "PEACE");
  set_relationship(next, next.active_player_id, target, "PEACE");
  clear_pending_offer(next, target, next.active_player_id);
  clear_pending_offer(next, next.active_player_id, target);
  return true;
}

bool apply_build_embassy(NativeGameState& next, const NativeAction& action) {
  const int target = action_int(action, "target_player_id", "tp", -1);
  if (target < 0 || !can_build_embassy(next, next.active_player_id, target)) {
    return false;
  }
  NativeCity* capital = capital_city_for_tribe(next, target);
  if (capital == nullptr) {
    return false;
  }
  NativeBuilding embassy;
  embassy.type = "EMBASSY";
  embassy.x = capital->x;
  embassy.y = capital->y;
  embassy.city_id = capital->id;
  embassy.owner_tribe_id = next.active_player_id;
  capital->buildings.push_back(embassy);
  const bool city_was_visible = city_payload_contains(next, capital->id);
  ensure_city_payload_visible(next, *capital);
  if (city_was_visible) {
    append_city_payload_building(next, capital->id, embassy);
  }
  append_tribe_payload_city(next, capital->tribe_id, capital->id);
  NativeTile* tile = tile_at(next, capital->x, capital->y);
  if (tile != nullptr) {
    tile->building = "EMBASSY";
    tile->road = true;
    sync_tile_to_payload(next, *tile);
  }
  for (NativeTile& candidate : next.tiles) {
    if (std::abs(candidate.x - capital->x) <= 1 && std::abs(candidate.y - capital->y) <= 1) {
      candidate.explored = true;
      candidate.visible = true;
      if (candidate.terrain.empty()) {
        candidate.terrain = (candidate.x == capital->x && candidate.y == capital->y) ? "CITY" : "PLAIN";
      }
      if (candidate.city_id <= 0) {
        candidate.city_id = capital->id;
      }
      sync_tile_to_payload(next, candidate);
    }
  }
  update_tribe_economy(next, next.active_player_id, -5, 0);
  return true;
}

bool apply_recover(NativeGameState& next, const NativeAction& action) {
  const int unit_id = action_int(action, "unit_id", "u", 0);
  NativeUnit* unit = unit_by_id(next, unit_id);
  if (unit == nullptr) {
    return false;
  }
  unit->current_hp = std::min(unit->max_hp, unit->current_hp + 4);
  unit->current_hp_exact = static_cast<double>(unit->current_hp);
  unit->status = "FINISHED";
  set_unit_payload_field(next, unit_id, "current_hp", "hp", py::int_(unit->current_hp));
  set_unit_payload_field(next, unit_id, "current_hp_exact", "hpx", py::float_(unit->current_hp_exact));
  set_unit_payload_field(next, unit_id, "status", "s", py::str("FINISHED"));
  return true;
}

bool apply_examine(NativeGameState& next, const NativeAction& action) {
  const int unit_id = action_int(action, "unit_id", "u", 0);
  NativeUnit* unit = unit_by_id(next, unit_id);
  if (unit == nullptr || !unit_is_fresh(*unit)) {
    return false;
  }
  NativeTile* tile = tile_at(next, unit->x, unit->y);
  NativeTribe* tribe = tribe_by_id(next, unit->tribe_id);
  if (tile == nullptr || tile->resource != "RUINS" || tribe == nullptr || tribe->city_ids.empty()) {
    return false;
  }

  std::string bonus = action_string(action, "bonus", "b");
  if (bonus.empty()) {
    // Java picks this from GameState's RNG at execute time, but the compact bot
    // payload does not yet include an RNG snapshot. Use the stable reward path
    // for native search continuity unless a parity fixture supplies a bonus.
    bonus = "RESOURCES";
  }

  tile->resource.clear();
  sync_tile_to_payload(next, *tile);

  if (bonus == "RESOURCES") {
    update_tribe_economy(next, unit->tribe_id, 10, 0);
    unit->status = "FINISHED";
    set_unit_payload_field(next, unit_id, "status", "s", py::str("FINISHED"));
    return true;
  }

  if (bonus == "POP_GROWTH") {
    int handler_city_id = -1;
    int best_capital_level = -1;
    for (int city_id : tribe->city_ids) {
      NativeCity* city = city_by_id(next, city_id);
      if (city != nullptr && city->capital && city->level > best_capital_level) {
        best_capital_level = city->level;
        handler_city_id = city_id;
      }
    }
    if (handler_city_id < 0) {
      handler_city_id = tribe->city_ids.front();
    }
    NativeCity* city = city_by_id(next, handler_city_id);
    if (city != nullptr) {
      city->population += 3;
      city->points_worth += 15;
      set_city_payload_field(next, city->id, "population", "pop", py::int_(city->population));
      set_city_payload_field(next, city->id, "points_worth", "pts", py::int_(city->points_worth));
      update_tribe_economy(next, unit->tribe_id, 0, 15);
    }
    unit->status = "FINISHED";
    set_unit_payload_field(next, unit_id, "status", "s", py::str("FINISHED"));
    return true;
  }

  if (bonus == "RESEARCH") {
    static const std::vector<std::string> techs = {
        "CLIMBING", "FISHING", "HUNTING", "ORGANIZATION", "RIDING", "ARCHERY", "FARMING",
        "FORESTRY", "FREE_SPIRIT", "MEDITATION", "MINING", "ROADS", "RAMMING", "SAILING",
        "STRATEGY", "AQUATISM", "CHIVALRY", "CONSTRUCTION", "DIPLOMACY", "MATHEMATICS",
        "NAVIGATION", "SMITHERY", "SPIRITUALISM", "TRADE", "PHILOSOPHY"};
    for (const std::string& tech : techs) {
      if (!has_tech(*tribe, tech)) {
        tribe->researched_tech_ids.push_back(tech);
        append_researched_tech_payload(next, unit->tribe_id, tech);
        break;
      }
    }
    unit->status = "FINISHED";
    set_unit_payload_field(next, unit_id, "status", "s", py::str("FINISHED"));
    return true;
  }

  if (bonus == "EXPLORER") {
    unit->status = "FINISHED";
    set_unit_payload_field(next, unit_id, "status", "s", py::str("FINISHED"));
    return true;
  }

  return bonus == "UNIT";
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
  unit.status = "FINISHED";
  unit.max_hp = unit_max_hp(type);
  unit.current_hp = unit.max_hp;
  unit.current_hp_exact = static_cast<double>(unit.max_hp);
  unit.attack = unit_attack(type);
  unit.defence = unit_defence(type);
  unit.movement = type == "RIDER" ? 2 : (type == "KNIGHT" ? 3 : 1);
  unit.range = 1;
  unit.cost = unit_cost(type);
  next.units.insert(next.units.begin(), unit);
  city->unit_ids.push_back(unit.id);
  append_city_payload_unit(next, city->id, unit.id);
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
    out["hpx"] = unit.current_hp_exact;
    out["current_hp_exact"] = unit.current_hp_exact;
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
    out["hts"] = false;
    out["hidden_at_turn_start"] = false;
    out["atk"] = py::float_(unit.attack);
    out["attack"] = py::float_(unit.attack);
    out["def"] = py::int_(static_cast<int>(unit.defence));
    out["defence"] = py::int_(static_cast<int>(unit.defence));
    out["mov"] = unit.movement;
    out["movement"] = unit.movement;
    out["r"] = unit.range;
    out["range"] = unit.range;
    out["cost"] = unit.cost;
    py::list reordered;
    reordered.append(out);
    py::list existing = py::reinterpret_borrow<py::list>(next.observation["units"]);
    for (const auto& item : existing) {
      reordered.append(item);
    }
    next.observation["units"] = reordered;
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
  city->points_worth += resource_bonus(resource) * 5;
  set_city_payload_field(next, city_id, "population", "pop", py::int_(city->population));
  set_city_payload_field(next, city_id, "points_worth", "pts", py::int_(city->points_worth));
  update_tribe_economy(next, city->tribe_id, 0, resource_bonus(resource) * 5);
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
    std::sort(tribe->researched_tech_ids.begin(), tribe->researched_tech_ids.end(), [](const std::string& left, const std::string& right) {
      return tech_order(left) < tech_order(right);
    });
    append_researched_tech_payload(next, tribe_id, tech);
  }
  int city_count = 0;
  for (const NativeCity& city : next.cities) {
    if (city.tribe_id == tribe_id) {
      city_count += 1;
    }
  }
  const int cost = 4 + tech_tier(tech) * std::max(1, city_count);
  update_tribe_economy(next, tribe_id, -cost, tech_tier(tech) * 100);
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
  tile->resource.clear();
  if (building == "LUMBER_HUT") {
    tile->terrain = "PLAIN";
  }
  sync_tile_to_payload(next, *tile);
  NativeBuilding native_building;
  native_building.type = building;
  native_building.x = x;
  native_building.y = y;
  native_building.city_id = city_id;
  native_building.owner_tribe_id = -1;
  if (building.find("TEMPLE") != std::string::npos) {
    native_building.level = 1;
    native_building.turns_to_score = 2;
  }
  city->buildings.push_back(native_building);
  append_city_payload_building(next, city_id, native_building);
  update_tribe_economy(next, city->tribe_id, -building_cost(building), 0);
  if (building == "LUMBER_HUT" || building == "FOREST_TEMPLE" || building == "TEMPLE" ||
      building == "WATER_TEMPLE" || building == "MOUNTAIN_TEMPLE") {
    city->population += 1;
    set_city_payload_field(next, city_id, "population", "pop", py::int_(city->population));
    city->points_worth += 5;
    set_city_payload_field(next, city_id, "points_worth", "pts", py::int_(city->points_worth));
    update_tribe_economy(next, city->tribe_id, 0, 5);
  }
  if (building == "FOREST_TEMPLE" || building == "TEMPLE" ||
      building == "WATER_TEMPLE" || building == "MOUNTAIN_TEMPLE") {
    city->points_worth += 100;
    set_city_payload_field(next, city_id, "points_worth", "pts", py::int_(city->points_worth));
    update_tribe_economy(next, city->tribe_id, 0, 100);
  }
  return true;
}

bool apply_level_up(NativeGameState& next, const NativeAction& action) {
  const int city_id = action_int(action, "city_id", "c", 0);
  NativeCity* city = city_by_id(next, city_id);
  if (city == nullptr || !city_can_level_up(*city)) {
    return false;
  }
  const std::string bonus = action_string(action, "bonus", "b");
  const int old_level = city->level;
  const int old_need = city->population_need;
  city->level += 1;
  city->population = std::max(0, city->population - old_need);
  city->population_need = city->level + 1;
  city->production += 1;
  int score_gain = 50 - (old_level + 1) * 5;
  if (bonus == "WORKSHOP") {
    city->production += 1;
  } else if (bonus == "EXPLORER") {
    for (NativeTile& tile : next.tiles) {
      tile.explored = true;
      tile.visible = true;
      sync_tile_to_payload(next, tile);
    }
  } else if (bonus == "RESOURCES") {
    update_tribe_economy(next, city->tribe_id, 5, 0);
  } else if (bonus == "POP_GROWTH") {
    city->population += 3;
  } else if (bonus == "PARK") {
    score_gain += 250;
    city->production += 1;
  } else if (bonus == "CITY_WALL") {
    city->walls = true;
    set_city_payload_field(next, city_id, "has_walls", "wall", py::bool_(true));
  }
  set_city_payload_field(next, city_id, "level", "lvl", py::int_(city->level));
  set_city_payload_field(next, city_id, "population", "pop", py::int_(city->population));
  set_city_payload_field(next, city_id, "population_need", "need", py::int_(city->population_need));
  set_city_payload_field(next, city_id, "production", "prod", py::int_(city->production));
  city->points_worth += score_gain;
  set_city_payload_field(next, city_id, "points_worth", "pts", py::int_(city->points_worth));
  update_tribe_economy(next, city->tribe_id, 0, score_gain);
  next.leveling_up = false;
  next.can_end_turn = true;
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
  if (burn) {
    tile->resource = "CROPS";
  }
  tile->building.clear();
  sync_tile_to_payload(next, *tile);
  update_tribe_economy(next, tribe_id, burn ? -5 : 1, 0);
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
  tile->resource.clear();
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
  remove_unit_ownership_payload(next, *unit);
  mark_unit_removed(next, *unit);
  update_tribe_economy(next, tribe_id, std::max(0, unit_cost(unit->type) / 2), -unit_points(unit->type));
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
  unit->current_hp_exact = static_cast<double>(unit->current_hp);
  set_unit_payload_field(next, unit_id, "is_veteran", "v", py::bool_(true));
  set_unit_payload_field(next, unit_id, "max_hp", "mhp", py::int_(unit->max_hp));
  set_unit_payload_field(next, unit_id, "current_hp", "hp", py::int_(unit->current_hp));
  set_unit_payload_field(next, unit_id, "current_hp_exact", "hpx", py::float_(unit->current_hp_exact));
  return true;
}

bool apply_upgrade_unit(NativeGameState& next, const NativeAction& action, const std::string& upgraded_type) {
  const int unit_id = action_int(action, "unit_id", "u", 0);
  NativeUnit* unit = unit_by_id(next, unit_id);
  if (unit == nullptr) {
    return false;
  }

  const int old_unit_id = unit->id;
  const std::string old_status = unit->status;
  int cost = 0;
  if (upgraded_type == "RAMMER") cost = 5;
  else if (upgraded_type == "SCOUT") cost = 5;
  else if (upgraded_type == "BOMBER") cost = 15;
  if (cost > 0) {
    update_tribe_economy(next, unit->tribe_id, -cost, 0);
  }
  update_tribe_economy(next, unit->tribe_id, 0, unit_points(upgraded_type) - unit_points(unit->type));

  const int new_unit_id = next_unit_id(next);
  NativeTile* tile = tile_at(next, unit->x, unit->y);
  if (tile != nullptr && tile->unit_id == old_unit_id) {
    tile->unit_id = new_unit_id;
    sync_tile_to_payload(next, *tile);
  }
  unit->id = new_unit_id;
  unit->type = upgraded_type;
  unit->status = old_status;
  unit->attack = unit_attack(upgraded_type);
  unit->defence = unit_defence(upgraded_type);
  unit->movement = upgraded_type == "RAMMER" || upgraded_type == "SCOUT" ? 3 : (upgraded_type == "BOMBER" ? 2 : unit->movement);
  unit->range = upgraded_type == "BOMBER" ? 3 : (upgraded_type == "SCOUT" ? 2 : 1);
  unit->cost = unit_cost(upgraded_type);
  set_unit_payload_field(next, old_unit_id, "id", nullptr, py::int_(new_unit_id));
  set_unit_payload_field(next, new_unit_id, "type", "t", py::str(unit->type));
  set_unit_payload_field(next, new_unit_id, "attack", "atk", py::float_(unit->attack));
  set_unit_payload_field(next, new_unit_id, "defence", "def", py::float_(unit->defence));
  set_unit_payload_field(next, new_unit_id, "movement", "mov", py::int_(unit->movement));
  set_unit_payload_field(next, new_unit_id, "range", "r", py::int_(unit->range));
  set_unit_payload_field(next, new_unit_id, "cost", nullptr, py::int_(unit->cost));
  if (unit->city_id < 0) {
    NativeUnit old_unit_ref = *unit;
    old_unit_ref.id = old_unit_id;
    remove_unit_from_owner_lists_payload(next, old_unit_ref);
    append_extra_unit_payload(next, unit->tribe_id, new_unit_id);
  } else if (NativeCity* city = city_by_id(next, unit->city_id); city != nullptr) {
    for (int& existing : city->unit_ids) {
      if (existing == old_unit_id) {
        existing = new_unit_id;
      }
    }
    NativeUnit old_unit_ref = *unit;
    old_unit_ref.id = old_unit_id;
    remove_unit_from_owner_lists_payload(next, old_unit_ref);
    append_city_payload_unit(next, city->id, new_unit_id);
  }
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
      require_present_key(tile_payload, "board tile", {"terrain"});
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
      tile.territory_city_id = tile.city_id;
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
    unit.current_hp_exact = read_double(payload, "current_hp_exact", read_double(payload, "hpx", static_cast<double>(unit.current_hp)));
    unit.max_hp = read_int(payload, "max_hp", read_int(payload, "mhp", 0));
    unit.kills = read_int(payload, "kills", read_int(payload, "k", 0));
    unit.veteran = read_bool(payload, "is_veteran", read_bool(payload, "v", false));
    unit.hidden = read_bool(payload, "is_hidden", read_bool(payload, "h", false));
    unit.hidden_at_turn_start = read_bool(payload, "hidden_at_turn_start", read_bool(payload, "hts", false));
    unit.hidden_enemy_hint = read_bool(payload, "hidden_enemy_hint", read_bool(payload, "hint", false));
    unit.attack = read_double(payload, "attack", read_double(payload, "atk", 0.0));
    unit.defence = read_double(payload, "defence", read_double(payload, "def", 0.0));
    unit.movement = read_int(payload, "movement", read_int(payload, "mov", 0));
    unit.range = read_int(payload, "range", read_int(payload, "r", 0));
    unit.cost = read_int(payload, "cost", 0);
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
    city.bound = read_int(payload, "bound", 0);
    city.points_worth = read_int(payload, "points_worth", read_int(payload, "pts", 0));
    city.capital = read_bool(payload, "is_capital", read_bool(payload, "cap", false));
    city.walls = read_bool(payload, "has_walls", read_bool(payload, "wall", false));
    city.infiltrated = read_bool(payload, "infiltrated", read_bool(payload, "inf", false));
    city.unit_ids = read_int_list(payload, "units");
    if (payload.contains("buildings") && py::isinstance<py::list>(payload["buildings"])) {
      py::list buildings = py::reinterpret_borrow<py::list>(payload["buildings"]);
      city.buildings.reserve(py::len(buildings));
      for (const auto& building_handle : buildings) {
        if (!py::isinstance<py::dict>(building_handle)) {
          throw std::runtime_error("Native strict payload parse failure: city building entry must be an object.");
        }
        py::dict building_payload = py::reinterpret_borrow<py::dict>(building_handle);
        NativeBuilding building;
        building.type = read_string_any(building_payload, "type", "t");
        building.x = read_int(building_payload, "x", 0);
        building.y = read_int(building_payload, "y", 0);
        building.city_id = read_int(building_payload, "city_id", read_int(building_payload, "city", city.id));
        building.owner_tribe_id = read_int(building_payload, "owner_tribe_id", read_int(building_payload, "owner", -1));
        building.level = read_int(building_payload, "level", read_int(building_payload, "lvl", 0));
        building.turns_to_score = read_int(building_payload, "turns_to_score", read_int(building_payload, "score_turns", 0));
        city.buildings.push_back(building);
      }
    } else if (payload.contains("b") && py::isinstance<py::list>(payload["b"])) {
      py::list buildings = py::reinterpret_borrow<py::list>(payload["b"]);
      city.buildings.reserve(py::len(buildings));
      for (const auto& building_handle : buildings) {
        if (!py::isinstance<py::dict>(building_handle)) {
          throw std::runtime_error("Native strict payload parse failure: city building entry must be an object.");
        }
        py::dict building_payload = py::reinterpret_borrow<py::dict>(building_handle);
        NativeBuilding building;
        building.type = read_string_any(building_payload, "type", "t");
        building.x = read_int(building_payload, "x", 0);
        building.y = read_int(building_payload, "y", 0);
        building.city_id = read_int(building_payload, "city_id", read_int(building_payload, "city", city.id));
        building.owner_tribe_id = read_int(building_payload, "owner_tribe_id", read_int(building_payload, "owner", -1));
        building.level = read_int(building_payload, "level", read_int(building_payload, "lvl", 0));
        building.turns_to_score = read_int(building_payload, "turns_to_score", read_int(building_payload, "score_turns", 0));
        city.buildings.push_back(building);
      }
    }
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
    tribe.kills = read_int(payload, "kills", 0);
    tribe.pacifist_count = read_int(payload, "pacifist_count", read_int(payload, "pacifist", 0));
    tribe.units_disabled_next_turn = read_bool(payload, "units_disabled_next_turn", read_bool(payload, "disabled", false));
    tribe.result = read_string(payload, "result");
    if (tribe.result.empty()) {
      tribe.result = read_string(payload, "res");
    }
    tribe.tribe_type = read_string(payload, "tribe");
    tribe.researched_tech_ids = read_string_list(payload, "researched_tech_ids");
    if (tribe.researched_tech_ids.empty()) {
      tribe.researched_tech_ids = read_string_list(payload, "tech");
    }
    tribe.city_ids = read_int_list(payload, "cities");
    tribe.extra_unit_ids = read_int_list(payload, "extra_units");
    if (tribe.extra_unit_ids.empty()) {
      tribe.extra_unit_ids = read_int_list(payload, "extra");
    }
    tribe.connected_city_ids = read_int_list(payload, "connected_city_ids");
    if (tribe.connected_city_ids.empty()) {
      tribe.connected_city_ids = read_int_list(payload, "conn");
    }
    tribe.met_tribe_ids = read_int_list(payload, "met_tribe_ids");
    if (tribe.met_tribe_ids.empty()) {
      tribe.met_tribe_ids = read_int_list(payload, "met");
    }
    tribe.known_capital_tribe_ids = read_int_list(payload, "known_capital_tribe_ids");
    if (tribe.known_capital_tribe_ids.empty()) {
      tribe.known_capital_tribe_ids = read_int_list(payload, "known_caps");
    }
    tribe.discovered_lighthouses = read_int_list(payload, "discovered_lighthouses");
    if (tribe.discovered_lighthouses.empty()) {
      tribe.discovered_lighthouses = read_int_list(payload, "lights");
    }
    tribe.monuments = read_string_map(payload, "monuments");
    if (tribe.monuments.empty()) {
      tribe.monuments = read_string_map(payload, "mon");
    }
    ensure_tribe_init_tech(tribe);
    state.tribes.push_back(tribe);
  }
}

void append_city_payload_unit(NativeGameState& state, int city_id, int unit_id) {
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
    std::vector<int> existing_units = read_int_list(city, "units");
    if (std::find(existing_units.begin(), existing_units.end(), unit_id) == existing_units.end()) {
      existing_units.push_back(unit_id);
    }
    py::list units;
    for (int existing : existing_units) {
      units.append(py::int_(existing));
    }
    city["units"] = units;
    if (city.contains("unit_ids")) {
      py::list unit_ids;
      for (int existing : existing_units) {
        unit_ids.append(py::int_(existing));
      }
      city["unit_ids"] = unit_ids;
    }
    return;
  }
}

void append_city_payload_building(NativeGameState& state, int city_id, const NativeBuilding& building) {
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
    py::dict entry;
    entry["t"] = building.type;
    entry["x"] = building.x;
    entry["y"] = building.y;
    entry["city"] = building.city_id;
    entry["owner"] = building.owner_tribe_id;
    if (building.type.find("TEMPLE") != std::string::npos) {
      entry["lvl"] = building.level <= 0 ? 1 : building.level;
      entry["score_turns"] = building.turns_to_score <= 0 ? 2 : building.turns_to_score;
    }
    py::list buildings = city.contains("b") && py::isinstance<py::list>(city["b"])
        ? py::reinterpret_borrow<py::list>(city["b"])
        : py::list();
    buildings.append(entry);
    city["b"] = buildings;
    if (city.contains("buildings")) {
      city.attr("pop")("buildings");
    }
    return;
  }
}

void ensure_city_payload_visible(NativeGameState& state, const NativeCity& native_city) {
  if (!state.observation.contains("cities") || !py::isinstance<py::list>(state.observation["cities"])) {
    return;
  }
  py::list cities = py::reinterpret_borrow<py::list>(state.observation["cities"]);
  for (const auto& item : cities) {
    if (py::isinstance<py::dict>(item) && read_int(py::reinterpret_borrow<py::dict>(item), "id", 0) == native_city.id) {
      return;
    }
  }
  py::dict city;
  city["id"] = native_city.id;
  city["p"] = native_city.tribe_id;
  city["x"] = native_city.x;
  city["y"] = native_city.y;
  city["lvl"] = native_city.level;
  city["pop"] = native_city.population;
  city["need"] = native_city.population_need;
  city["prod"] = native_city.production;
  city["cap"] = native_city.capital;
  city["wall"] = native_city.walls;
  city["bound"] = native_city.bound <= 0 ? 1 : native_city.bound;
  city["pts"] = native_city.points_worth <= 0 ? 180 : native_city.points_worth;
  city["inf"] = native_city.infiltrated;
  py::list units;
  for (int unit_id : native_city.unit_ids) {
    units.append(py::int_(unit_id));
  }
  city["units"] = units;
  py::list buildings;
  for (const NativeBuilding& building : native_city.buildings) {
    py::dict entry;
    entry["t"] = building.type;
    entry["x"] = building.x;
    entry["y"] = building.y;
    entry["city"] = building.city_id;
    entry["owner"] = building.owner_tribe_id;
    buildings.append(entry);
  }
  city["b"] = buildings;
  cities.append(city);
  std::vector<py::dict> sorted;
  sorted.reserve(py::len(cities));
  for (const auto& item : cities) {
    if (py::isinstance<py::dict>(item)) {
      sorted.push_back(py::reinterpret_borrow<py::dict>(item));
    }
  }
  std::sort(sorted.begin(), sorted.end(), [](const py::dict& left, const py::dict& right) {
    return read_int(left, "id", 0) < read_int(right, "id", 0);
  });
  py::list ordered;
  for (const py::dict& item : sorted) {
    ordered.append(item);
  }
  state.observation["cities"] = ordered;
}

void append_tribe_payload_city(NativeGameState& state, int tribe_id, int city_id) {
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
    std::vector<int> cities = read_int_list(tribe, "cities");
    if (std::find(cities.begin(), cities.end(), city_id) == cities.end()) {
      cities.push_back(city_id);
      std::sort(cities.begin(), cities.end());
    }
    py::list out;
    for (int existing : cities) {
      out.append(py::int_(existing));
    }
    tribe["cities"] = out;
    if (tribe.contains("city_ids")) {
      py::list city_ids;
      for (int existing : cities) {
        city_ids.append(py::int_(existing));
      }
      tribe["city_ids"] = city_ids;
    }
    return;
  }
}

void validate_typed_state(const NativeGameState& state) {
  std::set<int> tribe_ids;
  for (const NativeTribe& tribe : state.tribes) {
    if (!tribe_ids.insert(tribe.id).second) {
      throw std::runtime_error("Native strict payload parse failure: duplicate tribe id " + std::to_string(tribe.id) + ".");
    }
  }

  std::set<int> city_ids;
  for (const NativeCity& city : state.cities) {
    if (city.id <= 0 || !city_ids.insert(city.id).second) {
      throw std::runtime_error("Native strict payload parse failure: duplicate or invalid city id " + std::to_string(city.id) + ".");
    }
    if (city.tribe_id < -1 || (!tribe_ids.empty() && city.tribe_id >= 0 && tribe_ids.find(city.tribe_id) == tribe_ids.end())) {
      throw std::runtime_error("Native strict payload parse failure: city " + std::to_string(city.id) + " references unknown tribe " + std::to_string(city.tribe_id) + ".");
    }
    if (tile_at(const_cast<NativeGameState&>(state), city.x, city.y) == nullptr) {
      throw std::runtime_error("Native strict payload parse failure: city " + std::to_string(city.id) + " is outside the board.");
    }
    for (const NativeBuilding& building : city.buildings) {
      if (building.city_id != 0 && building.city_id != city.id) {
        throw std::runtime_error("Native strict payload parse failure: building in city " + std::to_string(city.id) + " references city " + std::to_string(building.city_id) + ".");
      }
      if (tile_at(const_cast<NativeGameState&>(state), building.x, building.y) == nullptr) {
        throw std::runtime_error("Native strict payload parse failure: building in city " + std::to_string(city.id) + " is outside the board.");
      }
    }
  }

  std::set<int> unit_ids;
  for (const NativeUnit& unit : state.units) {
    if (unit.id <= 0 || !unit_ids.insert(unit.id).second) {
      throw std::runtime_error("Native strict payload parse failure: duplicate or invalid unit id " + std::to_string(unit.id) + ".");
    }
    if (!tribe_ids.empty() && tribe_ids.find(unit.tribe_id) == tribe_ids.end()) {
      throw std::runtime_error("Native strict payload parse failure: unit " + std::to_string(unit.id) + " references unknown tribe " + std::to_string(unit.tribe_id) + ".");
    }
    if (!city_ids.empty() && unit.city_id > 0 && city_ids.find(unit.city_id) == city_ids.end()) {
      throw std::runtime_error("Native strict payload parse failure: unit " + std::to_string(unit.id) + " references unknown city " + std::to_string(unit.city_id) + ".");
    }
    if (tile_at(const_cast<NativeGameState&>(state), unit.x, unit.y) == nullptr) {
      throw std::runtime_error("Native strict payload parse failure: unit " + std::to_string(unit.id) + " is outside the board.");
    }
  }

  for (const NativeTile& tile : state.tiles) {
    if (tile.unit_id > 0 && unit_ids.find(tile.unit_id) == unit_ids.end()) {
      throw std::runtime_error("Native strict payload parse failure: tile " + std::to_string(tile.x) + "," + std::to_string(tile.y) + " references unknown unit " + std::to_string(tile.unit_id) + ".");
    }
  }
}

void parse_relationships(NativeGameState& state) {
  state.relationships.clear();
  state.pending_offer_from.clear();
  state.pending_offer_types.clear();
  const int tribe_count = static_cast<int>(state.tribes.size());
  if (tribe_count <= 0) {
    return;
  }
  state.relationships.assign(static_cast<size_t>(tribe_count), std::vector<std::string>(static_cast<size_t>(tribe_count), "PEACE"));
  state.pending_offer_from.assign(static_cast<size_t>(tribe_count), std::vector<int>(static_cast<size_t>(tribe_count), -1));
  state.pending_offer_types.assign(static_cast<size_t>(tribe_count), std::vector<std::string>(static_cast<size_t>(tribe_count), ""));

  if (!state.observation.contains("rel") || !py::isinstance<py::list>(state.observation["rel"])) {
    return;
  }
  py::list rows = py::reinterpret_borrow<py::list>(state.observation["rel"]);
  for (int y = 0; y < static_cast<int>(py::len(rows)) && y < tribe_count; ++y) {
    if (!py::isinstance<py::list>(rows[y])) {
      continue;
    }
    py::list row = py::reinterpret_borrow<py::list>(rows[y]);
    for (int x = 0; x < static_cast<int>(py::len(row)) && x < tribe_count; ++x) {
      if (row[x].is_none()) {
        state.relationships[static_cast<size_t>(y)][static_cast<size_t>(x)] = "";
      } else {
        state.relationships[static_cast<size_t>(y)][static_cast<size_t>(x)] =
            py::cast<std::string>(py::str(row[x]));
      }
    }
  }
}

int tribe_city_count(const NativeTribe& tribe, const NativeGameState& state) {
  if (!tribe.city_ids.empty()) {
    return static_cast<int>(tribe.city_ids.size());
  }
  int count = 0;
  for (const NativeCity& city : state.cities) {
    if (city.tribe_id == tribe.id) {
      count += 1;
    }
  }
  return count;
}

int tribe_max_production(const NativeTribe& tribe, const NativeGameState& state) {
  int max_production = 0;
  for (const NativeCity& city : state.cities) {
    if (city.tribe_id == tribe.id) {
      max_production = std::max(max_production, city.production);
    }
  }
  return max_production;
}

bool ranks_before(const NativeTribe& left, const NativeTribe& right, const NativeGameState& state) {
  const bool left_win = left.result == "WIN";
  const bool right_win = right.result == "WIN";
  if (left_win != right_win) {
    return left_win;
  }
  if (left.score != right.score) {
    return left.score > right.score;
  }
  const int left_techs = static_cast<int>(left.researched_tech_ids.size());
  const int right_techs = static_cast<int>(right.researched_tech_ids.size());
  if (left_techs != right_techs) {
    return left_techs > right_techs;
  }
  const int left_cities = tribe_city_count(left, state);
  const int right_cities = tribe_city_count(right, state);
  if (left_cities != right_cities) {
    return left_cities > right_cities;
  }
  const int left_production = tribe_max_production(left, state);
  const int right_production = tribe_max_production(right, state);
  if (left_production != right_production) {
    return left_production > right_production;
  }
  return left.id < right.id;
}

void sync_observation_ranking(NativeGameState& state) {
  std::vector<const NativeTribe*> ordered;
  ordered.reserve(state.tribes.size());
  for (const NativeTribe& tribe : state.tribes) {
    ordered.push_back(&tribe);
  }
  std::sort(ordered.begin(), ordered.end(), [&](const NativeTribe* left, const NativeTribe* right) {
    return ranks_before(*left, *right, state);
  });

  py::list rank;
  py::list ranking;
  for (const NativeTribe* tribe : ordered) {
    rank.append(tribe->id);
    ranking.append(tribe->id);
  }
  state.observation["rank"] = rank;
  state.observation["ranking"] = ranking;
}

void sync_observation_turn_flags(NativeGameState& state) {
  state.observation["active_player_id"] = state.active_player_id;
  state.observation["active"] = state.active_player_id;
  state.observation["can_end_turn"] = state.can_end_turn;
  state.observation["end"] = state.can_end_turn;
  state.observation["leveling_up"] = state.leveling_up;
  state.observation["lvlup"] = state.leveling_up;
  state.observation["tick"] = state.tick;
}

void sync_relationships_payload(NativeGameState& state) {
  if (state.relationships.empty()) {
    return;
  }
  py::list rows;
  for (const std::vector<std::string>& row : state.relationships) {
    py::list out_row;
    for (const std::string& value : row) {
      out_row.append(value.empty() ? py::object(py::none()) : py::object(py::str(value)));
    }
    rows.append(out_row);
  }
  state.observation["rel"] = rows;
}

std::string relationship_between(const NativeGameState& state, int from_tribe, int to_tribe) {
  if (from_tribe < 0 || to_tribe < 0 ||
      from_tribe >= static_cast<int>(state.relationships.size()) ||
      to_tribe >= static_cast<int>(state.relationships[from_tribe].size())) {
    return "PEACE";
  }
  return state.relationships[static_cast<size_t>(from_tribe)][static_cast<size_t>(to_tribe)];
}

void set_relationship(NativeGameState& state, int from_tribe, int to_tribe, const std::string& relationship) {
  if (from_tribe < 0 || to_tribe < 0 ||
      from_tribe >= static_cast<int>(state.relationships.size()) ||
      to_tribe >= static_cast<int>(state.relationships[from_tribe].size())) {
    return;
  }
  state.relationships[static_cast<size_t>(from_tribe)][static_cast<size_t>(to_tribe)] = relationship;
  sync_relationships_payload(state);
}

bool has_pending_offer(const NativeGameState& state, int from_tribe, int to_tribe) {
  if (from_tribe < 0 || to_tribe < 0 ||
      from_tribe >= static_cast<int>(state.pending_offer_from.size()) ||
      to_tribe >= static_cast<int>(state.pending_offer_from[from_tribe].size())) {
    return false;
  }
  return state.pending_offer_from[static_cast<size_t>(from_tribe)][static_cast<size_t>(to_tribe)] >= 0;
}

void set_pending_offer(NativeGameState& state, int from_tribe, int to_tribe, const std::string& offer_type) {
  if (from_tribe < 0 || to_tribe < 0 ||
      from_tribe >= static_cast<int>(state.pending_offer_from.size()) ||
      to_tribe >= static_cast<int>(state.pending_offer_from[from_tribe].size())) {
    return;
  }
  state.pending_offer_from[static_cast<size_t>(from_tribe)][static_cast<size_t>(to_tribe)] = from_tribe;
  state.pending_offer_types[static_cast<size_t>(from_tribe)][static_cast<size_t>(to_tribe)] = offer_type;
}

void clear_pending_offer(NativeGameState& state, int from_tribe, int to_tribe) {
  if (from_tribe < 0 || to_tribe < 0 ||
      from_tribe >= static_cast<int>(state.pending_offer_from.size()) ||
      to_tribe >= static_cast<int>(state.pending_offer_from[from_tribe].size())) {
    return;
  }
  state.pending_offer_from[static_cast<size_t>(from_tribe)][static_cast<size_t>(to_tribe)] = -1;
  state.pending_offer_types[static_cast<size_t>(from_tribe)][static_cast<size_t>(to_tribe)].clear();
}

void infer_pending_offers_from_actions(NativeGameState& state, const std::vector<NativeAction>& actions) {
  for (const NativeAction& action : actions) {
    const std::string type = canonical_action_type(action);
    const int target = action_int(action, "target_player_id", "tp", -1);
    if (target < 0) {
      continue;
    }
    if (type == "ACCEPT_PEACE") {
      set_pending_offer(state, target, state.active_player_id, "PEACE");
    } else if (type == "ACCEPT_TREATY") {
      set_pending_offer(state, target, state.active_player_id, "TREATY");
    }
  }
}

void ensure_tribe_init_tech(NativeTribe& tribe) {
  static const std::map<std::string, std::string> init_tech_by_type = {
      {"XIN_XI", "CLIMBING"}, {"IMPERIUS", "ORGANIZATION"}, {"BARDUR", "HUNTING"},
      {"OUMAJI", "RIDING"}, {"KICKOO", "FISHING"}, {"HOODRICK", "ARCHERY"},
      {"LUXIDOOR", "ORGANIZATION"}, {"VENGIR", "SMITHERY"}, {"ZEBASI", "FARMING"},
      {"AI_MO", "PHILOSOPHY"}, {"QUETZALI", "STRATEGY"}, {"YADAKK", "ROADS"}};
  std::string key = tribe.tribe_type;
  std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  for (const auto& item : init_tech_by_type) {
    if (key.find(item.first) == std::string::npos) {
      continue;
    }
    if (!has_tech(tribe, item.second)) {
      tribe.researched_tech_ids.push_back(tech_id(item.second));
    }
    return;
  }
}

int tribe_starting_stars(const NativeTribe& tribe) {
  static const std::map<std::string, int> stars_by_type = {
      {"XIN_XI", 7}, {"IMPERIUS", 5}, {"BARDUR", 5}, {"OUMAJI", 6}, {"KICKOO", 5},
      {"HOODRICK", 7}, {"LUXIDOOR", 2}, {"VENGIR", 5}, {"ZEBASI", 5}, {"AI_MO", 5},
      {"QUETZALI", 7}, {"YADAKK", 7}};
  std::string key = tribe.tribe_type;
  std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  for (const auto& item : stars_by_type) {
    if (key.find(item.first) != std::string::npos) {
      return item.second;
    }
  }
  return 5;
}

bool city_can_level_up(const NativeCity& city) {
  return city.population >= city.population_need && city.population_need > 0;
}

bool city_can_add_unit(const NativeCity& city, const NativeGameState& state) {
  NativeTile* tile = tile_at(const_cast<NativeGameState&>(state), city.x, city.y);
  if (tile != nullptr && tile->unit_id > 0) {
    return false;
  }
  return static_cast<int>(city.unit_ids.size()) < city.level + 1;
}

std::vector<NativeTile*> city_territory_tiles(NativeGameState& state, const NativeCity& city) {
  std::vector<NativeTile*> out;
  for (NativeTile& tile : state.tiles) {
    if (tile.city_id == city.id && tile.explored) {
      out.push_back(&tile);
    }
  }
  std::sort(out.begin(), out.end(), [](const NativeTile* left, const NativeTile* right) {
    if (left->x != right->x) {
      return left->x < right->x;
    }
    return left->y < right->y;
  });
  return out;
}

bool tile_in_city_territory(const NativeTile& tile, const NativeCity& city) {
  return tile.city_id == city.id;
}

bool is_action_unlocked(const NativeTribe& tribe, const std::string& action_type) {
  static const std::map<std::string, std::string> requirements = {
      {"BURN_FOREST", "CONSTRUCTION"}, {"CLEAR_FOREST", "FORESTRY"}, {"GROW_FOREST", "SPIRITUALISM"},
      {"DESTROY", "CHIVALRY"}, {"BUILD_EMBASSY", "DIPLOMACY"}, {"PROPOSE_PEACE", "STRATEGY"},
      {"PROPOSE_TREATY", "DIPLOMACY"}};
  auto it = requirements.find(action_type);
  return it == requirements.end() || has_tech(tribe, it->second);
}

bool is_building_unlocked(const NativeTribe& tribe, const std::string& building) {
  static const std::map<std::string, std::string> requirements = {
      {"PORT", "FISHING"}, {"MINE", "MINING"}, {"FARM", "FARMING"}, {"FORGE", "SMITHERY"},
      {"WINDMILL", "CONSTRUCTION"}, {"MARKET", "TRADE"}, {"LUMBER_HUT", "FORESTRY"},
      {"SAWMILL", "MATHEMATICS"}, {"TEMPLE", "FREE_SPIRIT"}, {"WATER_TEMPLE", "AQUATISM"},
      {"FOREST_TEMPLE", "SPIRITUALISM"}, {"MOUNTAIN_TEMPLE", "MEDITATION"},
      {"ALTAR_OF_PEACE", "MEDITATION"}, {"EMPERORS_TOMB", "TRADE"}, {"EYE_OF_GOD", "NAVIGATION"},
      {"GRAND_BAZAR", "ROADS"}, {"TOWER_OF_WISDOM", "PHILOSOPHY"}, {"EMBASSY", "DIPLOMACY"}};
  auto it = requirements.find(building);
  return it == requirements.end() || has_tech(tribe, it->second);
}

bool building_terrain_ok(const std::string& building, const NativeTile& tile) {
  static const std::map<std::string, std::set<std::string>> terrain = {
      {"PORT", {"SHALLOW_WATER"}}, {"MINE", {"MOUNTAIN"}}, {"FARM", {"PLAIN"}},
      {"FORGE", {"PLAIN"}}, {"WINDMILL", {"PLAIN"}}, {"MARKET", {"PLAIN"}},
      {"LUMBER_HUT", {"FOREST"}}, {"SAWMILL", {"PLAIN"}}, {"TEMPLE", {"PLAIN"}},
      {"WATER_TEMPLE", {"SHALLOW_WATER", "DEEP_WATER"}}, {"FOREST_TEMPLE", {"FOREST"}},
      {"MOUNTAIN_TEMPLE", {"MOUNTAIN"}},
      {"ALTAR_OF_PEACE", {"PLAIN", "SHALLOW_WATER"}}, {"EMPERORS_TOMB", {"PLAIN", "SHALLOW_WATER"}},
      {"EYE_OF_GOD", {"PLAIN", "SHALLOW_WATER"}}, {"GATE_OF_POWER", {"PLAIN", "SHALLOW_WATER"}},
      {"GRAND_BAZAR", {"PLAIN", "SHALLOW_WATER"}}, {"PARK_OF_FORTUNE", {"PLAIN", "SHALLOW_WATER"}},
      {"TOWER_OF_WISDOM", {"PLAIN", "SHALLOW_WATER"}}, {"EMBASSY", {"CITY"}}};
  auto it = terrain.find(building);
  if (it == terrain.end()) {
    return true;
  }
  return it->second.count(tile.terrain) > 0;
}

bool city_has_building(const NativeCity& city, const std::string& building) {
  for (const NativeBuilding& existing : city.buildings) {
    if (existing.type == building) {
      return true;
    }
  }
  return false;
}

bool city_has_building_at(const NativeCity& city, int x, int y) {
  for (const NativeBuilding& existing : city.buildings) {
    if (existing.x == x && existing.y == y && !existing.type.empty()) {
      return true;
    }
  }
  return false;
}

bool adjacent_building(const NativeGameState& state, const NativeTile& tile, const std::set<std::string>& buildings) {
  for (const NativeTile& candidate : state.tiles) {
    const int dx = std::abs(candidate.x - tile.x);
    const int dy = std::abs(candidate.y - tile.y);
    if (dx > 1 || dy > 1 || (dx == 0 && dy == 0)) {
      continue;
    }
    if (buildings.count(candidate.building) > 0) {
      return true;
    }
  }
  return false;
}

bool building_feasible_at(const NativeGameState& state, const NativeTribe& tribe, const NativeCity& city, const NativeTile& tile, const std::string& building) {
  if (tile.city_id != city.id || !tile.explored || !tile.building.empty() ||
      !is_building_unlocked(tribe, building) || !building_terrain_ok(building, tile)) {
    return false;
  }
  if (building == "MINE" && tile.resource != "ORE") {
    return false;
  }
  if (building == "FARM" && tile.resource != "CROPS") {
    return false;
  }
  if (building == "FORGE" && !adjacent_building(state, tile, {"MINE"})) {
    return false;
  }
  if (building == "WINDMILL" && !adjacent_building(state, tile, {"FARM"})) {
    return false;
  }
  if (building == "SAWMILL" && !adjacent_building(state, tile, {"LUMBER_HUT"})) {
    return false;
  }
  if (building == "MARKET" && !adjacent_building(state, tile, {"SAWMILL", "WINDMILL", "FORGE"})) {
    return false;
  }
  static const std::set<std::string> unique_city_buildings = {"SAWMILL", "MARKET", "WINDMILL", "FORGE"};
  if (unique_city_buildings.count(building) > 0 && city_has_building(city, building)) {
    return false;
  }
  static const std::set<std::string> monuments = {
      "ALTAR_OF_PEACE", "EMPERORS_TOMB", "EYE_OF_GOD", "GATE_OF_POWER",
      "GRAND_BAZAR", "PARK_OF_FORTUNE", "TOWER_OF_WISDOM"};
  if (monuments.count(building) > 0) {
    auto it = tribe.monuments.find(building);
    return it != tribe.monuments.end() && it->second == "AVAILABLE";
  }
  return true;
}

bool resource_tech_ok(const NativeTribe& tribe, const std::string& resource) {
  if (resource == "ANIMAL") {
    return has_tech(tribe, "HUNTING");
  }
  if (resource == "FISH") {
    return has_tech(tribe, "FISHING");
  }
  if (resource == "FRUIT") {
    return has_tech(tribe, "ORGANIZATION");
  }
  if (resource == "STARFISH") {
    return has_tech(tribe, "NAVIGATION");
  }
  return false;
}

int road_cost_at(const NativeTile& tile) {
  if (tile.terrain == "SHALLOW_WATER" || tile.terrain == "DEEP_WATER") {
    return 5;
  }
  return 3;
}

bool can_build_road_at(const NativeGameState& state, int tribe_id, const NativeTile& tile) {
  if (!tile.explored || tile.road) {
    return false;
  }
  if (!(tile.terrain == "PLAIN" || tile.terrain == "FOREST" || tile.terrain == "VILLAGE" ||
        tile.terrain == "SHALLOW_WATER")) {
    return false;
  }
  if (tile.unit_id > 0) {
    const NativeUnit* unit = nullptr;
    for (const NativeUnit& candidate : state.units) {
      if (candidate.id == tile.unit_id) {
        unit = &candidate;
        break;
      }
    }
    if (unit != nullptr && unit->tribe_id != tribe_id) {
      return false;
    }
  }
  if (tile.city_id > 0) {
    const NativeCity* city = nullptr;
    for (const NativeCity& candidate : state.cities) {
      if (candidate.id == tile.city_id) {
        city = &candidate;
        break;
      }
    }
    if (city != nullptr && city->tribe_id != tribe_id) {
      return false;
    }
  }
  return true;
}

bool tribe_met(const NativeTribe& tribe, int other_id) {
  return std::find(tribe.met_tribe_ids.begin(), tribe.met_tribe_ids.end(), other_id) != tribe.met_tribe_ids.end();
}

NativeCity* capital_city_for_tribe(NativeGameState& state, int tribe_id) {
  for (NativeCity& city : state.cities) {
    if (city.tribe_id == tribe_id && city.capital) {
      return &city;
    }
  }
  return nullptr;
}

bool can_build_embassy(const NativeGameState& state, int owner_id, int host_id) {
  if (owner_id == host_id || !is_action_unlocked(*tribe_by_id_const(state, owner_id), "BUILD_EMBASSY")) {
    return false;
  }
  const NativeTribe* owner = tribe_by_id_const(state, owner_id);
  if (owner == nullptr || owner->stars < 5 || !tribe_met(*owner, host_id)) {
    return false;
  }
  if (relationship_between(state, owner_id, host_id) == "WAR") {
    return false;
  }
  const NativeCity* capital = capital_city_for_tribe(const_cast<NativeGameState&>(state), host_id);
  if (capital == nullptr) {
    return false;
  }
  for (const NativeBuilding& building : capital->buildings) {
    if (building.type == "EMBASSY" && building.owner_tribe_id == owner_id) {
      return false;
    }
  }
  return true;
}

bool city_payload_contains(const NativeGameState& state, int city_id) {
  if (!state.observation.contains("cities") || !py::isinstance<py::list>(state.observation["cities"])) {
    return false;
  }
  py::list cities = py::reinterpret_borrow<py::list>(state.observation["cities"]);
  for (const auto& item : cities) {
    if (py::isinstance<py::dict>(item) && read_int(py::reinterpret_borrow<py::dict>(item), "id", 0) == city_id) {
      return true;
    }
  }
  return false;
}

void parse_observation_state(NativeGameState& state) {
  parse_tiles(state);
  parse_units(state);
  parse_cities(state);
  parse_tribes(state);
  ingest_cities_from_board(state);
  merge_capital_cities(state);
  assign_city_owners_from_capitals(state);
  parse_relationships(state);
  capture_capital_city_ids(state);
  sync_tribe_city_ids(state);
  state.leveling_up = read_bool(state.observation, "leveling_up", read_bool(state.observation, "lvlup", false));
  state.can_end_turn = read_bool(state.observation, "can_end_turn", read_bool(state.observation, "end", true));
  validate_typed_state(state);
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
  infer_pending_offers_from_actions(root.state, root.actions);
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
    reveal_from_current_assets(next);
    sync_all_tiles_to_payload(next);
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
  } else if (type == "INFILTRATE") {
    applied_ok = apply_infiltrate(next, applied);
  } else if (type == "HEAL_OTHERS") {
    applied_ok = apply_heal_others(next, applied);
  } else if (type == "EXAMINE") {
    applied_ok = apply_examine(next, applied);
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
  } else if (type == "BUILD_EMBASSY") {
    applied_ok = apply_build_embassy(next, applied);
  } else if (type == "PROPOSE_PEACE") {
    applied_ok = apply_propose_peace(next, applied);
  } else if (type == "ACCEPT_PEACE") {
    applied_ok = apply_accept_peace(next, applied);
  } else if (type == "PROPOSE_TREATY") {
    applied_ok = apply_propose_treaty(next, applied);
  } else if (type == "ACCEPT_TREATY") {
    applied_ok = apply_accept_treaty(next, applied);
  } else if (type == "CANCEL_TREATY") {
    applied_ok = apply_cancel_treaty(next, applied);
  }

  if (!applied_ok) {
    throw_transition_error("unsupported_or_failed_transition:" + type);
  }

  if (next.terminal) {
    return next;
  }
  const int newly_explored = reveal_from_current_assets(next);
  if (type == "MOVE" || type == "STEP_MOVE") {
    update_tribe_economy(next, next.active_player_id, 0, newly_explored * 5);
  }
  sync_all_tiles_to_payload(next);
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
