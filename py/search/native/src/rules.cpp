#include "rules.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

namespace tribes::native {
namespace {
void sync_observation_ranking(NativeGameState& state);

using TimingClock = std::chrono::steady_clock;
thread_local NativeTransitionTiming g_last_transition_timing;

double elapsed_ms(TimingClock::time_point started_at) {
  return std::chrono::duration<double, std::milli>(TimingClock::now() - started_at).count();
}

bool payload_sync_enabled(const NativeGameState& state) {
#ifdef TRIBES_NATIVE_MCTS_STANDALONE
  (void)state;
  return false;
#else
  return !state.observation.is_none();
#endif
}

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
  if (out.contains("_native_enemy_explored") && py::isinstance<py::dict>(out["_native_enemy_explored"])) {
    py::dict input_memory = py::reinterpret_borrow<py::dict>(out["_native_enemy_explored"]);
    py::dict output_memory;
    for (const auto& item : input_memory) {
      output_memory[item.first] = py::isinstance<py::list>(item.second) ? shallow_copy_list(item.second) : item.second;
    }
    out["_native_enemy_explored"] = output_memory;
  }
  return out;
}

NativeGameState copy_transition_state_without_observation(const NativeGameState& state) {
  NativeGameState next;
  next.board_size = state.board_size;
  next.tiles = state.tiles;
  next.units = state.units;
  next.cities = state.cities;
  next.tribes = state.tribes;
  next.capital_city_ids = state.capital_city_ids;
  next.relationships = state.relationships;
  next.pending_offer_from = state.pending_offer_from;
  next.pending_offer_types = state.pending_offer_types;
  next.actor_id_floor = state.actor_id_floor;
  next.root_player_id = state.root_player_id;
  next.active_player_id = state.active_player_id;
  next.tick = state.tick;
  next.terminal = state.terminal;
  next.leveling_up = state.leveling_up;
  next.can_end_turn = state.can_end_turn;
#ifdef TRIBES_NATIVE_MCTS_STANDALONE
  next.generated_action_ids_enabled = false;
#else
  next.generated_action_ids_enabled = state.generated_action_ids_enabled;
#endif
  return next;
}

std::string canonical_action_type(const NativeAction& action) {
  std::string type = action.type.empty() ? read_string(action.payload, "t") : action.type;
  return type;
}

int action_scalar_int(const NativeAction& action, const char* key, bool* found) {
  if (key == nullptr) {
    *found = false;
    return 0;
  }
  if (std::strcmp(key, "unit_id") == 0 || std::strcmp(key, "u") == 0) {
    if (action.unit_id != 0) {
      *found = true;
      return action.unit_id;
    }
  } else if (std::strcmp(key, "city_id") == 0 || std::strcmp(key, "c") == 0) {
    if (action.city_id != 0) {
      *found = true;
      return action.city_id;
    }
  } else if (std::strcmp(key, "tribe_id") == 0 || std::strcmp(key, "p") == 0) {
    if (action.tribe_id != 0) {
      *found = true;
      return action.tribe_id;
    }
  } else if (std::strcmp(key, "target_unit_id") == 0 || std::strcmp(key, "tu") == 0) {
    if (action.target_unit_id != 0) {
      *found = true;
      return action.target_unit_id;
    }
  } else if (std::strcmp(key, "target_city_id") == 0 || std::strcmp(key, "tc") == 0) {
    if (action.target_city_id != 0) {
      *found = true;
      return action.target_city_id;
    }
  } else if (std::strcmp(key, "target_player_id") == 0 || std::strcmp(key, "tp") == 0 ||
             std::strcmp(key, "target_id") == 0 || std::strcmp(key, "targetID") == 0) {
    if (action.target_player_id >= 0) {
      *found = true;
      return action.target_player_id;
    }
  } else if (std::strcmp(key, "x") == 0) {
    if (action.has_xy) {
      *found = true;
      return action.x;
    }
  } else if (std::strcmp(key, "y") == 0) {
    if (action.has_xy) {
      *found = true;
      return action.y;
    }
  }
  *found = false;
  return 0;
}

std::string action_scalar_string(const NativeAction& action, const char* key, bool* found) {
  if (key == nullptr) {
    *found = false;
    return "";
  }
  const std::string* value = nullptr;
  if (std::strcmp(key, "unit_type") == 0 || std::strcmp(key, "ut") == 0) value = &action.unit_type;
  else if (std::strcmp(key, "building_type") == 0 || std::strcmp(key, "bt") == 0) value = &action.building_type;
  else if (std::strcmp(key, "resource_type") == 0 || std::strcmp(key, "rt") == 0) value = &action.resource_type;
  else if (std::strcmp(key, "capture_type") == 0 || std::strcmp(key, "ct") == 0) value = &action.capture_type;
  else if (std::strcmp(key, "bonus") == 0 || std::strcmp(key, "b") == 0) value = &action.bonus;
  else if (std::strcmp(key, "technology") == 0 || std::strcmp(key, "tech") == 0) value = &action.tech;
  if (value != nullptr && !value->empty()) {
    *found = true;
    return *value;
  }
  *found = false;
  return "";
}

int action_int(const NativeAction& action, const char* primary, const char* fallback = nullptr, int default_value = 0) {
#ifdef TRIBES_NATIVE_MCTS_STANDALONE
  bool found = false;
  int scalar = action_scalar_int(action, primary, &found);
  if (found) {
    return scalar;
  }
  if (fallback != nullptr) {
    scalar = action_scalar_int(action, fallback, &found);
    if (found) {
      return scalar;
    }
  }
#endif
  int value = read_int(action.payload, primary, default_value);
  if (value == default_value && fallback != nullptr) {
    value = read_int(action.payload, fallback, default_value);
  }
  return value;
}

std::string action_string(const NativeAction& action, const char* primary, const char* fallback = nullptr) {
#ifdef TRIBES_NATIVE_MCTS_STANDALONE
  bool found = false;
  std::string scalar = action_scalar_string(action, primary, &found);
  if (found) {
    return scalar;
  }
  if (fallback != nullptr) {
    scalar = action_scalar_string(action, fallback, &found);
    if (found) {
      return scalar;
    }
  }
#endif
  return read_string_any(action.payload, primary, fallback);
}

void rebuild_state_indexes(NativeGameState& state) {
  const int board_cells = state.board_size > 0 ? state.board_size * state.board_size : 0;
  state.tile_coord_index.assign(static_cast<size_t>(board_cells), -1);
  for (int index = 0; index < static_cast<int>(state.tiles.size()); ++index) {
    const NativeTile& tile = state.tiles[static_cast<size_t>(index)];
    if (tile.x >= 0 && tile.y >= 0 && tile.x < state.board_size && tile.y < state.board_size) {
      state.tile_coord_index[static_cast<size_t>(tile.y * state.board_size + tile.x)] = index;
    }
  }

  int max_unit_id = 0;
  for (const NativeUnit& unit : state.units) {
    max_unit_id = std::max(max_unit_id, unit.id);
  }
  state.unit_id_index.assign(static_cast<size_t>(max_unit_id + 1), -1);
  for (int index = 0; index < static_cast<int>(state.units.size()); ++index) {
    const NativeUnit& unit = state.units[static_cast<size_t>(index)];
    if (unit.id >= 0) {
      state.unit_id_index[static_cast<size_t>(unit.id)] = index;
    }
  }

  int max_city_id = 0;
  for (const NativeCity& city : state.cities) {
    max_city_id = std::max(max_city_id, city.id);
  }
  state.city_id_index.assign(static_cast<size_t>(max_city_id + 1), -1);
  for (int index = 0; index < static_cast<int>(state.cities.size()); ++index) {
    const NativeCity& city = state.cities[static_cast<size_t>(index)];
    if (city.id >= 0) {
      state.city_id_index[static_cast<size_t>(city.id)] = index;
    }
  }
}

NativeTile* tile_at(NativeGameState& state, int x, int y) {
  if (x >= 0 && y >= 0 && x < state.board_size && y < state.board_size) {
    const int coord_index = y * state.board_size + x;
    if (coord_index >= 0 && coord_index < static_cast<int>(state.tile_coord_index.size())) {
      const int tile_index = state.tile_coord_index[static_cast<size_t>(coord_index)];
      if (tile_index >= 0 && tile_index < static_cast<int>(state.tiles.size())) {
        NativeTile& tile = state.tiles[static_cast<size_t>(tile_index)];
        if (tile.x == x && tile.y == y) {
          return &tile;
        }
      }
    }
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
  if (id >= 0 && id < static_cast<int>(state.unit_id_index.size())) {
    const int index = state.unit_id_index[static_cast<size_t>(id)];
    if (index >= 0 && index < static_cast<int>(state.units.size())) {
      NativeUnit& unit = state.units[static_cast<size_t>(index)];
      if (unit.id == id) {
        return &unit;
      }
    }
  }
  for (NativeUnit& unit : state.units) {
    if (unit.id == id) {
      return &unit;
    }
  }
  return nullptr;
}

const NativeUnit* unit_by_id_const(const NativeGameState& state, int id) {
  if (id >= 0 && id < static_cast<int>(state.unit_id_index.size())) {
    const int index = state.unit_id_index[static_cast<size_t>(id)];
    if (index >= 0 && index < static_cast<int>(state.units.size())) {
      const NativeUnit& unit = state.units[static_cast<size_t>(index)];
      if (unit.id == id) {
        return &unit;
      }
    }
  }
  for (const NativeUnit& unit : state.units) {
    if (unit.id == id) {
      return &unit;
    }
  }
  return nullptr;
}

NativeCity* city_by_id(NativeGameState& state, int id) {
  if (id >= 0 && id < static_cast<int>(state.city_id_index.size())) {
    const int index = state.city_id_index[static_cast<size_t>(id)];
    if (index >= 0 && index < static_cast<int>(state.cities.size())) {
      NativeCity& city = state.cities[static_cast<size_t>(index)];
      if (city.id == id) {
        return &city;
      }
    }
  }
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
  if (!payload_sync_enabled(state)) {
    return;
  }
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
    py::handle actual_value = value;
    const NativeUnit* native_unit = unit_by_id_const(state, unit_id);
    if (native_unit != nullptr && native_unit->tribe_id != state.root_player_id &&
        ((normalized != nullptr && strcmp(normalized, "kills") == 0) ||
         (compact != nullptr && strcmp(compact, "k") == 0))) {
      actual_value = py::int_(0);
    }
    unit[normalized] = actual_value;
    if (compact != nullptr) {
      unit[compact] = actual_value;
    }
    return;
  }
}

void set_city_payload_field(NativeGameState& state, int city_id, const char* normalized, const char* compact, const py::handle& value) {
  if (!payload_sync_enabled(state)) {
    return;
  }
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
  if (!payload_sync_enabled(state)) {
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

void increment_tribe_kills(NativeGameState& state, int tribe_id) {
  NativeTribe* tribe = tribe_by_id(state, tribe_id);
  if (tribe == nullptr) {
    return;
  }
  tribe->kills += 1;
  if (!payload_sync_enabled(state)) {
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
    py::dict payload = py::reinterpret_borrow<py::dict>(item);
    if (read_int(payload, "id", 0) != tribe_id) {
      continue;
    }
    if (payload.contains("kills")) {
      py::handle actual_value = tribe_id == state.root_player_id ? py::int_(tribe->kills) : py::int_(0);
      payload["kills"] = actual_value;
    }
    return;
  }
}

void sync_tile_to_payload(NativeGameState& state, const NativeTile& tile) {
  if (!payload_sync_enabled(state)) {
    return;
  }
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
  int visible_unit_id = tile.unit_id;
  if (tile.unit_id > 0) {
    const NativeUnit* occupant = unit_by_id(state, tile.unit_id);
    if (occupant != nullptr &&
        occupant->tribe_id != state.root_player_id &&
        (occupant->hidden || !tile.visible)) {
      visible_unit_id = 0;
    }
  }
  set_matrix_cell(board, "city", tile.x, tile.y, py::int_(visible_city_id));
  set_matrix_cell(board, "unit", tile.x, tile.y, py::int_(visible_unit_id));
  const bool visible_road = tile.explored && tile.road;
  set_matrix_cell(board, "road", tile.x, tile.y, py::int_(visible_road ? 1 : 0));
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
        out["unit_id"] = visible_unit_id;
        out["road"] = visible_road;
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
  if (!payload_sync_enabled(state)) {
    return;
  }
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
  if (!payload_sync_enabled(state)) {
    return;
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

void append_payload_int_to_list(py::dict owner, const char* key, int value) {
  py::list values = owner.contains(key) && py::isinstance<py::list>(owner[py::str(key)])
      ? py::reinterpret_borrow<py::list>(owner[py::str(key)])
      : py::list();
  values.append(value);
  owner[key] = values;
}

void append_city_payload_unit(NativeGameState& state, int city_id, int unit_id);
int reveal_square_for_root(NativeGameState& state, int cx, int cy, int radius);

void append_extra_unit_payload(NativeGameState& state, int tribe_id, int unit_id) {
  if (!payload_sync_enabled(state)) {
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

void erase_int(std::vector<int>& values, int value) {
  values.erase(std::remove(values.begin(), values.end(), value), values.end());
}

void sync_tribe_city_ids_payload(NativeGameState& state, int tribe_id) {
  if (!payload_sync_enabled(state)) {
    return;
  }
  NativeTribe* tribe = tribe_by_id(state, tribe_id);
  if (tribe == nullptr || !state.observation.contains("tribes") ||
      !py::isinstance<py::list>(state.observation["tribes"])) {
    return;
  }
  py::list ids;
  for (int city_id : tribe->city_ids) {
    ids.append(py::int_(city_id));
  }
  py::list tribes = py::reinterpret_borrow<py::list>(state.observation["tribes"]);
  for (const auto& item : tribes) {
    if (!py::isinstance<py::dict>(item)) {
      continue;
    }
    py::dict payload = py::reinterpret_borrow<py::dict>(item);
    if (read_int(payload, "id", -1) != tribe_id) {
      continue;
    }
    if (payload.contains("cities")) {
      payload["cities"] = ids;
    }
    if (payload.contains("city_ids")) {
      payload["city_ids"] = ids;
    }
    return;
  }
}

void sync_city_units_payload(NativeGameState& state, int city_id) {
  if (!payload_sync_enabled(state)) {
    return;
  }
  NativeCity* city = city_by_id(state, city_id);
  if (city == nullptr || !state.observation.contains("cities") ||
      !py::isinstance<py::list>(state.observation["cities"])) {
    return;
  }
  py::list ids;
  for (int unit_id : city->unit_ids) {
    ids.append(py::int_(unit_id));
  }
  py::list cities = py::reinterpret_borrow<py::list>(state.observation["cities"]);
  for (const auto& item : cities) {
    if (!py::isinstance<py::dict>(item)) {
      continue;
    }
    py::dict payload = py::reinterpret_borrow<py::dict>(item);
    if (read_int(payload, "id", -1) != city_id) {
      continue;
    }
    payload["units"] = ids;
    if (payload.contains("unit_ids")) {
      payload["unit_ids"] = ids;
    }
    return;
  }
}

void move_unit_to_city(NativeGameState& state, NativeUnit& unit, NativeCity& target_city) {
  const int old_city_id = unit.city_id;
  if (NativeCity* old_city = city_by_id(state, old_city_id); old_city != nullptr) {
    erase_int(old_city->unit_ids, unit.id);
    sync_city_units_payload(state, old_city->id);
  }
  if (NativeTribe* tribe = tribe_by_id(state, unit.tribe_id); tribe != nullptr) {
    erase_int(tribe->extra_unit_ids, unit.id);
  }
  erase_int(target_city.unit_ids, unit.id);
  target_city.unit_ids.push_back(unit.id);
  unit.city_id = target_city.id;
  remove_unit_from_owner_lists_payload(state, unit);
  append_city_payload_unit(state, target_city.id, unit.id);
  set_unit_payload_field(state, unit.id, "city_id", "c", py::int_(target_city.id));
}

bool move_last_unit_from_city_to_city(NativeGameState& state, NativeCity& source_city, NativeCity& target_city) {
  if (source_city.unit_ids.empty()) {
    return false;
  }
  const int unit_id = source_city.unit_ids.back();
  NativeUnit* unit = unit_by_id(state, unit_id);
  if (unit == nullptr) {
    return false;
  }
  move_unit_to_city(state, *unit, target_city);
  return true;
}

bool move_one_unit_to_new_city(NativeGameState& state, int tribe_id, NativeCity& target_city) {
  NativeTribe* tribe = tribe_by_id(state, tribe_id);
  if (tribe == nullptr) {
    return false;
  }
  NativeCity* capital = city_by_id(state, tribe->capital_id);
  if (capital != nullptr && capital->id != target_city.id && capital->tribe_id == tribe_id &&
      move_last_unit_from_city_to_city(state, *capital, target_city)) {
    return true;
  }
  for (int city_id : tribe->city_ids) {
    if (city_id == tribe->capital_id || city_id == target_city.id) {
      continue;
    }
    NativeCity* city = city_by_id(state, city_id);
    if (city != nullptr && city->tribe_id == tribe_id &&
        move_last_unit_from_city_to_city(state, *city, target_city)) {
      return true;
    }
  }
  return false;
}

void sync_city_buildings_payload(NativeGameState& state, NativeCity& city) {
  const bool has_embassy = std::any_of(city.buildings.begin(), city.buildings.end(), [](const NativeBuilding& building) {
    return building.type == "EMBASSY";
  });
  if (!has_embassy) {
    if (NativeTile* tile = tile_at(state, city.x, city.y); tile != nullptr) {
      tile->building.clear();
      sync_tile_to_payload(state, *tile);
    }
  }
  py::list buildings;
  for (const NativeBuilding& building : city.buildings) {
    py::dict entry;
    entry["t"] = building.type;
    entry["type"] = building.type;
    entry["x"] = building.x;
    entry["y"] = building.y;
    entry["city"] = building.city_id;
    entry["city_id"] = building.city_id;
    entry["owner"] = building.owner_tribe_id;
    entry["owner_tribe_id"] = building.owner_tribe_id;
    buildings.append(entry);
  }
  set_city_payload_field(state, city.id, "buildings", "b", buildings);
}

void remove_embassies_in_capital(NativeGameState& state, NativeCity& capital) {
  const size_t before = capital.buildings.size();
  capital.buildings.erase(
      std::remove_if(capital.buildings.begin(), capital.buildings.end(), [](const NativeBuilding& building) {
        return building.type == "EMBASSY";
      }),
      capital.buildings.end());
  if (before != capital.buildings.size()) {
    sync_city_buildings_payload(state, capital);
  }
}

void destroy_embassies_between(NativeGameState& state, int tribe_a, int tribe_b) {
  for (NativeCity& city : state.cities) {
    if (!city.capital || (city.tribe_id != tribe_a && city.tribe_id != tribe_b)) {
      continue;
    }
    const int owner_to_remove = city.tribe_id == tribe_a ? tribe_b : tribe_a;
    const size_t before = city.buildings.size();
    city.buildings.erase(
        std::remove_if(city.buildings.begin(), city.buildings.end(), [&](const NativeBuilding& building) {
          return building.type == "EMBASSY" && building.owner_tribe_id == owner_to_remove;
        }),
        city.buildings.end());
    if (before != city.buildings.size()) {
      sync_city_buildings_payload(state, city);
    }
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
      {"ARCHER", 15}, {"CATAPULT", 40}, {"KNIGHT", 40}, {"MIND_BENDER", 25}, {"CLOAK", 40},
      {"DAGGER", 10}, {"RAMMER", 0}, {"SCOUT", 0}, {"BOMBER", 0}, {"SUPERUNIT", 50},
      {"PIRATE", 0}};
  auto it = points.find(type);
  return it == points.end() ? 0 : it->second;
}

double unit_attack(const std::string& type) {
  static const std::map<std::string, double> values = {
      {"WARRIOR", 2}, {"RIDER", 2}, {"DEFENDER", 1}, {"SWORDMAN", 3}, {"SWORDSMAN", 3},
      {"ARCHER", 2}, {"CATAPULT", 4}, {"KNIGHT", 3.5}, {"MIND_BENDER", 0}, {"CLOAK", 2},
      {"DAGGER", 2}, {"RAMMER", 3}, {"SCOUT", 2}, {"BOMBER", 3}, {"SUPERUNIT", 5},
      {"JUGGERNAUT", 4}, {"DINGHY", 2}, {"PIRATE", 2}};
  auto it = values.find(type);
  return it == values.end() ? 2 : it->second;
}

int unit_defence(const std::string& type) {
  static const std::map<std::string, int> values = {
      {"WARRIOR", 2}, {"RIDER", 1}, {"DEFENDER", 3}, {"SWORDMAN", 3}, {"SWORDSMAN", 3},
      {"ARCHER", 1}, {"CATAPULT", 0}, {"KNIGHT", 1}, {"MIND_BENDER", 1}, {"CLOAK", 0},
      {"DAGGER", 2}, {"RAMMER", 3}, {"SCOUT", 1}, {"BOMBER", 2}, {"SUPERUNIT", 4},
      {"JUGGERNAUT", 4}, {"PIRATE", 2}};
  auto it = values.find(type);
  return it == values.end() ? 1 : it->second;
}

int unit_max_hp(const std::string& type) {
  static const std::map<std::string, int> values = {
      {"DEFENDER", 15}, {"SWORDMAN", 15}, {"SWORDSMAN", 15}, {"CLOAK", 5}, {"SUPERUNIT", 40}};
  auto it = values.find(type);
  return it == values.end() ? 10 : it->second;
}

int unit_movement(const std::string& type) {
  static const std::map<std::string, int> values = {
      {"RIDER", 2}, {"KNIGHT", 3}, {"CLOAK", 2}};
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

int building_population_bonus(const std::string& type) {
  if (type == "FARM" || type == "MINE") {
    return 2;
  }
  if (type == "LUMBER_HUT" || type == "PORT" ||
      type == "TEMPLE" || type == "WATER_TEMPLE" ||
      type == "FOREST_TEMPLE" || type == "MOUNTAIN_TEMPLE") {
    return 1;
  }
  if (type == "TOWER_OF_WISDOM") {
    return 3;
  }
  return 0;
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
  if (type == "STARFISH") {
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
  std::string normalized = tech;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  auto it = tiers.find(normalized);
  return it == tiers.end() ? 1 : it->second;
}

bool has_tech(const NativeTribe& tribe, const std::string& tech);

bool is_everything_researched(const NativeTribe& tribe) {
  static const std::vector<std::string> all_techs = {
      "CLIMBING", "FISHING", "HUNTING", "ORGANIZATION", "RIDING", "ARCHERY", "FARMING",
      "FORESTRY", "FREE_SPIRIT", "MEDITATION", "MINING", "ROADS", "RAMMING", "SAILING",
      "STRATEGY", "AQUATISM", "CHIVALRY", "CONSTRUCTION", "DIPLOMACY", "MATHEMATICS",
      "NAVIGATION", "SMITHERY", "SPIRITUALISM", "TRADE", "PHILOSOPHY"};
  for (const std::string& tech : all_techs) {
    if (!has_tech(tribe, tech)) {
      return false;
    }
  }
  return true;
}

void sync_tribe_monuments_payload(NativeGameState& state, const NativeTribe& tribe) {
  py::dict monuments;
  for (const auto& entry : tribe.monuments) {
    monuments[py::str(entry.first)] = py::str(entry.second);
  }
  set_tribe_payload_field(state, tribe.id, "monuments", monuments);
  set_tribe_payload_field(state, tribe.id, "mon", monuments);
}

int next_unit_id(const NativeGameState& state) {
  int max_id = state.actor_id_floor;
  for (const NativeCity& city : state.cities) {
    max_id = std::max(max_id, city.id);
  }
  for (const NativeUnit& unit : state.units) {
    max_id = std::max(max_id, unit.id);
  }
  return max_id + 1;
}

int next_city_id(const NativeGameState& state) {
  int max_id = state.actor_id_floor;
  for (const NativeCity& city : state.cities) {
    max_id = std::max(max_id, city.id);
  }
  for (const NativeUnit& unit : state.units) {
    max_id = std::max(max_id, unit.id);
  }
  return max_id + 1;
}

int inferred_hidden_capital_points_worth(NativeGameState& state, int city_id, int x, int y) {
  constexpr int kCityBorderPoints = 20;
  int assigned_tiles = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      NativeTile* tile = tile_at(state, x + dx, y + dy);
      if (tile == nullptr) {
        continue;
      }
      if (tile->city_id <= 0 || tile->city_id == city_id) {
        ++assigned_tiles;
      }
    }
  }
  return assigned_tiles > 0 ? assigned_tiles * kCityBorderPoints : 180;
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
std::string relationship_between(const NativeGameState& state, int from_tribe, int to_tribe);
void clear_pending_offer(NativeGameState& state, int from_tribe, int to_tribe);
std::string attacked_status_after(const NativeUnit& unit, bool dealt_kill = false);
bool unit_is_fresh(const NativeUnit& unit);
bool city_payload_contains(const NativeGameState& state, int city_id);
bool water_unit_type(const std::string& type);
void preserve_hidden_enemy_action_state(
    const NativeGameState& previous,
    const std::vector<NativeAction>& previous_actions,
    NativeGameState& next);
void infer_hidden_enemy_action_state_from_actions(
    NativeGameState& state,
    const std::vector<NativeAction>& actions);
void preserve_hidden_enemy_visible_actions(
    const NativeGameState& previous,
    const std::vector<NativeAction>& previous_actions,
    NativeGameState& next,
    std::vector<NativeAction>& actions,
    int max_actions);

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
  if (!payload_sync_enabled(state)) {
    return;
  }
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

void infer_hidden_city_center_from_territory(NativeCity& city, const NativeGameState& state) {
  int best_support = -1;
  int best_hidden_neighbors = -1;
  int best_center_distance = 0;
  int best_root_distance = 0;
  int best_x = city.x;
  int best_y = city.y;
  bool have_best = false;
  for (const NativeTile& tile : state.tiles) {
    if (tile.city_id != city.id) {
      continue;
    }
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0) {
          continue;
        }
        const int candidate_x = tile.x + dx;
        const int candidate_y = tile.y + dy;
        const NativeTile* candidate = tile_at(const_cast<NativeGameState&>(state), candidate_x, candidate_y);
        if (candidate == nullptr || candidate->explored) {
          continue;
        }
        int support = 0;
        int hidden_neighbors = 0;
        for (int support_dy = -1; support_dy <= 1; ++support_dy) {
          for (int support_dx = -1; support_dx <= 1; ++support_dx) {
            if (support_dx == 0 && support_dy == 0) {
              continue;
            }
            const NativeTile* neighbor = tile_at(
                const_cast<NativeGameState&>(state),
                candidate_x + support_dx,
                candidate_y + support_dy);
            if (neighbor != nullptr &&
                (neighbor->city_id == city.id || neighbor->territory_city_id == city.id)) {
              ++support;
            }
            if (neighbor != nullptr && !neighbor->explored) {
              ++hidden_neighbors;
            }
          }
        }
        int nearest_root_asset = 0;
        bool have_root_asset = false;
        for (const NativeCity& root_city : state.cities) {
          if (root_city.tribe_id != state.root_player_id) {
            continue;
          }
          const int distance = std::abs(candidate_x - root_city.x) + std::abs(candidate_y - root_city.y);
          nearest_root_asset = have_root_asset ? std::min(nearest_root_asset, distance) : distance;
          have_root_asset = true;
        }
        for (const NativeUnit& root_unit : state.units) {
          if (root_unit.tribe_id != state.root_player_id) {
            continue;
          }
          const int distance = std::abs(candidate_x - root_unit.x) + std::abs(candidate_y - root_unit.y);
          nearest_root_asset = have_root_asset ? std::min(nearest_root_asset, distance) : distance;
          have_root_asset = true;
        }
        const int center_distance =
            std::abs(candidate_x - (state.board_size / 2)) + std::abs(candidate_y - (state.board_size / 2));
        const int root_distance = have_root_asset ? nearest_root_asset : 0;
        if (!have_best ||
            support > best_support ||
            (support == best_support && hidden_neighbors > best_hidden_neighbors) ||
            (support == best_support && hidden_neighbors == best_hidden_neighbors &&
             center_distance < best_center_distance) ||
            (support == best_support && hidden_neighbors == best_hidden_neighbors &&
             center_distance == best_center_distance &&
             root_distance < best_root_distance) ||
            (support == best_support && hidden_neighbors == best_hidden_neighbors &&
             center_distance == best_center_distance &&
             root_distance == best_root_distance &&
             (candidate_y < best_y || (candidate_y == best_y && candidate_x < best_x)))) {
          have_best = true;
          best_support = support;
          best_hidden_neighbors = hidden_neighbors;
          best_center_distance = center_distance;
          best_root_distance = root_distance;
          best_x = candidate_x;
          best_y = candidate_y;
        }
      }
    }
  }
  if (have_best) {
    city.x = best_x;
    city.y = best_y;
  }
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
    if (city.capital && tile_at(state, city.x, city.y) != nullptr &&
        tile_at(state, city.x, city.y)->terrain != "CITY" &&
        tile_at(state, city.x, city.y)->terrain != "VILLAGE") {
      infer_hidden_city_center_from_territory(city, state);
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
    if (unit->tribe_id == next.root_player_id) {
      const int reveal_radius =
          (to->terrain == "MOUNTAIN" || unit->type == "SCOUT" || unit->type == "CLOAK" || unit->type == "DINGHY") ? 2 : 1;
      const int newly_explored = reveal_square_for_root(next, x, y, reveal_radius);
      if (newly_explored != 0) {
        update_tribe_economy(next, unit->tribe_id, 0, newly_explored * 5);
      }
    }
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
  NativeUnit* unit = unit_by_id(next, unit_id);
  if (unit == nullptr || unit->tribe_id < 0) {
    return false;
  }
  if (!unit_is_fresh(*unit)) {
    return false;
  }
  const std::string capture_type = action_string(action, "capture_type", "ct");
  NativeTile* tile = tile_at(next, unit->x, unit->y);
  int target_city_id = action_int(action, "target_city_id", "tc", action_int(action, "city_id", "c", 0));
  if (capture_type == "VILLAGE" || (capture_type.empty() && tile != nullptr && tile->terrain == "VILLAGE")) {
    target_city_id = -1;
  }
  if (target_city_id <= 0 && capture_type == "CITY" && tile != nullptr) {
    target_city_id = tile->city_id;
  }
  NativeCity* city = city_by_id(next, target_city_id);
  if (capture_type == "VILLAGE" || (capture_type.empty() && tile != nullptr && tile->terrain == "VILLAGE")) {
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
    new_city.points_worth = 220;
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
      out["bound"] = 0;
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
    if (NativeCity* created_city = city_by_id(next, new_city.id); created_city != nullptr) {
      move_one_unit_to_new_city(next, unit->tribe_id, *created_city);
    }
    unit->status = "FINISHED";
    set_unit_payload_field(next, unit->id, "status", "s", py::str("FINISHED"));
    update_tribe_economy(next, unit->tribe_id, 0, 220);
    return true;
  }
  if (city == nullptr) {
    return false;
  }
  if (tile == nullptr || tile->terrain != "CITY" || city->x != unit->x || city->y != unit->y) {
    return false;
  }
  if (city->tribe_id == unit->tribe_id || relationship_between(next, unit->tribe_id, city->tribe_id) == "TREATY") {
    return false;
  }
  const int old_city_owner = city->tribe_id;
  const int points = city->points_worth;
  clear_pending_offer(next, unit->tribe_id, old_city_owner);
  clear_pending_offer(next, old_city_owner, unit->tribe_id);
  city->tribe_id = unit->tribe_id;
  city->infiltrated = false;
  unit->status = "FINISHED";
  if (NativeTribe* old_owner = tribe_by_id(next, old_city_owner); old_owner != nullptr) {
    erase_int(old_owner->city_ids, city->id);
  }
  if (NativeTribe* new_owner = tribe_by_id(next, unit->tribe_id); new_owner != nullptr &&
      std::find(new_owner->city_ids.begin(), new_owner->city_ids.end(), city->id) == new_owner->city_ids.end()) {
    new_owner->city_ids.push_back(city->id);
  }
  update_tribe_economy(next, old_city_owner, 0, -points);
  update_tribe_economy(next, unit->tribe_id, 0, points);
  set_relationship(next, unit->tribe_id, old_city_owner, "WAR");
  set_relationship(next, old_city_owner, unit->tribe_id, "WAR");
  destroy_embassies_between(next, unit->tribe_id, old_city_owner);
  if (city->capital) {
    remove_embassies_in_capital(next, *city);
  }
  set_city_payload_field(next, city->id, "tribe_id", "p", py::int_(city->tribe_id));
  set_city_payload_field(next, city->id, "infiltrated", "inf", py::bool_(false));
  sync_tribe_city_ids_payload(next, old_city_owner);
  sync_tribe_city_ids_payload(next, unit->tribe_id);
  move_unit_to_city(next, *unit, *city);
  set_unit_payload_field(next, unit->id, "status", "s", py::str("FINISHED"));
  NativeTribe* previous_owner = tribe_by_id(next, old_city_owner);
  if (previous_owner != nullptr && previous_owner->city_ids.empty()) {
    previous_owner->result = "LOSS";
    set_tribe_payload_field(next, old_city_owner, "res", py::str("LOSS"));
  }
  evaluate_capital_terminal(next);
  return true;
}

bool melee_push_unit_type(const std::string& type) {
  return type == "DEFENDER" || type == "SWORDMAN" || type == "SWORDSMAN" ||
      type == "RIDER" || type == "WARRIOR" || type == "DAGGER" ||
      type == "KNIGHT" || type == "SUPERUNIT" || type == "RAMMER" ||
      type == "JUGGERNAUT" || type == "PIRATE";
}

bool fortifiable_unit_type(const std::string& type) {
  return type == "WARRIOR" || type == "RIDER" || type == "ARCHER" ||
      type == "DEFENDER" || type == "KNIGHT";
}

double terrain_defence_multiplier(NativeGameState& state, const NativeUnit& target) {
  NativeTile* tile = tile_at(state, target.x, target.y);
  if (tile == nullptr) {
    return 1.0;
  }
  NativeTribe* tribe = tribe_by_id(state, target.tribe_id);
  if (tribe == nullptr) {
    return 1.0;
  }
  if (tile->terrain == "CITY") {
    if (!fortifiable_unit_type(target.type)) {
      return 1.0;
    }
    NativeCity* city = city_by_id(state, tile->city_id);
    if (city == nullptr || city->tribe_id != target.tribe_id) {
      return 1.0;
    }
    return city->walls ? 4.0 : 1.5;
  }
  if ((tile->terrain == "MOUNTAIN" && has_tech(*tribe, "CLIMBING")) ||
      ((tile->terrain == "SHALLOW_WATER" || tile->terrain == "DEEP_WATER" || tile->terrain == "WATER") &&
       has_tech(*tribe, "AQUATISM")) ||
      (tile->terrain == "FOREST" && has_tech(*tribe, "ARCHERY"))) {
    return 1.5;
  }
  return 1.0;
}

int attack_damage_against(NativeGameState& state, const NativeUnit& attacker, const NativeUnit& target) {
  const double attacker_hp = attacker.current_hp_exact > 0.0 ? attacker.current_hp_exact : static_cast<double>(attacker.current_hp);
  const double target_hp = target.current_hp_exact > 0.0 ? target.current_hp_exact : static_cast<double>(target.current_hp);
  const double attacker_attack = attacker.attack > 0.0 ? attacker.attack : unit_attack(attacker.type);
  const double target_defence = target.defence > 0.0 ? target.defence : unit_defence(target.type);
  const double attack_force = attacker_attack * (attacker_hp / std::max(1, attacker.max_hp));
  const double defence_force =
      target_defence * (target_hp / std::max(1, target.max_hp)) * terrain_defence_multiplier(state, target);
  const double total_damage = attack_force + defence_force;
  return total_damage <= 0.0
      ? 0
      : static_cast<int>(std::round((attack_force / total_damage) * attacker_attack * 4.5));
}

bool try_push_after_lethal_attack(NativeGameState& next, NativeUnit& attacker, int target_x, int target_y) {
  if (!melee_push_unit_type(attacker.type)) {
    return false;
  }
  NativeTile* from = tile_at(next, attacker.x, attacker.y);
  NativeTile* to = tile_at(next, target_x, target_y);
  if (to == nullptr || to->unit_id > 0) {
    return false;
  }
  if (to->terrain == "MOUNTAIN") {
    NativeTribe* tribe = tribe_by_id(next, attacker.tribe_id);
    if (tribe == nullptr || !has_tech(*tribe, "CLIMBING")) {
      return false;
    }
  }
  if ((to->terrain == "SHALLOW_WATER" || to->terrain == "DEEP_WATER" || to->terrain == "WATER") &&
      !water_unit_type(attacker.type)) {
    return false;
  }
  if (from != nullptr && from->unit_id == attacker.id) {
    from->unit_id = 0;
    sync_tile_to_payload(next, *from);
  }
  to->unit_id = attacker.id;
  sync_tile_to_payload(next, *to);
  attacker.x = target_x;
  attacker.y = target_y;
  set_unit_payload_field(next, attacker.id, "x", nullptr, py::int_(target_x));
  set_unit_payload_field(next, attacker.id, "y", nullptr, py::int_(target_y));
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
  const int target_x = target->x;
  const int target_y = target->y;
  const double attack_force = attacker_attack * (attacker_hp / std::max(1, attacker->max_hp));
  const double defence_force =
      target_defence * (target_hp / std::max(1, target->max_hp)) * terrain_defence_multiplier(next, *target);
  const double total_damage = attack_force + defence_force;
  const int attack_damage = attack_damage_against(next, *attacker, *target);
  const int defence_damage = total_damage <= 0.0
      ? 0
      : static_cast<int>(std::round((defence_force / total_damage) * target_defence * 4.5));
  target->current_hp = std::max(0, target->current_hp - attack_damage);
  target->current_hp_exact = static_cast<double>(target->current_hp);
  set_unit_payload_field(next, target->id, "current_hp", "hp", py::int_(target->current_hp));
  set_unit_payload_field(next, target->id, "current_hp_exact", "hpx", py::float_(target->current_hp_exact));
  if (attacker->type == "BOMBER") {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0) {
          continue;
        }
        NativeTile* splash_tile = tile_at(next, target_x + dx, target_y + dy);
        if (splash_tile == nullptr || splash_tile->unit_id <= 0) {
          continue;
        }
        NativeUnit* splash_target = unit_by_id(next, splash_tile->unit_id);
        if (splash_target == nullptr || splash_target->tribe_id == attacker->tribe_id || splash_target->id == attacker->id) {
          continue;
        }
        const double splash_damage = attack_damage_against(next, *attacker, *splash_target) / 2.0;
        if (splash_damage <= 0.0) {
          continue;
        }
        const double splash_hp = splash_target->current_hp_exact > 0.0
            ? splash_target->current_hp_exact
            : static_cast<double>(splash_target->current_hp);
        if (splash_hp <= splash_damage) {
          update_tribe_economy(next, splash_target->tribe_id, 0, -unit_points(splash_target->type));
          remove_unit_ownership_payload(next, *splash_target);
          mark_unit_removed(next, *splash_target);
          attacker->kills += 1;
          set_unit_payload_field(next, attacker->id, "kills", "k", py::int_(attacker->kills));
          increment_tribe_kills(next, attacker->tribe_id);
        } else {
          splash_target->current_hp_exact = splash_hp - splash_damage;
          splash_target->current_hp = static_cast<int>(std::floor(splash_target->current_hp_exact));
          set_unit_payload_field(next, splash_target->id, "current_hp", "hp", py::int_(splash_target->current_hp));
          set_unit_payload_field(next, splash_target->id, "current_hp_exact", "hpx", py::float_(splash_target->current_hp_exact));
        }
      }
    }
  }
  if (target->current_hp <= 0) {
    update_tribe_economy(next, target->tribe_id, 0, -unit_points(target->type));
    remove_unit_ownership_payload(next, *target);
    mark_unit_removed(next, *target);
    attacker->kills += 1;
    set_unit_payload_field(next, attacker->id, "kills", "k", py::int_(attacker->kills));
    increment_tribe_kills(next, attacker->tribe_id);
    try_push_after_lethal_attack(next, *attacker, target_x, target_y);
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
        target->kills += 1;
        set_unit_payload_field(next, target->id, "kills", "k", py::int_(target->kills));
        increment_tribe_kills(next, target->tribe_id);
      }
    }
  }
  if (attacker->current_hp > 0) {
    attacker->status = attacked_status_after(*attacker, target->current_hp <= 0);
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

void populate_action_scalars_from_payload(NativeAction& action) {
  action.tribe_id = action_int(action, "tribe_id", "p", action.tribe_id);
  action.unit_id = action_int(action, "unit_id", "u", action.unit_id);
  action.city_id = action_int(action, "city_id", "c", action.city_id);
  action.target_unit_id = action_int(action, "target_unit_id", "tu", action.target_unit_id);
  action.target_city_id = action_int(action, "target_city_id", "tc", action.target_city_id);
  action.target_player_id = action_int(action, "target_player_id", "tp", action.target_player_id);
  if (action.target_player_id < 0) {
    action.target_player_id = action_int(action, "target_id", "targetID", action.target_player_id);
  }
  int x = 0;
  int y = 0;
  const bool has_x = read_int(action.payload, "x", -999999) != -999999;
  const bool has_y = read_int(action.payload, "y", -999999) != -999999;
  if (has_x && has_y) {
    action.x = read_int(action.payload, "x", 0);
    action.y = read_int(action.payload, "y", 0);
    action.has_xy = true;
  } else if (action.payload.contains("destination") && !action.payload[py::str("destination")].is_none()) {
    py::handle destination = action.payload[py::str("destination")];
    if (read_int(destination, "x", -999999) != -999999 && read_int(destination, "y", -999999) != -999999) {
      action.x = read_int(destination, "x", 0);
      action.y = read_int(destination, "y", 0);
      action.has_xy = true;
    }
  }
  (void)x;
  (void)y;
  action.unit_type = action_string(action, "unit_type", "ut");
  action.building_type = action_string(action, "building_type", "bt");
  action.resource_type = action_string(action, "resource_type", "rt");
  action.capture_type = action_string(action, "capture_type", "ct");
  action.bonus = action_string(action, "bonus", "b");
  action.tech = action_string(action, "technology", "tech");
  if (action.tech.empty()) {
    action.tech = action_string(action, "tech");
  }
}

void set_generated_actor(NativeAction& action, int tribe_id) {
  action.tribe_id = tribe_id;
#ifndef TRIBES_NATIVE_MCTS_STANDALONE
  action.payload["tribe_id"] = tribe_id;
  action.payload["p"] = tribe_id;
#endif
}

void set_generated_xy(NativeAction& action, int x, int y) {
  action.x = x;
  action.y = y;
  action.has_xy = true;
#ifndef TRIBES_NATIVE_MCTS_STANDALONE
  action.payload["x"] = x;
  action.payload["y"] = y;
#endif
}

void set_generated_target_unit(NativeAction& action, int target_unit_id) {
  action.target_unit_id = target_unit_id;
#ifndef TRIBES_NATIVE_MCTS_STANDALONE
  action.payload["target_unit_id"] = target_unit_id;
  action.payload["tu"] = target_unit_id;
#endif
}

void set_generated_target_city(NativeAction& action, int target_city_id) {
  action.target_city_id = target_city_id;
#ifndef TRIBES_NATIVE_MCTS_STANDALONE
  action.payload["target_city_id"] = target_city_id;
  action.payload["tc"] = target_city_id;
#endif
}

void set_generated_target_player(NativeAction& action, int target_player_id) {
  action.target_player_id = target_player_id;
#ifndef TRIBES_NATIVE_MCTS_STANDALONE
  action.payload["target_player_id"] = target_player_id;
  action.payload["tp"] = target_player_id;
#endif
}

void set_generated_unit_type(NativeAction& action, const std::string& unit_type) {
  action.unit_type = unit_type;
#ifndef TRIBES_NATIVE_MCTS_STANDALONE
  action.payload["unit_type"] = unit_type;
  action.payload["ut"] = unit_type;
#endif
}

void set_generated_building_type(NativeAction& action, const std::string& building_type) {
  action.building_type = building_type;
#ifndef TRIBES_NATIVE_MCTS_STANDALONE
  action.payload["building_type"] = building_type;
  action.payload["bt"] = building_type;
#endif
}

void set_generated_resource_type(NativeAction& action, const std::string& resource_type) {
  action.resource_type = resource_type;
#ifndef TRIBES_NATIVE_MCTS_STANDALONE
  action.payload["resource_type"] = resource_type;
  action.payload["rt"] = resource_type;
#endif
}

void set_generated_capture_type(NativeAction& action, const std::string& capture_type) {
  action.capture_type = capture_type;
#ifndef TRIBES_NATIVE_MCTS_STANDALONE
  action.payload["capture_type"] = capture_type;
  action.payload["ct"] = capture_type;
#endif
}

void set_generated_bonus(NativeAction& action, const std::string& bonus) {
  action.bonus = bonus;
#ifndef TRIBES_NATIVE_MCTS_STANDALONE
  action.payload["bonus"] = bonus;
  action.payload["b"] = bonus;
#endif
}

void set_generated_tech(NativeAction& action, const std::string& tech) {
  action.tech = tech;
#ifndef TRIBES_NATIVE_MCTS_STANDALONE
  action.payload["tech"] = tech;
#endif
}

void init_generated_payload(NativeAction& action) {
#ifndef TRIBES_NATIVE_MCTS_STANDALONE
  action.payload = py::dict();
#else
  (void)action;
#endif
}

void append_generated_action(
    NativeGameState& state,
    std::vector<NativeAction>& actions,
    int max_actions,
    NativeAction action) {
  if (max_actions >= 0 && static_cast<int>(state.legal_action_indexes.size()) >= max_actions) {
    return;
  }
  const int index = static_cast<int>(actions.size());
#ifdef TRIBES_NATIVE_MCTS_STANDALONE
  if (state.generated_action_ids_enabled && action.id.empty()) {
    action.id = "A" + std::to_string(index);
  }
#else
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
#endif
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
  if (state.generated_action_ids_enabled) {
    action.id = "sim:p" + std::to_string(tribe_id) + ":t" + std::to_string(state.tick) + id_suffix;
  }
  action.type = type;
  action.unit_id = unit_id;
  action.city_id = city_id;
  init_generated_payload(action);
  set_generated_actor(action, tribe_id);
  set_generated_xy(action, x, y);
#ifndef TRIBES_NATIVE_MCTS_STANDALONE
  action.payload[position_key] = position_payload(x, y);
#endif
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

bool can_use_road_at(const NativeGameState& state, int tribe_id, const NativeTile& tile) {
  const bool road_like = (tile.road && tile.terrain != "SHALLOW_WATER" && tile.terrain != "DEEP_WATER" &&
                          tile.terrain != "CITY") ||
      tile.terrain == "CITY" || tile.terrain == "VILLAGE";
  if (!road_like) {
    return false;
  }
  if (tile.city_id <= 0) {
    return true;
  }
  const NativeCity* city = city_by_id(const_cast<NativeGameState&>(state), tile.city_id);
  if (city == nullptr) {
    return false;
  }
  return city->tribe_id == tribe_id || relationship_between(state, tribe_id, city->tribe_id) == "TREATY";
}

bool adjacent_visible_enemy_unit(const NativeGameState& state, const NativeUnit& unit, int x, int y) {
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      if (dx == 0 && dy == 0) {
        continue;
      }
      NativeTile* neighbor = tile_at(const_cast<NativeGameState&>(state), x + dx, y + dy);
      if (neighbor == nullptr || neighbor->unit_id <= 0) {
        continue;
      }
      NativeUnit* other = unit_by_id(const_cast<NativeGameState&>(state), neighbor->unit_id);
      if (other != nullptr && other->tribe_id != unit.tribe_id &&
          relationship_between(state, unit.tribe_id, other->tribe_id) != "TREATY") {
        return true;
      }
    }
  }
  return false;
}

std::vector<std::pair<int, int>> reachable_move_targets(const NativeGameState& state, const NativeUnit& unit) {
  std::vector<std::pair<int, int>> targets;
  if (state.board_size <= 0 || water_unit_type(unit.type)) {
    return targets;
  }
  const double max_cost = static_cast<double>(std::max(1, unit.movement));
  const int board_size = state.board_size;
  const int board_cells = board_size * board_size;
  targets.reserve(static_cast<size_t>(board_cells));
  std::vector<double> best(static_cast<size_t>(board_cells), std::numeric_limits<double>::infinity());
  std::vector<std::pair<int, int>> frontier;
  frontier.reserve(static_cast<size_t>(board_cells));
  const std::pair<int, int> start{unit.x, unit.y};
  const auto board_index = [board_size](int x, int y) {
    return y * board_size + x;
  };
  best[static_cast<size_t>(board_index(unit.x, unit.y))] = 0.0;
  frontier.push_back(start);

  for (size_t cursor = 0; cursor < frontier.size(); ++cursor) {
    const auto [from_x, from_y] = frontier[cursor];
    const double cost_from = best[static_cast<size_t>(board_index(from_x, from_y))];
    if (cost_from >= max_cost) {
      continue;
    }
    const NativeTile* from_tile = tile_at(const_cast<NativeGameState&>(state), from_x, from_y);
    const bool on_road = from_tile != nullptr && can_use_road_at(state, unit.tribe_id, *from_tile);
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        if (dx == 0 && dy == 0) {
          continue;
        }
        const int x = from_x + dx;
        const int y = from_y + dy;
        if (x < 0 || y < 0 || x >= state.board_size || y >= state.board_size) {
          continue;
        }
        NativeTile* tile = tile_at(const_cast<NativeGameState&>(state), x, y);
        if (tile == nullptr || !(tile->explored || state.active_player_id != state.root_player_id) ||
            tile->terrain == "DEEP_WATER" || tile->terrain == "WATER") {
          continue;
        }
        if (tile->unit_id > 0) {
          NativeUnit* occupant = unit_by_id(const_cast<NativeGameState&>(state), tile->unit_id);
          if (occupant == nullptr || occupant->tribe_id != unit.tribe_id) {
            continue;
          }
        }
        if (unit.type == "MIND_BENDER" && tile->terrain == "CITY" && tile->city_id > 0) {
          NativeCity* city = city_by_id(const_cast<NativeGameState&>(state), tile->city_id);
          if (city != nullptr && city->tribe_id != unit.tribe_id) {
            continue;
          }
        }
        const bool zone_of_control = unit.type != "CLOAK" && adjacent_visible_enemy_unit(state, unit, x, y);
        if (zone_of_control && tile->unit_id > 0) {
          continue;
        }
        double step_cost = 1.0;
        if ((tile->terrain == "FOREST" || tile->terrain == "MOUNTAIN") && unit.type != "CLOAK") {
          step_cost = cost_from < max_cost ? max_cost - cost_from : max_cost;
        }
        if (unit.type != "CLOAK" && on_road && can_use_road_at(state, unit.tribe_id, *tile)) {
          step_cost = std::max(0.5, step_cost / 2.0);
        }
        if (zone_of_control) {
          step_cost = cost_from < max_cost ? max_cost - cost_from : max_cost;
        }
        const double next_cost = cost_from + step_cost;
        if (std::floor(next_cost) > max_cost) {
          continue;
        }
        const int next_index = board_index(x, y);
        if (next_cost + 1e-9 < best[static_cast<size_t>(next_index)]) {
          best[static_cast<size_t>(next_index)] = next_cost;
          frontier.emplace_back(x, y);
        }
      }
    }
  }

  for (int x = 0; x < board_size; ++x) {
    for (int y = 0; y < board_size; ++y) {
      if ((x == start.first && y == start.second) ||
          !std::isfinite(best[static_cast<size_t>(board_index(x, y))])) {
        continue;
      }
      NativeTile* tile = tile_at(const_cast<NativeGameState&>(state), x, y);
      if (tile != nullptr && tile->unit_id <= 0) {
        targets.emplace_back(x, y);
      }
    }
  }
  return targets;
}

std::string attacked_status_after(const NativeUnit& unit, bool dealt_kill) {
  if (unit.type == "KNIGHT" && dealt_kill) {
    return "ATTACKED";
  }
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
  if (!payload_sync_enabled(state)) {
    return;
  }
  for (const NativeTile& tile : state.tiles) {
    sync_tile_to_payload(state, tile);
  }
}

void append_visible_city_payload(NativeGameState& state, const NativeCity& city) {
  if (!payload_sync_enabled(state)) {
    return;
  }
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
  out["bound"] = city.bound > 0 ? city.bound : (city.capital ? 0 : 1);
  const int visible_points = city.points_worth > 0
      ? city.points_worth
      : (city.capital ? inferred_hidden_capital_points_worth(state, city.id, city.x, city.y) : 0);
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
  if (!payload_sync_enabled(state)) {
    return;
  }
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

bool tile_visible_for_asset(const NativeGameState& state, int x, int y);

bool unit_visible_to_root(const NativeGameState& state, const NativeUnit& unit) {
  if (unit.tribe_id == state.root_player_id) {
    return true;
  }
  return !unit.hidden && tile_visible_for_asset(state, unit.x, unit.y);
}

int reveal_square_for_root(NativeGameState& state, int cx, int cy, int radius) {
  int newly_explored = 0;
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
  return newly_explored;
}

int reveal_from_current_assets(NativeGameState& state) {
  int newly_explored = 0;
  for (const NativeCity& city : state.cities) {
    if (city.tribe_id == state.root_player_id) {
      newly_explored += reveal_square_for_root(state, city.x, city.y, 1);
    }
  }
  for (const NativeUnit& unit : state.units) {
    if (unit.tribe_id == state.root_player_id && unit.current_hp != 0) {
      newly_explored += reveal_square_for_root(state, unit.x, unit.y, unit.type == "SCOUT" || unit.type == "CLOAK" ? 2 : 1);
    }
  }
  return newly_explored;
}

int unit_reveal_radius_on_tile(const NativeGameState& state, const NativeUnit& unit, int x, int y) {
  const NativeTile* tile = tile_at(const_cast<NativeGameState&>(state), x, y);
  if ((tile != nullptr && tile->terrain == "MOUNTAIN") ||
      unit.type == "SCOUT" || unit.type == "CLOAK" || unit.type == "DINGHY") {
    return 2;
  }
  return 1;
}

bool tile_in_reveal_range(int source_x, int source_y, int radius, int x, int y) {
  return std::max(std::abs(x - source_x), std::abs(y - source_y)) <= radius;
}

bool tile_revealed_by_tribe_assets(
    const NativeGameState& state,
    int tribe_id,
    int x,
    int y,
    int excluded_unit_id) {
  for (const NativeCity& city : state.cities) {
    if (city.tribe_id == tribe_id && tile_in_reveal_range(city.x, city.y, 1, x, y)) {
      return true;
    }
  }
  for (const NativeUnit& unit : state.units) {
    if (unit.id == excluded_unit_id || unit.tribe_id != tribe_id || unit.current_hp == 0) {
      continue;
    }
    if (tile_in_reveal_range(unit.x, unit.y, unit_reveal_radius_on_tile(state, unit, unit.x, unit.y), x, y)) {
      return true;
    }
  }
  return false;
}

bool hidden_enemy_explored_memory_contains(
    const NativeGameState& state,
    int tribe_id,
    int x,
    int y) {
  const NativeTile* tile = tile_at(const_cast<NativeGameState&>(state), x, y);
  if (tile == nullptr) {
    return false;
  }
  return std::find(
             tile->hidden_explored_by_tribes.begin(),
             tile->hidden_explored_by_tribes.end(),
             tribe_id) != tile->hidden_explored_by_tribes.end();
}

void mark_hidden_enemy_explored_memory(
    NativeGameState& state,
    int tribe_id,
    int x,
    int y) {
  if (tribe_id == state.root_player_id) {
    return;
  }
  NativeTile* tile = tile_at(state, x, y);
  if (tile == nullptr || hidden_enemy_explored_memory_contains(state, tribe_id, x, y)) {
    return;
  }
  tile->hidden_explored_by_tribes.push_back(tribe_id);
}

void mark_hidden_enemy_unit_reveal_memory(
    NativeGameState& state,
    int tribe_id,
    const NativeUnit& unit,
    int x,
    int y) {
  const int radius = unit_reveal_radius_on_tile(state, unit, x, y);
  for (int tx = x - radius; tx <= x + radius; ++tx) {
    for (int ty = y - radius; ty <= y + radius; ++ty) {
      mark_hidden_enemy_explored_memory(state, tribe_id, tx, ty);
    }
  }
}

void prune_invisible_enemy_units_payload(NativeGameState& state) {
  if (!state.observation.contains("units") || !py::isinstance<py::list>(state.observation["units"])) {
    return;
  }
  py::list input = py::reinterpret_borrow<py::list>(state.observation["units"]);
  py::list output;
  for (const auto& item : input) {
    if (!py::isinstance<py::dict>(item)) {
      output.append(item);
      continue;
    }
    py::dict payload = py::reinterpret_borrow<py::dict>(item);
    const int unit_id = read_int(payload, "id", -1);
    const NativeUnit* unit = unit_by_id_const(state, unit_id);
    if (unit == nullptr || unit->tribe_id == state.root_player_id ||
        (!unit->hidden &&
         tile_revealed_by_tribe_assets(state, state.root_player_id, unit->x, unit->y, -1))) {
      output.append(item);
    }
  }
  state.observation["units"] = output;
}

int estimate_hidden_enemy_move_exploration_score(
    const NativeGameState& previous,
    NativeGameState& next,
    const NativeAction& action) {
  if (previous.active_player_id == previous.root_player_id) {
    return 0;
  }
  const int unit_id = action_int(action, "unit_id", "u", 0);
  const NativeUnit* previous_unit = unit_by_id_const(previous, unit_id);
  const NativeUnit* moved_unit = unit_by_id_const(next, unit_id);
  if (previous_unit == nullptr || moved_unit == nullptr || moved_unit->tribe_id != previous.active_player_id) {
    return 0;
  }

  mark_hidden_enemy_unit_reveal_memory(
      next, previous.active_player_id, *previous_unit, previous_unit->x, previous_unit->y);

  int newly_explored = 0;
  const int new_radius = unit_reveal_radius_on_tile(next, *moved_unit, moved_unit->x, moved_unit->y);
  for (const NativeTile& tile : next.tiles) {
    if (!tile_in_reveal_range(moved_unit->x, moved_unit->y, new_radius, tile.x, tile.y)) {
      continue;
    }
    if (hidden_enemy_explored_memory_contains(previous, previous.active_player_id, tile.x, tile.y) ||
        tile_revealed_by_tribe_assets(previous, previous.active_player_id, tile.x, tile.y, unit_id)) {
      mark_hidden_enemy_explored_memory(next, previous.active_player_id, tile.x, tile.y);
      continue;
    }
    const int old_radius = unit_reveal_radius_on_tile(previous, *previous_unit, previous_unit->x, previous_unit->y);
    if (tile_in_reveal_range(previous_unit->x, previous_unit->y, old_radius, tile.x, tile.y)) {
      mark_hidden_enemy_explored_memory(next, previous.active_player_id, tile.x, tile.y);
      continue;
    }
    mark_hidden_enemy_explored_memory(next, previous.active_player_id, tile.x, tile.y);
    newly_explored += 1;
  }
  return newly_explored * 5;
}

void preserve_hidden_enemy_action_state(
    const NativeGameState& previous,
    const std::vector<NativeAction>& previous_actions,
    NativeGameState& next) {
  if (previous.active_player_id == previous.root_player_id || next.active_player_id != previous.active_player_id) {
    return;
  }
  NativeTribe* next_active_tribe = tribe_by_id(next, next.active_player_id);
  if (next_active_tribe == nullptr) {
    return;
  }

  int minimum_stars_for_known_research = next_active_tribe->stars;
  for (int action_index : previous.legal_action_indexes) {
    if (action_index < 0 || action_index >= static_cast<int>(previous_actions.size())) {
      continue;
    }
    const NativeAction prior = previous_actions[action_index];
    const std::string type = canonical_action_type(prior);
    if (type == "BUILD_ROAD" && !has_tech(*next_active_tribe, "ROADS")) {
      next_active_tribe->researched_tech_ids.push_back(tech_id("ROADS"));
    }
    if (type == "SPAWN") {
      const std::string unit_type = action_string(prior, "unit_type", "ut");
      const std::string required_tech = unit_required_tech(unit_type);
      if (!required_tech.empty() && !has_tech(*next_active_tribe, required_tech)) {
        next_active_tribe->researched_tech_ids.push_back(tech_id(required_tech));
      }
      const int city_id = prior.city_id > 0 ? prior.city_id : action_int(prior, "city_id", "c", 0);
      if (city_id <= 0 || city_by_id(next, city_id) != nullptr) {
        continue;
      }
      const int city_x = action_int(prior, "x", nullptr, 0);
      const int city_y = action_int(prior, "y", nullptr, 0);
      if (tile_at(next, city_x, city_y) == nullptr) {
        continue;
      }
      NativeCity city;
      city.id = city_id;
      city.tribe_id = next.active_player_id;
      city.x = city_x;
      city.y = city_y;
      city.level = 1;
      city.population = 0;
      city.population_need = 2;
      city.production = 2;
      city.capital = next_active_tribe->capital_id == city_id;
      city.walls = false;
      city.infiltrated = false;
      city.bound = city.capital ? 0 : 1;
      city.points_worth = city.capital
          ? inferred_hidden_capital_points_worth(next, city.id, city.x, city.y)
          : 0;
      next.cities.push_back(city);
      if (std::find(next_active_tribe->city_ids.begin(), next_active_tribe->city_ids.end(), city_id) ==
          next_active_tribe->city_ids.end()) {
        next_active_tribe->city_ids.push_back(city_id);
      }
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          NativeTile* tile = tile_at(next, city_x + dx, city_y + dy);
          if (tile == nullptr || tile->explored || tile->city_id > 0) {
            continue;
          }
          tile->city_id = city.id;
          tile->territory_city_id = city.id;
        }
      }
      continue;
    }
    if (type == "RESEARCH_TECH") {
      const std::string tech = action_string(prior, "tech");
      if (tech.empty() || has_tech(*next_active_tribe, tech)) {
        continue;
      }
      minimum_stars_for_known_research = std::max(minimum_stars_for_known_research, tech_cost_for(*next_active_tribe, tech));
    }
  }

  if (minimum_stars_for_known_research > next_active_tribe->stars) {
    next_active_tribe->stars = minimum_stars_for_known_research;
  }
}

void infer_hidden_enemy_action_state_from_actions(
    NativeGameState& state,
    const std::vector<NativeAction>& actions) {
  preserve_hidden_enemy_action_state(state, actions, state);
}

bool actions_match_for_visibility(const NativeAction& left, const NativeAction& right) {
  const std::string left_type = canonical_action_type(left);
  const std::string right_type = canonical_action_type(right);
  if (left_type != right_type) {
    return false;
  }
  if (left_type == "RESEARCH_TECH") {
    return action_string(left, "tech") == action_string(right, "tech");
  }
  if (left_type == "SPAWN") {
    return action_int(left, "city_id", "c", 0) == action_int(right, "city_id", "c", 0) &&
        action_string(left, "unit_type", "ut") == action_string(right, "unit_type", "ut");
  }
  return left.id == right.id;
}

void preserve_hidden_enemy_visible_actions(
    const NativeGameState& previous,
    const std::vector<NativeAction>& previous_actions,
    NativeGameState& next,
    std::vector<NativeAction>& actions,
    int max_actions) {
  if (previous.active_player_id == previous.root_player_id || next.active_player_id != previous.active_player_id) {
    return;
  }
  for (int action_index : previous.legal_action_indexes) {
    if (action_index < 0 || action_index >= static_cast<int>(previous_actions.size())) {
      continue;
    }
    const NativeAction prior = previous_actions[action_index];
    const std::string type = canonical_action_type(prior);
    if (type != "RESEARCH_TECH" && type != "SPAWN") {
      continue;
    }
    bool already_present = false;
    for (int legal_index : next.legal_action_indexes) {
      if (legal_index < 0 || legal_index >= static_cast<int>(actions.size())) {
        continue;
      }
      if (actions_match_for_visibility(actions[legal_index], prior)) {
        already_present = true;
        break;
      }
    }
    if (already_present) {
      continue;
    }
    append_generated_action(next, actions, max_actions, prior);
  }
}

std::pair<int, int> infer_hidden_enemy_capital_spawn_xy(const NativeGameState& state) {
  const NativeTribe* active_tribe = tribe_by_id_const(state, state.active_player_id);
  int max_visible_city_id = 0;
  int latest_root_city_id = 0;
  int latest_root_city_x = 0;
  int latest_root_city_y = 0;
  int root_city_count = 0;
  bool root_capital_at_board_center = false;
  for (const NativeCity& city : state.cities) {
    max_visible_city_id = std::max(max_visible_city_id, city.id);
    if (city.tribe_id == state.root_player_id) {
      ++root_city_count;
      if (city.id > latest_root_city_id) {
        latest_root_city_id = city.id;
        latest_root_city_x = city.x;
        latest_root_city_y = city.y;
      }
      if (city.capital && city.x == state.board_size / 2 && city.y == state.board_size / 2) {
        root_capital_at_board_center = true;
      }
    }
  }
  int hidden_capital_id = active_tribe != nullptr && active_tribe->id != state.root_player_id
      ? active_tribe->capital_id
      : 0;
  if (hidden_capital_id <= max_visible_city_id) {
    for (const NativeTribe& tribe : state.tribes) {
      if (tribe.id != state.root_player_id && tribe.capital_id > max_visible_city_id) {
        hidden_capital_id = tribe.capital_id;
        break;
      }
    }
  }
  if (hidden_capital_id > 0) {
    int best_hint_x = 0;
    int best_hint_y = 0;
    int best_hint_support = -1;
    int best_hint_center_distance = 0;
    int best_hint_root_distance = 0;
    bool have_hint = false;
    for (const NativeTile& tile : state.tiles) {
      if (tile.explored) {
        continue;
      }
      int support = 0;
      int nearest_root_asset = 0;
      bool have_root_asset = false;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0) {
            continue;
          }
          const NativeTile* neighbor = tile_at(const_cast<NativeGameState&>(state), tile.x + dx, tile.y + dy);
          if (neighbor != nullptr &&
              (neighbor->city_id == hidden_capital_id || neighbor->territory_city_id == hidden_capital_id)) {
            ++support;
          }
        }
      }
      if (support <= 0) {
        continue;
      }
      for (const NativeCity& city : state.cities) {
        if (city.tribe_id != state.root_player_id) {
          continue;
        }
        const int distance = std::abs(tile.x - city.x) + std::abs(tile.y - city.y);
        nearest_root_asset = have_root_asset ? std::min(nearest_root_asset, distance) : distance;
        have_root_asset = true;
      }
      for (const NativeUnit& unit : state.units) {
        if (unit.tribe_id != state.root_player_id) {
          continue;
        }
        const int distance = std::abs(tile.x - unit.x) + std::abs(tile.y - unit.y);
        nearest_root_asset = have_root_asset ? std::min(nearest_root_asset, distance) : distance;
        have_root_asset = true;
      }
      const int center_distance = std::abs(tile.x - (state.board_size / 2)) + std::abs(tile.y - (state.board_size / 2));
      const int root_distance = have_root_asset ? nearest_root_asset : 0;
      if (!have_hint ||
          support > best_hint_support ||
          (support == best_hint_support && center_distance < best_hint_center_distance) ||
          (support == best_hint_support && center_distance == best_hint_center_distance &&
           root_distance < best_hint_root_distance) ||
          (support == best_hint_support && center_distance == best_hint_center_distance &&
           root_distance == best_hint_root_distance && (tile.y < best_hint_y || (tile.y == best_hint_y && tile.x < best_hint_x)))) {
        have_hint = true;
        best_hint_support = support;
        best_hint_center_distance = center_distance;
        best_hint_root_distance = root_distance;
        best_hint_x = tile.x;
        best_hint_y = tile.y;
      }
    }
    if (have_hint) {
      return {best_hint_x, best_hint_y};
    }
  }
  if (hidden_capital_id > max_visible_city_id && root_city_count >= 2) {
    NativeTile* inferred = tile_at(
        const_cast<NativeGameState&>(state),
        latest_root_city_x + 2,
        latest_root_city_y + 2);
    if (inferred != nullptr && !inferred->explored) {
      return {inferred->x, inferred->y};
    }
  }
  int best_full_x = 0;
  int best_full_y = 0;
  int best_full_distance = -1;
  int best_full_score = -1;
  int best_any_x = 0;
  int best_any_y = 0;
  int best_any_distance = -1;
  int best_any_score = -1;
  for (const NativeTile& tile : state.tiles) {
    if (tile.explored) {
      continue;
    }
    bool has_full_hidden_city_radius = true;
    for (int dy = -1; dy <= 1 && has_full_hidden_city_radius; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const NativeTile* neighbor = tile_at(const_cast<NativeGameState&>(state), tile.x + dx, tile.y + dy);
        if (neighbor == nullptr || neighbor->explored) {
          has_full_hidden_city_radius = false;
          break;
        }
      }
    }
    int nearest_root_asset = 0;
    bool have_root_asset = false;
    for (const NativeCity& city : state.cities) {
      if (city.tribe_id != state.root_player_id) {
        continue;
      }
      const int distance = std::abs(tile.x - city.x) + std::abs(tile.y - city.y);
      nearest_root_asset = have_root_asset ? std::min(nearest_root_asset, distance) : distance;
      have_root_asset = true;
    }
    for (const NativeUnit& unit : state.units) {
      if (unit.tribe_id != state.root_player_id) {
        continue;
      }
      const int distance = std::abs(tile.x - unit.x) + std::abs(tile.y - unit.y);
      nearest_root_asset = have_root_asset ? std::min(nearest_root_asset, distance) : distance;
      have_root_asset = true;
    }
    const int distance_score = have_root_asset ? nearest_root_asset : tile.x + tile.y;
    const int tie_score = tile.x + tile.y;
    if (distance_score > best_any_distance || (distance_score == best_any_distance && tie_score > best_any_score)) {
      best_any_distance = distance_score;
      best_any_score = tie_score;
      best_any_x = tile.x;
      best_any_y = tile.y;
    }
    if (has_full_hidden_city_radius &&
        (distance_score > best_full_distance || (distance_score == best_full_distance && tie_score > best_full_score))) {
      best_full_distance = distance_score;
      best_full_score = tie_score;
      best_full_x = tile.x;
      best_full_y = tile.y;
    }
  }
  // A hidden capital on the board edge has no complete 3x3 city radius. When
  // the observed capital occupies the board center, prefer the farthest hidden
  // tile over the otherwise-preferred complete-radius candidate so we do not
  // systematically place the unseen capital one tile too far inward.
  if (hidden_capital_id > 0 && root_city_count == 1 && root_capital_at_board_center &&
      best_any_distance > best_full_distance) {
    return {best_any_x, best_any_y};
  }
  if (best_full_distance >= 0) {
    return {best_full_x, best_full_y};
  }
  return {best_any_x, best_any_y};
}

NativeCity* capital_city_for_tribe(NativeGameState& state, int tribe_id);

NativeCity* ensure_hidden_active_enemy_capital_city(NativeGameState& state) {
  if (state.active_player_id == state.root_player_id) {
    return nullptr;
  }
  NativeTribe* active_tribe = tribe_by_id(state, state.active_player_id);
  if (active_tribe == nullptr || active_tribe->capital_id <= 0) {
    return nullptr;
  }
  const bool capital_was_hidden = city_by_id(state, active_tribe->capital_id) == nullptr;
  NativeCity* capital = capital_city_for_tribe(state, state.active_player_id);
  if (capital == nullptr) {
    return nullptr;
  }
  const bool has_active_enemy_units = std::any_of(
      state.units.begin(),
      state.units.end(),
      [&](const NativeUnit& unit) { return unit.tribe_id == state.active_player_id; });
  if (capital_was_hidden && !has_active_enemy_units &&
      relationship_between(state, state.active_player_id, state.root_player_id) == "WAR") {
    for (const std::string& tech : {"ROADS", "STRATEGY"}) {
      if (!has_tech(*active_tribe, tech)) {
        active_tribe->researched_tech_ids.push_back(tech);
      }
    }
    active_tribe->stars = std::max(active_tribe->stars, 5);
  }
  const int x = capital->x;
  const int y = capital->y;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      NativeTile* tile = tile_at(state, x + dx, y + dy);
      if (tile == nullptr || tile->explored || tile->city_id > 0) {
        continue;
      }
      tile->city_id = capital->id;
    }
  }
  return capital;
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
NativeCity* ensure_hidden_embassy_capital_city_for_tribe(NativeGameState& state, int tribe_id);
bool is_action_unlocked(const NativeTribe& tribe, const std::string& action_type);
bool is_building_unlocked(const NativeTribe& tribe, const std::string& building);
bool building_terrain_ok(const std::string& building, const NativeTile& tile);
bool building_feasible_at(const NativeGameState& state, const NativeTribe& tribe, const NativeCity& city, const NativeTile& tile, const std::string& building);
bool city_has_building_at(const NativeCity& city, int x, int y);
bool resource_tech_ok(const NativeTribe& tribe, const std::string& resource);
int road_cost_at(const NativeTile& tile);
bool can_build_road_at(const NativeGameState& state, int tribe_id, const NativeTile& tile);
bool tribe_met(const NativeTribe& tribe, int other_id);
bool can_build_embassy(NativeGameState& state, int owner_id, int host_id);
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
      init_generated_payload(upgrade);
      append_generated_action(state, actions, max_actions, std::move(upgrade));
    }
    if (has_tech(*tribe, "SAILING") && tribe->stars >= unit_cost("SCOUT")) {
      NativeAction upgrade;
      upgrade.type = "UPGRADE_SCOUT";
      upgrade.unit_id = unit.id;
      init_generated_payload(upgrade);
      append_generated_action(state, actions, max_actions, std::move(upgrade));
    }
    if (has_tech(*tribe, "NAVIGATION") && tribe->stars >= unit_cost("BOMBER")) {
      NativeAction upgrade;
      upgrade.type = "UPGRADE_BOMBER";
      upgrade.unit_id = unit.id;
      init_generated_payload(upgrade);
      append_generated_action(state, actions, max_actions, std::move(upgrade));
    }
  } else if (unit.type == "SCOUT" && has_tech(*tribe, "NAVIGATION") && tribe->stars >= unit_cost("BOMBER")) {
    NativeAction upgrade;
    upgrade.type = "UPGRADE_BOMBER";
    upgrade.unit_id = unit.id;
    init_generated_payload(upgrade);
    append_generated_action(state, actions, max_actions, std::move(upgrade));
  }
  if (unit.status == "FINISHED") {
    return;
  }
  NativeTile* current_tile = tile_at(state, unit.x, unit.y);

  if (unit_is_fresh(unit) && current_tile != nullptr && current_tile->resource == "RUINS" &&
      !tribe->city_ids.empty()) {
    NativeAction examine;
    if (state.generated_action_ids_enabled) {
      examine.id = "sim:p" + std::to_string(state.active_player_id) + ":t" + std::to_string(state.tick) +
          ":u" + std::to_string(unit.id) + ":examine";
    }
    examine.type = "EXAMINE";
    examine.unit_id = unit.id;
    init_generated_payload(examine);
    set_generated_actor(examine, state.active_player_id);
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
      if (state.generated_action_ids_enabled) {
        infiltrate.id = "sim:p" + std::to_string(state.active_player_id) + ":t" + std::to_string(state.tick) +
            ":u" + std::to_string(unit.id) + ":infiltrate:c" + std::to_string(city.id);
      }
      infiltrate.type = "INFILTRATE";
      infiltrate.unit_id = unit.id;
      infiltrate.city_id = city.id;
      init_generated_payload(infiltrate);
      set_generated_actor(infiltrate, state.active_player_id);
      set_generated_target_city(infiltrate, city.id);
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
        if (state.generated_action_ids_enabled) {
          attack.id = "sim:p" + std::to_string(state.active_player_id) + ":t" + std::to_string(state.tick) +
              ":u" + std::to_string(unit.id) + ":attack:u" + std::to_string(target->id);
        }
        attack.type = "ATTACK";
        attack.unit_id = unit.id;
        init_generated_payload(attack);
        set_generated_actor(attack, state.active_player_id);
        set_generated_target_unit(attack, target->id);
        append_generated_action(state, actions, max_actions, std::move(attack));
      }
    }
  }

  if (unit_is_fresh(unit) && current_tile != nullptr && (current_tile->terrain == "VILLAGE" || current_tile->terrain == "CITY")) {
    NativeCity* city = current_tile->city_id > 0 ? city_by_id(state, current_tile->city_id) : nullptr;
    if (current_tile->terrain == "VILLAGE" || (city != nullptr && city->tribe_id != unit.tribe_id &&
        relationship_between(state, unit.tribe_id, city->tribe_id) != "TREATY")) {
      const bool village_capture = current_tile->terrain == "VILLAGE";
      NativeAction capture;
      if (state.generated_action_ids_enabled) {
        capture.id = "sim:p" + std::to_string(state.active_player_id) + ":t" + std::to_string(state.tick) +
            ":u" + std::to_string(unit.id) + ":capture";
      }
      capture.type = "CAPTURE";
      capture.unit_id = unit.id;
      capture.city_id = village_capture || city == nullptr ? 0 : city->id;
      init_generated_payload(capture);
      set_generated_actor(capture, state.active_player_id);
      set_generated_target_city(capture, village_capture || city == nullptr ? -1 : city->id);
      set_generated_capture_type(capture, current_tile->terrain);
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
        if (state.generated_action_ids_enabled) {
          convert.id = "sim:p" + std::to_string(state.active_player_id) + ":t" + std::to_string(state.tick) +
              ":u" + std::to_string(unit.id) + ":convert:u" + std::to_string(target->id);
        }
        convert.type = "CONVERT";
        convert.unit_id = unit.id;
        init_generated_payload(convert);
        set_generated_actor(convert, state.active_player_id);
        set_generated_target_unit(convert, target->id);
        append_generated_action(state, actions, max_actions, std::move(convert));
      }
    }
  }

  if (unit.type == "MIND_BENDER" && heal_others_target_exists(state, unit)) {
    NativeAction heal;
    if (state.generated_action_ids_enabled) {
      heal.id = "sim:p" + std::to_string(state.active_player_id) + ":t" + std::to_string(state.tick) +
          ":u" + std::to_string(unit.id) + ":heal_others";
    }
    heal.type = "HEAL_OTHERS";
    heal.unit_id = unit.id;
    init_generated_payload(heal);
    set_generated_actor(heal, state.active_player_id);
    append_generated_action(state, actions, max_actions, std::move(heal));
  }

  if (unit_is_fresh(unit) && has_tech(*tribe, "FREE_SPIRIT")) {
    NativeAction disband;
    disband.type = "DISBAND";
    disband.unit_id = unit.id;
    init_generated_payload(disband);
    append_generated_action(state, actions, max_actions, std::move(disband));
  }

  if (unit.kills >= 3 && !unit.veteran && !water_unit_type(unit.type) &&
      unit.type != "SUPERUNIT" && unit.type != "CLOAK" && unit.type != "DAGGER") {
    NativeAction veteran;
    veteran.type = "MAKE_VETERAN";
    veteran.unit_id = unit.id;
    init_generated_payload(veteran);
    append_generated_action(state, actions, max_actions, std::move(veteran));
  }

  if (unit_can_move(unit)) {
    std::vector<std::pair<int, int>> java_priority_order;
    java_priority_order.reserve(8);
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
      move_targets.reserve(8);
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
      std::set<std::pair<int, int>> emitted_moves;
      for (const auto& target : java_priority_order) {
        emitted_moves.insert(target);
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
      for (const auto& target : reachable_move_targets(state, unit)) {
        if (emitted_moves.count(target) > 0) {
          continue;
        }
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
    if (state.generated_action_ids_enabled) {
      recover.id = "sim:p" + std::to_string(state.active_player_id) + ":t" + std::to_string(state.tick) +
          ":u" + std::to_string(unit.id) + ":recover";
    }
    recover.type = "RECOVER";
    recover.unit_id = unit.id;
    init_generated_payload(recover);
    set_generated_actor(recover, state.active_player_id);
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
      init_generated_payload(build);
      set_generated_actor(build, state.active_player_id);
      set_generated_xy(build, tile->x, tile->y);
      set_generated_building_type(build, building);
      append_generated_action(state, actions, max_actions, std::move(build));
    }
  }

  for (NativeTile* tile : tiles) {
    if (tile->terrain == "FOREST" && tile->building.empty() && tile_in_city_territory(*tile, city) &&
        is_action_unlocked(*tribe, "BURN_FOREST") && stars >= 5) {
      NativeAction burn;
      burn.type = "BURN_FOREST";
      burn.city_id = city.id;
      init_generated_payload(burn);
      set_generated_actor(burn, state.active_player_id);
      set_generated_xy(burn, tile->x, tile->y);
      append_generated_action(state, actions, max_actions, std::move(burn));
    }
  }
  for (NativeTile* tile : tiles) {
    if (tile->terrain == "FOREST" && tile->building.empty() && tile_in_city_territory(*tile, city) &&
        is_action_unlocked(*tribe, "CLEAR_FOREST")) {
      NativeAction clear;
      clear.type = "CLEAR_FOREST";
      clear.city_id = city.id;
      init_generated_payload(clear);
      set_generated_actor(clear, state.active_player_id);
      set_generated_xy(clear, tile->x, tile->y);
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
      init_generated_payload(destroy);
      set_generated_actor(destroy, state.active_player_id);
      set_generated_xy(destroy, tile->x, tile->y);
      append_generated_action(state, actions, max_actions, std::move(destroy));
    }
  }
  for (NativeTile* tile : tiles) {
    if (tile->terrain == "PLAIN" && tile->building.empty() &&
        tile_in_city_territory(*tile, city) && is_action_unlocked(*tribe, "GROW_FOREST") && stars >= 5) {
      NativeAction grow;
      grow.type = "GROW_FOREST";
      grow.city_id = city.id;
      init_generated_payload(grow);
      set_generated_actor(grow, state.active_player_id);
      set_generated_xy(grow, tile->x, tile->y);
      append_generated_action(state, actions, max_actions, std::move(grow));
    }
  }
  for (NativeTile* tile : tiles) {
    if (!tile->resource.empty() && tile_in_city_territory(*tile, city) &&
        resource_tech_ok(*tribe, tile->resource) && stars >= resource_cost(tile->resource)) {
      NativeAction gather;
      gather.type = "RESOURCE_GATHERING";
      gather.city_id = city.id;
      init_generated_payload(gather);
      set_generated_actor(gather, state.active_player_id);
      set_generated_xy(gather, tile->x, tile->y);
      set_generated_resource_type(gather, tile->resource);
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
      init_generated_payload(spawn);
      set_generated_xy(spawn, city.x, city.y);
      set_generated_unit_type(spawn, unit_type);
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
    for (int x = 0; x < state.board_size; ++x) {
      for (int y = 0; y < state.board_size; ++y) {
        NativeTile* tile = tile_at(state, x, y);
        if (tile == nullptr ||
            !can_build_road_at(state, state.active_player_id, *tile) ||
            stars < road_cost_at(*tile)) {
          continue;
        }
        NativeAction road;
        road.type = "BUILD_ROAD";
        init_generated_payload(road);
        set_generated_actor(road, state.active_player_id);
        set_generated_xy(road, tile->x, tile->y);
        append_generated_action(state, actions, max_actions, std::move(road));
      }
    }
  }

  for (int target = 0; target < tribe_count; ++target) {
    if (target == state.active_player_id) {
      continue;
    }
    if (can_build_embassy(state, state.active_player_id, target)) {
      NativeAction embassy;
      embassy.type = "BUILD_EMBASSY";
      init_generated_payload(embassy);
      set_generated_actor(embassy, state.active_player_id);
      set_generated_target_player(embassy, target);
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
    init_generated_payload(research);
    set_generated_actor(research, state.active_player_id);
    set_generated_tech(research, tech);
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
      init_generated_payload(propose);
      set_generated_actor(propose, state.active_player_id);
      set_generated_target_player(propose, target);
      append_generated_action(state, actions, max_actions, std::move(propose));
    }
    if (has_pending_offer(state, target, state.active_player_id) &&
        state.pending_offer_types[static_cast<size_t>(target)][static_cast<size_t>(state.active_player_id)] == "PEACE") {
      NativeAction accept;
      accept.type = "ACCEPT_PEACE";
      init_generated_payload(accept);
      set_generated_actor(accept, state.active_player_id);
      set_generated_target_player(accept, target);
      append_generated_action(state, actions, max_actions, std::move(accept));
    }
    if (is_action_unlocked(*tribe, "PROPOSE_TREATY") &&
        relationship_between(state, state.active_player_id, target) == "PEACE" &&
        !has_pending_offer(state, state.active_player_id, target) &&
        !has_pending_offer(state, target, state.active_player_id)) {
      NativeAction propose;
      propose.type = "PROPOSE_TREATY";
      init_generated_payload(propose);
      set_generated_actor(propose, state.active_player_id);
      set_generated_target_player(propose, target);
      append_generated_action(state, actions, max_actions, std::move(propose));
    }
    if (has_pending_offer(state, target, state.active_player_id) &&
        state.pending_offer_types[static_cast<size_t>(target)][static_cast<size_t>(state.active_player_id)] == "TREATY") {
      NativeAction accept;
      accept.type = "ACCEPT_TREATY";
      init_generated_payload(accept);
      set_generated_actor(accept, state.active_player_id);
      set_generated_target_player(accept, target);
      append_generated_action(state, actions, max_actions, std::move(accept));
    }
    if (relationship_between(state, state.active_player_id, target) == "TREATY") {
      NativeAction cancel;
      cancel.type = "CANCEL_TREATY";
      init_generated_payload(cancel);
      set_generated_actor(cancel, state.active_player_id);
      set_generated_target_player(cancel, target);
      append_generated_action(state, actions, max_actions, std::move(cancel));
    }
  }

  if (state.can_end_turn) {
    NativeAction end_turn;
    if (state.generated_action_ids_enabled) {
      end_turn.id = "sim:p" + std::to_string(state.active_player_id) + ":t" + std::to_string(state.tick) + ":end";
    }
    end_turn.type = "END_TURN";
    init_generated_payload(end_turn);
    set_generated_actor(end_turn, state.active_player_id);
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
    const NativeCity* city = city_by_id(state, city_id);
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
        init_generated_payload(level_up);
        set_generated_actor(level_up, state.active_player_id);
        set_generated_xy(level_up, city->x, city->y);
        set_generated_bonus(level_up, bonus);
        append_generated_action(state, actions, max_actions, std::move(level_up));
      }
      sync_observation_turn_flags(state);
      return;
    }
  }

  state.leveling_up = false;
  state.can_end_turn = true;
  const bool allow_hidden_capital_bootstrap = state.transition_kind == "END_TURN";
  if (allow_hidden_capital_bootstrap) {
    ensure_hidden_active_enemy_capital_city(state);
  }
  regenerate_tribe_actions(state, actions, max_actions);
  for (int city_id : active_tribe->city_ids) {
    const NativeCity* city = city_by_id(state, city_id);
    if (city != nullptr && city->tribe_id == state.active_player_id) {
      regenerate_city_actions(state, actions, max_actions, *city);
    }
  }
  for (const NativeUnit& unit : state.units) {
    regenerate_unit_actions(state, actions, max_actions, unit);
  }
  sync_observation_turn_flags(state);
}

bool try_reuse_legal_actions_after_simple_unit_update(
    const NativeGameState& previous,
    NativeGameState& next,
    const std::vector<NativeAction>& actions,
    const NativeAction& applied,
    const std::string& type,
    int max_actions) {
  if (type != "RECOVER" && type != "MAKE_VETERAN") {
    return false;
  }
  if (max_actions >= 0 && static_cast<int>(previous.legal_action_indexes.size()) >= max_actions) {
    return false;
  }
  const int unit_id = action_int(applied, "unit_id", "u", 0);
  if (unit_id <= 0) {
    return false;
  }

  next.tile_coord_index = previous.tile_coord_index;
  next.unit_id_index = previous.unit_id_index;
  next.city_id_index = previous.city_id_index;
  next.legal_action_indexes.clear();
  next.legal_action_indexes.reserve(previous.legal_action_indexes.size());
  for (int action_index : previous.legal_action_indexes) {
    if (action_index < 0 || action_index >= static_cast<int>(actions.size())) {
      continue;
    }
    const NativeAction& action = actions[static_cast<size_t>(action_index)];
    const int action_unit_id = action_int(action, "unit_id", "u", 0);
    const std::string action_type = canonical_action_type(action);
    if (type == "RECOVER" && action_unit_id == unit_id) {
      continue;
    }
    if (type == "MAKE_VETERAN" && action_unit_id == unit_id &&
        (action_type == "MAKE_VETERAN" || action_type == "RECOVER")) {
      continue;
    }
    next.legal_action_indexes.push_back(action_index);
  }
  if (next.legal_action_indexes.empty()) {
    return false;
  }
  sync_observation_turn_flags(next);
  return true;
}

bool active_player_has_level_up_action(const NativeGameState& state) {
  const NativeTribe* active_tribe = tribe_by_id(const_cast<NativeGameState&>(state), state.active_player_id);
  if (active_tribe == nullptr) {
    return false;
  }
  for (int city_id : active_tribe->city_ids) {
    const NativeCity* city = city_by_id(const_cast<NativeGameState&>(state), city_id);
    if (city != nullptr && city_can_level_up(*city)) {
      return true;
    }
  }
  return false;
}

int active_city_center_id_at(const NativeGameState& state, int x, int y) {
  const NativeTile* tile = tile_at(const_cast<NativeGameState&>(state), x, y);
  if (tile == nullptr || tile->city_id <= 0) {
    return 0;
  }
  const NativeCity* city = city_by_id(const_cast<NativeGameState&>(state), tile->city_id);
  if (city == nullptr || city->tribe_id != state.active_player_id || city->x != x || city->y != y) {
    return 0;
  }
  return city->id;
}

void add_active_city_center_if_present(const NativeGameState& state, int x, int y, std::set<int>& city_ids) {
  const int city_id = active_city_center_id_at(state, x, y);
  if (city_id > 0) {
    city_ids.insert(city_id);
  }
}

bool append_legal_indexes_with_cap(std::vector<int>& out, const std::vector<int>& indexes, int max_actions) {
  for (int action_index : indexes) {
    if (max_actions >= 0 && static_cast<int>(out.size()) >= max_actions) {
      return false;
    }
    out.push_back(action_index);
  }
  return true;
}

std::vector<int> regenerate_city_action_indexes(
    NativeGameState& state,
    std::vector<NativeAction>& actions,
    const NativeCity& city) {
  state.legal_action_indexes.clear();
  regenerate_city_actions(state, actions, -1, city);
  return state.legal_action_indexes;
}

std::vector<int> regenerate_unit_action_indexes(
    NativeGameState& state,
    std::vector<NativeAction>& actions,
    const NativeUnit& unit) {
  state.legal_action_indexes.clear();
  regenerate_unit_actions(state, actions, -1, unit);
  return state.legal_action_indexes;
}

std::vector<int> regenerate_all_active_unit_action_indexes(
    NativeGameState& state,
    std::vector<NativeAction>& actions) {
  std::vector<int> out;
  for (const NativeUnit& unit : state.units) {
    if (unit.tribe_id != state.active_player_id) {
      continue;
    }
    state.legal_action_indexes.clear();
    regenerate_unit_actions(state, actions, -1, unit);
    out.insert(out.end(), state.legal_action_indexes.begin(), state.legal_action_indexes.end());
  }
  return out;
}

bool partially_regenerate_all_unit_actions(
    const NativeGameState& previous,
    NativeGameState& next,
    std::vector<NativeAction>& actions,
    const std::set<int>& dirty_city_ids,
    int max_actions) {
  const NativeTribe* active_tribe = tribe_by_id(next, next.active_player_id);
  if (active_tribe == nullptr || active_player_has_level_up_action(next)) {
    return false;
  }

  std::vector<int> tribe_action_indexes;
  std::map<int, std::vector<int>> city_action_indexes;
  for (int action_index : previous.legal_action_indexes) {
    if (action_index < 0 || action_index >= static_cast<int>(actions.size())) {
      continue;
    }
    const NativeAction& action = actions[static_cast<size_t>(action_index)];
    const int unit_id = action_int(action, "unit_id", "u", 0);
    if (unit_id > 0) {
      continue;
    }
    const int city_id = action_int(action, "city_id", "c", 0);
    if (city_id > 0) {
      city_action_indexes[city_id].push_back(action_index);
    } else {
      tribe_action_indexes.push_back(action_index);
    }
  }

  std::map<int, std::vector<int>> regenerated_city_action_indexes;
  for (int city_id : dirty_city_ids) {
    const NativeCity* city = city_by_id(next, city_id);
    if (city != nullptr && city->tribe_id == next.active_player_id) {
      regenerated_city_action_indexes[city_id] = regenerate_city_action_indexes(next, actions, *city);
    }
  }
  const std::vector<int> regenerated_unit_action_indexes =
      regenerate_all_active_unit_action_indexes(next, actions);

  std::vector<int> merged;
  if (!append_legal_indexes_with_cap(merged, tribe_action_indexes, max_actions)) {
    next.legal_action_indexes = std::move(merged);
    sync_observation_turn_flags(next);
    return true;
  }
  for (int city_id : active_tribe->city_ids) {
    const auto regenerated = regenerated_city_action_indexes.find(city_id);
    if (regenerated != regenerated_city_action_indexes.end()) {
      if (!append_legal_indexes_with_cap(merged, regenerated->second, max_actions)) {
        next.legal_action_indexes = std::move(merged);
        sync_observation_turn_flags(next);
        return true;
      }
      continue;
    }
    const auto preserved = city_action_indexes.find(city_id);
    if (preserved != city_action_indexes.end() &&
        !append_legal_indexes_with_cap(merged, preserved->second, max_actions)) {
      next.legal_action_indexes = std::move(merged);
      sync_observation_turn_flags(next);
      return true;
    }
  }
  append_legal_indexes_with_cap(merged, regenerated_unit_action_indexes, max_actions);
  next.legal_action_indexes = std::move(merged);
  sync_observation_turn_flags(next);
  return true;
}

bool partially_regenerate_single_unit_actions(
    const NativeGameState& previous,
    NativeGameState& next,
    std::vector<NativeAction>& actions,
    int dirty_unit_id,
    int max_actions) {
  const NativeUnit* unit = unit_by_id(next, dirty_unit_id);
  const std::vector<int> regenerated_unit_action_indexes =
      unit == nullptr ? std::vector<int>{} : regenerate_unit_action_indexes(next, actions, *unit);

  bool inserted_dirty_unit_actions = false;
  std::vector<int> merged;
  for (int action_index : previous.legal_action_indexes) {
    if (action_index < 0 || action_index >= static_cast<int>(actions.size())) {
      continue;
    }
    const NativeAction& action = actions[static_cast<size_t>(action_index)];
    const int unit_id = action_int(action, "unit_id", "u", 0);
    if (unit_id == dirty_unit_id) {
      if (!inserted_dirty_unit_actions) {
        append_legal_indexes_with_cap(merged, regenerated_unit_action_indexes, max_actions);
        inserted_dirty_unit_actions = true;
      }
      continue;
    }
    if (max_actions >= 0 && static_cast<int>(merged.size()) >= max_actions) {
      break;
    }
    merged.push_back(action_index);
  }
  if (!inserted_dirty_unit_actions) {
    append_legal_indexes_with_cap(merged, regenerated_unit_action_indexes, max_actions);
  }
  next.legal_action_indexes = std::move(merged);
  sync_observation_turn_flags(next);
  return true;
}

std::vector<int> regenerate_tribe_action_indexes(
    NativeGameState& state,
    std::vector<NativeAction>& actions) {
  state.legal_action_indexes.clear();
  regenerate_tribe_actions(state, actions, -1);
  return state.legal_action_indexes;
}

std::vector<int> regenerate_all_active_city_action_indexes(
    NativeGameState& state,
    std::vector<NativeAction>& actions) {
  std::vector<int> out;
  const NativeTribe* active_tribe = tribe_by_id(state, state.active_player_id);
  if (active_tribe == nullptr) {
    return out;
  }
  for (int city_id : active_tribe->city_ids) {
    const NativeCity* city = city_by_id(state, city_id);
    if (city == nullptr || city->tribe_id != state.active_player_id) {
      continue;
    }
    state.legal_action_indexes.clear();
    regenerate_city_actions(state, actions, -1, *city);
    out.insert(out.end(), state.legal_action_indexes.begin(), state.legal_action_indexes.end());
  }
  return out;
}

bool active_unit_actions_may_depend_on_research_or_stars(
    const NativeGameState& state,
    const std::string& researched_tech) {
  if (researched_tech == "FREE_SPIRIT" || researched_tech == "RAMMING" ||
      researched_tech == "SAILING" || researched_tech == "NAVIGATION" ||
      researched_tech == "DIPLOMACY") {
    return true;
  }
  for (const NativeUnit& unit : state.units) {
    if (unit.tribe_id != state.active_player_id || unit.current_hp <= 0) {
      continue;
    }
    if (unit.type == "RAFT" || unit.type == "SCOUT") {
      return true;
    }
  }
  return false;
}

bool partially_regenerate_research_preserving_unit_actions(
    const NativeGameState& previous,
    NativeGameState& next,
    std::vector<NativeAction>& actions,
    const NativeAction& applied,
    int max_actions,
    int newly_explored) {
  if (max_actions >= 0 && static_cast<int>(previous.legal_action_indexes.size()) >= max_actions) {
    return false;
  }
  if (newly_explored != 0 || previous.leveling_up || next.leveling_up ||
      active_player_has_level_up_action(next)) {
    return false;
  }
  const std::string researched_tech = action_string(applied, "tech");
  if (researched_tech.empty() ||
      active_unit_actions_may_depend_on_research_or_stars(previous, researched_tech)) {
    return false;
  }

  const std::vector<int> regenerated_tribe_action_indexes = regenerate_tribe_action_indexes(next, actions);
  const std::vector<int> regenerated_city_action_indexes = regenerate_all_active_city_action_indexes(next, actions);
  std::vector<int> preserved_unit_action_indexes;
  for (int action_index : previous.legal_action_indexes) {
    if (action_index < 0 || action_index >= static_cast<int>(actions.size())) {
      continue;
    }
    const NativeAction& action = actions[static_cast<size_t>(action_index)];
    if (action_int(action, "unit_id", "u", 0) > 0) {
      preserved_unit_action_indexes.push_back(action_index);
    }
  }

  std::vector<int> merged;
  if (!append_legal_indexes_with_cap(merged, regenerated_tribe_action_indexes, max_actions) ||
      !append_legal_indexes_with_cap(merged, regenerated_city_action_indexes, max_actions)) {
    next.legal_action_indexes = std::move(merged);
    sync_observation_turn_flags(next);
    return true;
  }
  append_legal_indexes_with_cap(merged, preserved_unit_action_indexes, max_actions);
  next.legal_action_indexes = std::move(merged);
  sync_observation_turn_flags(next);
  return true;
}

bool try_partially_regenerate_actions_after_attack_or_move(
    const NativeGameState& previous,
    NativeGameState& next,
    std::vector<NativeAction>& actions,
    const NativeAction& applied,
    const std::string& type,
    int max_actions,
    int newly_explored) {
  if (max_actions >= 0 && static_cast<int>(previous.legal_action_indexes.size()) >= max_actions) {
    return false;
  }
  if (newly_explored != 0 || previous.leveling_up || next.leveling_up) {
    return false;
  }

  if (type == "MOVE" || type == "STEP_MOVE") {
    const int unit_id = action_int(applied, "unit_id", "u", 0);
    const NativeUnit* previous_unit = unit_by_id(const_cast<NativeGameState&>(previous), unit_id);
    if (previous_unit == nullptr) {
      return false;
    }
    const int destination_x = action_int(applied, "x", nullptr, previous_unit->x);
    const int destination_y = action_int(applied, "y", nullptr, previous_unit->y);
    if (destination_x == previous_unit->x && destination_y == previous_unit->y) {
      return false;
    }
    std::set<int> dirty_city_ids;
    add_active_city_center_if_present(next, previous_unit->x, previous_unit->y, dirty_city_ids);
    add_active_city_center_if_present(next, destination_x, destination_y, dirty_city_ids);
    return partially_regenerate_all_unit_actions(previous, next, actions, dirty_city_ids, max_actions);
  }

  if (type != "ATTACK") {
    return false;
  }

  const int attacker_id = action_int(applied, "unit_id", "u", 0);
  const int target_id = action_int(applied, "target_unit_id", "tu", 0);
  const NativeUnit* previous_attacker = unit_by_id(const_cast<NativeGameState&>(previous), attacker_id);
  const NativeUnit* previous_target = unit_by_id(const_cast<NativeGameState&>(previous), target_id);
  if (previous_attacker == nullptr || previous_target == nullptr) {
    return false;
  }
  if (relationship_between(previous, previous_attacker->tribe_id, previous_target->tribe_id) != "WAR") {
    return false;
  }

  const NativeUnit* next_attacker = unit_by_id(next, attacker_id);
  const NativeUnit* next_target = unit_by_id(next, target_id);
  const bool simple_nonlethal =
      next_attacker != nullptr && next_attacker->current_hp > 0 &&
      next_target != nullptr && next_target->current_hp > 0 &&
      next_attacker->x == previous_attacker->x &&
      next_attacker->y == previous_attacker->y;
  if (simple_nonlethal) {
    return partially_regenerate_single_unit_actions(previous, next, actions, attacker_id, max_actions);
  }

  std::set<int> dirty_city_ids;
  add_active_city_center_if_present(next, previous_attacker->x, previous_attacker->y, dirty_city_ids);
  add_active_city_center_if_present(next, previous_target->x, previous_target->y, dirty_city_ids);
  if (next_attacker != nullptr) {
    add_active_city_center_if_present(next, next_attacker->x, next_attacker->y, dirty_city_ids);
  }
  if (next_target != nullptr) {
    add_active_city_center_if_present(next, next_target->x, next_target->y, dirty_city_ids);
  }
  return partially_regenerate_all_unit_actions(previous, next, actions, dirty_city_ids, max_actions);
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
  if (payload_sync_enabled(state)) {
    state.observation["active_player_id"] = player_id;
    state.observation["active"] = player_id;
    state.observation["can_end_turn"] = true;
    state.observation["end"] = true;
  }
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
  set_active_player(next, next_player_id);
  if (next_player_id != next.root_player_id) {
    ensure_hidden_active_enemy_capital_city(next);
  }
  apply_init_turn(next, next_player_id);
  sync_observation_ranking(next);
  sync_observation_turn_flags(next);
}

void append_city_payload_unit(NativeGameState& state, int city_id, int unit_id);
void append_city_payload_building(NativeGameState& state, int city_id, const NativeBuilding& building);
void remove_city_payload_building(NativeGameState& state, int city_id, int x, int y);
void ensure_city_payload_visible(NativeGameState& state, const NativeCity& native_city);
void append_tribe_payload_city(NativeGameState& state, int tribe_id, int city_id);
void sync_relationships_payload(NativeGameState& state);

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
  int target = action_int(action, "target_player_id", "tp", -1);
  if (target < 0) {
    target = action_int(action, "target_id", "targetID", -1);
  }
  if (target < 0 || !can_build_embassy(next, next.active_player_id, target)) {
    return false;
  }
  NativeCity* capital = ensure_hidden_embassy_capital_city_for_tribe(next, target);
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
  set_city_payload_field(next, capital->id, "bound", nullptr, py::int_(0));
  if (city_was_visible) {
    append_city_payload_building(next, capital->id, embassy);
  }
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
  int spawn_unit_id = std::max(next_unit_id(next), city_id + 1);
  for (const NativeTribe& tribe : next.tribes) {
    spawn_unit_id = std::max(spawn_unit_id, tribe.capital_id + 1);
  }
  unit.id = spawn_unit_id;
  unit.tribe_id = city->tribe_id;
  const bool root_owned_spawn = city->tribe_id == next.root_player_id;
  unit.city_id = root_owned_spawn ? city_id : -1;
  unit.x = city->x;
  unit.y = city->y;
  unit.type = type;
  unit.status = "FINISHED";
  unit.max_hp = unit_max_hp(type);
  unit.current_hp = unit.max_hp;
  unit.current_hp_exact = static_cast<double>(unit.max_hp);
  unit.attack = unit_attack(type);
  unit.defence = unit_defence(type);
  unit.movement = unit_movement(type);
  const StaticEvalVariant variant = static_eval_variant();
  unit.range = (variant == StaticEvalVariant::Experimental ||
                variant == StaticEvalVariant::Experimental2 ||
                variant == StaticEvalVariant::ExperimentalTraining)
      ? (type == "CATAPULT" ? 3 : (type == "ARCHER" ? 2 : 1))
      : 1;
  unit.cost = unit_cost(type);
  next.units.insert(next.units.begin(), unit);
  if (root_owned_spawn) {
    city->unit_ids.push_back(unit.id);
  }
  tile->unit_id = unit.id;
  const bool visible_to_root = unit_visible_to_root(next, unit);
  if (root_owned_spawn && visible_to_root) {
    append_city_payload_unit(next, city->id, unit.id);
  }
  sync_tile_to_payload(next, *tile);
  if (payload_sync_enabled(next) &&
      visible_to_root &&
      next.observation.contains("units") && py::isinstance<py::list>(next.observation["units"])) {
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
  if (resource == "STARFISH") {
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

void reveal_known_capital_for_tribe(NativeGameState& state, NativeTribe& observer, int target_tribe_id) {
  if (target_tribe_id == observer.id) {
    return;
  }
  NativeCity* capital = capital_city_for_tribe(state, target_tribe_id);
  if (capital == nullptr) {
    return;
  }
  if (std::find(
          observer.known_capital_tribe_ids.begin(),
          observer.known_capital_tribe_ids.end(),
          target_tribe_id) == observer.known_capital_tribe_ids.end()) {
    observer.known_capital_tribe_ids.push_back(target_tribe_id);
  }
  ensure_city_payload_visible(state, *capital);
  set_city_payload_field(state, capital->id, "bound", nullptr, py::int_(0));
  NativeTile* tile = tile_at(state, capital->x, capital->y);
  if (tile == nullptr) {
    return;
  }
  tile->explored = true;
  tile->city_id = capital->id;
  tile->territory_city_id = capital->id;
  tile->terrain = "CITY";
  tile->road = true;
  sync_tile_to_payload(state, *tile);
}

bool apply_research(NativeGameState& next, const NativeAction& action) {
  const int tribe_id = action_int(action, "tribe_id", "p", next.active_player_id);
  const std::string tech = action_string(action, "tech");
  NativeTribe* tribe = tribe_by_id(next, tribe_id);
  if (tribe == nullptr || tech.empty()) {
    return false;
  }
  const bool had_philosophy = has_tech(*tribe, "PHILOSOPHY");
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
  int cost = 4 + tech_tier(tech) * std::max(1, city_count);
  if (had_philosophy) {
    cost = static_cast<int>(std::ceil(cost * (2.0 / 3.0)));
  }
  update_tribe_economy(next, tribe_id, -cost, tech_tier(tech) * 100);
  if (tech == "DIPLOMACY") {
    std::vector<int> met_tribes = tribe->met_tribe_ids;
    for (int met_tribe_id : met_tribes) {
      reveal_known_capital_for_tribe(next, *tribe, met_tribe_id);
    }
    for (size_t index = 0; index < next.relationships.size(); ++index) {
      if (index < next.relationships[index].size() && next.relationships[index][index].empty()) {
        next.relationships[index][index] = "PEACE";
      }
    }
    sync_relationships_payload(next);
  }
  if (is_everything_researched(*tribe)) {
    tribe->monuments["TOWER_OF_WISDOM"] = "AVAILABLE";
    sync_tribe_monuments_payload(next, *tribe);
  }
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
  const int population_bonus = building_population_bonus(building);
  if (population_bonus > 0) {
    city->population += population_bonus;
    set_city_payload_field(next, city_id, "population", "pop", py::int_(city->population));
    city->points_worth += population_bonus * 5;
    set_city_payload_field(next, city_id, "points_worth", "pts", py::int_(city->points_worth));
    update_tribe_economy(next, city->tribe_id, 0, population_bonus * 5);
  }
  if (building == "FOREST_TEMPLE" || building == "TEMPLE" ||
      building == "WATER_TEMPLE" || building == "MOUNTAIN_TEMPLE") {
    city->points_worth += 100;
    set_city_payload_field(next, city_id, "points_worth", "pts", py::int_(city->points_worth));
    update_tribe_economy(next, city->tribe_id, 0, 100);
  }
  if (building == "TOWER_OF_WISDOM") {
    city->points_worth += 400;
    set_city_payload_field(next, city_id, "points_worth", "pts", py::int_(city->points_worth));
    update_tribe_economy(next, city->tribe_id, 0, 400);
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

bool is_monument_building(const std::string& building) {
  static const std::set<std::string> monuments = {
      "ALTAR_OF_PEACE", "EMPERORS_TOMB", "EYE_OF_GOD", "GATE_OF_POWER",
      "GRAND_BAZAR", "PARK_OF_FORTUNE", "TOWER_OF_WISDOM"};
  return monuments.count(building) > 0;
}

bool monument_build_available(const NativeTribe& tribe, const std::string& building) {
  auto it = tribe.monuments.find(building);
  if (it != tribe.monuments.end()) {
    return it->second == "AVAILABLE";
  }
  return building == "TOWER_OF_WISDOM" && is_everything_researched(tribe);
}

int destroy_population_bonus(const std::string& building) {
  if (building == "FARM" || building == "MINE") {
    return 2;
  }
  if (building == "LUMBER_HUT" || building == "PORT" ||
      building == "TEMPLE" || building == "WATER_TEMPLE" ||
      building == "FOREST_TEMPLE" || building == "MOUNTAIN_TEMPLE") {
    return 1;
  }
  if (is_monument_building(building)) {
    return 3;
  }
  return 0;
}

int destroy_score_delta(const NativeBuilding& building) {
  int delta = -destroy_population_bonus(building.type) * 5;
  if (building.type == "TEMPLE" || building.type == "WATER_TEMPLE" ||
      building.type == "FOREST_TEMPLE" || building.type == "MOUNTAIN_TEMPLE") {
    const int level = std::max(1, building.level);
    delta -= level * 100;
  } else if (is_monument_building(building.type)) {
    delta -= 400;
  }
  return delta;
}

bool apply_destroy(NativeGameState& next, const NativeAction& action) {
  const int x = action_int(action, "x", nullptr, -1);
  const int y = action_int(action, "y", nullptr, -1);
  NativeTile* tile = tile_at(next, x, y);
  if (tile == nullptr) {
    return false;
  }
  NativeCity* city = city_by_id(next, action_int(action, "city_id", "c", tile->city_id));
  if (city == nullptr && tile->city_id > 0) {
    city = city_by_id(next, tile->city_id);
  }
  if (tile->resource == "RUINS") {
    tile->resource.clear();
    sync_tile_to_payload(next, *tile);
    return true;
  }
  if (city != nullptr) {
    auto building_it = std::find_if(city->buildings.begin(), city->buildings.end(), [&](const NativeBuilding& building) {
      return building.x == x && building.y == y && !building.type.empty();
    });
    if (building_it != city->buildings.end()) {
      const NativeBuilding removed = *building_it;
      const int population_delta = -destroy_population_bonus(removed.type);
      const int score_delta = destroy_score_delta(removed);
      city->population += population_delta;
      city->points_worth += score_delta;
      set_city_payload_field(next, city->id, "population", "pop", py::int_(city->population));
      set_city_payload_field(next, city->id, "points_worth", "pts", py::int_(city->points_worth));
      update_tribe_economy(next, city->tribe_id, 0, score_delta);
      city->buildings.erase(building_it);
      remove_city_payload_building(next, city->id, x, y);
    }
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
  next.actor_id_floor = std::max(next.actor_id_floor, unit->id);
  if (payload_sync_enabled(next)) {
    next.observation["_native_actor_id_floor"] = next.actor_id_floor;
  }
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
      tile.city_id = read_int(tile_payload, "city_id", read_int(tile_payload, "city", 0));
      tile.territory_city_id = tile.city_id;
      tile.unit_id = read_int(tile_payload, "unit_id", read_int(tile_payload, "unit", 0));
      state.tiles.push_back(tile);
    }
  }
}

void parse_hidden_enemy_explored_memory(NativeGameState& state) {
  if (!state.observation.contains("_native_enemy_explored") ||
      !py::isinstance<py::dict>(state.observation["_native_enemy_explored"]) ||
      state.board_size <= 0) {
    return;
  }
  py::dict memory = py::reinterpret_borrow<py::dict>(state.observation["_native_enemy_explored"]);
  for (const auto& entry : memory) {
    int tribe_id = -1;
    try {
      tribe_id = std::stoi(py::cast<std::string>(py::str(entry.first)));
    } catch (const std::exception&) {
      continue;
    }
    if (!py::isinstance<py::list>(entry.second)) {
      continue;
    }
    py::list codes = py::reinterpret_borrow<py::list>(entry.second);
    for (const auto& item : codes) {
      int code = -1;
      try {
        code = py::cast<int>(item);
      } catch (const py::cast_error&) {
        continue;
      }
      const int x = code / state.board_size;
      const int y = code % state.board_size;
      mark_hidden_enemy_explored_memory(state, tribe_id, x, y);
    }
  }
}

void parse_units(NativeGameState& state) {
  if (!state.observation.contains("units") || !py::isinstance<py::list>(state.observation["units"])) {
    throw std::runtime_error("Native strict payload parse failure: observation.units must be a list.");
  }
  state.actor_id_floor = std::max(
      state.actor_id_floor,
      read_int(state.observation, "_native_actor_id_floor", read_int(state.observation, "actor_id_floor", 0)));
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
    if (city.unit_ids.empty()) {
      city.unit_ids = read_int_list(payload, "unit_ids");
    }
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
  if (!payload_sync_enabled(state)) {
    return;
  }
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
  if (!payload_sync_enabled(state)) {
    return;
  }
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

void sync_city_unit_ids_from_units(NativeGameState& state) {
  for (NativeCity& city : state.cities) {
    city.unit_ids.clear();
  }
  for (const NativeUnit& unit : state.units) {
    if (unit.city_id <= 0) {
      continue;
    }
    NativeCity* city = city_by_id(state, unit.city_id);
    if (city == nullptr || city->tribe_id != unit.tribe_id) {
      continue;
    }
    if (std::find(city->unit_ids.begin(), city->unit_ids.end(), unit.id) == city->unit_ids.end()) {
      city->unit_ids.push_back(unit.id);
    }
  }
}

py::list remove_building_from_payload_list(const py::handle& value, int x, int y) {
  py::list output;
  if (!py::isinstance<py::list>(value)) {
    return output;
  }
  py::list buildings = py::reinterpret_borrow<py::list>(value);
  for (const auto& item : buildings) {
    if (py::isinstance<py::dict>(item)) {
      py::dict building = py::reinterpret_borrow<py::dict>(item);
      if (read_int(building, "x", 0) == x && read_int(building, "y", 0) == y) {
        continue;
      }
    }
    output.append(item);
  }
  return output;
}

void remove_city_payload_building(NativeGameState& state, int city_id, int x, int y) {
  if (!payload_sync_enabled(state)) {
    return;
  }
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
    if (city.contains("b")) {
      city["b"] = remove_building_from_payload_list(city["b"], x, y);
    }
    if (city.contains("buildings")) {
      city["buildings"] = remove_building_from_payload_list(city["buildings"], x, y);
    }
    return;
  }
}

void ensure_city_payload_visible(NativeGameState& state, const NativeCity& native_city) {
  if (!payload_sync_enabled(state)) {
    return;
  }
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
  city["bound"] = native_city.bound > 0 ? native_city.bound : (native_city.capital ? 0 : 1);
  city["pts"] = native_city.points_worth > 0
      ? native_city.points_worth
      : (native_city.capital
          ? inferred_hidden_capital_points_worth(state, native_city.id, native_city.x, native_city.y)
          : 0);
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
  if (!payload_sync_enabled(state)) {
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

  py::object relationship_rows;
  if (state.observation.contains("rel") && py::isinstance<py::list>(state.observation["rel"])) {
    relationship_rows = state.observation["rel"];
  } else if (state.observation.contains("relationships") && py::isinstance<py::list>(state.observation["relationships"])) {
    relationship_rows = state.observation["relationships"];
  } else {
    return;
  }
  py::list rows = py::reinterpret_borrow<py::list>(relationship_rows);
  for (int y = 0; y < static_cast<int>(py::len(rows)) && y < tribe_count; ++y) {
    if (!py::isinstance<py::list>(rows[y])) {
      continue;
    }
    py::list row = py::reinterpret_borrow<py::list>(rows[y]);
    for (int x = 0; x < static_cast<int>(py::len(row)) && x < tribe_count; ++x) {
      if (row[x].is_none()) {
        state.relationships[static_cast<size_t>(y)][static_cast<size_t>(x)] = "";
      } else {
        const std::string relationship = py::cast<std::string>(py::str(row[x]));
        state.relationships[static_cast<size_t>(y)][static_cast<size_t>(x)] = relationship;
        bool has_target_city_hint = false;
        for (const NativeCity& city : state.cities) {
          if (city.tribe_id == x || (city.capital && [&]() {
                const NativeTribe* target_tribe = tribe_by_id_const(state, x);
                return target_tribe != nullptr && target_tribe->capital_id == city.id;
              }())) {
            has_target_city_hint = true;
            break;
          }
        }
        if (!has_target_city_hint) {
          const NativeTribe* target_tribe = tribe_by_id_const(state, x);
          const int capital_id = target_tribe == nullptr ? -1 : target_tribe->capital_id;
          for (const NativeTile& tile : state.tiles) {
            if (capital_id > 0 && (tile.city_id == capital_id || tile.territory_city_id == capital_id)) {
              has_target_city_hint = true;
              break;
            }
          }
          if (!has_target_city_hint && capital_id > 0) {
            int max_visible_city_id = 0;
            int root_city_count = 0;
            for (const NativeCity& city : state.cities) {
              max_visible_city_id = std::max(max_visible_city_id, city.id);
              if (city.tribe_id == state.root_player_id) {
                ++root_city_count;
              }
            }
            has_target_city_hint = capital_id > max_visible_city_id && root_city_count >= 2;
          }
        }
        if (x != y && !relationship.empty() && has_target_city_hint) {
          NativeTribe* tribe = tribe_by_id(state, y);
          if (tribe != nullptr &&
              std::find(tribe->met_tribe_ids.begin(), tribe->met_tribe_ids.end(), x) == tribe->met_tribe_ids.end()) {
            tribe->met_tribe_ids.push_back(x);
          }
        }
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
  if (!payload_sync_enabled(state)) {
    return;
  }
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
  if (!payload_sync_enabled(state)) {
    return;
  }
  state.observation["active_player_id"] = state.active_player_id;
  state.observation["active"] = state.active_player_id;
  state.observation["can_end_turn"] = state.can_end_turn;
  state.observation["end"] = state.can_end_turn;
  state.observation["leveling_up"] = state.leveling_up;
  state.observation["lvlup"] = state.leveling_up;
  state.observation["tick"] = state.tick;
}

void sync_relationships_payload(NativeGameState& state) {
  if (!payload_sync_enabled(state)) {
    return;
  }
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
  state.observation["relationships"] = rows;
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

void add_met_tribe_if_missing(NativeGameState& state, int owner_id, int target_id) {
  if (owner_id < 0 || target_id < 0 || owner_id == target_id) {
    return;
  }
  NativeTribe* owner = tribe_by_id(state, owner_id);
  if (owner == nullptr) {
    return;
  }
  if (std::find(owner->met_tribe_ids.begin(), owner->met_tribe_ids.end(), target_id) == owner->met_tribe_ids.end()) {
    owner->met_tribe_ids.push_back(target_id);
  }
}

void add_known_capital_tribe_if_missing(NativeGameState& state, int owner_id, int target_id) {
  if (owner_id < 0 || target_id < 0 || owner_id == target_id) {
    return;
  }
  NativeTribe* owner = tribe_by_id(state, owner_id);
  if (owner == nullptr) {
    return;
  }
  if (std::find(owner->known_capital_tribe_ids.begin(), owner->known_capital_tribe_ids.end(), target_id) ==
      owner->known_capital_tribe_ids.end()) {
    owner->known_capital_tribe_ids.push_back(target_id);
  }
}

void infer_pending_offers_from_actions(NativeGameState& state, const std::vector<NativeAction>& actions) {
  std::set<std::pair<std::string, int>> present_targeted_actions;
  for (const NativeAction& action : actions) {
    const std::string type = canonical_action_type(action);
    const int target = action_int(action, "target_player_id", "tp", -1);
    if (target < 0) {
      continue;
    }
    present_targeted_actions.insert({type, target});
    if (type == "BUILD_EMBASSY" || type == "PROPOSE_PEACE" || type == "ACCEPT_PEACE" ||
        type == "PROPOSE_TREATY" || type == "ACCEPT_TREATY" || type == "CANCEL_TREATY") {
      const int owner = action_int(action, "tribe_id", "p", state.active_player_id);
      add_met_tribe_if_missing(state, owner, target);
      if (type == "BUILD_EMBASSY") {
        add_known_capital_tribe_if_missing(state, owner, target);
      }
    }
    if (type == "ACCEPT_PEACE") {
      set_pending_offer(state, target, state.active_player_id, "PEACE");
    } else if (type == "ACCEPT_TREATY") {
      set_pending_offer(state, target, state.active_player_id, "TREATY");
    }
  }
  const NativeTribe* active_tribe = tribe_by_id_const(state, state.active_player_id);
  if (active_tribe == nullptr) {
    return;
  }
  for (const NativeTribe& target_tribe : state.tribes) {
    const int target = target_tribe.id;
    if (target == state.active_player_id) {
      continue;
    }
    if (is_action_unlocked(*active_tribe, "PROPOSE_PEACE") &&
        relationship_between(state, state.active_player_id, target) == "WAR" &&
        present_targeted_actions.find({"PROPOSE_PEACE", target}) == present_targeted_actions.end() &&
        !has_pending_offer(state, target, state.active_player_id)) {
      set_pending_offer(state, state.active_player_id, target, "PEACE");
    }
    if (is_action_unlocked(*active_tribe, "PROPOSE_TREATY") &&
        relationship_between(state, state.active_player_id, target) == "PEACE" &&
        present_targeted_actions.find({"PROPOSE_TREATY", target}) == present_targeted_actions.end() &&
        !has_pending_offer(state, target, state.active_player_id)) {
      set_pending_offer(state, state.active_player_id, target, "TREATY");
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
    const bool hidden_active_enemy_city =
        state.active_player_id != state.root_player_id && city.tribe_id == state.active_player_id;
    if (tile.city_id == city.id && (tile.explored || hidden_active_enemy_city)) {
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
    return monument_build_available(tribe, building);
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
  bool explored_for_actor = tile.explored;
  bool hidden_owned_city_territory = false;
  const NativeCity* hidden_owner_city = nullptr;
  if (!explored_for_actor && state.active_player_id != state.root_player_id && tile.city_id > 0) {
    for (const NativeCity& candidate : state.cities) {
      if (candidate.id == tile.city_id) {
        hidden_owner_city = &candidate;
        break;
      }
    }
    hidden_owned_city_territory = hidden_owner_city != nullptr && hidden_owner_city->tribe_id == tribe_id;
    explored_for_actor = hidden_owned_city_territory;
  }
  if (!explored_for_actor || tile.road) {
    return false;
  }
  if (hidden_owned_city_territory && hidden_owner_city != nullptr &&
      tile.x == hidden_owner_city->x && tile.y == hidden_owner_city->y) {
    return false;
  }
  if (!(tile.terrain == "PLAIN" || tile.terrain == "FOREST" || tile.terrain == "VILLAGE" ||
        (hidden_owned_city_territory && tile.terrain.empty()) ||
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
  NativeTribe* tribe = tribe_by_id(state, tribe_id);
  if (tribe == nullptr || tribe->capital_id <= 0) {
    return nullptr;
  }
  NativeCity* capital = city_by_id(state, tribe->capital_id);
  if (capital != nullptr) {
    capital->tribe_id = tribe_id;
    capital->capital = true;
    if (std::find(tribe->city_ids.begin(), tribe->city_ids.end(), capital->id) == tribe->city_ids.end()) {
      tribe->city_ids.push_back(capital->id);
    }
    return capital;
  }
  const auto [x, y] = infer_hidden_enemy_capital_spawn_xy(state);
  if (tile_at(state, x, y) == nullptr) {
    return nullptr;
  }
  NativeCity city;
  city.id = tribe->capital_id;
  city.tribe_id = tribe_id;
  city.x = x;
  city.y = y;
  city.level = 1;
  city.population = 0;
  city.population_need = 2;
  city.production = 2;
  city.capital = true;
  city.walls = false;
  city.infiltrated = false;
  city.bound = 0;
  city.points_worth = inferred_hidden_capital_points_worth(state, city.id, city.x, city.y);
  state.cities.push_back(city);
  tribe->city_ids.push_back(city.id);
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      NativeTile* tile = tile_at(state, x + dx, y + dy);
      if (tile == nullptr || tile->explored || tile->city_id > 0) {
        continue;
      }
      tile->city_id = city.id;
    }
  }
  return &state.cities.back();
}

std::pair<int, int> infer_hidden_embassy_capital_xy(const NativeGameState& state) {
  int best_x = 0;
  int best_y = 0;
  int best_distance = -1;
  int best_score = -1;
  for (const NativeTile& tile : state.tiles) {
    if (tile.explored) {
      continue;
    }
    bool full_hidden_radius = true;
    for (int dy = -1; dy <= 1 && full_hidden_radius; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const NativeTile* neighbor = tile_at(const_cast<NativeGameState&>(state), tile.x + dx, tile.y + dy);
        if (neighbor == nullptr || neighbor->explored) {
          full_hidden_radius = false;
          break;
        }
      }
    }
    if (!full_hidden_radius) {
      continue;
    }
    int nearest_root_asset = 0;
    bool have_root_asset = false;
    for (const NativeCity& city : state.cities) {
      if (city.tribe_id != state.root_player_id) {
        continue;
      }
      const int distance = std::abs(tile.x - city.x) + std::abs(tile.y - city.y);
      nearest_root_asset = have_root_asset ? std::min(nearest_root_asset, distance) : distance;
      have_root_asset = true;
    }
    for (const NativeUnit& unit : state.units) {
      if (unit.tribe_id != state.root_player_id) {
        continue;
      }
      const int distance = std::abs(tile.x - unit.x) + std::abs(tile.y - unit.y);
      nearest_root_asset = have_root_asset ? std::min(nearest_root_asset, distance) : distance;
      have_root_asset = true;
    }
    const int distance_score = have_root_asset ? nearest_root_asset : tile.x + tile.y;
    const int tie_score = tile.x + tile.y;
    if (distance_score > best_distance || (distance_score == best_distance && tie_score > best_score)) {
      best_distance = distance_score;
      best_score = tie_score;
      best_x = tile.x;
      best_y = tile.y;
    }
  }
  if (best_distance >= 0) {
    return {best_x, best_y};
  }
  return infer_hidden_enemy_capital_spawn_xy(state);
}

NativeCity* ensure_hidden_embassy_capital_city_for_tribe(NativeGameState& state, int tribe_id) {
  NativeTribe* tribe = tribe_by_id(state, tribe_id);
  if (tribe == nullptr) {
    return nullptr;
  }
  int capital_id = tribe->capital_id;
  if (capital_id <= 0) {
    for (const NativeCity& city : state.cities) {
      if (city.tribe_id == tribe_id && city.capital) {
        capital_id = city.id;
        break;
      }
    }
  }
  if (capital_id <= 0) {
    return nullptr;
  }
  NativeCity* capital = city_by_id(state, capital_id);
  if (capital != nullptr) {
    capital->tribe_id = tribe_id;
    capital->capital = true;
    return capital;
  }
  const auto [x, y] = infer_hidden_embassy_capital_xy(state);
  if (tile_at(state, x, y) == nullptr) {
    return nullptr;
  }
  NativeCity city;
  city.id = capital_id;
  city.tribe_id = tribe_id;
  city.x = x;
  city.y = y;
  city.level = 1;
  city.population = 0;
  city.population_need = 2;
  city.production = 2;
  city.capital = true;
  city.walls = false;
  city.infiltrated = false;
  city.bound = 0;
  city.points_worth = inferred_hidden_capital_points_worth(state, city.id, city.x, city.y);
  state.cities.push_back(city);
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      NativeTile* tile = tile_at(state, x + dx, y + dy);
      if (tile == nullptr || tile->explored || tile->city_id > 0) {
        continue;
      }
      tile->city_id = city.id;
    }
  }
  return &state.cities.back();
}

bool can_build_embassy(NativeGameState& state, int owner_id, int host_id) {
  const NativeTribe* owner = tribe_by_id_const(state, owner_id);
  if (owner_id == host_id || owner == nullptr || !is_action_unlocked(*owner, "BUILD_EMBASSY")) {
    return false;
  }
  if (owner->stars < 5 || !tribe_met(*owner, host_id)) {
    return false;
  }
  if (relationship_between(state, owner_id, host_id) == "WAR") {
    return false;
  }
  const NativeTribe* host = tribe_by_id_const(state, host_id);
  const int capital_id = host == nullptr ? -1 : host->capital_id;
  bool capital_known = std::find(owner->known_capital_tribe_ids.begin(), owner->known_capital_tribe_ids.end(), host_id) !=
      owner->known_capital_tribe_ids.end();
  for (const NativeCity& city : state.cities) {
    if ((city.tribe_id == host_id && city.capital) || (capital_id > 0 && city.id == capital_id)) {
      capital_known = true;
      break;
    }
  }
  if (!capital_known && capital_id > 0) {
    for (const NativeTile& tile : state.tiles) {
      if (tile.city_id == capital_id || tile.territory_city_id == capital_id) {
        capital_known = true;
        break;
      }
    }
  }
  if (!capital_known && capital_id > 0) {
    int max_visible_city_id = 0;
    int root_city_count = 0;
    for (const NativeCity& city : state.cities) {
      max_visible_city_id = std::max(max_visible_city_id, city.id);
      if (city.tribe_id == state.root_player_id) {
        ++root_city_count;
      }
    }
    capital_known = capital_id > max_visible_city_id && root_city_count >= 2;
  }
  if (!capital_known) {
    return false;
  }
  const NativeCity* capital = ensure_hidden_embassy_capital_city_for_tribe(state, host_id);
  if (capital == nullptr) {
    return false;
  }
  if (capital->tribe_id != host_id) {
    return false;
  }
  for (const NativeBuilding& building : capital->buildings) {
    if (building.type == "EMBASSY" && building.owner_tribe_id == owner_id) {
      return false;
    }
  }
  return true;
}

std::string build_embassy_failure_reason(const NativeGameState& state, const NativeAction& action) {
  int target = action_int(action, "target_player_id", "tp", -1);
  if (target < 0) {
    target = action_int(action, "target_id", "targetID", -1);
  }
  const int owner_id = state.active_player_id;
  const int action_player = action_int(action, "p", "tribe_id", -1);
  const int action_index = action_int(action, "i", "index", -1);
  const std::string action_context =
      ":active=" + std::to_string(owner_id) +
      ":target=" + std::to_string(target) +
      ":p=" + std::to_string(action_player) +
      ":i=" + std::to_string(action_index);
  if (target < 0) {
    return "missing_target" + action_context;
  }
  if (owner_id == target) {
    return "self_target" + action_context;
  }
  const NativeTribe* owner = tribe_by_id_const(state, owner_id);
  if (owner == nullptr) {
    return "missing_owner:" + std::to_string(owner_id) + action_context;
  }
  if (!is_action_unlocked(*owner, "BUILD_EMBASSY")) {
    return "locked_diplomacy" + action_context;
  }
  if (owner->stars < 5) {
    return "stars:" + std::to_string(owner->stars) + action_context;
  }
  if (!tribe_met(*owner, target)) {
    return "unmet:" + std::to_string(target) + action_context;
  }
  if (relationship_between(state, owner_id, target) == "WAR") {
    return "war:" + std::to_string(target) + action_context;
  }
  const NativeCity* capital = ensure_hidden_embassy_capital_city_for_tribe(const_cast<NativeGameState&>(state), target);
  if (capital == nullptr) {
    return "missing_capital:" + std::to_string(target) + action_context;
  }
  if (capital->tribe_id != target) {
    return "captured_capital:" + std::to_string(capital->id) + action_context;
  }
  for (const NativeBuilding& building : capital->buildings) {
    if (building.type == "EMBASSY" && building.owner_tribe_id == owner_id) {
      return "duplicate_embassy:" + std::to_string(capital->id) + action_context;
    }
  }
  return "unknown" + action_context;
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
  parse_hidden_enemy_explored_memory(state);
  parse_units(state);
  parse_cities(state);
  sync_city_unit_ids_from_units(state);
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
  rebuild_state_indexes(state);
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
#ifdef TRIBES_NATIVE_MCTS_STANDALONE
    populate_action_scalars_from_payload(action);
#endif
    root.state.legal_action_indexes.push_back(static_cast<int>(root.actions.size()));
    root.actions.push_back(action);
  }
  root.state.terminal = root.state.terminal || root.state.legal_action_indexes.empty();
  if (root.state.terminal && root.state.terminal_reason.empty()) {
    root.state.terminal_reason = "no_legal_actions";
  }
  infer_pending_offers_from_actions(root.state, root.actions);
  infer_hidden_enemy_action_state_from_actions(root.state, root.actions);
  return root;
}

NativeGameState apply_action_strict(
    const NativeGameState& state,
    std::vector<NativeAction>& actions,
    int global_action_index,
    int max_actions) {
  g_last_transition_timing = NativeTransitionTiming{};
  auto started_at = TimingClock::now();
  NativeGameState next = copy_transition_state_without_observation(state);
  g_last_transition_timing.state_copy_ms += elapsed_ms(started_at);
  started_at = TimingClock::now();
#ifndef TRIBES_NATIVE_MCTS_STANDALONE
  next.observation = deepish_copy_observation(state.observation);
#endif
  g_last_transition_timing.observation_copy_ms += elapsed_ms(started_at);
  next.terminal_reason.clear();
  next.transition_kind.clear();
  next.terminal_value_known = false;
  next.terminal_value = 0.0;
  next.winner_id = -1;

  if (global_action_index < 0 || global_action_index >= static_cast<int>(actions.size())) {
    throw_transition_error("invalid_action_index:" + std::to_string(global_action_index));
  }

  const NativeAction applied = actions[global_action_index];
  const std::string type = canonical_action_type(applied);
  if (type == "END_TURN") {
    started_at = TimingClock::now();
    apply_end_turn_transition(next);
    evaluate_capital_terminal(next);
    g_last_transition_timing.action_mutation_ms += elapsed_ms(started_at);
    if (next.terminal) {
      return next;
    }
    started_at = TimingClock::now();
    reveal_from_current_assets(next);
    sync_all_tiles_to_payload(next);
    g_last_transition_timing.reveal_sync_ms += elapsed_ms(started_at);
    next.transition_kind = "END_TURN";
    started_at = TimingClock::now();
    rebuild_state_indexes(next);
    regenerate_actions(next, actions, max_actions);
    g_last_transition_timing.regenerate_actions_ms += elapsed_ms(started_at);
    next.terminal = next.legal_action_indexes.empty();
    if (next.terminal) {
      next.terminal_reason = "no_regenerated_actions";
    }
    return next;
  }

  bool applied_ok = false;
  started_at = TimingClock::now();
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
  g_last_transition_timing.action_mutation_ms += elapsed_ms(started_at);

  if (!applied_ok) {
    if (type == "BUILD_EMBASSY") {
      throw_transition_error("unsupported_or_failed_transition:" + type + ":" + build_embassy_failure_reason(next, applied));
    }
    throw_transition_error("unsupported_or_failed_transition:" + type);
  }

  if (next.terminal) {
    return next;
  }
  started_at = TimingClock::now();
  const int newly_explored = reveal_from_current_assets(next);
  if (type == "MOVE" || type == "STEP_MOVE" || type == "ATTACK" || type == "SPAWN" ||
      type == "UPGRADE_RAMMER" || type == "UPGRADE_SCOUT" || type == "UPGRADE_BOMBER") {
    update_tribe_economy(next, next.active_player_id, 0, newly_explored * 5);
  }
  if (type == "MOVE" || type == "STEP_MOVE") {
    const int hidden_enemy_score = estimate_hidden_enemy_move_exploration_score(state, next, applied);
    if (hidden_enemy_score != 0) {
      update_tribe_economy(next, next.active_player_id, 0, hidden_enemy_score);
    }
    if (state.active_player_id != state.root_player_id) {
      prune_invisible_enemy_units_payload(next);
    }
  }
  g_last_transition_timing.reveal_sync_ms += elapsed_ms(started_at);
  if (type == "ATTACK") {
    started_at = TimingClock::now();
    preserve_hidden_enemy_action_state(state, actions, next);
    g_last_transition_timing.hidden_enemy_ms += elapsed_ms(started_at);
  }
  started_at = TimingClock::now();
  sync_all_tiles_to_payload(next);
  g_last_transition_timing.reveal_sync_ms += elapsed_ms(started_at);
  next.transition_kind = type;
  started_at = TimingClock::now();
  if (!try_reuse_legal_actions_after_simple_unit_update(state, next, actions, applied, type, max_actions)) {
    rebuild_state_indexes(next);
    regenerate_actions(next, actions, max_actions);
  }
  g_last_transition_timing.regenerate_actions_ms += elapsed_ms(started_at);
  if (type == "ATTACK") {
    started_at = TimingClock::now();
    preserve_hidden_enemy_visible_actions(state, actions, next, actions, max_actions);
    g_last_transition_timing.hidden_enemy_ms += elapsed_ms(started_at);
  }
  next.terminal = next.legal_action_indexes.empty();
  if (next.terminal) {
    next.terminal_reason = "no_regenerated_actions";
  }
  return next;
}

NativeTransitionTiming last_transition_timing() {
  return g_last_transition_timing;
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
