#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "../third_party/nlohmann/json.hpp"

#ifndef TRIBES_NATIVE_MCTS_STANDALONE
#define TRIBES_NATIVE_MCTS_STANDALONE
#endif
#include "../src/native_mcts.cpp"

using json = nlohmann::json;

namespace {

struct CliConfig {
  int simulations = 64;
  double wall_clock_seconds = 0.0;
  int top_k_actions = 64;
  int max_actions = 512;
  int batch_size = 64;
  double c_puct = 1.5;
  double dirichlet_alpha = 0.3;
  double dirichlet_epsilon = 0.25;
  double root_temperature = 1.0;
  bool sample_action = true;
  bool reuse_tree = false;
  bool progressive_widening = true;
  bool profile_json = false;
  bool profile_timing = false;
  uint64_t seed = 13;
  std::string static_eval_variant = "baseline";
};

py::object py_from_json(const json& value) {
  if (value.is_null()) return py::none();
  if (value.is_boolean()) return py::bool_(value.get<bool>());
  if (value.is_number_integer()) return py::int_(value.get<long long>());
  if (value.is_number_unsigned()) return py::int_(value.get<unsigned long long>());
  if (value.is_number_float()) return py::float_(value.get<double>());
  if (value.is_string()) return py::str(value.get<std::string>());
  if (value.is_array()) {
    py::list out;
    for (const json& item : value) out.append(py_from_json(item));
    return std::move(out);
  }
  if (value.is_object()) {
    py::dict out;
    for (auto it = value.begin(); it != value.end(); ++it) out[py::str(it.key())] = py_from_json(it.value());
    return std::move(out);
  }
  return py::none();
}

json json_from_py(py::handle value) {
  if (value.is_none()) return nullptr;
  if (py::isinstance<py::bool_>(value)) return py::cast<bool>(value);
  if (py::isinstance<py::int_>(value)) return py::cast<long long>(value);
  if (py::isinstance<py::float_>(value)) return py::cast<double>(value);
  if (py::isinstance<py::str>(value)) return py::cast<std::string>(py::str(value));
  if (py::isinstance<py::dict>(value)) {
    json out = json::object();
    py::dict dict = py::reinterpret_borrow<py::dict>(value);
    for (const auto& item : dict) out[py::cast<std::string>(py::str(item.first))] = json_from_py(item.second);
    return out;
  }
  if (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value)) {
    json out = json::array();
    py::list values = py::reinterpret_borrow<py::list>(value);
    for (const auto& item : values) out.push_back(json_from_py(item));
    return out;
  }
  return py::cast<std::string>(py::str(value));
}

int json_int(const json& object, const char* key, int fallback = 0) {
  if (!object.is_object() || !object.contains(key) || object[key].is_null()) return fallback;
  try {
    if (object[key].is_number_integer()) return object[key].get<int>();
    if (object[key].is_number()) return static_cast<int>(object[key].get<double>());
    if (object[key].is_string()) return std::stoi(object[key].get<std::string>());
  } catch (...) {
  }
  return fallback;
}

std::string json_string(const json& object, const char* key, const std::string& fallback = "") {
  if (!object.is_object() || !object.contains(key) || object[key].is_null()) return fallback;
  if (object[key].is_string()) return object[key].get<std::string>();
  return object[key].dump();
}


bool json_truthy(const json& value) {
  if (value.is_null()) return false;
  if (value.is_boolean()) return value.get<bool>();
  if (value.is_number_integer()) return value.get<long long>() != 0;
  if (value.is_number_unsigned()) return value.get<unsigned long long>() != 0;
  if (value.is_number_float()) return value.get<double>() != 0.0;
  if (value.is_string()) {
    const std::string s = value.get<std::string>();
    return !(s.empty() || s == "0" || s == "false" || s == "False" || s == "FALSE");
  }
  if (value.is_array() || value.is_object()) return !value.empty();
  return false;
}

int json_to_int(const json& value, int fallback = 0) {
  try {
    if (value.is_number_integer()) return value.get<int>();
    if (value.is_number_unsigned()) return static_cast<int>(value.get<unsigned long long>());
    if (value.is_number_float()) return static_cast<int>(value.get<double>());
    if (value.is_string()) return std::stoi(value.get<std::string>());
  } catch (...) {
  }
  return fallback;
}

json matrix_value(const json& matrix, int x, int y) {
  if (!matrix.is_array() || y < 0 || y >= static_cast<int>(matrix.size())) return nullptr;
  const json& row = matrix[static_cast<size_t>(y)];
  if (!row.is_array() || x < 0 || x >= static_cast<int>(row.size())) return nullptr;
  return row[static_cast<size_t>(x)];
}

bool has_usable_key(const json& object, const char* key) {
  return object.is_object() && object.contains(key) && !object[key].is_null();
}

void set_default(json& object, const char* key, const json& value) {
  if (!object.is_object()) return;
  if (!object.contains(key)) object[key] = value;
}

json value_or(const json& object, const char* key, const json& fallback = nullptr) {
  if (!object.is_object() || !object.contains(key)) return fallback;
  return object[key];
}

json value_or_any(const json& object, std::initializer_list<const char*> keys, const json& fallback = nullptr) {
  if (!object.is_object()) return fallback;
  for (const char* key : keys) {
    if (object.contains(key)) return object[key];
  }
  return fallback;
}

json normalize_board_json(const json& board) {
  json out = board.is_object() ? board : json::object();
  int size = json_int(out, "size", 0);
  if (!out.contains("tiles") && size > 0) {
    json rows = json::array();
    const json terrain = value_or(out, "terrain", json::array());
    const json resource = value_or(out, "resource", json::array());
    const json building = value_or(out, "building", json::array());
    const json city = value_or(out, "city", json::array());
    const json unit = value_or(out, "unit", json::array());
    const json exp = value_or(out, "exp", json::array());
    const json road = value_or(out, "road", json::array());
    for (int y = 0; y < size; ++y) {
      json row = json::array();
      for (int x = 0; x < size; ++x) {
        json tile = json::object();
        tile["x"] = x;
        tile["y"] = y;
        tile["terrain"] = matrix_value(terrain, x, y);
        tile["resource"] = matrix_value(resource, x, y);
        tile["building"] = matrix_value(building, x, y);
        tile["city_id"] = json_to_int(matrix_value(city, x, y), 0);
        tile["unit_id"] = json_to_int(matrix_value(unit, x, y), 0);
        tile["explored"] = json_truthy(matrix_value(exp, x, y));
        tile["road"] = json_truthy(matrix_value(road, x, y));
        row.push_back(std::move(tile));
      }
      rows.push_back(std::move(row));
    }
    out["tiles"] = std::move(rows);
  }

  json normalized_rows = json::array();
  const json raw_rows = value_or(out, "tiles", json::array());
  if (raw_rows.is_array()) {
    for (int y = 0; y < static_cast<int>(raw_rows.size()); ++y) {
      const json& raw_row = raw_rows[static_cast<size_t>(y)];
      json normalized_row = json::array();
      if (raw_row.is_array()) {
        for (int x = 0; x < static_cast<int>(raw_row.size()); ++x) {
          json tile = raw_row[static_cast<size_t>(x)].is_object() ? raw_row[static_cast<size_t>(x)] : json::object();
          set_default(tile, "x", x);
          set_default(tile, "y", y);
          set_default(tile, "terrain", nullptr);
          set_default(tile, "resource", nullptr);
          set_default(tile, "building", nullptr);
          set_default(tile, "city_id", value_or(tile, "city", 0));
          set_default(tile, "unit_id", value_or(tile, "unit", 0));
          set_default(tile, "explored", json_truthy(value_or(tile, "visible", false)));
          set_default(tile, "visible", json_truthy(value_or(tile, "explored", false)));
          set_default(tile, "road", false);
          set_default(tile, "territory_city_id", value_or_any(tile, {"territory", "territory_city", "city_id"}, 0));
          normalized_row.push_back(std::move(tile));
        }
      }
      normalized_rows.push_back(std::move(normalized_row));
    }
  }
  out["tiles"] = std::move(normalized_rows);
  if (size <= 0) out["size"] = static_cast<int>(out["tiles"].size());
  return out;
}

json normalize_unit_json(const json& unit) {
  json out = unit.is_object() ? unit : json::object();
  set_default(out, "tribe_id", value_or(out, "p", -1));
  set_default(out, "city_id", value_or(out, "c", 0));
  set_default(out, "type", value_or(out, "t", nullptr));
  set_default(out, "current_hp", value_or(out, "hp", 0));
  set_default(out, "current_hp_exact", value_or_any(out, {"current_hp", "hp"}, 0));
  set_default(out, "max_hp", value_or(out, "mhp", 0));
  set_default(out, "kills", value_or(out, "k", 0));
  set_default(out, "is_veteran", value_or(out, "v", false));
  set_default(out, "status", value_or(out, "s", nullptr));
  set_default(out, "is_hidden", value_or(out, "h", false));
  set_default(out, "hidden_enemy_hint", value_or(out, "hint", false));
  set_default(out, "attack", value_or(out, "atk", 0));
  set_default(out, "defence", value_or(out, "def", 0));
  set_default(out, "movement", value_or(out, "mov", 0));
  set_default(out, "range", value_or(out, "r", 0));
  set_default(out, "tribe_id", -1);
  set_default(out, "city_id", 0);
  set_default(out, "type", nullptr);
  set_default(out, "current_hp", 0);
  set_default(out, "current_hp_exact", value_or(out, "current_hp", 0));
  set_default(out, "max_hp", 0);
  set_default(out, "kills", 0);
  set_default(out, "is_veteran", false);
  set_default(out, "status", nullptr);
  set_default(out, "is_hidden", false);
  set_default(out, "hidden_at_turn_start", false);
  set_default(out, "hidden_enemy_hint", false);
  set_default(out, "attack", 0);
  set_default(out, "defence", 0);
  set_default(out, "movement", 0);
  set_default(out, "range", 0);
  return out;
}

json normalize_city_json(const json& city) {
  json out = city.is_object() ? city : json::object();
  set_default(out, "tribe_id", value_or(out, "p", -1));
  set_default(out, "level", value_or(out, "lvl", 0));
  set_default(out, "population", value_or(out, "pop", 0));
  set_default(out, "population_need", value_or(out, "need", 0));
  set_default(out, "production", value_or(out, "prod", 0));
  set_default(out, "is_capital", value_or(out, "cap", false));
  set_default(out, "has_walls", value_or(out, "wall", false));
  set_default(out, "points_worth", value_or(out, "pts", 0));
  set_default(out, "infiltrated", value_or(out, "inf", false));
  set_default(out, "buildings", value_or(out, "b", json::array()));
  json buildings = json::array();
  if (out["buildings"].is_array()) {
    for (const json& raw_building : out["buildings"]) {
      if (!raw_building.is_object()) continue;
      json building = raw_building;
      set_default(building, "type", value_or(building, "t", nullptr));
      buildings.push_back(std::move(building));
    }
  }
  out["buildings"] = std::move(buildings);
  set_default(out, "tribe_id", -1);
  set_default(out, "level", 0);
  set_default(out, "population", 0);
  set_default(out, "population_need", 0);
  set_default(out, "production", 0);
  set_default(out, "is_capital", false);
  set_default(out, "has_walls", false);
  set_default(out, "bound", 0);
  set_default(out, "points_worth", 0);
  set_default(out, "infiltrated", false);
  set_default(out, "unit_ids", json::array());
  set_default(out, "buildings", json::array());
  return out;
}

json normalize_tribe_json(const json& tribe) {
  json out = tribe.is_object() ? tribe : json::object();
  set_default(out, "result", value_or(out, "res", nullptr));
  set_default(out, "researched_tech_ids", value_or(out, "tech", json::array()));
  set_default(out, "capital_id", value_or(out, "cap", 0));
  set_default(out, "researched_tech_ids", json::array());
  set_default(out, "capital_id", 0);
  set_default(out, "city_ids", json::array());
  set_default(out, "extra_unit_ids", json::array());
  set_default(out, "connected_city_ids", json::array());
  set_default(out, "met_tribe_ids", json::array());
  set_default(out, "known_capital_tribe_ids", json::array());
  set_default(out, "discovered_lighthouses", json::array());
  set_default(out, "pacifist_count", 0);
  set_default(out, "units_disabled_next_turn", false);
  set_default(out, "monuments", json::object());
  return out;
}

json normalize_action_json(const json& action) {
  json out = action.is_object() ? action : json::object();
  set_default(out, "id", "A" + std::to_string(json_int(out, "i", 0)));
  set_default(out, "type", value_or(out, "t", nullptr));
  if (out.contains("type") && out["type"].is_string()) {
    const std::string type = out["type"].get<std::string>();
    if (type == "GATHER") out["type"] = "RESOURCE_GATHERING";
    else if (type == "RESEARCH") out["type"] = "RESEARCH_TECH";
  }
  set_default(out, "unit_id", value_or(out, "u", 0));
  set_default(out, "city_id", value_or(out, "c", 0));
  set_default(out, "tribe_id", value_or(out, "p", 0));
  set_default(out, "target_unit_id", value_or(out, "tu", 0));
  set_default(out, "target_city_id", value_or(out, "tc", 0));
  set_default(out, "target_player_id", value_or(out, "tp", -1));
  set_default(out, "unit_type", value_or(out, "ut", nullptr));
  set_default(out, "building_type", value_or(out, "bt", nullptr));
  set_default(out, "resource_type", value_or(out, "rt", nullptr));
  set_default(out, "capture_type", value_or(out, "ct", nullptr));
  set_default(out, "bonus", value_or(out, "b", nullptr));
  if (out.contains("x") && out.contains("y")) {
    json pos = json{{"x", out["x"]}, {"y", out["y"]}};
    const std::string type = json_string(out, "type");
    if (type == "MOVE") set_default(out, "destination", pos);
    else if (type == "BUILD_ROAD") set_default(out, "position", pos);
    else set_default(out, "target_pos", pos);
  }
  return out;
}

bool board_tiles_look_expanded(const json& board) {
  if (!board.is_object() || !board.contains("tiles") || !board["tiles"].is_array() || board["tiles"].empty()) {
    return false;
  }
  const json& first_row = board["tiles"][0];
  if (!first_row.is_array() || first_row.empty() || !first_row[0].is_object()) {
    return false;
  }
  const json& tile = first_row[0];
  return tile.contains("x") && tile.contains("y") && tile.contains("terrain") &&
      tile.contains("city_id") && tile.contains("unit_id") && tile.contains("explored");
}

bool entries_have_required_keys(const json& values, std::initializer_list<const char*> keys) {
  if (!values.is_array()) {
    return false;
  }
  for (const json& value : values) {
    if (!value.is_object()) {
      return false;
    }
    for (const char* key : keys) {
      if (!value.contains(key)) {
        return false;
      }
    }
  }
  return true;
}

bool message_looks_normalized(const json& message) {
  if (!message.is_object() || !message.contains("observation") || !message["observation"].is_object()) {
    return false;
  }
  const json& observation = message["observation"];
  if (!observation.contains("board") || !board_tiles_look_expanded(observation["board"])) {
    return false;
  }
  if (!entries_have_required_keys(value_or(observation, "units", json::array()), {"id", "tribe_id", "city_id", "type", "current_hp", "max_hp", "status"})) {
    return false;
  }
  if (!entries_have_required_keys(value_or(observation, "cities", json::array()), {"id", "tribe_id", "x", "y", "level", "population", "population_need", "production"})) {
    return false;
  }
  if (!entries_have_required_keys(value_or(observation, "tribes", json::array()), {"id", "stars", "score", "researched_tech_ids"})) {
    return false;
  }
  const json& actions = value_or(message, "actions", json::array());
  if (!actions.is_array()) {
    return false;
  }
  for (const json& action : actions) {
    if (!action.is_object() || !action.contains("id") || !action.contains("type")) {
      return false;
    }
  }
  return true;
}

json normalize_cli_message(const json& message) {
  if (message_looks_normalized(message)) {
    return message;
  }
  json out = message.is_object() ? message : json::object();
  if (!out.contains("observation") && out.contains("obs")) out["observation"] = out["obs"];
  if (!out.contains("forward_model") && out.contains("fm")) out["forward_model"] = out["fm"];
  if (out.contains("observation") && out["observation"].is_object()) {
    json observation = out["observation"];
    set_default(observation, "active_player_id", value_or(observation, "active", value_or(out, "player_id", 0)));
    set_default(observation, "can_end_turn", value_or(observation, "end", false));
    set_default(observation, "leveling_up", value_or(observation, "lvlup", false));
    set_default(observation, "ranking", value_or(observation, "rank", json::array()));
    observation["board"] = normalize_board_json(value_or(observation, "board", json::object()));

    json units = json::array();
    const json raw_units = value_or(observation, "units", json::array());
    if (raw_units.is_array()) {
      for (const json& unit : raw_units) units.push_back(normalize_unit_json(unit));
    }
    observation["units"] = std::move(units);

    json cities = json::array();
    const json raw_cities = value_or(observation, "cities", json::array());
    if (raw_cities.is_array()) {
      for (const json& city : raw_cities) cities.push_back(normalize_city_json(city));
    }
    observation["cities"] = std::move(cities);

    json tribes = json::array();
    const json raw_tribes = value_or(observation, "tribes", json::array());
    if (raw_tribes.is_array()) {
      for (const json& tribe : raw_tribes) tribes.push_back(normalize_tribe_json(tribe));
    }
    observation["tribes"] = std::move(tribes);
    out["observation"] = std::move(observation);
  }

  json actions = json::array();
  const json raw_actions = value_or(out, "actions", json::array());
  if (raw_actions.is_array()) {
    for (const json& action : raw_actions) actions.push_back(normalize_action_json(action));
  }
  out["actions"] = std::move(actions);
  return out;
}

std::vector<json> json_actions(const json& payload, int max_actions) {
  std::vector<json> out;
  if (!payload.contains("actions") || !payload["actions"].is_array()) return out;
  int count = static_cast<int>(payload["actions"].size());
  if (max_actions >= 0) count = std::min(count, max_actions);
  out.reserve(count);
  for (int i = 0; i < count; ++i) {
    if (payload["actions"][i].is_object()) out.push_back(payload["actions"][i]);
  }
  return out;
}

std::string cli_action_type(const json& action) {
  std::string type = json_string(action, "type");
  if (type.empty()) type = json_string(action, "t");
  if (type.empty() && action.contains("payload") && action["payload"].is_object()) {
    type = json_string(action["payload"], "type");
    if (type.empty()) type = json_string(action["payload"], "t");
  }
  return type;
}

std::string cli_action_id(const json& action, int index) {
  std::string id = json_string(action, "id");
  if (id.empty()) id = "A" + std::to_string(json_int(action, "i", index));
  return id;
}

int cli_action_unit_id(const json& action) {
  int value = json_int(action, "unit_id", 0);
  if (value == 0) value = json_int(action, "u", 0);
  if (value == 0 && action.contains("payload") && action["payload"].is_object()) {
    value = json_int(action["payload"], "unit_id", json_int(action["payload"], "u", 0));
  }
  return value;
}

std::pair<int, int> cli_action_xy(const json& action) {
  const json* source = &action;
  if (action.contains("payload") && action["payload"].is_object()) source = &action["payload"];
  if (source->contains("destination") && (*source)["destination"].is_array() && (*source)["destination"].size() >= 2) {
    return {(*source)["destination"][0].get<int>(), (*source)["destination"][1].get<int>()};
  }
  return {json_int(*source, "x", 0), json_int(*source, "y", 0)};
}

bool cli_keep_root_type(const std::string& type) {
  static const std::unordered_set<std::string> keep = {
      "CAPTURE", "MAKE_VETERAN", "ATTACK", "CONVERT", "EXAMINE", "RESOURCE_GATHERING",
      "LEVEL_UP", "RESEARCH_TECH", "BUILD", "SPAWN", "RECOVER", "HEAL_OTHERS",
      "UPGRADE_SHIP", "UPGRADE_BOAT", "UPGRADE_RAMMER", "UPGRADE_SCOUT", "UPGRADE_BOMBER"};
  return keep.count(type) > 0;
}

std::vector<int> select_root_indexes(const std::vector<json>& actions, const std::vector<double>& priors, int top_k_actions) {
  std::vector<int> indexes(actions.size());
  for (int i = 0; i < static_cast<int>(actions.size()); ++i) indexes[i] = i;
  if (top_k_actions <= 0 || static_cast<int>(indexes.size()) <= top_k_actions) return indexes;

  std::vector<int> forced;
  std::unordered_set<int> selected;
  for (int i = 0; i < static_cast<int>(actions.size()); ++i) {
    if (cli_keep_root_type(cli_action_type(actions[i]))) {
      forced.push_back(i);
      selected.insert(i);
    }
  }
  const int budget = std::max(top_k_actions, static_cast<int>(selected.size()));
  std::sort(indexes.begin(), indexes.end(), [&](int left, int right) {
    const double l = left < static_cast<int>(priors.size()) ? priors[left] : 0.0;
    const double r = right < static_cast<int>(priors.size()) ? priors[right] : 0.0;
    return l > r;
  });
  std::unordered_set<std::string> seen_moves;
  for (int index : indexes) {
    if (static_cast<int>(selected.size()) >= budget) break;
    if (cli_action_type(actions[index]) == "MOVE") {
      auto [x, y] = cli_action_xy(actions[index]);
      const std::string sig = std::to_string(cli_action_unit_id(actions[index])) + ":" + std::to_string(x) + ":" + std::to_string(y);
      if (seen_moves.count(sig)) continue;
      seen_moves.insert(sig);
    }
    selected.insert(index);
  }

  std::vector<int> ordered = forced;
  for (int index : indexes) {
    if (selected.count(index) && std::find(forced.begin(), forced.end(), index) == forced.end()) ordered.push_back(index);
    if (static_cast<int>(ordered.size()) >= budget) break;
  }
  return ordered;
}

std::vector<double> coerce_priors(py::handle raw_priors, int action_count) {
  std::vector<double> priors;
  if (py::isinstance<py::list>(raw_priors) || py::isinstance<py::tuple>(raw_priors)) {
    py::list values = py::reinterpret_borrow<py::list>(raw_priors);
    for (py::handle item : values) {
      if (static_cast<int>(priors.size()) >= action_count) break;
      priors.push_back(py::cast<double>(item));
    }
  }
  if (static_cast<int>(priors.size()) != action_count) {
    throw std::runtime_error("Static eval prior/action mismatch.");
  }
  double total = 0.0;
  for (double prior : priors) total += std::max(0.0, prior);
  if (total <= 0.0) throw std::runtime_error("Static eval returned non-positive total prior mass.");
  for (double& prior : priors) prior = std::max(0.0, prior) / total;
  return priors;
}

json choose_action_with_native_tree(const json& message, const CliConfig& cfg, std::mt19937_64& rng) {
  std::vector<json> root_actions = json_actions(message, cfg.max_actions);
  if (root_actions.empty()) return json{{"actionId", nullptr}, {"rankedActionIds", json::array()}};

  const bool collect_timing = cfg.profile_json || cfg.profile_timing;
  json timing_ms = {
      {"select_ms", 0.0},
      {"apply_action_ms", 0.0},
      {"static_eval_ms", 0.0},
      {"node_allocation_ms", 0.0},
      {"backup_ms", 0.0},
      {"transition_state_copy_ms", 0.0},
      {"transition_observation_copy_ms", 0.0},
      {"transition_action_mutation_ms", 0.0},
      {"transition_reveal_sync_ms", 0.0},
      {"transition_regenerate_actions_ms", 0.0},
      {"transition_hidden_enemy_ms", 0.0},
      {"search_loop_ms", 0.0},
      {"root_static_eval_ms", 0.0},
      {"root_setup_ms", 0.0},
      {"result_distribution_ms", 0.0},
  };

  const auto root_setup_started = std::chrono::steady_clock::now();
  py::dict root_payload = py::reinterpret_borrow<py::dict>(py_from_json(message));
  py::dict eval_payload;
  eval_payload["player_id"] = py::int_(json_int(message, "player_id", 0));
  eval_payload["observation"] = root_payload["observation"];
  eval_payload["actions"] = root_payload["actions"];
  const auto root_eval_started = std::chrono::steady_clock::now();
  py::dict root_eval = tribes::native::evaluate_static(eval_payload, cfg.max_actions);
  if (collect_timing) {
    timing_ms["root_static_eval_ms"] = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - root_eval_started).count();
  }
  std::vector<double> all_priors = coerce_priors(root_eval["priors"], static_cast<int>(root_actions.size()));
  const double root_value = py::cast<double>(root_eval["value"]);

  std::vector<int> root_indexes = select_root_indexes(root_actions, all_priors, cfg.top_k_actions);
  std::vector<std::string> searched_ids;
  std::vector<double> priors;
  for (int index : root_indexes) {
    searched_ids.push_back(cli_action_id(root_actions[index], index));
    priors.push_back(all_priors[index]);
  }
  double total = 0.0;
  for (double prior : priors) total += std::max(0.0, prior);
  if (total <= 0.0) {
    std::fill(priors.begin(), priors.end(), 1.0 / std::max<size_t>(1, priors.size()));
  } else {
    for (double& prior : priors) prior = std::max(0.0, prior) / total;
  }

  NativeMCTS tree(
      root_payload,
      root_indexes,
      priors,
      root_value,
      message.value("is_terminal", message.value("terminal", false)),
      cfg.seed,
      cfg.max_actions,
      cfg.progressive_widening);
  if (collect_timing) {
    tree.set_static_timing_enabled(true);
  }
  tree.add_root_dirichlet_noise(cfg.dirichlet_alpha, cfg.dirichlet_epsilon);
  const int reserve_capacity = cfg.wall_clock_seconds > 0.0
      ? std::max(cfg.simulations + 1, static_cast<int>(std::ceil(cfg.wall_clock_seconds * 25000.0)))
      : cfg.simulations + 1;
  tree.reserve_tree_capacity(std::max(2, reserve_capacity));
  if (collect_timing) {
    timing_ms["root_setup_ms"] = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - root_setup_started).count();
  }

  const auto started = std::chrono::steady_clock::now();
  int expanded = 0;
  int completed_paths = 0;
  int batches = 0;
  int depth_sum = 0;
  int max_selected_depth = 0;
  int turn_depth_sum = 0;
  int max_turn_depth = 0;
  while ((cfg.wall_clock_seconds > 0.0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() < cfg.wall_clock_seconds) ||
         (cfg.wall_clock_seconds <= 0.0 && expanded < cfg.simulations)) {
    const int frontier = cfg.wall_clock_seconds > 0.0 ? std::max(1, cfg.batch_size) : std::min(std::max(1, cfg.batch_size), std::max(1, cfg.simulations - expanded));
    py::tuple batch = tree.run_static_search_batch(frontier, cfg.c_puct);
    batches += 1;
    const int expanded_count = py::cast<int>(batch[0]);
    const int completed = py::cast<int>(batch[1]);
    if (collect_timing) {
      py::dict batch_timing = tree.last_static_timing();
      timing_ms["select_ms"] = timing_ms["select_ms"].get<double>() + py::cast<double>(batch_timing["select_ms"]);
      timing_ms["apply_action_ms"] = timing_ms["apply_action_ms"].get<double>() + py::cast<double>(batch_timing["apply_action_ms"]);
      timing_ms["static_eval_ms"] = timing_ms["static_eval_ms"].get<double>() + py::cast<double>(batch_timing["static_eval_ms"]);
      timing_ms["node_allocation_ms"] = timing_ms["node_allocation_ms"].get<double>() + py::cast<double>(batch_timing["node_allocation_ms"]);
      timing_ms["backup_ms"] = timing_ms["backup_ms"].get<double>() + py::cast<double>(batch_timing["backup_ms"]);
      timing_ms["transition_state_copy_ms"] = timing_ms["transition_state_copy_ms"].get<double>() + py::cast<double>(batch_timing["transition_state_copy_ms"]);
      timing_ms["transition_observation_copy_ms"] = timing_ms["transition_observation_copy_ms"].get<double>() + py::cast<double>(batch_timing["transition_observation_copy_ms"]);
      timing_ms["transition_action_mutation_ms"] = timing_ms["transition_action_mutation_ms"].get<double>() + py::cast<double>(batch_timing["transition_action_mutation_ms"]);
      timing_ms["transition_reveal_sync_ms"] = timing_ms["transition_reveal_sync_ms"].get<double>() + py::cast<double>(batch_timing["transition_reveal_sync_ms"]);
      timing_ms["transition_regenerate_actions_ms"] = timing_ms["transition_regenerate_actions_ms"].get<double>() + py::cast<double>(batch_timing["transition_regenerate_actions_ms"]);
      timing_ms["transition_hidden_enemy_ms"] = timing_ms["transition_hidden_enemy_ms"].get<double>() + py::cast<double>(batch_timing["transition_hidden_enemy_ms"]);
    }
    py::tuple batch_stats = tree.last_batch_stats();
    expanded += std::max(0, expanded_count);
    completed_paths += std::max(0, completed);
    depth_sum += py::cast<int>(batch_stats[0]);
    max_selected_depth = std::max(max_selected_depth, py::cast<int>(batch_stats[1]));
    turn_depth_sum += py::cast<int>(batch_stats[2]);
    max_turn_depth = std::max(max_turn_depth, py::cast<int>(batch_stats[3]));
    if (completed <= 0 || (cfg.wall_clock_seconds <= 0.0 && expanded_count <= 0)) break;
  }
  const double search_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  if (collect_timing) {
    timing_ms["search_loop_ms"] = search_elapsed * 1000.0;
  }

  const auto result_started = std::chrono::steady_clock::now();
  std::vector<double> probs = tree.root_visit_distribution_by_index(cfg.root_temperature);
  if (probs.empty()) probs = priors;
  int selected = 0;
  if (cfg.sample_action) {
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    selected = dist(rng);
  } else {
    for (int i = 1; i < static_cast<int>(probs.size()); ++i) {
      if (probs[i] > probs[selected]) selected = i;
    }
  }

  std::vector<int> rank(probs.size());
  for (int i = 0; i < static_cast<int>(rank.size()); ++i) rank[i] = i;
  std::sort(rank.begin(), rank.end(), [&](int left, int right) {
    return probs[left] > probs[right];
  });
  json ranked = json::array();
  std::unordered_set<std::string> seen;
  for (int index : rank) {
    if (index >= 0 && index < static_cast<int>(searched_ids.size())) {
      ranked.push_back(searched_ids[index]);
      seen.insert(searched_ids[index]);
    }
  }
  for (int i = 0; i < static_cast<int>(root_actions.size()); ++i) {
    const std::string id = cli_action_id(root_actions[i], i);
    if (!seen.count(id)) ranked.push_back(id);
  }
  const std::string selected_id = selected < static_cast<int>(searched_ids.size()) ? searched_ids[selected] : searched_ids.front();
  json response{{"actionId", selected_id}, {"rankedActionIds", ranked}};
  if (collect_timing) {
    timing_ms["result_distribution_ms"] = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - result_started).count();
  }
  if (cfg.profile_json) {
    response["_profile"] = {
        {"elapsed_sec", search_elapsed},
        {"simulations", expanded},
        {"selected_paths", completed_paths},
        {"completed_paths", completed_paths},
        {"expanded_nodes", expanded},
        {"batches", batches},
        {"depth_sum", depth_sum},
        {"max_depth", max_selected_depth},
        {"turn_depth_sum", turn_depth_sum},
        {"max_turn_depth", max_turn_depth},
        {"avg_selected_depth", completed_paths > 0 ? static_cast<double>(depth_sum) / completed_paths : 0.0},
        {"avg_turn_depth", completed_paths > 0 ? static_cast<double>(turn_depth_sum) / completed_paths : 0.0},
        {"node_count", tree.node_count()},
        {"timing_ms", timing_ms},
    };
  }
  return response;
}

void parse_args(int argc, char** argv, CliConfig& cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
      return argv[++i];
    };
    if (arg == "--simulations") cfg.simulations = std::stoi(next());
    else if (arg == "--wall-clock-per-action-seconds") cfg.wall_clock_seconds = std::stod(next());
    else if (arg == "--top-k-actions") cfg.top_k_actions = std::stoi(next());
    else if (arg == "--max-actions") cfg.max_actions = std::stoi(next());
    else if (arg == "--search-batch-size") cfg.batch_size = std::stoi(next());
    else if (arg == "--static-eval-variant") cfg.static_eval_variant = next();
    else if (arg == "--deterministic") {
      cfg.sample_action = false;
      cfg.root_temperature = 1e-6;
      cfg.dirichlet_epsilon = 0.0;
    } else if (arg == "--reuse-tree") {
      cfg.reuse_tree = true;
    } else if (arg == "--profile-json") {
      cfg.profile_json = true;
      cfg.profile_timing = true;
    } else if (arg == "--profile-timing") {
      cfg.profile_timing = true;
    } else if (arg == "--seed") {
      cfg.seed = static_cast<uint64_t>(std::stoull(next()));
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "native_static_mcts_bot.exe [--simulations N] [--wall-clock-per-action-seconds SEC]\n"
          << "  [--top-k-actions N] [--max-actions N] [--search-batch-size N]\n"
          << "  [--static-eval-variant baseline|experimental] [--deterministic] [--reuse-tree]\n"
          << "  [--profile-json] [--profile-timing] [--seed N]\n";
      std::exit(0);
    }
  }
}

void set_static_eval_variant_env(const std::string& variant) {
#ifdef _WIN32
  _putenv_s("TRIBES_STATIC_EVAL_VARIANT", variant.c_str());
#else
  setenv("TRIBES_STATIC_EVAL_VARIANT", variant.c_str(), 1);
#endif
}

}  // namespace

int main(int argc, char** argv) {
  CliConfig cfg;
  try {
    parse_args(argc, argv, cfg);
    set_static_eval_variant_env(cfg.static_eval_variant);
    std::mt19937_64 rng(cfg.seed);
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      json message = json::parse(line, nullptr, false);
      if (message.is_discarded()) continue;
      const std::string type = message.value("type", "");
      if (type == "action_request") {
        message = normalize_cli_message(message);
        std::cout << choose_action_with_native_tree(message, cfg, rng).dump() << std::endl;
      } else if (type == "game_over") {
        break;
      }
    }
  } catch (const std::exception& exc) {
    std::cerr << "[native_static_mcts_bot] error: " << exc.what() << std::endl;
    return 1;
  }
  return 0;
}
