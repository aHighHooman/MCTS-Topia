#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../py/search/native/third_party/nlohmann/json.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

struct Args {
  fs::path static_exe = "out/native/static_mcts_bot.exe";
  fs::path record_dir;
  std::string branch_name = "branch";
  double sample_rate = 1.0;
  uint64_t record_seed = 0;
  std::vector<std::string> child_args;
};

double now_seconds() {
  using clock = std::chrono::system_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

std::string sanitize_branch(const std::string& raw) {
  std::string out;
  for (char ch : raw) {
    if (std::isalnum(static_cast<unsigned char>(ch)) || ch == '-' || ch == '_') out.push_back(ch);
    else out.push_back('_');
  }
  return out.empty() ? "branch" : out;
}

void write_text_file(const fs::path& path, const std::string& text) {
  fs::create_directories(path.parent_path());
#ifdef _WIN32
  std::wstring wide = fs::absolute(path).wstring();
  if (wide.rfind(LR"(\\?\)", 0) != 0) {
    if (wide.rfind(LR"(\\)", 0) == 0) {
      wide = LR"(\\?\UNC\)" + wide.substr(2);
    } else {
      wide = LR"(\\?\)" + wide;
    }
  }
  FILE* file = _wfopen(wide.c_str(), L"wb");
  if (!file) throw std::runtime_error("failed to open output file: " + path.string());
  const char* data = text.data();
  size_t remaining = text.size();
  while (remaining > 0) {
    const size_t written = std::fwrite(data, 1, remaining, file);
    if (written == 0) {
      std::fclose(file);
      throw std::runtime_error("failed to write output file: " + path.string());
    }
    data += written;
    remaining -= written;
  }
  if (std::fclose(file) != 0) throw std::runtime_error("failed to close output file: " + path.string());
#else
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("failed to open output file: " + path.string());
  out << text;
  if (!out) throw std::runtime_error("failed to write output file: " + path.string());
#endif
}

#ifdef _WIN32
std::wstring windows_extended_path(const fs::path& path) {
  std::wstring wide = fs::absolute(path).wstring();
  if (wide.rfind(LR"(\\?\)", 0) != 0) {
    if (wide.rfind(LR"(\\)", 0) == 0) {
      wide = LR"(\\?\UNC\)" + wide.substr(2);
    } else {
      wide = LR"(\\?\)" + wide;
    }
  }
  return wide;
}
#endif

void replace_file(const fs::path& from, const fs::path& to) {
#ifdef _WIN32
  const std::wstring wide_from = windows_extended_path(from);
  const std::wstring wide_to = windows_extended_path(to);
  if (!MoveFileExW(wide_from.c_str(), wide_to.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    const DWORD error = GetLastError();
    throw std::runtime_error("failed to replace output file: " + from.string() + " -> " + to.string() + " error=" + std::to_string(error));
  }
#else
  fs::rename(from, to);
#endif
}

int as_int(const json& value, int fallback = 0) {
  try {
    if (value.is_number_integer()) return value.get<int>();
    if (value.is_number()) return static_cast<int>(value.get<double>());
    if (value.is_string()) return std::stoi(value.get<std::string>());
  } catch (...) {
  }
  return fallback;
}

std::string as_string(const json& value, const std::string& fallback = "") {
  if (value.is_string()) return value.get<std::string>();
  if (value.is_null()) return fallback;
  return value.dump();
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

int json_int(const json& object, const char* key, int fallback = 0) {
  if (!object.is_object() || !object.contains(key) || object[key].is_null()) return fallback;
  return json_to_int(object[key], fallback);
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

json matrix_value(const json& matrix, int x, int y) {
  if (!matrix.is_array() || y < 0 || y >= static_cast<int>(matrix.size())) return nullptr;
  const json& row = matrix[static_cast<size_t>(y)];
  if (!row.is_array() || x < 0 || x >= static_cast<int>(row.size())) return nullptr;
  return row[static_cast<size_t>(x)];
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

json array_value(const json& array, size_t index, const json& fallback = nullptr) {
  if (!array.is_array() || index >= array.size()) return fallback;
  return array[index];
}

int array_int(const json& array, size_t index, int fallback = 0) {
  return json_to_int(array_value(array, index), fallback);
}

json enum_value(const std::vector<const char*>& names, const json& value) {
  if (value.is_string()) return value;
  const int index = json_to_int(value, -1);
  if (index < 0 || index >= static_cast<int>(names.size())) return value;
  return json(names[static_cast<size_t>(index)]);
}

json action_type_value(const json& value) {
  static const std::vector<const char*> names = {
      "BUILD", "BURN_FOREST", "CLEAR_FOREST", "DESTROY", "GROW_FOREST", "LEVEL_UP",
      "GATHER", "SPAWN", "BUILD_ROAD", "BUILD_EMBASSY", "END_TURN", "RESEARCH",
      "PROPOSE_PEACE", "ACCEPT_PEACE", "PROPOSE_TREATY", "ACCEPT_TREATY", "CANCEL_TREATY",
      "ATTACK", "CAPTURE", "CONVERT", "DISBAND", "EXAMINE", "HEAL_OTHERS", "INFILTRATE",
      "MAKE_VETERAN", "MOVE", "RECOVER", "UPGRADE_RAMMER", "UPGRADE_SCOUT", "UPGRADE_BOMBER"};
  return enum_value(names, value);
}

json unit_type_value(const json& value) {
  static const std::vector<const char*> names = {
      "WARRIOR", "RIDER", "DEFENDER", "SWORDMAN", "ARCHER", "CATAPULT", "KNIGHT",
      "MIND_BENDER", "RAFT", "SCOUT", "BOMBER", "SUPERUNIT", "CLOAK", "DAGGER",
      "RAMMER", "JUGGERNAUT", "DINGHY", "PIRATE"};
  return enum_value(names, value);
}

json building_type_value(const json& value) {
  static const std::vector<const char*> names = {
      "PORT", "MINE", "FORGE", "FARM", "WINDMILL", "MARKET", "LUMBER_HUT", "SAWMILL",
      "TEMPLE", "WATER_TEMPLE", "FOREST_TEMPLE", "MOUNTAIN_TEMPLE", "ALTAR_OF_PEACE",
      "EMPERORS_TOMB", "EYE_OF_GOD", "GATE_OF_POWER", "GRAND_BAZAR", "PARK_OF_FORTUNE",
      "TOWER_OF_WISDOM", "EMBASSY"};
  return enum_value(names, value);
}

json resource_type_value(const json& value) {
  static const std::vector<const char*> names = {
      "FISH", "FRUIT", "ANIMAL", "STARFISH", "LIGHTHOUSE", "ORE", "CROPS", "RUINS"};
  return enum_value(names, value);
}

json terrain_type_value(const json& value) {
  static const std::vector<const char*> names = {
      "PLAIN", "SHALLOW_WATER", "DEEP_WATER", "MOUNTAIN", "VILLAGE", "CITY", "FOREST", "FOG"};
  return enum_value(names, value);
}

json result_type_value(const json& value) {
  static const std::vector<const char*> names = {"WIN", "LOSS", "INCOMPLETE"};
  return enum_value(names, value);
}

json status_type_value(const json& value) {
  static const std::vector<const char*> names = {
      "FRESH", "MOVED", "ATTACKED", "MOVED_AND_ATTACKED", "PUSHED", "FINISHED"};
  return enum_value(names, value);
}

json tech_type_value(const json& value) {
  static const std::vector<const char*> names = {
      "CLIMBING", "FISHING", "HUNTING", "ORGANIZATION", "RIDING", "ARCHERY",
      "FARMING", "FORESTRY", "FREE_SPIRIT", "MEDITATION", "MINING", "ROADS",
      "RAMMING", "SAILING", "STRATEGY", "AQUATISM", "CHIVALRY", "CONSTRUCTION",
      "DIPLOMACY", "MATHEMATICS", "NAVIGATION", "SMITHERY", "SPIRITUALISM",
      "TRADE", "PHILOSOPHY"};
  return enum_value(names, value);
}

json level_up_value(const json& value) {
  static const std::vector<const char*> names = {
      "WORKSHOP", "EXPLORER", "CITY_WALL", "RESOURCES", "POP_GROWTH", "BORDER_GROWTH",
      "PARK", "SUPERUNIT"};
  return enum_value(names, value);
}

json examine_bonus_value(const json& value) {
  static const std::vector<const char*> names = {
      "UNIT", "RESEARCH", "POP_GROWTH", "EXPLORER", "RESOURCES"};
  return enum_value(names, value);
}

json relationship_type_value(const json& value) {
  static const std::vector<const char*> names = {"WAR", "PEACE", "TREATY"};
  return enum_value(names, value);
}

json convert_matrix_codes(const json& matrix, json (*converter)(const json&)) {
  if (!matrix.is_array()) return json::array();
  json rows = json::array();
  for (const json& row : matrix) {
    json out_row = json::array();
    if (row.is_array()) {
      for (const json& value : row) out_row.push_back(value.is_null() ? json(nullptr) : converter(value));
    }
    rows.push_back(std::move(out_row));
  }
  return rows;
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

json normalize_dense_board_json(const json& board) {
  json out = json::object();
  out["size"] = array_int(board, 0, 0);
  out["terrain"] = convert_matrix_codes(array_value(board, 1, json::array()), terrain_type_value);
  out["resource"] = convert_matrix_codes(array_value(board, 2, json::array()), resource_type_value);
  out["building"] = convert_matrix_codes(array_value(board, 3, json::array()), building_type_value);
  out["city"] = array_value(board, 4, json::array());
  out["unit"] = array_value(board, 5, json::array());
  out["exp"] = array_value(board, 6, json::array());
  out["road"] = array_value(board, 7, json::array());
  return normalize_board_json(out);
}

json normalize_dense_buildings_json(const json& buildings) {
  json out = json::array();
  if (!buildings.is_array()) return out;
  for (const json& building : buildings) {
    if (!building.is_array()) continue;
    out.push_back(json{
        {"type", building_type_value(array_value(building, 0))},
        {"x", array_value(building, 1, 0)},
        {"y", array_value(building, 2, 0)}});
  }
  return out;
}

json normalize_dense_tribe_json(const json& tribe) {
  json techs = json::array();
  const json raw_techs = array_value(tribe, 6, json::array());
  if (raw_techs.is_array()) {
    for (const json& tech : raw_techs) techs.push_back(tech_type_value(tech));
  }
  return normalize_tribe_json(json{
      {"id", array_value(tribe, 0, 0)},
      {"type", array_value(tribe, 1, nullptr)},
      {"stars", array_value(tribe, 2, 0)},
      {"score", array_value(tribe, 3, 0)},
      {"result", result_type_value(array_value(tribe, 4))},
      {"capital_id", array_value(tribe, 5, 0)},
      {"researched_tech_ids", std::move(techs)}});
}

json normalize_dense_city_json(const json& city) {
  return normalize_city_json(json{
      {"id", array_value(city, 0, 0)},
      {"tribe_id", array_value(city, 1, -1)},
      {"x", array_value(city, 2, 0)},
      {"y", array_value(city, 3, 0)},
      {"level", array_value(city, 4, 0)},
      {"population", array_value(city, 5, 0)},
      {"population_need", array_value(city, 6, 0)},
      {"production", array_value(city, 7, 0)},
      {"is_capital", json_truthy(array_value(city, 8, false))},
      {"has_walls", json_truthy(array_value(city, 9, false))},
      {"points_worth", array_value(city, 10, 0)},
      {"buildings", normalize_dense_buildings_json(array_value(city, 11, json::array()))}});
}

json normalize_dense_unit_json(const json& unit) {
  return normalize_unit_json(json{
      {"id", array_value(unit, 0, 0)},
      {"tribe_id", array_value(unit, 1, -1)},
      {"city_id", array_value(unit, 2, 0)},
      {"type", unit_type_value(array_value(unit, 3))},
      {"x", array_value(unit, 4, 0)},
      {"y", array_value(unit, 5, 0)},
      {"current_hp", array_value(unit, 6, 0)},
      {"max_hp", array_value(unit, 7, 0)},
      {"kills", array_value(unit, 8, 0)},
      {"is_veteran", json_truthy(array_value(unit, 9, false))},
      {"status", status_type_value(array_value(unit, 10))},
      {"is_hidden", json_truthy(array_value(unit, 11, false))},
      {"hidden_enemy_hint", json_truthy(array_value(unit, 12, false))},
      {"attack", array_value(unit, 13, 0)},
      {"defence", array_value(unit, 14, 0)},
      {"movement", array_value(unit, 15, 0)},
      {"range", array_value(unit, 16, 0)},
      {"cost", array_value(unit, 17, 0)}});
}

json normalize_dense_relationships_json(const json& relationships) {
  json rows = json::array();
  if (!relationships.is_array()) return rows;
  for (const json& row : relationships) {
    json out_row = json::array();
    if (row.is_array()) {
      for (const json& relationship : row) {
        out_row.push_back(relationship.is_null() ? json(nullptr) : relationship_type_value(relationship));
      }
    }
    rows.push_back(std::move(out_row));
  }
  return rows;
}

json normalize_dense_observation_json(const json& observation) {
  json out = json::object();
  out["tick"] = array_value(observation, 0, 0);
  out["map_type"] = array_value(observation, 1, nullptr);
  out["active_player_id"] = array_value(observation, 2, 0);
  out["can_end_turn"] = json_truthy(array_value(observation, 3, false));
  out["leveling_up"] = json_truthy(array_value(observation, 4, false));
  json tribes = json::array();
  for (const json& tribe : array_value(observation, 5, json::array())) tribes.push_back(normalize_dense_tribe_json(tribe));
  out["tribes"] = std::move(tribes);
  json cities = json::array();
  for (const json& city : array_value(observation, 6, json::array())) cities.push_back(normalize_dense_city_json(city));
  out["cities"] = std::move(cities);
  json units = json::array();
  for (const json& unit : array_value(observation, 7, json::array())) units.push_back(normalize_dense_unit_json(unit));
  out["units"] = std::move(units);
  out["board"] = normalize_dense_board_json(array_value(observation, 8, json::array()));
  out["ranking"] = array_value(observation, 9, json::array());
  out["relationships"] = normalize_dense_relationships_json(array_value(observation, 10, json::array()));
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

json normalize_dense_action_json(const json& action, int index) {
  if (!action.is_array() || action.empty()) {
    return normalize_action_json(json{{"i", index}, {"id", "A" + std::to_string(index)}});
  }
  json out = json::object();
  out["i"] = index;
  out["_wire_index"] = index;
  out["id"] = "A" + std::to_string(index);
  out["type"] = action_type_value(array_value(action, 0));
  std::string type = json_string(out, "type");
  if (type == "GATHER") type = "RESOURCE_GATHERING";
  else if (type == "RESEARCH") type = "RESEARCH_TECH";
  out["type"] = type;

  if (type == "MOVE") {
    out["unit_id"] = array_value(action, 1, 0);
    out["x"] = array_value(action, 2, 0);
    out["y"] = array_value(action, 3, 0);
  } else if (type == "ATTACK" || type == "CONVERT") {
    out["unit_id"] = array_value(action, 1, 0);
    out["target_unit_id"] = array_value(action, 2, 0);
  } else if (type == "CAPTURE") {
    out["unit_id"] = array_value(action, 1, 0);
    out["target_city_id"] = array_value(action, 2, 0);
    out["capture_type"] = terrain_type_value(array_value(action, 3));
  } else if (type == "INFILTRATE") {
    out["unit_id"] = array_value(action, 1, 0);
    out["target_city_id"] = array_value(action, 2, 0);
  } else if (type == "BUILD" || type == "RESOURCE_GATHERING" || type == "SPAWN" || type == "LEVEL_UP") {
    out["city_id"] = array_value(action, 1, 0);
    out["x"] = array_value(action, 2, 0);
    out["y"] = array_value(action, 3, 0);
    if (type == "BUILD") out["building_type"] = building_type_value(array_value(action, 4));
    else if (type == "RESOURCE_GATHERING") out["resource_type"] = resource_type_value(array_value(action, 4));
    else if (type == "SPAWN") out["unit_type"] = unit_type_value(array_value(action, 4));
    else out["bonus"] = level_up_value(array_value(action, 4));
  } else if (type == "BURN_FOREST" || type == "CLEAR_FOREST" || type == "DESTROY" || type == "GROW_FOREST") {
    out["city_id"] = array_value(action, 1, 0);
    out["x"] = array_value(action, 2, 0);
    out["y"] = array_value(action, 3, 0);
  } else if (type == "BUILD_ROAD") {
    out["tribe_id"] = array_value(action, 1, 0);
    out["x"] = array_value(action, 2, 0);
    out["y"] = array_value(action, 3, 0);
  } else if (type == "RESEARCH_TECH") {
    out["tribe_id"] = array_value(action, 1, 0);
    out["tech"] = tech_type_value(array_value(action, 2));
    out["technology"] = out["tech"];
  } else if (type == "BUILD_EMBASSY" || type == "PROPOSE_PEACE" || type == "ACCEPT_PEACE" ||
             type == "PROPOSE_TREATY" || type == "ACCEPT_TREATY" || type == "CANCEL_TREATY") {
    out["tribe_id"] = array_value(action, 1, 0);
    out["target_player_id"] = array_value(action, 2, -1);
  } else if (type == "EXAMINE") {
    out["unit_id"] = array_value(action, 1, 0);
    out["bonus"] = examine_bonus_value(array_value(action, 2));
  } else if (type == "END_TURN") {
    out["tribe_id"] = array_value(action, 1, 0);
  } else {
    out["unit_id"] = array_value(action, 1, 0);
  }
  return normalize_action_json(out);
}

json normalize_payload_message(const json& message) {
  json out = message.is_object() ? message : json::object();
  if (!out.contains("observation") && out.contains("obs")) out["observation"] = out["obs"];
  if (out.contains("observation") && out["observation"].is_array()) {
    out["observation"] = normalize_dense_observation_json(out["observation"]);
  }
  if (out.contains("observation") && out["observation"].is_object()) {
    json observation = out["observation"];
    set_default(observation, "active_player_id", value_or(observation, "active", value_or(out, "player_id", 0)));
    set_default(observation, "can_end_turn", value_or(observation, "end", false));
    set_default(observation, "leveling_up", value_or(observation, "lvlup", false));
    set_default(observation, "ranking", value_or(observation, "rank", json::array()));
    observation["board"] = normalize_board_json(value_or(observation, "board", json::object()));
    json units = json::array();
    for (const json& unit : value_or(observation, "units", json::array())) units.push_back(normalize_unit_json(unit));
    observation["units"] = std::move(units);
    json cities = json::array();
    for (const json& city : value_or(observation, "cities", json::array())) cities.push_back(normalize_city_json(city));
    observation["cities"] = std::move(cities);
    json tribes = json::array();
    for (const json& tribe : value_or(observation, "tribes", json::array())) tribes.push_back(normalize_tribe_json(tribe));
    observation["tribes"] = std::move(tribes);
    out["observation"] = std::move(observation);
  }
  json actions = json::array();
  const json raw_actions = value_or(out, "actions", json::array());
  if (raw_actions.is_array()) {
    for (int i = 0; i < static_cast<int>(raw_actions.size()); ++i) {
      const json& action = raw_actions[static_cast<size_t>(i)];
      actions.push_back(action.is_array() ? normalize_dense_action_json(action, i) : normalize_action_json(action));
    }
  }
  out["actions"] = std::move(actions);
  return out;
}

std::string terminal_winner_string(const json& game_over) {
  for (const std::string key : {"winner_id", "winner"}) {
    if (!game_over.contains(key) || game_over[key].is_null()) continue;
    std::string value = as_string(game_over[key]);
    std::string upper = value;
    std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
    if (upper.empty() || upper == "NONE" || upper == "DRAW" || upper == "NULL" || upper == "-1") continue;
    return value;
  }
  return "";
}

json final_scores(const json& game_over) {
  if (game_over.contains("final_scores")) return game_over["final_scores"];
  if (game_over.contains("scores")) return game_over["scores"];
  return nullptr;
}

json ranking(const json& game_over) {
  if (game_over.contains("ranking")) return game_over["ranking"];
  if (game_over.contains("rank")) return game_over["rank"];
  return nullptr;
}

double terminal_target_for_player(const json& game_over, int player_id) {
  std::string winner = terminal_winner_string(game_over);
  if (!winner.empty()) return as_int(winner, -999999) == player_id ? 1.0 : -1.0;
  json rank = ranking(game_over);
  if (rank.is_array() && !rank.empty()) {
    if (as_int(rank[0], -999999) == player_id) return 1.0;
    for (const json& item : rank) {
      if (as_int(item, -999999) == player_id) return -1.0;
    }
  }
  json scores = final_scores(game_over);
  if (scores.is_array() && !scores.empty()) {
    double best = -1e300;
    double player_score = -1e300;
    bool found = false;
    for (const json& item : scores) {
      if (!item.is_array() || item.size() < 2) continue;
      const int pid = as_int(item[0], -999999);
      const double score = item[1].is_number() ? item[1].get<double>() : 0.0;
      best = std::max(best, score);
      if (pid == player_id) {
        player_score = score;
        found = true;
      }
    }
    if (found) return player_score == best ? 1.0 : -1.0;
  }
  return 0.0;
}

int player_id_from_message(const json& message) {
  if (message.contains("observation") && message["observation"].is_object()) {
    const json& obs = message["observation"];
    if (obs.contains("active_player_id")) return as_int(obs["active_player_id"], as_int(message.value("player_id", 0), 0));
  }
  return as_int(message.value("player_id", 0), 0);
}

int tick_from_message(const json& message) {
  if (message.contains("observation") && message["observation"].is_object()) {
    const json& obs = message["observation"];
    if (obs.contains("tick")) return as_int(obs["tick"], 0);
  }
  return as_int(message.value("tick", 0), 0);
}

json root_payload(const json& message) {
  json normalized = normalize_payload_message(message);
  json out = json::object();
  out["player_id"] = as_int(normalized.value("player_id", player_id_from_message(message)), 0);
  out["observation"] = normalized.value("observation", json::object());
  out["actions"] = normalized.value("actions", json::array());
  for (const std::string key : {"request_id", "match_seed", "level_seed", "game_seed", "agent_seed"}) {
    if (message.contains(key)) out[key] = message[key];
  }
  return out;
}

std::string selected_action_id(const json& response) {
  if (response.contains("actionId")) return as_string(response["actionId"]);
  if (response.contains("action_id")) return as_string(response["action_id"]);
  return "";
}

std::string random_hex(std::mt19937_64& rng, int chars) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(chars);
  for (int i = 0; i < chars; ++i) out.push_back(kHex[static_cast<size_t>(rng() & 15)]);
  return out;
}

std::string hex_u64(uint64_t value) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out(16, '0');
  for (int i = 15; i >= 0; --i) {
    out[static_cast<size_t>(i)] = kHex[value & 15ULL];
    value >>= 4;
  }
  return out;
}

std::string content_digest(const std::string& text) {
  uint64_t h1 = 1469598103934665603ULL;
  uint64_t h2 = 1099511628211ULL ^ static_cast<uint64_t>(text.size());
  for (unsigned char ch : text) {
    h1 ^= static_cast<uint64_t>(ch);
    h1 *= 1099511628211ULL;
    h2 ^= static_cast<uint64_t>(ch) + 0x9e3779b97f4a7c15ULL + (h2 << 6) + (h2 >> 2);
    h2 *= 1099511628211ULL;
  }
  return hex_u64(h1) + hex_u64(h2);
}

std::string store_payload(const json& payload, const fs::path& payload_dir, std::mt19937_64& rng) {
  const std::string text = payload.dump();
  const std::string digest = content_digest(text);
  const fs::path path = payload_dir / digest.substr(0, 2) / digest.substr(2, 2) / (digest + ".json");
  if (fs::exists(path)) return digest;
  fs::path tmp = path;
  tmp += "." + random_hex(rng, 16) + ".tmp";
  write_text_file(tmp, text);
  try {
    replace_file(tmp, path);
  } catch (...) {
    if (fs::exists(path)) {
      std::error_code ec;
      fs::remove(tmp, ec);
      return digest;
    }
    throw;
  }
  return digest;
}

fs::path record_path(const fs::path& record_dir, const std::string& branch_name, std::mt19937_64& rng) {
  const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  return record_dir / ("records_" + sanitize_branch(branch_name) + "_" + std::to_string(millis) + "_" + random_hex(rng, 8) + ".jsonl");
}

void write_labeled_records(
    const fs::path& record_dir,
    const std::string& branch_name,
    const std::vector<json>& pending,
    const json& game_over,
    std::mt19937_64& rng,
    double elapsed_sec,
    const Args& args) {
  if (pending.empty()) return;
  fs::create_directories(record_dir);
  fs::path path = record_path(record_dir, branch_name, rng);
  fs::path tmp = path;
  tmp += ".tmp";
  std::ostringstream out;
  const std::string winner = terminal_winner_string(game_over);
  for (size_t i = 0; i < pending.size(); ++i) {
    json row = pending[i];
    const int player_id = as_int(row.value("player_id", 0), 0);
    row["row_index"] = i;
    row["target"] = terminal_target_for_player(game_over, player_id);
    row["winner_id"] = winner.empty() ? json(nullptr) : json(as_int(winner, -1));
    row["final_scores"] = final_scores(game_over);
    row["ranking"] = ranking(game_over);
    row["decisive"] = !winner.empty();
    row["game_over"] = game_over;
    row["proxy_elapsed_sec"] = elapsed_sec;
    row["record_seed"] = args.record_seed;
    row["static_exe"] = args.static_exe.string();
    row["static_args"] = args.child_args;
    out << row.dump() << "\n";
  }
  write_text_file(tmp, out.str());
  replace_file(tmp, path);
}

std::string quote_arg(const std::string& raw) {
  if (raw.find_first_of(" \t\"") == std::string::npos) return raw;
  std::string out = "\"";
  for (char ch : raw) {
    if (ch == '"') out += "\\\"";
    else out.push_back(ch);
  }
  out += "\"";
  return out;
}

#ifdef _WIN32
class ChildProcess {
 public:
  ChildProcess(const fs::path& exe, const std::vector<std::string>& args) {
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(sa);
    sa.lpSecurityDescriptor = nullptr;
    sa.bInheritHandle = TRUE;
    HANDLE child_stdout_read = nullptr;
    HANDLE child_stdout_write = nullptr;
    HANDLE child_stdin_read = nullptr;
    HANDLE child_stdin_write = nullptr;
    if (!CreatePipe(&child_stdout_read, &child_stdout_write, &sa, 0)) throw std::runtime_error("CreatePipe stdout failed");
    if (!SetHandleInformation(child_stdout_read, HANDLE_FLAG_INHERIT, 0)) throw std::runtime_error("SetHandleInformation stdout failed");
    if (!CreatePipe(&child_stdin_read, &child_stdin_write, &sa, 0)) throw std::runtime_error("CreatePipe stdin failed");
    if (!SetHandleInformation(child_stdin_write, HANDLE_FLAG_INHERIT, 0)) throw std::runtime_error("SetHandleInformation stdin failed");

    STARTUPINFOA si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    si.hStdOutput = child_stdout_write;
    si.hStdInput = child_stdin_read;
    si.dwFlags |= STARTF_USESTDHANDLES;
    ZeroMemory(&pi_, sizeof(pi_));

    std::string command = quote_arg(exe.string());
    for (const std::string& arg : args) command += " " + quote_arg(arg);
    std::vector<char> mutable_command(command.begin(), command.end());
    mutable_command.push_back('\0');
    BOOL ok = CreateProcessA(
        nullptr,
        mutable_command.data(),
        nullptr,
        nullptr,
        TRUE,
        0,
        nullptr,
        nullptr,
        &si,
        &pi_);
    CloseHandle(child_stdout_write);
    CloseHandle(child_stdin_read);
    if (!ok) {
      CloseHandle(child_stdout_read);
      CloseHandle(child_stdin_write);
      throw std::runtime_error("CreateProcess failed");
    }
    stdout_read_ = child_stdout_read;
    stdin_write_ = child_stdin_write;
  }

  ~ChildProcess() {
    if (stdin_write_) CloseHandle(stdin_write_);
    if (stdout_read_) CloseHandle(stdout_read_);
    if (pi_.hProcess) CloseHandle(pi_.hProcess);
    if (pi_.hThread) CloseHandle(pi_.hThread);
  }

  void write_line(const std::string& line) {
    std::string value = line + "\n";
    DWORD written = 0;
    if (!WriteFile(stdin_write_, value.data(), static_cast<DWORD>(value.size()), &written, nullptr)) {
      throw std::runtime_error("failed writing to child stdin");
    }
  }

  std::string read_line() {
    std::string line;
    char ch = '\0';
    DWORD read = 0;
    while (ReadFile(stdout_read_, &ch, 1, &read, nullptr) && read == 1) {
      if (ch == '\n') break;
      if (ch != '\r') line.push_back(ch);
    }
    return line;
  }

 private:
  PROCESS_INFORMATION pi_{};
  HANDLE stdout_read_ = nullptr;
  HANDLE stdin_write_ = nullptr;
};
#else
class ChildProcess {
 public:
  ChildProcess(const fs::path&, const std::vector<std::string>&) {
    throw std::runtime_error("static_recording_proxy_native currently supports Windows process pipes only");
  }
};
#endif

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
      return argv[++i];
    };
    if (arg == "--static-exe") args.static_exe = next();
    else if (arg == "--record-dir") args.record_dir = next();
    else if (arg == "--branch-name") args.branch_name = next();
    else if (arg == "--record-sample-rate") args.sample_rate = std::stod(next());
    else if (arg == "--record-seed") args.record_seed = static_cast<uint64_t>(std::stoull(next()));
    else {
      for (; i < argc; ++i) args.child_args.push_back(argv[i]);
      break;
    }
  }
  if (args.record_dir.empty()) throw std::runtime_error("--record-dir is required");
  args.sample_rate = std::max(0.0, std::min(1.0, args.sample_rate));
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);
    fs::path payload_dir = args.record_dir / "payloads";
    std::mt19937_64 rng(args.record_seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    ChildProcess child(args.static_exe, args.child_args);
    std::vector<json> pending;
    const double started_at = now_seconds();
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      json message = json::parse(line, nullptr, false);
      if (!message.is_object()) continue;
      const std::string type = as_string(message.value("type", ""));
      child.write_line(message.dump());
      if (type == "action_request") {
        std::string response_line = child.read_line();
        if (response_line.empty()) throw std::runtime_error("static child exited before responding to action_request");
        std::cout << response_line << std::endl;
        if (args.sample_rate > 0.0 && uniform(rng) <= args.sample_rate) {
          json response = json::parse(response_line, nullptr, false);
          json payload = root_payload(message);
          std::string digest = store_payload(payload, payload_dir, rng);
          json row;
          row["schema_version"] = 1;
          row["branch_name"] = args.branch_name;
          row["payload_hash"] = digest;
          row["payload_path"] = (payload_dir / digest.substr(0, 2) / digest.substr(2, 2) / (digest + ".json")).string();
          row["player_id"] = as_int(payload.value("player_id", 0), 0);
          row["active_player_id"] = player_id_from_message(message);
          row["tick"] = tick_from_message(message);
          row["action_count"] = payload.value("actions", json::array()).size();
          row["selected_action_id"] = response.is_object() ? selected_action_id(response) : "";
          for (const std::string key : {"request_id", "match_seed", "level_seed", "game_seed", "agent_seed"}) {
            if (message.contains(key)) row[key] = message[key];
          }
          row["recorded_at"] = now_seconds();
          pending.push_back(std::move(row));
        }
      } else if (type == "game_over") {
        write_labeled_records(args.record_dir, args.branch_name, pending, message, rng, now_seconds() - started_at, args);
        pending.clear();
      }
    }
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
}
