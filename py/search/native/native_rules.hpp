#pragma once

#include <pybind11/pybind11.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace py = pybind11;

namespace tribes::native {

struct NativeAction {
  std::string id;
  std::string type;
  int unit_id = 0;
  int city_id = 0;
  py::dict payload;
};

struct NativeTile {
  int x = 0;
  int y = 0;
  bool visible = false;
  bool explored = false;
  bool road = false;
  std::string terrain;
  std::string resource;
  std::string building;
  int city_id = 0;
  int unit_id = 0;
  int territory_city_id = 0;
};

struct NativeUnit {
  int id = 0;
  int tribe_id = -1;
  int city_id = 0;
  int x = 0;
  int y = 0;
  int current_hp = 0;
  double current_hp_exact = 0.0;
  int max_hp = 0;
  int kills = 0;
  bool veteran = false;
  bool hidden = false;
  bool hidden_at_turn_start = false;
  bool hidden_enemy_hint = false;
  double attack = 0.0;
  double defence = 0.0;
  int movement = 0;
  int range = 0;
  int cost = 0;
  std::string type;
  std::string status;
};

struct NativeBuilding {
  std::string type;
  int x = 0;
  int y = 0;
  int city_id = 0;
  int owner_tribe_id = -1;
  int level = 0;
  int turns_to_score = 0;
};

struct NativeCity {
  int id = 0;
  int tribe_id = -1;
  int x = 0;
  int y = 0;
  int level = 0;
  int population = 0;
  int population_need = 0;
  int production = 0;
  int bound = 0;
  int points_worth = 0;
  bool capital = false;
  bool walls = false;
  bool infiltrated = false;
  std::vector<int> unit_ids;
  std::vector<NativeBuilding> buildings;
};

struct NativeTribe {
  int id = 0;
  int stars = 0;
  int score = 0;
  int capital_id = 0;
  int kills = 0;
  int pacifist_count = 0;
  bool units_disabled_next_turn = false;
  std::string result;
  std::string tribe_type;
  std::vector<std::string> researched_tech_ids;
  std::vector<int> city_ids;
  std::vector<int> extra_unit_ids;
  std::vector<int> connected_city_ids;
  std::vector<int> met_tribe_ids;
  std::vector<int> known_capital_tribe_ids;
  std::vector<int> discovered_lighthouses;
  std::map<std::string, std::string> monuments;
};

struct NativeGameState {
  py::dict observation;
  int board_size = 0;
  std::vector<NativeTile> tiles;
  std::vector<NativeUnit> units;
  std::vector<NativeCity> cities;
  std::vector<NativeTribe> tribes;
  std::vector<int> capital_city_ids;
  std::vector<std::vector<std::string>> relationships;
  std::vector<std::vector<int>> pending_offer_from;
  std::vector<std::vector<std::string>> pending_offer_types;
  std::vector<int> legal_action_indexes;
  int actor_id_floor = 0;
  int root_player_id = 0;
  int active_player_id = 0;
  int tick = 0;
  bool terminal = false;
  bool terminal_value_known = false;
  double terminal_value = 0.0;
  std::string transition_kind;
  int winner_id = -1;
  std::string terminal_reason;
  bool leveling_up = false;
  bool can_end_turn = true;
};

struct NativeRoot {
  std::vector<NativeAction> actions;
  NativeGameState state;
};

NativeRoot parse_root_payload(const py::dict& payload, int max_actions);
NativeGameState apply_action_strict(
    const NativeGameState& state,
    std::vector<NativeAction>& actions,
    int global_action_index,
    int max_actions);
py::dict serialize_evaluation_payload(
    const NativeGameState& state,
    const std::vector<NativeAction>& actions);
double value_to_root_perspective(double active_player_value, int root_player_id, int leaf_active_player_id);

}  // namespace tribes::native
