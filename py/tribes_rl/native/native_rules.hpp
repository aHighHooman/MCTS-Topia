#pragma once

#include <pybind11/pybind11.h>

#include <cstdint>
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
};

struct NativeUnit {
  int id = 0;
  int tribe_id = -1;
  int city_id = 0;
  int x = 0;
  int y = 0;
  int current_hp = 0;
  int max_hp = 0;
  int kills = 0;
  bool veteran = false;
  bool hidden = false;
  std::string type;
  std::string status;
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
  bool capital = false;
  bool walls = false;
};

struct NativeTribe {
  int id = 0;
  int stars = 0;
  int score = 0;
  int capital_id = 0;
  std::string result;
  std::vector<std::string> researched_tech_ids;
};

struct NativeGameState {
  py::dict observation;
  int board_size = 0;
  std::vector<NativeTile> tiles;
  std::vector<NativeUnit> units;
  std::vector<NativeCity> cities;
  std::vector<NativeTribe> tribes;
  std::vector<int> legal_action_indexes;
  int root_player_id = 0;
  int active_player_id = 0;
  int tick = 0;
  bool terminal = false;
  bool terminal_value_known = false;
  double terminal_value = 0.0;
  std::string transition_kind;
  int winner_id = -1;
  std::string terminal_reason;
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
