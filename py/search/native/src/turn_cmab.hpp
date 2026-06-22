#pragma once

#ifdef TRIBES_NATIVE_MCTS_STANDALONE
#include "json_py.hpp"
#else
#include <pybind11/pybind11.h>
#endif

#include "rules.hpp"

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef TRIBES_NATIVE_MCTS_STANDALONE
namespace py = pybind11;
#endif

namespace tribes::native {

enum class TurnCmabOpponentMode {
  RootMax,
  RootAdversarial,
  ActiveSelfish,
};

struct TurnCmabConfig {
  int simulations = 256;
  int max_actions = 512;
  int max_turn_depth = 3;
  int max_primitives_per_turn = 32;
  int max_new_edges_per_node = 64;
  int progressive_base = 4;
  double progressive_scale = 1.5;
  double outer_c = 1.4;
  double cmab_c = 1.0;
  double cmab_prior_weight = 0.35;
  double cmab_temperature = 1.0;
  int max_choices_per_factor = 12;
  int max_road_choices = 8;
  int max_resource_choices = 12;
  int max_city_choices = 8;
  int max_unit_choices = 8;
  bool deterministic = false;
  bool profile_json = false;
  TurnCmabOpponentMode opponent_mode = TurnCmabOpponentMode::RootAdversarial;
};

enum class FactorStage {
  Forced,
  Tactical,
  Research,
  City,
  ResourceBuild,
  Road,
  Diplomacy,
  Cleanup,
  EndTurn,
};

enum class FactorKind {
  ForcedLevelUp,
  Unit,
  City,
  Research,
  Resource,
  Build,
  Road,
  Diplomacy,
  EndTurn,
};

struct CmabChoice {
  std::string choice_key;
  std::string action_signature;
  int initial_action_index = -1;
  double prior = 0.0;
  FactorStage stage = FactorStage::Cleanup;
  FactorKind kind = FactorKind::EndTurn;
  int actor_id = 0;
};

struct CmabFactor {
  std::string factor_key;
  FactorStage stage = FactorStage::Cleanup;
  FactorKind kind = FactorKind::EndTurn;
  std::vector<CmabChoice> choices;
};

struct CmabChoiceStats {
  int visits = 0;
  double value_sum_root = 0.0;
  double prior = 0.0;
};

struct CmabStats {
  int samples = 0;
  std::unordered_map<std::string, CmabChoiceStats> choice_stats;
};

struct TurnPlanChoice {
  std::string choice_key;
  std::string action_signature;
  FactorStage stage = FactorStage::Cleanup;
  FactorKind kind = FactorKind::EndTurn;
};

struct TurnPlan {
  std::vector<TurnPlanChoice> choices;
  std::vector<std::string> executed_action_signatures;
  std::vector<std::string> executed_action_ids;
  std::string first_action_id;
  std::string first_action_signature;
  int first_action_root_index = -1;
  NativeGameState result_state;
  bool reached_turn_boundary = false;
  bool terminal = false;
  int primitives_executed = 0;
  double value_root = 0.0;
};

std::string action_signature(const NativeAction& action);
int find_legal_action_by_signature(
    const NativeGameState& state,
    const std::vector<NativeAction>& actions,
    const std::string& signature);
bool is_end_turn_signature(const std::string& signature);

std::vector<CmabFactor> build_factors_for_stage(
    const NativeGameState& state,
    const std::vector<NativeAction>& actions,
    FactorStage stage,
    const TurnCmabConfig& config);

class TurnCmabMCTS {
 public:
  TurnCmabMCTS(
      const py::dict& root_payload,
      const TurnCmabConfig& config,
      uint64_t seed);

  void run(int simulations);
  py::dict result_py(double temperature, bool sample_action);
  py::dict profile_py() const;
  int root_edge_count() const;

 private:
  struct TurnEdge;
  struct TurnNode;

  std::vector<NativeAction> actions_;
  std::vector<NativeGameState> states_;
  std::vector<TurnNode> nodes_;
  TurnCmabConfig config_;
  std::mt19937_64 rng_;
  int root_player_id_ = 0;
  int primitive_actions_executed_ = 0;
  int max_primitives_per_turn_sample_ = 0;
  int turn_depth_sum_ = 0;
  int max_turn_depth_reached_ = 0;
  double search_loop_ms_ = 0.0;
  double factor_build_ms_ = 0.0;
  double cmab_select_ms_ = 0.0;
  double apply_action_ms_ = 0.0;
  double static_eval_ms_ = 0.0;
  double backup_ms_ = 0.0;

  int make_turn_node(NativeGameState state);
  TurnEdge sample_new_turn_edge(int node_id);
  int select_existing_turn_edge(const TurnNode& node) const;
  int progressive_edge_limit(const TurnNode& node) const;
  void backup_turn_path(
      const std::vector<int>& node_ids,
      const std::vector<int>& edge_ids,
      double value_root);
  double evaluate_state_root_perspective(const NativeGameState& state);
  double player_sign_for_state(const NativeGameState& state) const;
};

}  // namespace tribes::native
