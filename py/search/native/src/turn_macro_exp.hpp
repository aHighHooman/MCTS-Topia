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

enum class TurnMacroExpOpponentMode {
  RootMax,
  Maximalist,
};

struct TurnMacroExpConfig {
  int simulations = 256;
  int max_actions = 512;
  // A non-positive value means no primitive-action limit per macro turn.
  int max_primitives_per_turn = 0;
  int max_new_edges_per_node = 8;
  int progressive_base = 1;
  double progressive_scale = 1.0;
  double outer_c = 1.4;
  double macro_exp_c = 1.0;
  double macro_exp_prior_weight = 0.35;
  double macro_exp_temperature = 1.0;
  int inner_simulations = 128;
  double inner_c_puct = 1.5;
  int greedy_eval_top_k = 1;
  int max_choices_per_factor = 12;
  int max_road_choices = 8;
  int max_resource_choices = 12;
  int max_city_choices = 8;
  int max_unit_choices = 8;
  bool deterministic = false;
  bool profile_json = false;
  TurnMacroExpOpponentMode opponent_mode = TurnMacroExpOpponentMode::Maximalist;
};

enum class MacroExpStage {
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

enum class MacroExpKind {
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

struct MacroExpChoice {
  std::string choice_key;
  std::string macro_exp_action_signature;
  int initial_action_index = -1;
  double prior = 0.0;
  MacroExpStage stage = MacroExpStage::Cleanup;
  MacroExpKind kind = MacroExpKind::EndTurn;
  int actor_id = 0;
};

struct MacroExpFactor {
  std::string factor_key;
  MacroExpStage stage = MacroExpStage::Cleanup;
  MacroExpKind kind = MacroExpKind::EndTurn;
  std::vector<MacroExpChoice> choices;
};

struct MacroExpChoiceStats {
  int visits = 0;
  double value_sum_root = 0.0;
  double prior = 0.0;
};

struct MacroExpStats {
  int samples = 0;
  std::unordered_map<std::string, MacroExpChoiceStats> choice_stats;
};

struct MacroExpTurnPlanChoice {
  std::string choice_key;
  std::string macro_exp_action_signature;
  MacroExpStage stage = MacroExpStage::Cleanup;
  MacroExpKind kind = MacroExpKind::EndTurn;
};

struct MacroExpTurnPlan {
  std::vector<MacroExpTurnPlanChoice> choices;
  std::vector<std::string> executed_macro_exp_action_signatures;
  std::vector<std::string> executed_action_ids;
  std::string first_action_id;
  std::string first_macro_exp_action_signature;
  int first_action_root_index = -1;
  NativeGameState result_state;
  bool reached_turn_boundary = false;
  bool terminal = false;
  int primitives_executed = 0;
  double value_root = 0.0;
  double value_actor = 0.0;
};

std::string macro_exp_action_signature(const NativeAction& action);
int macro_exp_find_legal_action_by_signature(
    const NativeGameState& state,
    const std::vector<NativeAction>& actions,
    const std::string& signature);
bool macro_exp_is_end_turn_signature(const std::string& signature);

std::vector<MacroExpFactor> macro_exp_build_factors_for_stage(
    const NativeGameState& state,
    const std::vector<NativeAction>& actions,
    MacroExpStage stage,
    const TurnMacroExpConfig& config);

class TurnMacroExpMCTS {
 public:
  TurnMacroExpMCTS(
      const py::dict& root_payload,
      const TurnMacroExpConfig& config,
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
  TurnMacroExpConfig config_;
  std::mt19937_64 rng_;
  int root_player_id_ = 0;
  int primitive_actions_executed_ = 0;
  int max_primitives_per_turn_sample_ = 0;
  int turn_depth_sum_ = 0;
  int max_turn_depth_reached_ = 0;
  double search_loop_ms_ = 0.0;
  double factor_build_ms_ = 0.0;
  double macro_exp_select_ms_ = 0.0;
  double inner_search_ms_ = 0.0;
  double apply_action_ms_ = 0.0;
  double static_eval_ms_ = 0.0;
  double backup_ms_ = 0.0;
  int static_eval_calls_ = 0;
  int greedy_static_calls_ = 0;
  int greedy_static_candidates_considered_ = 0;
  int greedy_static_child_evals_ = 0;
  int greedy_static_child_eval_skips_ = 0;
  int inner_searches_ = 0;
  int inner_simulations_executed_ = 0;
  int inner_nodes_expanded_ = 0;

  int make_turn_node(NativeGameState state);
  int make_turn_node_with_value(NativeGameState state, double value_estimate_root);
  TurnEdge sample_new_turn_edge(int node_id);
  std::vector<TurnEdge> generate_turn_edges(int node_id, int max_edges);
  int select_existing_turn_edge(const TurnNode& node) const;
  int visible_edge_limit(const TurnNode& node) const;
  void backup_turn_path(
      const std::vector<int>& node_ids,
      const std::vector<int>& edge_ids,
      double value_root);
  double evaluate_state_for_player(const NativeGameState& state, int player_id);
  double evaluate_state_root_perspective(const NativeGameState& state);
};

}  // namespace tribes::native
