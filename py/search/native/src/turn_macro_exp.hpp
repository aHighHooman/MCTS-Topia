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
  double macro_exp_prior_weight = 0.35;
  double macro_exp_temperature = 1.0;
  int inner_simulations = 128;
  double inner_c_puct = 1.5;
  bool profile_json = false;
  TurnMacroExpOpponentMode opponent_mode = TurnMacroExpOpponentMode::Maximalist;
};

struct MacroExpTurnPlan {
  std::vector<std::string> executed_macro_exp_action_signatures;
  std::vector<std::string> commitment_signatures;
  std::string first_action_id;
  NativeGameState result_state;
  bool reached_turn_boundary = false;
  bool terminal = false;
  int primitives_executed = 0;
  double value_root = 0.0;
};

std::string macro_exp_action_signature(const NativeAction& action);
std::string macro_exp_state_fingerprint(
    const NativeGameState& state,
    const std::vector<NativeAction>& actions);
int macro_exp_find_legal_action_by_signature(
    const NativeGameState& state,
    const std::vector<NativeAction>& actions,
    const std::string& signature);
bool macro_exp_is_end_turn_signature(const std::string& signature);

class TurnMacroExpMCTS {
 public:
  TurnMacroExpMCTS(
      const py::dict& root_payload,
      const TurnMacroExpConfig& config,
      uint64_t seed);

  void run(int simulations);
  void run_for(double seconds);
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
  double inner_search_ms_ = 0.0;
  double inner_setup_ms_ = 0.0;
  double inner_simulation_ms_ = 0.0;
  double inner_candidate_ms_ = 0.0;
  double apply_action_ms_ = 0.0;
  double static_eval_ms_ = 0.0;
  double backup_ms_ = 0.0;
  double edge_generation_ms_ = 0.0;
  double edge_generation_static_eval_ms_ = 0.0;
  double tree_policy_ms_ = 0.0;
  double node_creation_ms_ = 0.0;
  double edge_install_ms_ = 0.0;
  NativeTransitionTiming transition_timing_;
  NativeTransitionTiming inner_transition_timing_;
  int static_eval_calls_ = 0;
  int inner_searches_ = 0;
  int inner_simulations_executed_ = 0;
  int inner_nodes_expanded_ = 0;
  int macro_exp_raw_candidates_ = 0;
  int macro_exp_visited_candidates_ = 0;
  int macro_exp_value_candidates_ = 0;
  int macro_exp_prior_candidates_ = 0;
  int macro_exp_exact_duplicates_removed_ = 0;
  int macro_exp_material_duplicates_removed_ = 0;
  int opponent_leaf_evaluations_ = 0;
  int deadline_stops_ = 0;
  int64_t deadline_ns_ = 0;
  std::vector<int> path_node_scratch_;
  std::vector<int> path_edge_scratch_;

  int make_turn_node(NativeGameState state);
  int make_turn_node_with_value(NativeGameState state, double value_estimate_root);
  std::vector<TurnEdge> generate_turn_edges(int node_id, int max_edges);
  int select_existing_turn_edge(const TurnNode& node) const;
  int visible_edge_limit(const TurnNode& node) const;
  void backup_turn_path(
      const std::vector<int>& node_ids,
      const std::vector<int>& edge_ids,
      double value_root,
      const NativeGameState& leaf_state);
  double evaluate_state_for_player(const NativeGameState& state, int player_id);
  double evaluate_state_root_perspective(const NativeGameState& state);
};

}  // namespace tribes::native
