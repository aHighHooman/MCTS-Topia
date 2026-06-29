#ifdef TRIBES_NATIVE_MCTS_STANDALONE
#include "json_py.hpp"
#else
#include <torch/extension.h>

#include <pybind11/stl.h>
#endif

#include "rules.hpp"
#include "static_eval.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#ifndef TRIBES_NATIVE_MCTS_STANDALONE
namespace py = pybind11;
#endif
using tribes::native::NativeAction;
using tribes::native::NativeCity;
using tribes::native::NativeGameState;
using tribes::native::NativeRoot;
using tribes::native::NativeTile;
using tribes::native::NativeUnit;
using tribes::native::NativeTransitionTiming;
using tribes::native::StaticEvaluation;
using tribes::native::apply_action_strict;
using tribes::native::evaluate_static_state;
using tribes::native::last_transition_timing;
using tribes::native::parse_root_payload;
using tribes::native::serialize_evaluation_payload;
using tribes::native::value_to_root_perspective;

namespace {

struct Node {
  int state_index = 0;
  std::vector<double> priors;
  std::vector<int64_t> visits;
  std::vector<double> value_sums;
  std::vector<int> child_node_ids;
  int64_t total_visits = 0;
  double value_estimate = 0.0;
  bool terminal = false;
};

struct PendingSelection {
  std::vector<int> path_node_ids;
  std::vector<int> path_action_indexes;
  int leaf_active_player_id = 0;
  bool leaf_value_root_perspective = false;
  int parent_node_id = -1;
  int parent_action_index = -1;
  int repeat_count = 1;
};

int terrain_index(const std::string& value) {
  if (value == "PLAIN") return 0;
  if (value == "SHALLOW_WATER") return 1;
  if (value == "DEEP_WATER") return 2;
  if (value == "MOUNTAIN") return 3;
  if (value == "VILLAGE") return 4;
  if (value == "CITY") return 5;
  if (value == "FOREST") return 6;
  if (value == "FOG") return 7;
  return -1;
}

int resource_index(const std::string& value) {
  if (value == "FISH") return 0;
  if (value == "FRUIT") return 1;
  if (value == "ANIMAL") return 2;
  if (value == "STARFISH") return 3;
  if (value == "LIGHTHOUSE") return 4;
  if (value == "ORE") return 5;
  if (value == "CROPS") return 6;
  if (value == "RUINS") return 7;
  return -1;
}

int building_index(const std::string& value) {
  if (value == "PORT") return 0;
  if (value == "MINE") return 1;
  if (value == "FORGE") return 2;
  if (value == "FARM") return 3;
  if (value == "WINDMILL") return 4;
  if (value == "MARKET") return 5;
  if (value == "LUMBER_HUT") return 6;
  if (value == "SAWMILL") return 7;
  if (value == "TEMPLE") return 8;
  if (value == "WATER_TEMPLE") return 9;
  if (value == "FOREST_TEMPLE") return 10;
  if (value == "MOUNTAIN_TEMPLE") return 11;
  if (value == "ALTAR_OF_PEACE") return 12;
  if (value == "EMPERORS_TOMB") return 13;
  if (value == "EYE_OF_GOD") return 14;
  if (value == "GATE_OF_POWER") return 15;
  if (value == "GRAND_BAZAR") return 16;
  if (value == "PARK_OF_FORTUNE") return 17;
  if (value == "TOWER_OF_WISDOM") return 18;
  if (value == "EMBASSY") return 19;
  return -1;
}

int unit_status_index(const std::string& value) {
  if (value == "FRESH") return 0;
  if (value == "MOVED") return 1;
  if (value == "ATTACKED") return 2;
  if (value == "MOVED_AND_ATTACKED") return 3;
  if (value == "PUSHED") return 4;
  if (value == "FINISHED") return 5;
  return -1;
}

class NativeMCTS {
 public:
  NativeMCTS(
      const py::dict& root_payload,
      const std::vector<int>& root_action_indexes,
      const std::vector<double>& root_priors,
      double root_value,
      bool root_terminal,
      uint64_t seed,
      int max_actions,
      bool use_progressive_widening = true,
      bool adversarial_opponent = false) :
      max_actions_(max_actions),
      use_progressive_widening_(use_progressive_widening),
      adversarial_opponent_(adversarial_opponent),
      rng_(seed) {
    NativeRoot root = parse_root_payload(root_payload, max_actions);
    actions_ = root.actions;
    if (!root_action_indexes.empty()) {
      root.state.legal_action_indexes.clear();
      for (int index : root_action_indexes) {
        if (index < 0 || index >= static_cast<int>(actions_.size())) {
          throw std::out_of_range("Root action index out of range.");
        }
        root.state.legal_action_indexes.push_back(index);
      }
    }
    root_action_indexes_ = root.state.legal_action_indexes;
    root.state.terminal = root.state.terminal || root_terminal || root.state.legal_action_indexes.empty();
    if (root.state.terminal && root.state.terminal_reason.empty()) {
      root.state.terminal_reason = "root_terminal";
    }
    const bool effective_root_terminal = root.state.terminal;
    states_.push_back(std::move(root.state));
    nodes_.push_back(make_node(0, root_priors, root_value, effective_root_terminal));
  }

  void add_root_dirichlet_noise(double alpha, double epsilon) {
    if (epsilon <= 0.0 || alpha <= 0.0 || nodes_.empty() || nodes_[0].priors.empty()) {
      return;
    }
    std::gamma_distribution<double> gamma(alpha, 1.0);
    std::vector<double> noise(nodes_[0].priors.size(), 0.0);
    double total = 0.0;
    for (double& value : noise) {
      value = gamma(rng_);
      total += value;
    }
    if (total <= 0.0) {
      return;
    }
    for (size_t i = 0; i < noise.size(); ++i) {
      noise[i] /= total;
      nodes_[0].priors[i] = (1.0 - epsilon) * nodes_[0].priors[i] + epsilon * noise[i];
    }
  }

  void reserve_tree_capacity(int capacity) {
    if (capacity <= 0) {
      return;
    }
    const size_t target = static_cast<size_t>(capacity);
    states_.reserve(std::max(states_.capacity(), target));
    nodes_.reserve(std::max(nodes_.capacity(), target));
    pending_selections_.reserve(std::max(pending_selections_.capacity(), target));
  }

  py::dict select_leaf(double c_puct) {
    std::vector<int> path_node_ids;
    std::vector<int> path_action_indexes;
    path_node_ids.reserve(8);
    path_action_indexes.reserve(8);
    int current_node_id = 0;
    double leaf_value = leaf_value_for(nodes_[0], states_[nodes_[0].state_index]);
    bool needs_expansion = false;
    int parent_node_id = -1;
    int parent_action_index = -1;
    std::vector<int> leaf_action_indexes;
    bool leaf_terminal = nodes_[0].terminal;
    const NativeGameState* selected_pending_child_state = nullptr;

    while (true) {
      const Node& node = nodes_[current_node_id];
      const NativeGameState& state = states_[node.state_index];
      leaf_terminal = node.terminal || state.terminal;
      if (leaf_terminal || state.legal_action_indexes.empty()) {
        leaf_value = leaf_value_for(node, state);
        leaf_action_indexes = state.legal_action_indexes;
        break;
      }

      const int local_action_index = select_action_index(node, c_puct);
      path_node_ids.push_back(current_node_id);
      path_action_indexes.push_back(local_action_index);

      const int child_node_id = node.child_node_ids[local_action_index];
      if (child_node_id < 0) {
        NativeGameState child_state = apply_action(state, state.legal_action_indexes[local_action_index]);
        needs_expansion = true;
        parent_node_id = current_node_id;
        parent_action_index = local_action_index;
        leaf_value = child_state.terminal ? terminal_value_for(child_state) : node.value_estimate;
        leaf_terminal = child_state.terminal;
        leaf_action_indexes = child_state.legal_action_indexes;
        auto [pending_it, inserted] = pending_child_states_.insert_or_assign(
            pending_child_key(parent_node_id, parent_action_index),
            std::move(child_state));
        (void)inserted;
        selected_pending_child_state = &pending_it->second;
        break;
      }

      current_node_id = child_node_id;
      const Node& child = nodes_[current_node_id];
      const NativeGameState& child_state = states_[child.state_index];
      leaf_terminal = child.terminal || child_state.terminal;
      leaf_value = leaf_value_for(child, child_state);
      leaf_action_indexes = child_state.legal_action_indexes;
      if (leaf_terminal) {
        break;
      }
    }
    if (!needs_expansion && leaf_action_indexes.empty()) {
      const Node& node = nodes_[current_node_id];
      const NativeGameState& state = states_[node.state_index];
      leaf_terminal = node.terminal || state.terminal;
      leaf_value = leaf_value_for(node, state);
      leaf_action_indexes = state.legal_action_indexes;
    }

    py::dict out;
    out["path_node_ids"] = path_node_ids;
    out["path_action_indexes"] = path_action_indexes;
    out["needs_expansion"] = needs_expansion;
    out["parent_node_id"] = parent_node_id;
    out["parent_action_index"] = parent_action_index;
    out["leaf_value"] = leaf_value;
    out["leaf_terminal"] = leaf_terminal;
    out["leaf_action_indexes"] = leaf_action_indexes;
    int leaf_state_index = needs_expansion
        ? -1
        : nodes_[current_node_id].state_index;
    if (needs_expansion) {
      const NativeGameState& child_state = selected_pending_child_state != nullptr
          ? *selected_pending_child_state
          : pending_child_states_[pending_child_key(parent_node_id, parent_action_index)];
      out["leaf_active_player_id"] = child_state.active_player_id;
      out["leaf_payload"] = serialize_leaf_payload(child_state);
    } else if (leaf_state_index >= 0) {
      out["leaf_active_player_id"] = states_[leaf_state_index].active_player_id;
      out["leaf_payload"] = serialize_leaf_payload(states_[leaf_state_index]);
    } else {
      out["leaf_active_player_id"] = states_[nodes_[0].state_index].active_player_id;
      out["leaf_payload"] = serialize_leaf_payload(states_[nodes_[0].state_index]);
    }
    return out;
  }

  py::list select_leaf_batch(int frontier, double c_puct) {
    py::list out;
    if (frontier <= 0) {
      return out;
    }
    for (int i = 0; i < frontier; ++i) {
      std::vector<int> path_node_ids;
      std::vector<int> path_action_indexes;
      path_node_ids.reserve(8);
      path_action_indexes.reserve(8);
      int current_node_id = 0;
      double leaf_value = leaf_value_for(nodes_[0], states_[nodes_[0].state_index]);
      bool needs_expansion = false;
      int parent_node_id = -1;
      int parent_action_index = -1;
      bool leaf_terminal = nodes_[0].terminal;
      int leaf_active_player_id = states_[nodes_[0].state_index].active_player_id;
      bool leaf_value_root_perspective = states_[nodes_[0].state_index].terminal_value_known;
      const NativeGameState* selected_leaf_state = &states_[nodes_[0].state_index];
      const NativeGameState* selected_pending_child_state = nullptr;

      while (true) {
        const Node& node = nodes_[current_node_id];
        const NativeGameState& state = states_[node.state_index];
        leaf_terminal = node.terminal || state.terminal;
        leaf_active_player_id = state.active_player_id;
        selected_leaf_state = &state;
        leaf_value_root_perspective = leaf_terminal && state.terminal_value_known;
        if (leaf_terminal || state.legal_action_indexes.empty()) {
          leaf_value = leaf_value_for(node, state);
          break;
        }

        const int local_action_index = select_action_index(node, c_puct);
        path_node_ids.push_back(current_node_id);
        path_action_indexes.push_back(local_action_index);

        const int child_node_id = node.child_node_ids[local_action_index];
        if (child_node_id < 0) {
          NativeGameState child_state = apply_action(state, state.legal_action_indexes[local_action_index]);
          needs_expansion = true;
          parent_node_id = current_node_id;
          parent_action_index = local_action_index;
          leaf_value = child_state.terminal ? terminal_value_for(child_state) : node.value_estimate;
          leaf_terminal = child_state.terminal;
          auto [pending_it, inserted] = pending_child_states_.insert_or_assign(
              pending_child_key(parent_node_id, parent_action_index),
              std::move(child_state));
          (void)inserted;
          selected_pending_child_state = &pending_it->second;
          selected_leaf_state = selected_pending_child_state;
          leaf_active_player_id = selected_pending_child_state->active_player_id;
          leaf_value_root_perspective = leaf_terminal && selected_pending_child_state->terminal_value_known;
          break;
        }

        current_node_id = child_node_id;
        const Node& child = nodes_[current_node_id];
        const NativeGameState& child_state = states_[child.state_index];
        leaf_terminal = child.terminal || child_state.terminal;
        leaf_value = leaf_value_for(child, child_state);
        leaf_active_player_id = child_state.active_player_id;
        selected_leaf_state = &child_state;
        leaf_value_root_perspective = leaf_terminal && child_state.terminal_value_known;
        if (leaf_terminal) {
          break;
        }
      }

      reserve_path_internal(path_node_ids, path_action_indexes);

      PendingSelection pending;
      pending.path_node_ids = std::move(path_node_ids);
      pending.path_action_indexes = std::move(path_action_indexes);
      pending.leaf_active_player_id = leaf_active_player_id;
      pending.leaf_value_root_perspective = leaf_value_root_perspective;
      pending.parent_node_id = parent_node_id;
      pending.parent_action_index = parent_action_index;
      const int selection_id = static_cast<int>(pending_selections_.size());
      pending_selections_.push_back(std::move(pending));

      py::dict selection;
      selection["selection_id"] = selection_id;
      selection["needs_expansion"] = needs_expansion;
      selection["parent_node_id"] = parent_node_id;
      selection["parent_action_index"] = parent_action_index;
      selection["leaf_value"] = leaf_value;
      selection["leaf_terminal"] = leaf_terminal;
      selection["leaf_active_player_id"] = leaf_active_player_id;
      selection["state_key"] = py::int_(
          static_cast<int64_t>(parent_node_id) * 1000000LL +
          static_cast<int64_t>(parent_action_index));
      if (needs_expansion && !leaf_terminal && selected_leaf_state != nullptr) {
        selection["leaf_payload"] = serialize_leaf_payload(*selected_leaf_state);
      }
      out.append(selection);
    }
    return out;
  }

  py::list select_leaf_batch_compact(int frontier, double c_puct) {
    py::list out;
    if (frontier <= 0) {
      return out;
    }
    for (int i = 0; i < frontier; ++i) {
      std::vector<int> path_node_ids;
      std::vector<int> path_action_indexes;
      path_node_ids.reserve(8);
      path_action_indexes.reserve(8);
      int current_node_id = 0;
      double leaf_value = leaf_value_for(nodes_[0], states_[nodes_[0].state_index]);
      bool needs_expansion = false;
      int parent_node_id = -1;
      int parent_action_index = -1;
      bool leaf_terminal = nodes_[0].terminal;
      int leaf_active_player_id = states_[nodes_[0].state_index].active_player_id;
      bool leaf_value_root_perspective = states_[nodes_[0].state_index].terminal_value_known;
      const NativeGameState* selected_leaf_state = &states_[nodes_[0].state_index];
      const NativeGameState* selected_pending_child_state = nullptr;

      while (true) {
        const Node& node = nodes_[current_node_id];
        const NativeGameState& state = states_[node.state_index];
        leaf_terminal = node.terminal || state.terminal;
        leaf_active_player_id = state.active_player_id;
        selected_leaf_state = &state;
        leaf_value_root_perspective = leaf_terminal && state.terminal_value_known;
        if (leaf_terminal || state.legal_action_indexes.empty()) {
          leaf_value = leaf_value_for(node, state);
          break;
        }

        const int local_action_index = select_action_index(node, c_puct);
        path_node_ids.push_back(current_node_id);
        path_action_indexes.push_back(local_action_index);

        const int child_node_id = node.child_node_ids[local_action_index];
        if (child_node_id < 0) {
          NativeGameState child_state = apply_action(state, state.legal_action_indexes[local_action_index]);
          needs_expansion = true;
          parent_node_id = current_node_id;
          parent_action_index = local_action_index;
          leaf_value = child_state.terminal ? terminal_value_for(child_state) : node.value_estimate;
          leaf_terminal = child_state.terminal;
          auto [pending_it, inserted] = pending_child_states_.insert_or_assign(
              pending_child_key(parent_node_id, parent_action_index),
              std::move(child_state));
          (void)inserted;
          selected_pending_child_state = &pending_it->second;
          selected_leaf_state = selected_pending_child_state;
          leaf_active_player_id = selected_pending_child_state->active_player_id;
          leaf_value_root_perspective = leaf_terminal && selected_pending_child_state->terminal_value_known;
          break;
        }

        current_node_id = child_node_id;
        const Node& child = nodes_[current_node_id];
        const NativeGameState& child_state = states_[child.state_index];
        leaf_terminal = child.terminal || child_state.terminal;
        leaf_value = leaf_value_for(child, child_state);
        leaf_active_player_id = child_state.active_player_id;
        selected_leaf_state = &child_state;
        leaf_value_root_perspective = leaf_terminal && child_state.terminal_value_known;
        if (leaf_terminal) {
          break;
        }
      }

      reserve_path_internal_unchecked(path_node_ids, path_action_indexes);

      PendingSelection pending;
      pending.path_node_ids = std::move(path_node_ids);
      pending.path_action_indexes = std::move(path_action_indexes);
      pending.leaf_active_player_id = leaf_active_player_id;
      pending.leaf_value_root_perspective = leaf_value_root_perspective;
      pending.parent_node_id = parent_node_id;
      pending.parent_action_index = parent_action_index;
      const int selection_id = static_cast<int>(pending_selections_.size());
      pending_selections_.push_back(std::move(pending));

      py::object leaf_payload = py::none();
      if (needs_expansion && !leaf_terminal && selected_leaf_state != nullptr) {
        leaf_payload = serialize_leaf_payload(*selected_leaf_state);
      }
      const int64_t state_key =
          static_cast<int64_t>(parent_node_id) * 1000000LL +
          static_cast<int64_t>(parent_action_index);
      out.append(py::make_tuple(
          selection_id,
          needs_expansion,
          parent_node_id,
          parent_action_index,
          leaf_value,
          leaf_terminal,
          state_key,
          leaf_payload));
    }
    return out;
  }

  py::list select_leaf_batch_evals_only(int frontier, double c_puct) {
    py::list out;
    last_batch_depth_sum_ = 0;
    last_batch_max_depth_ = 0;
    last_batch_turn_depth_sum_ = 0;
    last_batch_max_turn_depth_ = 0;
    if (frontier <= 0) {
      return out;
    }
    for (int i = 0; i < frontier; ++i) {
      std::vector<int> path_node_ids;
      std::vector<int> path_action_indexes;
      path_node_ids.reserve(8);
      path_action_indexes.reserve(8);
      int turn_depth = 0;
      int current_node_id = 0;
      double leaf_value = leaf_value_for(nodes_[0], states_[nodes_[0].state_index]);
      bool needs_expansion = false;
      int parent_node_id = -1;
      int parent_action_index = -1;
      bool leaf_terminal = nodes_[0].terminal;
      int leaf_active_player_id = states_[nodes_[0].state_index].active_player_id;
      bool leaf_value_root_perspective = states_[nodes_[0].state_index].terminal_value_known;
      const NativeGameState* selected_leaf_state = &states_[nodes_[0].state_index];
      const NativeGameState* selected_pending_child_state = nullptr;

      while (true) {
        const Node& node = nodes_[current_node_id];
        const NativeGameState& state = states_[node.state_index];
        leaf_terminal = node.terminal || state.terminal;
        leaf_active_player_id = state.active_player_id;
        selected_leaf_state = &state;
        leaf_value_root_perspective = leaf_terminal && state.terminal_value_known;
        if (leaf_terminal || state.legal_action_indexes.empty()) {
          leaf_value = leaf_value_for(node, state);
          break;
        }

        const int local_action_index = select_action_index(node, c_puct);
        path_node_ids.push_back(current_node_id);
        path_action_indexes.push_back(local_action_index);
        if (is_end_turn_action(state.legal_action_indexes[local_action_index])) {
          turn_depth += 1;
        }

        const int child_node_id = node.child_node_ids[local_action_index];
        if (child_node_id < 0) {
          NativeGameState child_state = apply_action(state, state.legal_action_indexes[local_action_index]);
          needs_expansion = true;
          parent_node_id = current_node_id;
          parent_action_index = local_action_index;
          leaf_value = child_state.terminal ? terminal_value_for(child_state) : node.value_estimate;
          leaf_terminal = child_state.terminal;
          auto [pending_it, inserted] = pending_child_states_.insert_or_assign(
              pending_child_key(parent_node_id, parent_action_index),
              std::move(child_state));
          (void)inserted;
          selected_pending_child_state = &pending_it->second;
          selected_leaf_state = selected_pending_child_state;
          leaf_active_player_id = selected_pending_child_state->active_player_id;
          leaf_value_root_perspective = leaf_terminal && selected_pending_child_state->terminal_value_known;
          break;
        }

        current_node_id = child_node_id;
        const Node& child = nodes_[current_node_id];
        const NativeGameState& child_state = states_[child.state_index];
        leaf_terminal = child.terminal || child_state.terminal;
        leaf_value = leaf_value_for(child, child_state);
        leaf_active_player_id = child_state.active_player_id;
        selected_leaf_state = &child_state;
        leaf_value_root_perspective = leaf_terminal && child_state.terminal_value_known;
        if (leaf_terminal) {
          break;
        }
      }

      const int selected_depth = static_cast<int>(path_node_ids.size());
      last_batch_depth_sum_ += selected_depth;
      last_batch_max_depth_ = std::max(last_batch_max_depth_, selected_depth);
      last_batch_turn_depth_sum_ += turn_depth;
      last_batch_max_turn_depth_ = std::max(last_batch_max_turn_depth_, turn_depth);

      if (!needs_expansion || leaf_terminal || selected_leaf_state == nullptr) {
        complete_path_internal_unchecked(
            path_node_ids,
            path_action_indexes,
            leaf_value,
            leaf_active_player_id,
            leaf_value_root_perspective);
        continue;
      }

      reserve_path_internal_unchecked(path_node_ids, path_action_indexes);

      PendingSelection pending;
      pending.path_node_ids = std::move(path_node_ids);
      pending.path_action_indexes = std::move(path_action_indexes);
      pending.leaf_active_player_id = leaf_active_player_id;
      pending.leaf_value_root_perspective = leaf_value_root_perspective;
      pending.parent_node_id = parent_node_id;
      pending.parent_action_index = parent_action_index;
      const int selection_id = static_cast<int>(pending_selections_.size());
      pending_selections_.push_back(std::move(pending));

      const int64_t state_key =
          static_cast<int64_t>(parent_node_id) * 1000000LL +
          static_cast<int64_t>(parent_action_index);
      out.append(py::make_tuple(
          selection_id,
          parent_node_id,
          parent_action_index,
          state_key,
          selected_depth,
          turn_depth,
          serialize_leaf_payload(*selected_leaf_state)));
    }
    return out;
  }

  py::tuple last_batch_stats() const {
    return py::make_tuple(
        last_batch_depth_sum_,
        last_batch_max_depth_,
        last_batch_turn_depth_sum_,
        last_batch_max_turn_depth_);
  }

  py::tuple select_leaf_batches_evals_only(int frontier, int max_batches, double c_puct) {
    py::list out;
    last_batch_depth_sum_ = 0;
    last_batch_max_depth_ = 0;
    last_batch_turn_depth_sum_ = 0;
    last_batch_max_turn_depth_ = 0;
    int completed_simulations = 0;
    if (frontier <= 0) {
      return py::make_tuple(out, completed_simulations);
    }
    const size_t target_evaluations = static_cast<size_t>(std::max(1, frontier));
    const int batches = std::max(1, max_batches);
    std::unordered_map<int64_t, int> primary_selection_by_eval_key;
    primary_selection_by_eval_key.reserve(target_evaluations);
    for (int batch = 0; batch < batches; ++batch) {
      for (int i = 0; i < frontier; ++i) {
        std::vector<int> path_node_ids;
        std::vector<int> path_action_indexes;
        path_node_ids.reserve(8);
        path_action_indexes.reserve(8);
        int turn_depth = 0;
        int current_node_id = 0;
        double leaf_value = leaf_value_for(nodes_[0], states_[nodes_[0].state_index]);
        bool needs_expansion = false;
        int parent_node_id = -1;
        int parent_action_index = -1;
        bool leaf_terminal = nodes_[0].terminal;
        int leaf_active_player_id = states_[nodes_[0].state_index].active_player_id;
        bool leaf_value_root_perspective = states_[nodes_[0].state_index].terminal_value_known;
        const NativeGameState* selected_leaf_state = &states_[nodes_[0].state_index];
        const NativeGameState* selected_pending_child_state = nullptr;
        bool grouped_duplicate = false;

        while (true) {
          const Node& node = nodes_[current_node_id];
          const NativeGameState& state = states_[node.state_index];
          leaf_terminal = node.terminal || state.terminal;
          leaf_active_player_id = state.active_player_id;
          selected_leaf_state = &state;
          leaf_value_root_perspective = leaf_terminal && state.terminal_value_known;
          if (leaf_terminal || state.legal_action_indexes.empty()) {
            leaf_value = leaf_value_for(node, state);
            break;
          }

          const int local_action_index = select_action_index(node, c_puct);
          path_node_ids.push_back(current_node_id);
          path_action_indexes.push_back(local_action_index);
          if (is_end_turn_action(state.legal_action_indexes[local_action_index])) {
            turn_depth += 1;
          }

          const int child_node_id = node.child_node_ids[local_action_index];
          if (child_node_id < 0) {
            needs_expansion = true;
            parent_node_id = current_node_id;
            parent_action_index = local_action_index;
            const int64_t duplicate_state_key =
                static_cast<int64_t>(parent_node_id) * 1000000LL +
                static_cast<int64_t>(parent_action_index);
            auto primary_it = primary_selection_by_eval_key.find(duplicate_state_key);
            if (primary_it != primary_selection_by_eval_key.end()) {
              const int selected_depth = static_cast<int>(path_node_ids.size());
              last_batch_depth_sum_ += selected_depth;
              last_batch_max_depth_ = std::max(last_batch_max_depth_, selected_depth);
              last_batch_turn_depth_sum_ += turn_depth;
              last_batch_max_turn_depth_ = std::max(last_batch_max_turn_depth_, turn_depth);
              completed_simulations += 1;
              reserve_path_internal_unchecked(path_node_ids, path_action_indexes);
              pending_selections_[primary_it->second].repeat_count += 1;
              grouped_duplicate = true;
              break;
            }

            NativeGameState child_state = apply_action(state, state.legal_action_indexes[local_action_index]);
            leaf_value = child_state.terminal ? terminal_value_for(child_state) : node.value_estimate;
            leaf_terminal = child_state.terminal;
            auto [pending_it, inserted] = pending_child_states_.insert_or_assign(
                pending_child_key(parent_node_id, parent_action_index),
                std::move(child_state));
            (void)inserted;
            selected_pending_child_state = &pending_it->second;
            selected_leaf_state = selected_pending_child_state;
            leaf_active_player_id = selected_pending_child_state->active_player_id;
            leaf_value_root_perspective = leaf_terminal && selected_pending_child_state->terminal_value_known;
            break;
          }

          current_node_id = child_node_id;
          const Node& child = nodes_[current_node_id];
          const NativeGameState& child_state = states_[child.state_index];
          leaf_terminal = child.terminal || child_state.terminal;
          leaf_value = leaf_value_for(child, child_state);
          leaf_active_player_id = child_state.active_player_id;
          selected_leaf_state = &child_state;
          leaf_value_root_perspective = leaf_terminal && child_state.terminal_value_known;
          if (leaf_terminal) {
            break;
          }
        }

        if (grouped_duplicate) {
          continue;
        }

        const int selected_depth = static_cast<int>(path_node_ids.size());
        last_batch_depth_sum_ += selected_depth;
        last_batch_max_depth_ = std::max(last_batch_max_depth_, selected_depth);
        last_batch_turn_depth_sum_ += turn_depth;
        last_batch_max_turn_depth_ = std::max(last_batch_max_turn_depth_, turn_depth);
        completed_simulations += 1;

        if (!needs_expansion || leaf_terminal || selected_leaf_state == nullptr) {
          complete_path_internal_unchecked(
              path_node_ids,
              path_action_indexes,
              leaf_value,
              leaf_active_player_id,
              leaf_value_root_perspective);
          continue;
        }

        reserve_path_internal_unchecked(path_node_ids, path_action_indexes);

        PendingSelection pending;
        pending.path_node_ids = std::move(path_node_ids);
        pending.path_action_indexes = std::move(path_action_indexes);
        pending.leaf_active_player_id = leaf_active_player_id;
        pending.leaf_value_root_perspective = leaf_value_root_perspective;
        pending.parent_node_id = parent_node_id;
        pending.parent_action_index = parent_action_index;
        const int selection_id = static_cast<int>(pending_selections_.size());
        pending_selections_.push_back(std::move(pending));

        const int64_t state_key =
            static_cast<int64_t>(parent_node_id) * 1000000LL +
            static_cast<int64_t>(parent_action_index);
        primary_selection_by_eval_key.emplace(state_key, selection_id);
        out.append(py::make_tuple(
            selection_id,
            parent_node_id,
            parent_action_index,
            state_key,
            selected_depth,
            turn_depth,
            serialize_leaf_payload(*selected_leaf_state)));
      }
      if (py::len(out) >= target_evaluations) {
        break;
      }
    }
    return py::make_tuple(out, completed_simulations);
  }

  py::tuple run_static_search_batch(int frontier, double c_puct) {
    last_batch_depth_sum_ = 0;
    last_batch_max_depth_ = 0;
    last_batch_turn_depth_sum_ = 0;
    last_batch_max_turn_depth_ = 0;
    reset_last_static_timing();
    int expanded_nodes = 0;
    int completed_simulations = 0;
    if (frontier <= 0) {
      return py::make_tuple(expanded_nodes, completed_simulations);
    }
    std::vector<int> path_node_ids;
    std::vector<int> path_action_indexes;
    path_node_ids.reserve(32);
    path_action_indexes.reserve(32);
    for (int i = 0; i < frontier; ++i) {
      path_node_ids.clear();
      path_action_indexes.clear();
      int turn_depth = 0;
      int current_node_id = 0;
      double leaf_value = leaf_value_for(nodes_[0], states_[nodes_[0].state_index]);
      bool needs_expansion = false;
      int parent_node_id = -1;
      int parent_action_index = -1;
      bool leaf_terminal = nodes_[0].terminal;
      int leaf_active_player_id = states_[nodes_[0].state_index].active_player_id;
      bool leaf_value_root_perspective = states_[nodes_[0].state_index].terminal_value_known;
      const NativeGameState* selected_leaf_state = &states_[nodes_[0].state_index];

      const auto select_started_at = static_timing_now();
      while (true) {
        const Node& node = nodes_[current_node_id];
        const NativeGameState& state = states_[node.state_index];
        leaf_terminal = node.terminal || state.terminal;
        leaf_active_player_id = state.active_player_id;
        selected_leaf_state = &state;
        leaf_value_root_perspective = leaf_terminal && state.terminal_value_known;
        if (leaf_terminal || state.legal_action_indexes.empty()) {
          leaf_value = leaf_value_for(node, state);
          break;
        }

        const int local_action_index = select_action_index(node, c_puct);
        path_node_ids.push_back(current_node_id);
        path_action_indexes.push_back(local_action_index);
        if (is_end_turn_action(state.legal_action_indexes[local_action_index])) {
          turn_depth += 1;
        }
        const int child_node_id = node.child_node_ids[local_action_index];
        if (child_node_id < 0) {
          needs_expansion = true;
          parent_node_id = current_node_id;
          parent_action_index = local_action_index;
          break;
        }

        current_node_id = child_node_id;
        const Node& child = nodes_[current_node_id];
        const NativeGameState& child_state = states_[child.state_index];
        leaf_terminal = child.terminal || child_state.terminal;
        leaf_value = leaf_value_for(child, child_state);
        leaf_active_player_id = child_state.active_player_id;
        selected_leaf_state = &child_state;
        leaf_value_root_perspective = leaf_terminal && child_state.terminal_value_known;
        if (leaf_terminal) {
          break;
        }
      }
      add_static_timing_ms(last_static_select_ms_, select_started_at);

      const int selected_depth = static_cast<int>(path_node_ids.size());
      last_batch_depth_sum_ += selected_depth;
      last_batch_max_depth_ = std::max(last_batch_max_depth_, selected_depth);
      last_batch_turn_depth_sum_ += turn_depth;
      last_batch_max_turn_depth_ = std::max(last_batch_max_turn_depth_, turn_depth);
      completed_simulations += 1;

      if (!needs_expansion) {
        const auto backup_started_at = static_timing_now();
        complete_path_internal_unchecked(
            path_node_ids,
            path_action_indexes,
            leaf_value,
            leaf_active_player_id,
            leaf_value_root_perspective);
        add_static_timing_ms(last_static_backup_ms_, backup_started_at);
        continue;
      }

      const auto apply_started_at = static_timing_now();
      NativeGameState child_state = apply_action(
          states_[nodes_[parent_node_id].state_index],
          states_[nodes_[parent_node_id].state_index].legal_action_indexes[parent_action_index]);
      add_transition_timing(last_transition_timing());
      add_static_timing_ms(last_static_apply_ms_, apply_started_at);
      leaf_terminal = child_state.terminal;
      leaf_active_player_id = child_state.active_player_id;
      leaf_value_root_perspective = leaf_terminal && child_state.terminal_value_known;
      StaticEvaluation evaluation;
      const auto eval_started_at = static_timing_now();
      if (leaf_terminal) {
        evaluation.value = terminal_value_for(child_state);
      } else {
        evaluation = evaluate_static_state(child_state, actions_);
      }
      add_static_timing_ms(last_static_eval_ms_, eval_started_at);

      const auto allocation_started_at = static_timing_now();
      child_state.terminal = child_state.terminal || leaf_terminal;
      const int state_index = static_cast<int>(states_.size());
      states_.push_back(std::move(child_state));
      const int child_node_id = static_cast<int>(nodes_.size());
      nodes_[parent_node_id].child_node_ids[parent_action_index] = child_node_id;
      nodes_.push_back(make_node(
          state_index,
          evaluation.priors,
          evaluation.value,
          states_[state_index].terminal));
      expanded_nodes += 1;
      add_static_timing_ms(last_static_allocation_ms_, allocation_started_at);
      const auto backup_started_at = static_timing_now();
      complete_path_internal_unchecked(
          path_node_ids,
          path_action_indexes,
          evaluation.value,
          leaf_active_player_id,
          leaf_value_root_perspective);
      add_static_timing_ms(last_static_backup_ms_, backup_started_at);
    }
    return py::make_tuple(expanded_nodes, completed_simulations);
  }

  void set_static_timing_enabled(bool enabled) {
    static_timing_enabled_ = enabled;
    reset_last_static_timing();
  }

  py::dict last_static_timing() const {
    py::dict out;
    out["select_ms"] = last_static_select_ms_;
    out["apply_action_ms"] = last_static_apply_ms_;
    out["static_eval_ms"] = last_static_eval_ms_;
    out["node_allocation_ms"] = last_static_allocation_ms_;
    out["backup_ms"] = last_static_backup_ms_;
    out["transition_state_copy_ms"] = last_transition_state_copy_ms_;
    out["transition_observation_copy_ms"] = last_transition_observation_copy_ms_;
    out["transition_action_mutation_ms"] = last_transition_action_mutation_ms_;
    out["transition_reveal_sync_ms"] = last_transition_reveal_sync_ms_;
    out["transition_regenerate_actions_ms"] = last_transition_regenerate_actions_ms_;
    out["transition_hidden_enemy_ms"] = last_transition_hidden_enemy_ms_;
    return out;
  }

  void complete_selected_paths(
      const std::vector<int>& selection_ids,
      const std::vector<double>& leaf_values) {
    if (selection_ids.size() != leaf_values.size()) {
      throw std::invalid_argument("Selection ids and leaf values must have the same length.");
    }
    for (size_t i = 0; i < selection_ids.size(); ++i) {
      const int selection_id = selection_ids[i];
      if (selection_id < 0 || selection_id >= static_cast<int>(pending_selections_.size())) {
        throw std::out_of_range("Selection id out of range.");
      }
      const PendingSelection& pending = pending_selections_[selection_id];
      for (int repeat = 0; repeat < std::max(1, pending.repeat_count); ++repeat) {
        complete_reserved_path_internal_unchecked(
            pending.path_node_ids,
            pending.path_action_indexes,
            leaf_values[i],
            pending.leaf_active_player_id,
            pending.leaf_value_root_perspective);
      }
      if (pending.parent_node_id >= 0 && pending.parent_action_index >= 0) {
        pending_child_states_.erase(pending_child_key(pending.parent_node_id, pending.parent_action_index));
      }
      pending_selections_[selection_id] = PendingSelection();
    }
  }

  int expand(
      int parent_node_id,
      int parent_action_index,
      const std::vector<double>& child_priors,
      double child_value,
      bool child_terminal) {
    validate_node_id(parent_node_id);
    if (parent_action_index < 0 || parent_action_index >= static_cast<int>(nodes_[parent_node_id].child_node_ids.size())) {
      throw std::out_of_range("Parent action index out of range.");
    }
    if (nodes_[parent_node_id].child_node_ids[parent_action_index] >= 0) {
      return nodes_[parent_node_id].child_node_ids[parent_action_index];
    }
    NativeGameState child_state;
    const int64_t pending_key = pending_child_key(parent_node_id, parent_action_index);
    auto pending_it = pending_child_states_.find(pending_key);
    if (pending_it != pending_child_states_.end()) {
      child_state = std::move(pending_it->second);
      pending_child_states_.erase(pending_it);
    } else {
      child_state = apply_action(
          states_[nodes_[parent_node_id].state_index],
          states_[nodes_[parent_node_id].state_index].legal_action_indexes[parent_action_index]);
    }
    child_state.terminal = child_state.terminal || child_terminal;
    if (!child_state.terminal && child_priors.size() != child_state.legal_action_indexes.size()) {
      throw std::invalid_argument("Child priors must match child legal action count.");
    }
    int state_index = static_cast<int>(states_.size());
    const bool effective_child_terminal = child_state.terminal;
    states_.push_back(std::move(child_state));
    int child_node_id = static_cast<int>(nodes_.size());
    nodes_[parent_node_id].child_node_ids[parent_action_index] = child_node_id;
    nodes_.push_back(make_node(state_index, child_priors, child_value, effective_child_terminal));
    return child_node_id;
  }

  void backprop(
      const std::vector<int>& path_node_ids,
      const std::vector<int>& path_action_indexes,
      double leaf_value) {
    if (path_node_ids.size() != path_action_indexes.size()) {
      throw std::invalid_argument("Path node ids and action indexes must have the same length.");
    }
    for (size_t i = 0; i < path_node_ids.size(); ++i) {
      validate_node_id(path_node_ids[i]);
      Node& node = nodes_[path_node_ids[i]];
      int action_index = path_action_indexes[i];
      if (action_index < 0 || action_index >= static_cast<int>(node.visits.size())) {
        throw std::out_of_range("Action index out of range during backup.");
      }
      node.visits[action_index] += 1;
      node.total_visits += 1;
      node.value_sums[action_index] += leaf_value;
    }
  }

  void reserve_path(
      const std::vector<int>& path_node_ids,
      const std::vector<int>& path_action_indexes) {
    reserve_path_internal(path_node_ids, path_action_indexes);
  }

  void reserve_path_internal(
      const std::vector<int>& path_node_ids,
      const std::vector<int>& path_action_indexes) {
    if (path_node_ids.size() != path_action_indexes.size()) {
      throw std::invalid_argument("Path node ids and action indexes must have the same length.");
    }
    for (size_t i = 0; i < path_node_ids.size(); ++i) {
      validate_node_id(path_node_ids[i]);
      Node& node = nodes_[path_node_ids[i]];
      int action_index = path_action_indexes[i];
      if (action_index < 0 || action_index >= static_cast<int>(node.visits.size())) {
        throw std::out_of_range("Action index out of range during reservation.");
      }
      node.visits[action_index] += 1;
      node.total_visits += 1;
    }
  }

  void reserve_path_internal_unchecked(
      const std::vector<int>& path_node_ids,
      const std::vector<int>& path_action_indexes) {
    for (size_t i = 0; i < path_node_ids.size(); ++i) {
      Node& node = nodes_[path_node_ids[i]];
      const int action_index = path_action_indexes[i];
      node.visits[action_index] += 1;
      node.total_visits += 1;
    }
  }

  void complete_reserved_path(
      const std::vector<int>& path_node_ids,
      const std::vector<int>& path_action_indexes,
      double leaf_value,
      int leaf_active_player_id) {
    complete_reserved_path_internal(path_node_ids, path_action_indexes, leaf_value, leaf_active_player_id);
  }

  void complete_reserved_path_internal(
      const std::vector<int>& path_node_ids,
      const std::vector<int>& path_action_indexes,
      double leaf_value,
      int leaf_active_player_id,
      bool leaf_value_root_perspective = false) {
    if (path_node_ids.size() != path_action_indexes.size()) {
      throw std::invalid_argument("Path node ids and action indexes must have the same length.");
    }
    for (size_t i = 0; i < path_node_ids.size(); ++i) {
      validate_node_id(path_node_ids[i]);
      Node& node = nodes_[path_node_ids[i]];
      int action_index = path_action_indexes[i];
      if (action_index < 0 || action_index >= static_cast<int>(node.value_sums.size())) {
        throw std::out_of_range("Action index out of range during reserved backup.");
      }
      const NativeGameState& edge_state = states_[node.state_index];
      node.value_sums[action_index] += leaf_value_root_perspective
          ? leaf_value
          : value_to_root_perspective(
                leaf_value,
                edge_state.root_player_id,
                leaf_active_player_id);
    }
  }

  void complete_reserved_path_internal_unchecked(
      const std::vector<int>& path_node_ids,
      const std::vector<int>& path_action_indexes,
      double leaf_value,
      int leaf_active_player_id,
      bool leaf_value_root_perspective = false) {
    for (size_t i = 0; i < path_node_ids.size(); ++i) {
      Node& node = nodes_[path_node_ids[i]];
      const int action_index = path_action_indexes[i];
      const NativeGameState& edge_state = states_[node.state_index];
      node.value_sums[action_index] += leaf_value_root_perspective
          ? leaf_value
          : value_to_root_perspective(
                leaf_value,
                edge_state.root_player_id,
                leaf_active_player_id);
    }
  }

  void complete_path_internal(
      const std::vector<int>& path_node_ids,
      const std::vector<int>& path_action_indexes,
      double leaf_value,
      int leaf_active_player_id,
      bool leaf_value_root_perspective = false) {
    if (path_node_ids.size() != path_action_indexes.size()) {
      throw std::invalid_argument("Path node ids and action indexes must have the same length.");
    }
    for (size_t i = 0; i < path_node_ids.size(); ++i) {
      validate_node_id(path_node_ids[i]);
      Node& node = nodes_[path_node_ids[i]];
      int action_index = path_action_indexes[i];
      if (action_index < 0 || action_index >= static_cast<int>(node.value_sums.size())) {
        throw std::out_of_range("Action index out of range during backup.");
      }
      const NativeGameState& edge_state = states_[node.state_index];
      node.visits[action_index] += 1;
      node.total_visits += 1;
      node.value_sums[action_index] += leaf_value_root_perspective
          ? leaf_value
          : value_to_root_perspective(
                leaf_value,
                edge_state.root_player_id,
                leaf_active_player_id);
    }
  }

  void complete_path_internal_unchecked(
      const std::vector<int>& path_node_ids,
      const std::vector<int>& path_action_indexes,
      double leaf_value,
      int leaf_active_player_id,
      bool leaf_value_root_perspective = false) {
    for (size_t i = 0; i < path_node_ids.size(); ++i) {
      Node& node = nodes_[path_node_ids[i]];
      const int action_index = path_action_indexes[i];
      const NativeGameState& edge_state = states_[node.state_index];
      node.visits[action_index] += 1;
      node.total_visits += 1;
      node.value_sums[action_index] += leaf_value_root_perspective
          ? leaf_value
          : value_to_root_perspective(
                leaf_value,
                edge_state.root_player_id,
                leaf_active_player_id);
    }
  }

  py::dict root_visit_distribution(double temperature) const {
    py::dict out;
    if (nodes_.empty()) {
      return out;
    }
    const Node& root = nodes_[0];
    const NativeGameState& root_state = states_[root.state_index];
    if (root_state.legal_action_indexes.empty()) {
      return out;
    }

    std::vector<double> probs(root.visits.size(), 0.0);
    if (temperature <= 1e-6) {
      auto best_it = std::max_element(root.visits.begin(), root.visits.end());
      probs[std::distance(root.visits.begin(), best_it)] = 1.0;
    } else {
      double total = 0.0;
      for (size_t i = 0; i < root.visits.size(); ++i) {
        probs[i] = std::pow(static_cast<double>(root.visits[i]), 1.0 / temperature);
        total += probs[i];
      }
      if (total <= 0.0) {
        std::fill(probs.begin(), probs.end(), 1.0 / static_cast<double>(probs.size()));
      } else {
        for (double& prob : probs) {
          prob /= total;
        }
      }
    }
    const std::vector<int>& output_indexes = root_action_indexes_.empty()
        ? root_state.legal_action_indexes
        : root_action_indexes_;
    for (size_t i = 0; i < output_indexes.size() && i < probs.size(); ++i) {
      const int action_index = output_indexes[i];
      if (action_index >= 0 && action_index < static_cast<int>(actions_.size())) {
        out[py::str(actions_[action_index].id)] = probs[i];
      }
    }
    return out;
  }

  std::vector<double> root_visit_distribution_by_index(double temperature) const {
    std::vector<double> probs;
    if (nodes_.empty()) {
      return probs;
    }
    const Node& root = nodes_[0];
    if (root.visits.empty()) {
      return probs;
    }
    probs.assign(root.visits.size(), 0.0);
    if (temperature <= 1e-6) {
      auto best_it = std::max_element(root.visits.begin(), root.visits.end());
      probs[std::distance(root.visits.begin(), best_it)] = 1.0;
      return probs;
    }
    double total = 0.0;
    for (size_t i = 0; i < root.visits.size(); ++i) {
      probs[i] = std::pow(static_cast<double>(root.visits[i]), 1.0 / temperature);
      total += probs[i];
    }
    if (total <= 0.0) {
      std::fill(probs.begin(), probs.end(), 1.0 / static_cast<double>(probs.size()));
    } else {
      for (double& prob : probs) {
        prob /= total;
      }
    }
    return probs;
  }

  py::list root_action_stats() const {
    py::list out;
    if (nodes_.empty()) {
      return out;
    }
    const Node& root = nodes_[0];
    const NativeGameState& root_state = states_[root.state_index];
    const std::vector<int>& output_indexes = root_action_indexes_.empty()
        ? root_state.legal_action_indexes
        : root_action_indexes_;
    int total_visits = 0;
    for (int visits : root.visits) {
      total_visits += visits;
    }
    for (size_t local = 0; local < output_indexes.size() && local < root.visits.size(); ++local) {
      const int global = output_indexes[local];
      py::dict row;
      row["action_id"] = global >= 0 && global < static_cast<int>(actions_.size())
          ? py::str(actions_[global].id)
          : py::str("");
      row["global_action_index"] = global;
      row["local_action_index"] = static_cast<int>(local);
      row["prior"] = local < root.priors.size() ? root.priors[local] : 0.0;
      row["visits"] = root.visits[local];
      row["visit_share"] = total_visits > 0
          ? static_cast<double>(root.visits[local]) / static_cast<double>(total_visits)
          : 0.0;
      row["value_sum"] = local < root.value_sums.size() ? root.value_sums[local] : 0.0;
      row["q_mean"] = root.visits[local] > 0 && local < root.value_sums.size()
          ? root.value_sums[local] / static_cast<double>(root.visits[local])
          : 0.0;
      row["child_node_id"] = local < root.child_node_ids.size() ? root.child_node_ids[local] : -1;
      out.append(row);
    }
    return out;
  }

  py::dict root_payload() const {
    py::dict out;
    if (nodes_.empty()) {
      return out;
    }
    return serialize_evaluation_payload(states_[nodes_[0].state_index], actions_);
  }

  py::list root_action_payloads() const {
    py::list out;
    if (nodes_.empty()) {
      return out;
    }
    const NativeGameState& root_state = states_[nodes_[0].state_index];
    for (int action_index : root_state.legal_action_indexes) {
      if (action_index >= 0 && action_index < static_cast<int>(actions_.size())) {
        out.append(actions_[action_index].payload);
      }
    }
    return out;
  }

  int node_count() const {
    return static_cast<int>(nodes_.size());
  }

  py::dict promote_root_child_by_action_id(const std::string& action_id) {
    py::dict out;
    out["ok"] = false;
    out["reason"] = py::str("not_found");
    out["previous_nodes"] = static_cast<int>(nodes_.size());
    out["promoted_subtree_nodes"] = 0;
    if (nodes_.empty()) {
      out["reason"] = py::str("empty_tree");
      return out;
    }
    const Node& root = nodes_[0];
    const NativeGameState& root_state = states_[root.state_index];
    int local_action_index = -1;
    for (size_t i = 0; i < root_state.legal_action_indexes.size() && i < root.child_node_ids.size(); ++i) {
      const int global_action_index = root_state.legal_action_indexes[i];
      if (global_action_index >= 0 &&
          global_action_index < static_cast<int>(actions_.size()) &&
          actions_[global_action_index].id == action_id) {
        local_action_index = static_cast<int>(i);
        break;
      }
    }
    if (local_action_index < 0) {
      return out;
    }
    const int child_node_id = root.child_node_ids[local_action_index];
    if (child_node_id < 0) {
      out["reason"] = py::str("child_not_expanded");
      return out;
    }
    validate_node_id(child_node_id);

    std::vector<int> order;
    std::vector<int> stack;
    std::vector<char> seen(nodes_.size(), 0);
    stack.push_back(child_node_id);
    seen[child_node_id] = 1;
    while (!stack.empty()) {
      const int node_id = stack.back();
      stack.pop_back();
      order.push_back(node_id);
      const Node& node = nodes_[node_id];
      for (int nested_child_id : node.child_node_ids) {
        if (nested_child_id >= 0) {
          validate_node_id(nested_child_id);
          if (!seen[nested_child_id]) {
            seen[nested_child_id] = 1;
            stack.push_back(nested_child_id);
          }
        }
      }
    }

    std::vector<int> node_remap(nodes_.size(), -1);
    for (size_t i = 0; i < order.size(); ++i) {
      node_remap[order[i]] = static_cast<int>(i);
    }
    std::unordered_map<int, int> state_remap;
    std::vector<NativeGameState> compact_states;
    compact_states.reserve(order.size());
    for (int old_node_id : order) {
      const int old_state_index = nodes_[old_node_id].state_index;
      if (state_remap.find(old_state_index) == state_remap.end()) {
        const int new_state_index = static_cast<int>(compact_states.size());
        state_remap.emplace(old_state_index, new_state_index);
        compact_states.push_back(states_[old_state_index]);
      }
    }

    std::vector<Node> compact_nodes;
    compact_nodes.reserve(order.size());
    for (int old_node_id : order) {
      Node node = nodes_[old_node_id];
      node.state_index = state_remap.at(node.state_index);
      for (int& nested_child_id : node.child_node_ids) {
        nested_child_id = nested_child_id >= 0 ? node_remap[nested_child_id] : -1;
      }
      compact_nodes.push_back(std::move(node));
    }

    states_ = std::move(compact_states);
    nodes_ = std::move(compact_nodes);
    pending_child_states_.clear();
    pending_selections_.clear();
    last_batch_depth_sum_ = 0;
    last_batch_max_depth_ = 0;
    last_batch_turn_depth_sum_ = 0;
    last_batch_max_turn_depth_ = 0;
    root_action_indexes_ = states_[nodes_[0].state_index].legal_action_indexes;

    out["ok"] = true;
    out["reason"] = py::str("");
    out["promoted_subtree_nodes"] = static_cast<int>(nodes_.size());
    out["root_payload"] = root_payload();
    return out;
  }

 private:
  std::vector<NativeAction> actions_;
  std::vector<int> root_action_indexes_;
  std::vector<NativeGameState> states_;
  std::vector<Node> nodes_;
  std::unordered_map<int64_t, NativeGameState> pending_child_states_;
  std::vector<PendingSelection> pending_selections_;
  int64_t last_batch_depth_sum_ = 0;
  int last_batch_max_depth_ = 0;
  int64_t last_batch_turn_depth_sum_ = 0;
  int last_batch_max_turn_depth_ = 0;
  int max_actions_ = 0;
  bool use_progressive_widening_ = true;
  bool adversarial_opponent_ = true;
  bool static_timing_enabled_ = false;
  double last_static_select_ms_ = 0.0;
  double last_static_apply_ms_ = 0.0;
  double last_static_eval_ms_ = 0.0;
  double last_static_allocation_ms_ = 0.0;
  double last_static_backup_ms_ = 0.0;
  double last_transition_state_copy_ms_ = 0.0;
  double last_transition_observation_copy_ms_ = 0.0;
  double last_transition_action_mutation_ms_ = 0.0;
  double last_transition_reveal_sync_ms_ = 0.0;
  double last_transition_regenerate_actions_ms_ = 0.0;
  double last_transition_hidden_enemy_ms_ = 0.0;
  mutable std::mt19937_64 rng_;

  static int64_t pending_child_key(int parent_node_id, int parent_action_index) {
    return (static_cast<int64_t>(parent_node_id) << 32) |
        static_cast<uint32_t>(parent_action_index);
  }

  static double terminal_value_for(const NativeGameState& state) {
    return state.terminal_value_known ? state.terminal_value : 0.0;
  }

  static double leaf_value_for(const Node& node, const NativeGameState& state) {
    return (node.terminal || state.terminal) ? terminal_value_for(state) : node.value_estimate;
  }

  Node make_node(int state_index, const std::vector<double>& priors, double value, bool terminal) const {
    const NativeGameState& state = states_[state_index];
    Node node;
    node.state_index = state_index;
    if (!terminal) {
      normalize_priors_into(node.priors, priors, state.legal_action_indexes.size());
    }
    node.visits.assign(node.priors.size(), 0);
    node.value_sums.assign(node.priors.size(), 0.0);
    node.child_node_ids.assign(node.priors.size(), -1);
    node.value_estimate = value;
    node.terminal = terminal;
    return node;
  }

  static void normalize_priors_into(std::vector<double>& out, const std::vector<double>& priors, size_t expected) {
    if (priors.size() != expected) {
      throw std::invalid_argument("Prior count does not match legal action count.");
    }
    out.resize(expected);
    double total = 0.0;
    for (size_t i = 0; i < expected; ++i) {
      const double value = std::max(0.0, priors[i]);
      out[i] = value;
      total += value;
    }
    if (total <= 0.0) {
      std::fill(out.begin(), out.end(), expected > 0 ? 1.0 / static_cast<double>(expected) : 0.0);
      return;
    }
    for (double& value : out) {
      value /= total;
    }
  }

  std::vector<double> normalize_priors(const std::vector<double>& priors, size_t expected) const {
    if (priors.size() != expected) {
      throw std::invalid_argument("Prior count does not match legal action count.");
    }
    std::vector<double> out = priors;
    double total = 0.0;
    for (double value : out) {
      total += std::max(0.0, value);
    }
    if (total <= 0.0) {
      std::fill(out.begin(), out.end(), expected > 0 ? 1.0 / static_cast<double>(expected) : 0.0);
      return out;
    }
    for (double& value : out) {
      value = std::max(0.0, value) / total;
    }
    return out;
  }

  NativeGameState apply_action(const NativeGameState& state, int global_action_index) {
    return apply_action_strict(state, actions_, global_action_index, max_actions_);
  }

  bool is_end_turn_action(int global_action_index) const {
    return global_action_index >= 0 &&
        global_action_index < static_cast<int>(actions_.size()) &&
        actions_[global_action_index].type == "END_TURN";
  }

  py::dict serialize_leaf_payload(const NativeGameState& state) const {
    py::dict payload = serialize_evaluation_payload(state, actions_);
#ifdef TRIBES_NATIVE_MCTS_STANDALONE
    return payload;
#else
    const int board_size = state.board_size;
    if (board_size <= 0) {
      return payload;
    }
    constexpr int board_channels = 82;
    constexpr int terrain_offset = 10;
    constexpr int resource_offset = 18;
    constexpr int building_offset = 26;
    constexpr int visible_index = 66;
    constexpr int visible_unit_owner_offset = 67;
    constexpr int visible_city_owner_offset = 70;
    constexpr int visible_unit_status_offset = 73;
    constexpr int visible_unit_hp_fraction_index = 79;
    constexpr int visible_city_level_index = 80;
    constexpr int visible_territory_owner_signed_index = 81;
    auto tensor = torch::zeros({board_channels, board_size, board_size}, torch::TensorOptions().dtype(torch::kFloat32));
    auto values = tensor.accessor<float, 3>();
    const float coord_scale = board_size > 1 ? static_cast<float>(board_size - 1) : 1.0f;
    for (int y = 0; y < board_size; ++y) {
      for (int x = 0; x < board_size; ++x) {
        values[3][y][x] = static_cast<float>(x) / coord_scale;
        values[4][y][x] = static_cast<float>(y) / coord_scale;
      }
    }

    std::unordered_map<int, const NativeUnit*> unit_by_id;
    unit_by_id.reserve(state.units.size());
    for (const NativeUnit& unit : state.units) {
      if (unit.id > 0) {
        unit_by_id.emplace(unit.id, &unit);
      }
    }
    std::unordered_map<int, const NativeCity*> city_by_id;
    city_by_id.reserve(state.cities.size());
    for (const NativeCity& city : state.cities) {
      if (city.id > 0) {
        city_by_id.emplace(city.id, &city);
      }
    }
    std::vector<bool> observation_visible(static_cast<size_t>(board_size * board_size), false);
    if (state.observation.contains("board") && py::isinstance<py::dict>(state.observation["board"])) {
      py::dict board = py::reinterpret_borrow<py::dict>(state.observation["board"]);
      if (board.contains("tiles") && py::isinstance<py::list>(board["tiles"])) {
        py::list rows = py::reinterpret_borrow<py::list>(board["tiles"]);
        for (int y = 0; y < std::min(board_size, static_cast<int>(py::len(rows))); ++y) {
          py::handle row_handle = rows[y];
          if (!py::isinstance<py::list>(row_handle)) {
            continue;
          }
          py::list row = py::reinterpret_borrow<py::list>(row_handle);
          for (int x = 0; x < std::min(board_size, static_cast<int>(py::len(row))); ++x) {
            py::handle tile_handle = row[x];
            if (!py::isinstance<py::dict>(tile_handle)) {
              continue;
            }
            py::dict tile = py::reinterpret_borrow<py::dict>(tile_handle);
            if (tile.contains("visible") && !tile["visible"].is_none()) {
              try {
                observation_visible[static_cast<size_t>(y * board_size + x)] = py::cast<bool>(tile["visible"]);
              } catch (const py::cast_error&) {
              }
            }
          }
        }
      }
    }

    int explored_tiles = 0;
    int visible_tiles = 0;
    for (const NativeTile& tile : state.tiles) {
      const int x = tile.x;
      const int y = tile.y;
      if (x < 0 || y < 0 || x >= board_size || y >= board_size) {
        continue;
      }
      values[0][y][x] = 1.0f;
      if (tile.explored) {
        explored_tiles += 1;
        values[2][y][x] = 1.0f;
      }
      const bool visible = observation_visible[static_cast<size_t>(y * board_size + x)];
      if (visible) {
        visible_tiles += 1;
        values[visible_index][y][x] = 1.0f;
      }
      values[5][y][x] = tile.unit_id > 0 ? 1.0f : 0.0f;
      values[6][y][x] = tile.city_id > 0 ? 1.0f : 0.0f;
      values[7][y][x] = tile.road ? 1.0f : 0.0f;
      const int terrain = terrain_index(tile.terrain);
      if (terrain >= 0) {
        values[terrain_offset + terrain][y][x] = 1.0f;
      }
      const int resource = resource_index(tile.resource);
      if (resource >= 0) {
        values[resource_offset + resource][y][x] = 1.0f;
      }
      const int building = building_index(tile.building);
      if (building >= 0) {
        values[building_offset + building][y][x] = 1.0f;
      }

      auto unit_it = unit_by_id.find(tile.unit_id);
      if (unit_it != unit_by_id.end()) {
        const NativeUnit& unit = *unit_it->second;
        const int owner_bucket = unit.tribe_id == state.active_player_id ? 0 : (unit.tribe_id >= 0 ? 1 : 2);
        values[visible_unit_owner_offset + owner_bucket][y][x] = 1.0f;
        values[visible_unit_hp_fraction_index][y][x] =
            static_cast<float>(unit.current_hp) / static_cast<float>(std::max(1, unit.max_hp));
        const int status = unit_status_index(unit.status);
        if (status >= 0) {
          values[visible_unit_status_offset + status][y][x] = 1.0f;
        }
      }
      auto city_it = city_by_id.find(tile.city_id);
      if (city_it != city_by_id.end()) {
        const NativeCity& city = *city_it->second;
        const int owner_bucket = city.tribe_id == state.active_player_id ? 0 : (city.tribe_id >= 0 ? 1 : 2);
        values[visible_city_owner_offset + owner_bucket][y][x] = 1.0f;
        values[visible_city_level_index][y][x] = std::min(1.0f, static_cast<float>(city.level) / 10.0f);
      }
      const int territory_city_id = tile.territory_city_id > 0 ? tile.territory_city_id : tile.city_id;
      auto territory_it = city_by_id.find(territory_city_id);
      if (territory_it != city_by_id.end()) {
        values[visible_territory_owner_signed_index][y][x] =
            territory_it->second->tribe_id == state.active_player_id ? 1.0f : -1.0f;
      }
    }
    payload["_native_board_tensor"] = tensor;
    payload["_native_explored_tiles"] = explored_tiles;
    payload["_native_visible_tiles"] = visible_tiles;
    return payload;
#endif
  }

  using StaticTimingClock = std::chrono::steady_clock;
  using StaticTimingPoint = StaticTimingClock::time_point;

  StaticTimingPoint static_timing_now() const {
    return static_timing_enabled_ ? StaticTimingClock::now() : StaticTimingPoint{};
  }

  void add_static_timing_ms(double& target, StaticTimingPoint started_at) {
    if (!static_timing_enabled_) {
      return;
    }
    const auto elapsed = StaticTimingClock::now() - started_at;
    target += std::chrono::duration<double, std::milli>(elapsed).count();
  }

  void reset_last_static_timing() {
    last_static_select_ms_ = 0.0;
    last_static_apply_ms_ = 0.0;
    last_static_eval_ms_ = 0.0;
    last_static_allocation_ms_ = 0.0;
    last_static_backup_ms_ = 0.0;
    last_transition_state_copy_ms_ = 0.0;
    last_transition_observation_copy_ms_ = 0.0;
    last_transition_action_mutation_ms_ = 0.0;
    last_transition_reveal_sync_ms_ = 0.0;
    last_transition_regenerate_actions_ms_ = 0.0;
    last_transition_hidden_enemy_ms_ = 0.0;
  }

  void add_transition_timing(const NativeTransitionTiming& timing) {
    if (!static_timing_enabled_) {
      return;
    }
    last_transition_state_copy_ms_ += timing.state_copy_ms;
    last_transition_observation_copy_ms_ += timing.observation_copy_ms;
    last_transition_action_mutation_ms_ += timing.action_mutation_ms;
    last_transition_reveal_sync_ms_ += timing.reveal_sync_ms;
    last_transition_regenerate_actions_ms_ += timing.regenerate_actions_ms;
    last_transition_hidden_enemy_ms_ += timing.hidden_enemy_ms;
  }

  static size_t progressive_action_count(const Node& node) {
    const size_t total = node.priors.size();
    if (total <= 8) {
      return total;
    }
    const size_t base = std::min<size_t>(8, total);
    const size_t unlocked = base + static_cast<size_t>(std::sqrt(std::max<int64_t>(0, node.total_visits)));
    return std::max<size_t>(1, std::min(total, unlocked));
  }

  int select_action_index(const Node& node, double c_puct) const {
    if (node.priors.size() <= 1) {
      return 0;
    }
    const NativeGameState& state = states_[node.state_index];
    const double player_sign =
        adversarial_opponent_ && state.active_player_id != state.root_player_id ? -1.0 : 1.0;
    const size_t action_count = use_progressive_widening_ ? progressive_action_count(node) : node.priors.size();
    const double sqrt_total = std::sqrt(std::max(1.0, static_cast<double>(node.total_visits)));
    double best_score = -std::numeric_limits<double>::infinity();
    int best_index = 0;
    for (size_t i = 0; i < action_count; ++i) {
      const double visit_count = static_cast<double>(node.visits[i]);
      const double q = visit_count > 0.0 ? node.value_sums[i] / visit_count : 0.0;
      const double u = c_puct * node.priors[i] * sqrt_total / (1.0 + visit_count);
      const double score = player_sign * q + u;
      if (score > best_score) {
        best_score = score;
        best_index = static_cast<int>(i);
      }
    }
    return best_index;
  }

  void validate_node_id(int node_id) const {
    if (node_id < 0 || node_id >= static_cast<int>(nodes_.size())) {
      throw std::out_of_range("Node id out of range.");
    }
  }
};

}  // namespace

#ifndef TRIBES_NATIVE_MCTS_STANDALONE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
      py::class_<NativeMCTS>(m, "NativeMCTS")
      .def(py::init<
           const py::dict&,
           const std::vector<int>&,
           const std::vector<double>&,
           double,
           bool,
           uint64_t,
           int,
           bool,
           bool>(),
           py::arg("root_payload"),
           py::arg("root_action_indexes"),
           py::arg("root_priors"),
           py::arg("root_value"),
           py::arg("root_terminal"),
           py::arg("seed"),
           py::arg("max_actions"),
           py::arg("use_progressive_widening") = true,
           py::arg("adversarial_opponent") = false)
      .def("add_root_dirichlet_noise", &NativeMCTS::add_root_dirichlet_noise)
      .def("reserve_tree_capacity", &NativeMCTS::reserve_tree_capacity)
      .def("select_leaf", &NativeMCTS::select_leaf)
      .def("select_leaf_batch", &NativeMCTS::select_leaf_batch)
      .def("select_leaf_batch_compact", &NativeMCTS::select_leaf_batch_compact)
      .def("select_leaf_batch_evals_only", &NativeMCTS::select_leaf_batch_evals_only)
      .def("select_leaf_batches_evals_only", &NativeMCTS::select_leaf_batches_evals_only)
      .def("run_static_search_batch", &NativeMCTS::run_static_search_batch)
      .def("set_static_timing_enabled", &NativeMCTS::set_static_timing_enabled)
      .def("last_static_timing", &NativeMCTS::last_static_timing)
      .def("last_batch_stats", &NativeMCTS::last_batch_stats)
      .def("expand", &NativeMCTS::expand)
      .def("backprop", &NativeMCTS::backprop)
      .def("reserve_path", &NativeMCTS::reserve_path)
      .def("complete_reserved_path", &NativeMCTS::complete_reserved_path)
      .def("complete_selected_paths", &NativeMCTS::complete_selected_paths)
      .def("root_visit_distribution", &NativeMCTS::root_visit_distribution)
      .def("root_visit_distribution_by_index", &NativeMCTS::root_visit_distribution_by_index)
      .def("root_action_stats", &NativeMCTS::root_action_stats)
      .def("root_payload", &NativeMCTS::root_payload)
      .def("root_action_payloads", &NativeMCTS::root_action_payloads)
      .def("node_count", &NativeMCTS::node_count)
      .def("promote_root_child_by_action_id", &NativeMCTS::promote_root_child_by_action_id);
  m.def("evaluate_static", &tribes::native::evaluate_static);
  m.def("evaluate_static_batch", &tribes::native::evaluate_static_batch);
  m.def("evaluate_static_breakdown", &tribes::native::evaluate_static_breakdown);
  m.def("evaluate_static_breakdown_batch", &tribes::native::evaluate_static_breakdown_batch);
  m.def("evaluate_action_breakdown", &tribes::native::evaluate_action_breakdown);
}
#endif
