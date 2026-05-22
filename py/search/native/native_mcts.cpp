#include <torch/extension.h>

#include <pybind11/stl.h>

#include "native_rules.hpp"
#include "native_static_eval.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
using tribes::native::NativeAction;
using tribes::native::NativeGameState;
using tribes::native::NativeRoot;
using tribes::native::apply_action_strict;
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

class NativeMCTS {
 public:
  NativeMCTS(
      const py::dict& root_payload,
      const std::vector<int>& root_action_indexes,
      const std::vector<double>& root_priors,
      double root_value,
      bool root_terminal,
      uint64_t seed,
      int max_actions) : max_actions_(max_actions), rng_(seed) {
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
    states_.push_back(root.state);
    nodes_.push_back(make_node(0, root_priors, root_value, root.state.terminal));
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

  py::dict select_leaf(int max_depth, double c_puct) {
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
    const bool unlimited_depth = max_depth <= 0;

    for (int depth = 0; unlimited_depth || depth < max_depth; ++depth) {
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
      out["leaf_payload"] = serialize_evaluation_payload(child_state, actions_);
    } else if (leaf_state_index >= 0) {
      out["leaf_active_player_id"] = states_[leaf_state_index].active_player_id;
      out["leaf_payload"] = serialize_evaluation_payload(states_[leaf_state_index], actions_);
    } else {
      out["leaf_active_player_id"] = states_[nodes_[0].state_index].active_player_id;
      out["leaf_payload"] = serialize_evaluation_payload(states_[nodes_[0].state_index], actions_);
    }
    return out;
  }

  py::list select_leaf_batch(int frontier, int max_depth, double c_puct) {
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
      const bool unlimited_depth = max_depth <= 0;

      for (int depth = 0; unlimited_depth || depth < max_depth; ++depth) {
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
        selection["leaf_payload"] = serialize_evaluation_payload(*selected_leaf_state, actions_);
      }
      out.append(selection);
    }
    return out;
  }

  py::list select_leaf_batch_compact(int frontier, int max_depth, double c_puct) {
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
      const bool unlimited_depth = max_depth <= 0;

      for (int depth = 0; unlimited_depth || depth < max_depth; ++depth) {
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
        leaf_payload = serialize_evaluation_payload(*selected_leaf_state, actions_);
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

  py::list select_leaf_batch_evals_only(int frontier, int max_depth, double c_puct) {
    py::list out;
    last_batch_depth_sum_ = 0;
    last_batch_max_depth_ = 0;
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
      const bool unlimited_depth = max_depth <= 0;

      for (int depth = 0; unlimited_depth || depth < max_depth; ++depth) {
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

      const int selected_depth = static_cast<int>(path_node_ids.size());
      last_batch_depth_sum_ += selected_depth;
      last_batch_max_depth_ = std::max(last_batch_max_depth_, selected_depth);

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
          serialize_evaluation_payload(*selected_leaf_state, actions_)));
    }
    return out;
  }

  py::tuple last_batch_stats() const {
    return py::make_tuple(last_batch_depth_sum_, last_batch_max_depth_);
  }

  py::tuple select_leaf_batches_evals_only(int frontier, int max_batches, int max_depth, double c_puct) {
    py::list out;
    last_batch_depth_sum_ = 0;
    last_batch_max_depth_ = 0;
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
        const bool unlimited_depth = max_depth <= 0;
        bool grouped_duplicate = false;

        for (int depth = 0; unlimited_depth || depth < max_depth; ++depth) {
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
            serialize_evaluation_payload(*selected_leaf_state, actions_)));
      }
      if (py::len(out) >= target_evaluations) {
        break;
      }
    }
    return py::make_tuple(out, completed_simulations);
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
    states_.push_back(child_state);
    int child_node_id = static_cast<int>(nodes_.size());
    nodes_[parent_node_id].child_node_ids[parent_action_index] = child_node_id;
    nodes_.push_back(make_node(state_index, child_priors, child_value, child_state.terminal));
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

 private:
  std::vector<NativeAction> actions_;
  std::vector<int> root_action_indexes_;
  std::vector<NativeGameState> states_;
  std::vector<Node> nodes_;
  std::unordered_map<int64_t, NativeGameState> pending_child_states_;
  std::vector<PendingSelection> pending_selections_;
  int64_t last_batch_depth_sum_ = 0;
  int last_batch_max_depth_ = 0;
  int max_actions_ = 0;
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
    node.priors = terminal ? std::vector<double>() : normalize_priors(priors, state.legal_action_indexes.size());
    node.visits.assign(node.priors.size(), 0);
    node.value_sums.assign(node.priors.size(), 0.0);
    node.child_node_ids.assign(node.priors.size(), -1);
    node.value_estimate = value;
    node.terminal = terminal;
    return node;
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

  static int select_action_index(const Node& node, double c_puct) {
    if (node.priors.size() <= 1) {
      return 0;
    }
    const double sqrt_total = std::sqrt(std::max(1.0, static_cast<double>(node.total_visits)));
    double best_score = -std::numeric_limits<double>::infinity();
    int best_index = 0;
    for (size_t i = 0; i < node.priors.size(); ++i) {
      const double visit_count = static_cast<double>(node.visits[i]);
      const double q = visit_count > 0.0 ? node.value_sums[i] / visit_count : 0.0;
      const double u = c_puct * node.priors[i] * sqrt_total / (1.0 + visit_count);
      const double score = q + u;
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
      py::class_<NativeMCTS>(m, "NativeMCTS")
      .def(py::init<
           const py::dict&,
           const std::vector<int>&,
           const std::vector<double>&,
           double,
           bool,
           uint64_t,
           int>())
      .def("add_root_dirichlet_noise", &NativeMCTS::add_root_dirichlet_noise)
      .def("select_leaf", &NativeMCTS::select_leaf)
      .def("select_leaf_batch", &NativeMCTS::select_leaf_batch)
      .def("select_leaf_batch_compact", &NativeMCTS::select_leaf_batch_compact)
      .def("select_leaf_batch_evals_only", &NativeMCTS::select_leaf_batch_evals_only)
      .def("select_leaf_batches_evals_only", &NativeMCTS::select_leaf_batches_evals_only)
      .def("last_batch_stats", &NativeMCTS::last_batch_stats)
      .def("expand", &NativeMCTS::expand)
      .def("backprop", &NativeMCTS::backprop)
      .def("reserve_path", &NativeMCTS::reserve_path)
      .def("complete_reserved_path", &NativeMCTS::complete_reserved_path)
      .def("complete_selected_paths", &NativeMCTS::complete_selected_paths)
      .def("root_visit_distribution", &NativeMCTS::root_visit_distribution);
  m.def("evaluate_static", &tribes::native::evaluate_static);
  m.def("evaluate_static_batch", &tribes::native::evaluate_static_batch);
}
