#include "native_turn_cmab.hpp"

#include "native_static_eval.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace tribes::native {

namespace {

using Clock = std::chrono::steady_clock;

const char* stage_name(FactorStage stage) {
  switch (stage) {
    case FactorStage::Forced: return "forced";
    case FactorStage::Tactical: return "tactical";
    case FactorStage::Research: return "research";
    case FactorStage::City: return "city";
    case FactorStage::ResourceBuild: return "resource_build";
    case FactorStage::Road: return "road";
    case FactorStage::Diplomacy: return "diplomacy";
    case FactorStage::Cleanup: return "cleanup";
    case FactorStage::EndTurn: return "end_turn";
  }
  return "unknown";
}

bool is_forced_type(const std::string& type) {
  return type == "LEVEL_UP" || type == "MAKE_VETERAN" || type == "CAPTURE" || type == "EXAMINE";
}

bool is_tactical_type(const std::string& type) {
  return type == "ATTACK" || type == "CONVERT" || type == "INFILTRATE" ||
      type == "HEAL_OTHERS" || type == "MOVE" || type == "STEP_MOVE" ||
      type == "RECOVER" || type == "UPGRADE_RAMMER" || type == "UPGRADE_SCOUT" ||
      type == "UPGRADE_BOMBER" || type == "UPGRADE_SHIP" || type == "UPGRADE_BOAT";
}

bool is_resource_build_type(const std::string& type) {
  return type == "RESOURCE_GATHERING" || type == "BUILD" || type == "BURN_FOREST" ||
      type == "CLEAR_FOREST" || type == "GROW_FOREST" || type == "DESTROY";
}

bool is_diplomacy_type(const std::string& type) {
  return type == "BUILD_EMBASSY" || type == "PROPOSE_PEACE" || type == "ACCEPT_PEACE" ||
      type == "PROPOSE_TREATY" || type == "ACCEPT_TREATY" || type == "CANCEL_TREATY" ||
      type == "SEND_STARS";
}

bool belongs_to_stage(const NativeAction& action, FactorStage stage) {
  const std::string& type = action.type;
  switch (stage) {
    case FactorStage::Forced:
      return is_forced_type(type);
    case FactorStage::Tactical:
      return is_tactical_type(type);
    case FactorStage::Research:
      return type == "RESEARCH_TECH";
    case FactorStage::City:
      return type == "SPAWN";
    case FactorStage::ResourceBuild:
      return is_resource_build_type(type);
    case FactorStage::Road:
      return type == "BUILD_ROAD";
    case FactorStage::Diplomacy:
      return is_diplomacy_type(type);
    case FactorStage::Cleanup:
      return !is_forced_type(type) && !is_tactical_type(type) && type != "RESEARCH_TECH" &&
          type != "SPAWN" && !is_resource_build_type(type) && type != "BUILD_ROAD" &&
          !is_diplomacy_type(type) && type != "END_TURN";
    case FactorStage::EndTurn:
      return type == "END_TURN";
  }
  return false;
}

FactorKind kind_for_action(const NativeAction& action, FactorStage stage) {
  if (stage == FactorStage::Forced) return FactorKind::ForcedLevelUp;
  if (stage == FactorStage::Tactical) return FactorKind::Unit;
  if (stage == FactorStage::Research) return FactorKind::Research;
  if (stage == FactorStage::City) return FactorKind::City;
  if (stage == FactorStage::Road) return FactorKind::Road;
  if (stage == FactorStage::Diplomacy) return FactorKind::Diplomacy;
  if (stage == FactorStage::EndTurn) return FactorKind::EndTurn;
  if (action.type == "BUILD") return FactorKind::Build;
  if (action.type == "RESOURCE_GATHERING") return FactorKind::Resource;
  return FactorKind::Build;
}

int actor_id_for_action(const NativeAction& action, FactorStage stage) {
  if (stage == FactorStage::Tactical || action.unit_id > 0) return action.unit_id;
  if (stage == FactorStage::City || stage == FactorStage::ResourceBuild || action.city_id > 0) {
    return action.city_id;
  }
  return action.tribe_id;
}

std::string factor_key_for_action(const NativeAction& action, FactorStage stage, const NativeGameState& state) {
  std::ostringstream out;
  switch (stage) {
    case FactorStage::Forced:
      out << "forced:" << action.type << ":" << actor_id_for_action(action, stage);
      break;
    case FactorStage::Tactical:
      out << "unit:" << action.unit_id;
      break;
    case FactorStage::Research:
      out << "research:" << state.active_player_id;
      break;
    case FactorStage::City:
      out << "city:" << action.city_id;
      break;
    case FactorStage::ResourceBuild:
      out << "resource_build:" << action.city_id;
      break;
    case FactorStage::Road:
      out << "road:" << state.active_player_id;
      break;
    case FactorStage::Diplomacy:
      out << "diplomacy:" << state.active_player_id;
      break;
    case FactorStage::Cleanup:
      out << "cleanup:" << action.type << ":" << actor_id_for_action(action, stage);
      break;
    case FactorStage::EndTurn:
      out << "end_turn:" << state.active_player_id;
      break;
  }
  return out.str();
}

int cap_for_stage(FactorStage stage, const TurnCmabConfig& config) {
  switch (stage) {
    case FactorStage::Tactical: return config.max_unit_choices;
    case FactorStage::City: return config.max_city_choices;
    case FactorStage::ResourceBuild: return config.max_resource_choices;
    case FactorStage::Road: return config.max_road_choices;
    default: return config.max_choices_per_factor;
  }
}

double cmab_choice_score(
    const CmabChoiceStats& stats,
    double prior,
    int total_samples,
    double c,
    double prior_weight,
    double player_sign) {
  const double q = stats.visits > 0 ? stats.value_sum_root / static_cast<double>(stats.visits) : 0.0;
  const double u = c * std::sqrt(std::log(1.0 + static_cast<double>(total_samples)) / (1.0 + stats.visits));
  return player_sign * q + u + prior_weight * prior;
}

void record_choice(TurnPlan& plan, const CmabChoice& choice) {
  TurnPlanChoice selected;
  selected.choice_key = choice.choice_key;
  selected.action_signature = choice.action_signature;
  selected.stage = choice.stage;
  selected.kind = choice.kind;
  plan.choices.push_back(std::move(selected));
}

void record_execution(TurnPlan& plan, const NativeAction& action) {
  const std::string signature = action_signature(action);
  plan.executed_action_signatures.push_back(signature);
  plan.executed_action_ids.push_back(action.id);
  if (plan.first_action_id.empty()) {
    plan.first_action_id = action.id;
    plan.first_action_signature = signature;
  }
}

std::vector<FactorStage> stage_order_for_state(const NativeGameState& state) {
  if (state.leveling_up) {
    return {FactorStage::Forced};
  }
  return {
      FactorStage::Forced,
      FactorStage::Tactical,
      FactorStage::Research,
      FactorStage::City,
      FactorStage::ResourceBuild,
      FactorStage::Road,
      FactorStage::Diplomacy,
      FactorStage::Cleanup,
      FactorStage::EndTurn,
  };
}

}  // namespace

struct TurnCmabMCTS::TurnEdge {
  TurnPlan plan;
  int child_node_id = -1;
  int visits = 0;
  double value_sum_root = 0.0;
  double prior = 0.0;
};

struct TurnCmabMCTS::TurnNode {
  int state_index = -1;
  bool terminal = false;
  double value_estimate_root = 0.0;
  int visits = 0;
  std::vector<TurnEdge> edges;
  CmabStats cmab;
};

std::string action_signature(const NativeAction& action) {
  std::ostringstream out;
  out << action.type;
  if (action.unit_id > 0) out << ":u=" << action.unit_id;
  if (action.city_id > 0) out << ":c=" << action.city_id;
  if (action.tribe_id >= 0) out << ":p=" << action.tribe_id;
  if (action.target_unit_id > 0) out << ":tu=" << action.target_unit_id;
  if (action.target_city_id > 0) out << ":tc=" << action.target_city_id;
  if (action.target_player_id >= 0) out << ":tp=" << action.target_player_id;
  if (action.has_xy) out << ":x=" << action.x << ":y=" << action.y;
  if (!action.unit_type.empty()) out << ":unit=" << action.unit_type;
  if (!action.building_type.empty()) out << ":building=" << action.building_type;
  if (!action.resource_type.empty()) out << ":resource=" << action.resource_type;
  if (!action.capture_type.empty()) out << ":capture=" << action.capture_type;
  if (!action.bonus.empty()) out << ":bonus=" << action.bonus;
  if (!action.tech.empty()) out << ":tech=" << action.tech;
  return out.str();
}

int find_legal_action_by_signature(
    const NativeGameState& state,
    const std::vector<NativeAction>& actions,
    const std::string& signature) {
  if (signature.empty()) return -1;
  for (int action_index : state.legal_action_indexes) {
    if (action_index < 0 || action_index >= static_cast<int>(actions.size())) continue;
    if (action_signature(actions[action_index]) == signature) return action_index;
  }
  return -1;
}

bool is_end_turn_signature(const std::string& signature) {
  return signature == "END_TURN" || signature.rfind("END_TURN:", 0) == 0;
}

std::vector<CmabFactor> build_factors_for_stage(
    const NativeGameState& state,
    const std::vector<NativeAction>& actions,
    FactorStage stage,
    const TurnCmabConfig& config) {
  StaticEvaluation eval = evaluate_static_state(state, actions);
  std::unordered_map<int, double> prior_by_global_index;
  for (size_t local = 0; local < state.legal_action_indexes.size() && local < eval.priors.size(); ++local) {
    prior_by_global_index[state.legal_action_indexes[local]] = eval.priors[local];
  }

  std::unordered_map<std::string, size_t> factor_offsets;
  std::vector<CmabFactor> factors;
  for (int global_index : state.legal_action_indexes) {
    if (global_index < 0 || global_index >= static_cast<int>(actions.size())) continue;
    const NativeAction& action = actions[global_index];
    if (!belongs_to_stage(action, stage)) continue;

    const std::string factor_key = factor_key_for_action(action, stage, state);
    auto offset_it = factor_offsets.find(factor_key);
    if (offset_it == factor_offsets.end()) {
      offset_it = factor_offsets.emplace(factor_key, factors.size()).first;
      CmabFactor factor;
      factor.factor_key = factor_key;
      factor.stage = stage;
      factor.kind = kind_for_action(action, stage);
      factors.push_back(std::move(factor));
    }

    CmabChoice choice;
    choice.action_signature = action_signature(action);
    choice.choice_key = std::string(stage_name(stage)) + "|" + factor_key + "|" + choice.action_signature;
    choice.initial_action_index = global_index;
    choice.prior = prior_by_global_index.count(global_index) ? prior_by_global_index[global_index] : 0.0;
    choice.stage = stage;
    choice.kind = kind_for_action(action, stage);
    choice.actor_id = actor_id_for_action(action, stage);
    factors[offset_it->second].choices.push_back(std::move(choice));
  }

  const bool optional = stage != FactorStage::Forced && stage != FactorStage::EndTurn;
  for (CmabFactor& factor : factors) {
    std::sort(factor.choices.begin(), factor.choices.end(), [](const CmabChoice& left, const CmabChoice& right) {
      return left.prior > right.prior;
    });
    const int executable_cap = std::max(1, cap_for_stage(stage, config));
    if (static_cast<int>(factor.choices.size()) > executable_cap) {
      factor.choices.resize(executable_cap);
    }
    if (optional) {
      CmabChoice none;
      none.choice_key = std::string(stage_name(stage)) + "|" + factor.factor_key + "|none";
      none.prior = 0.05;
      none.stage = stage;
      none.kind = factor.kind;
      factor.choices.push_back(std::move(none));
    }
    double total = 0.0;
    for (const CmabChoice& choice : factor.choices) total += std::max(0.0, choice.prior);
    if (total <= 0.0) {
      const double uniform = factor.choices.empty() ? 0.0 : 1.0 / static_cast<double>(factor.choices.size());
      for (CmabChoice& choice : factor.choices) choice.prior = uniform;
    } else {
      for (CmabChoice& choice : factor.choices) choice.prior = std::max(0.0, choice.prior) / total;
    }
  }
  return factors;
}

TurnCmabMCTS::TurnCmabMCTS(
    const py::dict& root_payload,
    const TurnCmabConfig& config,
    uint64_t seed) :
    config_(config),
    rng_(seed) {
  NativeRoot root = parse_root_payload(root_payload, config_.max_actions);
  actions_ = std::move(root.actions);
  root_player_id_ = root.state.root_player_id;
  root.state.terminal = root.state.terminal || root.state.legal_action_indexes.empty();
  make_turn_node(std::move(root.state));
}

int TurnCmabMCTS::make_turn_node(NativeGameState state) {
  const int state_index = static_cast<int>(states_.size());
  const bool terminal = state.terminal || state.legal_action_indexes.empty();
  states_.push_back(std::move(state));
  TurnNode node;
  node.state_index = state_index;
  node.terminal = terminal;
  node.value_estimate_root = evaluate_state_root_perspective(states_[state_index]);
  const int node_id = static_cast<int>(nodes_.size());
  nodes_.push_back(std::move(node));
  return node_id;
}

double TurnCmabMCTS::evaluate_state_root_perspective(const NativeGameState& state) {
  if (state.terminal) {
    return state.terminal_value_known ? state.terminal_value : 0.0;
  }
  const auto started = Clock::now();
  StaticEvaluation eval = evaluate_static_state(state, actions_);
  if (config_.profile_json) {
    static_eval_ms_ += std::chrono::duration<double, std::milli>(Clock::now() - started).count();
  }
  return value_to_root_perspective(eval.value, root_player_id_, state.active_player_id);
}

double TurnCmabMCTS::player_sign_for_state(const NativeGameState& state) const {
  if (config_.opponent_mode == TurnCmabOpponentMode::RootMax) return 1.0;
  if (config_.opponent_mode == TurnCmabOpponentMode::ActiveSelfish) {
    return state.active_player_id == root_player_id_ ? 1.0 : -1.0;
  }
  return state.active_player_id == root_player_id_ ? 1.0 : -1.0;
}

int TurnCmabMCTS::progressive_edge_limit(const TurnNode& node) const {
  const int limit = config_.progressive_base +
      static_cast<int>(config_.progressive_scale * std::sqrt(std::max(1, node.visits)));
  return std::min(config_.max_new_edges_per_node, std::max(1, limit));
}

int TurnCmabMCTS::select_existing_turn_edge(const TurnNode& node) const {
  if (node.edges.empty()) return -1;
  const NativeGameState& state = states_[node.state_index];
  const double player_sign = player_sign_for_state(state);
  const double log_n = std::log(1.0 + static_cast<double>(std::max(1, node.visits)));
  double best_score = -std::numeric_limits<double>::infinity();
  int best = 0;
  for (int i = 0; i < static_cast<int>(node.edges.size()); ++i) {
    const TurnEdge& edge = node.edges[i];
    const double q = edge.visits > 0 ? edge.value_sum_root / static_cast<double>(edge.visits) : 0.0;
    const double u = config_.outer_c * std::sqrt(log_n / (1.0 + edge.visits));
    const double score = player_sign * q + u + 0.10 * edge.prior;
    if (score > best_score) {
      best_score = score;
      best = i;
    }
  }
  return best;
}

TurnCmabMCTS::TurnEdge TurnCmabMCTS::sample_new_turn_edge(int node_id) {
  TurnNode& node = nodes_[node_id];
  const NativeGameState root_state = states_[node.state_index];
  NativeGameState current = root_state;
  TurnPlan plan;
  const int starting_player = current.active_player_id;
  int no_progress_rounds = 0;

  while (plan.primitives_executed < config_.max_primitives_per_turn) {
    if (current.terminal) break;
    if (current.active_player_id != starting_player) {
      plan.reached_turn_boundary = true;
      break;
    }

    bool applied_in_round = false;
    bool sampled_factor = false;
    for (FactorStage stage : stage_order_for_state(current)) {
      if (plan.primitives_executed >= config_.max_primitives_per_turn) break;
      const auto factor_started = Clock::now();
      std::vector<CmabFactor> factors = build_factors_for_stage(current, actions_, stage, config_);
      if (config_.profile_json) {
        factor_build_ms_ += std::chrono::duration<double, std::milli>(Clock::now() - factor_started).count();
      }
      if (factors.empty()) continue;

      for (const CmabFactor& factor : factors) {
        if (factor.choices.empty() || plan.primitives_executed >= config_.max_primitives_per_turn) continue;
        sampled_factor = true;
        const auto select_started = Clock::now();
        std::vector<double> scores;
        scores.reserve(factor.choices.size());
        for (const CmabChoice& choice : factor.choices) {
          CmabChoiceStats& stats = node.cmab.choice_stats[choice.choice_key];
          if (stats.visits == 0 && stats.prior == 0.0) stats.prior = choice.prior;
          scores.push_back(cmab_choice_score(
              stats,
              choice.prior,
              std::max(1, node.cmab.samples),
              config_.cmab_c,
              config_.cmab_prior_weight,
              player_sign_for_state(root_state)));
        }
        int selected = 0;
        if (config_.deterministic || config_.cmab_temperature <= 1e-9) {
          selected = static_cast<int>(std::max_element(scores.begin(), scores.end()) - scores.begin());
        } else {
          const double max_score = *std::max_element(scores.begin(), scores.end());
          std::vector<double> weights(scores.size(), 0.0);
          for (size_t i = 0; i < scores.size(); ++i) {
            weights[i] = std::exp((scores[i] - max_score) / std::max(1e-6, config_.cmab_temperature));
          }
          std::discrete_distribution<int> dist(weights.begin(), weights.end());
          selected = dist(rng_);
        }
        if (config_.profile_json) {
          cmab_select_ms_ += std::chrono::duration<double, std::milli>(Clock::now() - select_started).count();
        }

        const CmabChoice& choice = factor.choices[selected];
        record_choice(plan, choice);
        if (choice.action_signature.empty()) continue;

        const int action_index = find_legal_action_by_signature(current, actions_, choice.action_signature);
        if (action_index < 0) continue;
        const NativeAction& action = actions_[action_index];
        record_execution(plan, action);
        if (plan.first_action_root_index < 0) {
          for (size_t local = 0; local < root_state.legal_action_indexes.size(); ++local) {
            const int root_action_index = root_state.legal_action_indexes[local];
            if (root_action_index >= 0 &&
                root_action_index < static_cast<int>(actions_.size()) &&
                actions_[root_action_index].id == action.id) {
              plan.first_action_root_index = static_cast<int>(local);
              break;
            }
          }
        }

        const auto apply_started = Clock::now();
        current = apply_action_strict(current, actions_, action_index, config_.max_actions);
        if (config_.profile_json) {
          apply_action_ms_ += std::chrono::duration<double, std::milli>(Clock::now() - apply_started).count();
        }
        applied_in_round = true;
        plan.primitives_executed += 1;
        primitive_actions_executed_ += 1;

        if (is_end_turn_signature(choice.action_signature) || current.active_player_id != starting_player) {
          plan.reached_turn_boundary = true;
        }
        if (current.terminal || plan.reached_turn_boundary || current.leveling_up) break;
      }
      if (current.terminal || plan.reached_turn_boundary || current.leveling_up || applied_in_round) break;
    }

    if (current.terminal || plan.reached_turn_boundary) break;
    if (!applied_in_round) {
      no_progress_rounds += sampled_factor ? 1 : 2;
      if (no_progress_rounds >= 2) break;
    } else {
      no_progress_rounds = 0;
    }
  }

  if (!current.terminal && current.active_player_id == starting_player && !plan.reached_turn_boundary &&
      plan.primitives_executed < config_.max_primitives_per_turn) {
    const int end_index = find_legal_action_by_signature(current, actions_, "END_TURN:p=" + std::to_string(current.active_player_id));
    const int fallback_end_index = end_index >= 0 ? end_index : find_legal_action_by_signature(current, actions_, "END_TURN");
    if (fallback_end_index >= 0) {
      const NativeAction& action = actions_[fallback_end_index];
      CmabChoice choice;
      choice.choice_key = std::string("end_turn|end_turn:") + std::to_string(current.active_player_id) + "|" + action_signature(action);
      choice.action_signature = action_signature(action);
      choice.stage = FactorStage::EndTurn;
      choice.kind = FactorKind::EndTurn;
      record_choice(plan, choice);
      record_execution(plan, action);
      if (plan.first_action_root_index < 0) {
        for (size_t local = 0; local < root_state.legal_action_indexes.size(); ++local) {
          const int root_action_index = root_state.legal_action_indexes[local];
          if (root_action_index >= 0 &&
              root_action_index < static_cast<int>(actions_.size()) &&
              actions_[root_action_index].id == action.id) {
            plan.first_action_root_index = static_cast<int>(local);
            break;
          }
        }
      }
      const auto apply_started = Clock::now();
      current = apply_action_strict(current, actions_, fallback_end_index, config_.max_actions);
      if (config_.profile_json) {
        apply_action_ms_ += std::chrono::duration<double, std::milli>(Clock::now() - apply_started).count();
      }
      plan.primitives_executed += 1;
      primitive_actions_executed_ += 1;
      plan.reached_turn_boundary = true;
    }
  }

  plan.result_state = std::move(current);
  plan.terminal = plan.result_state.terminal;
  plan.value_root = evaluate_state_root_perspective(plan.result_state);
  max_primitives_per_turn_sample_ = std::max(max_primitives_per_turn_sample_, plan.primitives_executed);

  node.cmab.samples += 1;
  std::unordered_set<std::string> unique_choices;
  for (const TurnPlanChoice& choice : plan.choices) {
    if (choice.choice_key.empty() || !unique_choices.insert(choice.choice_key).second) continue;
    CmabChoiceStats& stats = node.cmab.choice_stats[choice.choice_key];
    stats.visits += 1;
    stats.value_sum_root += plan.value_root;
  }

  TurnEdge edge;
  edge.plan = std::move(plan);
  double prior_sum = 0.0;
  int prior_count = 0;
  for (const TurnPlanChoice& choice : edge.plan.choices) {
    auto stats_it = node.cmab.choice_stats.find(choice.choice_key);
    if (stats_it != node.cmab.choice_stats.end()) {
      prior_sum += stats_it->second.prior;
      prior_count += 1;
    }
  }
  edge.prior = prior_count > 0 ? prior_sum / static_cast<double>(prior_count) : 0.0;
  return edge;
}

void TurnCmabMCTS::run(int simulations) {
  if (nodes_.empty()) return;
  const auto started = Clock::now();
  for (int sim = 0; sim < simulations; ++sim) {
    std::vector<int> path_node_ids;
    std::vector<int> path_edge_ids;
    int node_id = 0;
    int sim_turn_depth = 0;
    double leaf_value_root = nodes_[0].value_estimate_root;

    for (int depth = 0; depth < std::max(1, config_.max_turn_depth); ++depth) {
      TurnNode& node = nodes_[node_id];
      const NativeGameState& state = states_[node.state_index];
      if (node.terminal || state.terminal) {
        leaf_value_root = evaluate_state_root_perspective(state);
        break;
      }

      const int allowed_edges = progressive_edge_limit(node);
      if (static_cast<int>(node.edges.size()) < allowed_edges) {
        TurnEdge edge = sample_new_turn_edge(node_id);
        const int edge_id = static_cast<int>(node.edges.size());
        node.edges.push_back(std::move(edge));
        path_node_ids.push_back(node_id);
        path_edge_ids.push_back(edge_id);
        sim_turn_depth += 1;
        leaf_value_root = node.edges[edge_id].plan.value_root;
        if (!node.edges[edge_id].plan.terminal && depth + 1 < config_.max_turn_depth &&
            node.edges[edge_id].plan.result_state.active_player_id != state.active_player_id) {
          NativeGameState child_state = node.edges[edge_id].plan.result_state;
          const int child_node_id = make_turn_node(std::move(child_state));
          nodes_[node_id].edges[edge_id].child_node_id = child_node_id;
        }
        break;
      }

      const int edge_id = select_existing_turn_edge(node);
      if (edge_id < 0) break;
      path_node_ids.push_back(node_id);
      path_edge_ids.push_back(edge_id);
      sim_turn_depth += 1;
      TurnEdge& edge = node.edges[edge_id];
      if (edge.child_node_id < 0) {
        leaf_value_root = edge.plan.value_root;
        break;
      }
      node_id = edge.child_node_id;
    }

    const auto backup_started = Clock::now();
    backup_turn_path(path_node_ids, path_edge_ids, leaf_value_root);
    turn_depth_sum_ += sim_turn_depth;
    max_turn_depth_reached_ = std::max(max_turn_depth_reached_, sim_turn_depth);
    if (config_.profile_json) {
      backup_ms_ += std::chrono::duration<double, std::milli>(Clock::now() - backup_started).count();
    }
  }
  search_loop_ms_ += std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

void TurnCmabMCTS::backup_turn_path(
    const std::vector<int>& node_ids,
    const std::vector<int>& edge_ids,
    double value_root) {
  for (size_t i = 0; i < node_ids.size() && i < edge_ids.size(); ++i) {
    TurnNode& node = nodes_[node_ids[i]];
    TurnEdge& edge = node.edges[edge_ids[i]];
    node.visits += 1;
    edge.visits += 1;
    edge.value_sum_root += value_root;
  }
}

py::dict TurnCmabMCTS::result_py(double temperature, bool sample_action) {
  py::dict response;
  py::list ranked;
  if (nodes_.empty()) {
    response["actionId"] = py::none();
    response["rankedActionIds"] = ranked;
    return response;
  }
  const TurnNode& root = nodes_[0];
  const NativeGameState& root_state = states_[root.state_index];
  std::unordered_map<std::string, int> visits_by_id;
  std::unordered_map<std::string, double> value_by_id;
  for (const TurnEdge& edge : root.edges) {
    if (edge.plan.first_action_id.empty()) continue;
    visits_by_id[edge.plan.first_action_id] += edge.visits;
    value_by_id[edge.plan.first_action_id] += edge.value_sum_root;
  }

  std::vector<std::string> ids;
  ids.reserve(visits_by_id.size());
  for (const auto& item : visits_by_id) ids.push_back(item.first);
  std::sort(ids.begin(), ids.end(), [&](const std::string& left, const std::string& right) {
    const int lv = visits_by_id[left];
    const int rv = visits_by_id[right];
    if (lv != rv) return lv > rv;
    const double lq = lv > 0 ? value_by_id[left] / static_cast<double>(lv) : 0.0;
    const double rq = rv > 0 ? value_by_id[right] / static_cast<double>(rv) : 0.0;
    return lq > rq;
  });

  std::unordered_set<std::string> legal_root_ids;
  for (int action_index : root_state.legal_action_indexes) {
    if (action_index >= 0 && action_index < static_cast<int>(actions_.size())) {
      legal_root_ids.insert(actions_[action_index].id);
    }
  }

  std::string selected_id;
  if (!ids.empty() && sample_action && temperature > 1e-9) {
    std::vector<double> weights;
    std::vector<std::string> legal_ids;
    for (const std::string& id : ids) {
      if (!legal_root_ids.count(id)) continue;
      legal_ids.push_back(id);
      weights.push_back(std::pow(std::max(0, visits_by_id[id]), 1.0 / std::max(1e-6, temperature)));
    }
    if (!legal_ids.empty()) {
      std::discrete_distribution<int> dist(weights.begin(), weights.end());
      selected_id = legal_ids[dist(rng_)];
    }
  }
  if (selected_id.empty()) {
    for (const std::string& id : ids) {
      if (legal_root_ids.count(id)) {
        selected_id = id;
        break;
      }
    }
  }

  std::unordered_set<std::string> seen;
  for (const std::string& id : ids) {
    if (legal_root_ids.count(id) && seen.insert(id).second) ranked.append(id);
  }
  for (int action_index : root_state.legal_action_indexes) {
    if (action_index < 0 || action_index >= static_cast<int>(actions_.size())) continue;
    const std::string& id = actions_[action_index].id;
    if (seen.insert(id).second) ranked.append(id);
    if (selected_id.empty()) selected_id = id;
  }

  response["actionId"] = selected_id.empty() ? py::none() : py::str(selected_id);
  response["rankedActionIds"] = ranked;
  if (config_.profile_json) {
    response["_profile"] = profile_py();
  }
  return response;
}

py::dict TurnCmabMCTS::profile_py() const {
  py::dict profile;
  int edge_count = 0;
  int cmab_samples = 0;
  for (const TurnNode& node : nodes_) {
    edge_count += static_cast<int>(node.edges.size());
    cmab_samples += node.cmab.samples;
  }
  py::dict root_first_action_visits;
  if (!nodes_.empty()) {
    for (const TurnEdge& edge : nodes_[0].edges) {
      if (edge.plan.first_action_id.empty()) continue;
      int current = 0;
      if (root_first_action_visits.contains(edge.plan.first_action_id)) {
        current = py::cast<int>(root_first_action_visits[edge.plan.first_action_id]);
      }
      root_first_action_visits[edge.plan.first_action_id] = current + edge.visits;
    }
  }
  py::dict timing;
  timing["search_loop_ms"] = search_loop_ms_;
  timing["factor_build_ms"] = factor_build_ms_;
  timing["cmab_select_ms"] = cmab_select_ms_;
  timing["apply_action_ms"] = apply_action_ms_;
  timing["static_eval_ms"] = static_eval_ms_;
  timing["backup_ms"] = backup_ms_;

  profile["search_mode"] = py::str("turn-cmab");
  profile["elapsed_sec"] = search_loop_ms_ / 1000.0;
  profile["outer_simulations"] = nodes_.empty() ? 0 : nodes_[0].visits;
  profile["simulations"] = nodes_.empty() ? 0 : nodes_[0].visits;
  profile["selected_paths"] = nodes_.empty() ? 0 : nodes_[0].visits;
  profile["completed_paths"] = nodes_.empty() ? 0 : nodes_[0].visits;
  profile["turn_nodes"] = static_cast<int>(nodes_.size());
  profile["turn_edges"] = edge_count;
  profile["expanded_nodes"] = static_cast<int>(nodes_.size());
  profile["node_count"] = static_cast<int>(nodes_.size());
  profile["depth_sum"] = turn_depth_sum_;
  profile["max_depth"] = max_turn_depth_reached_;
  profile["turn_depth_sum"] = turn_depth_sum_;
  profile["max_turn_depth"] = max_turn_depth_reached_;
  profile["avg_turn_depth"] = nodes_.empty() || nodes_[0].visits <= 0
      ? 0.0
      : static_cast<double>(turn_depth_sum_) / static_cast<double>(nodes_[0].visits);
  profile["root_turn_edges"] = nodes_.empty() ? 0 : static_cast<int>(nodes_[0].edges.size());
  profile["cmab_samples"] = cmab_samples;
  profile["primitive_actions_executed"] = primitive_actions_executed_;
  profile["avg_primitives_per_turn_sample"] = cmab_samples > 0
      ? static_cast<double>(primitive_actions_executed_) / static_cast<double>(cmab_samples)
      : 0.0;
  profile["max_primitives_per_turn_sample"] = max_primitives_per_turn_sample_;
  profile["fallback_used"] = false;
  profile["root_first_action_visits"] = root_first_action_visits;
  profile["timing_ms"] = timing;
  return profile;
}

int TurnCmabMCTS::root_edge_count() const {
  return nodes_.empty() ? 0 : static_cast<int>(nodes_[0].edges.size());
}

}  // namespace tribes::native
