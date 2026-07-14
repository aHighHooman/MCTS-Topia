#include "turn_macro_exp.hpp"

#include "static_eval.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace tribes::native {

namespace {

using Clock = std::chrono::steady_clock;

const char* macro_exp_stage_name(MacroExpStage stage) {
  switch (stage) {
    case MacroExpStage::Forced: return "forced";
    case MacroExpStage::Tactical: return "tactical";
    case MacroExpStage::Research: return "research";
    case MacroExpStage::City: return "city";
    case MacroExpStage::ResourceBuild: return "resource_build";
    case MacroExpStage::Road: return "road";
    case MacroExpStage::Diplomacy: return "diplomacy";
    case MacroExpStage::Cleanup: return "cleanup";
    case MacroExpStage::EndTurn: return "end_turn";
  }
  return "unknown";
}

bool macro_exp_is_forced_type(const std::string& type) {
  return type == "LEVEL_UP" || type == "MAKE_VETERAN" || type == "CAPTURE" || type == "EXAMINE";
}

bool macro_exp_is_tactical_type(const std::string& type) {
  return type == "ATTACK" || type == "CONVERT" || type == "INFILTRATE" ||
      type == "HEAL_OTHERS" || type == "MOVE" || type == "STEP_MOVE" ||
      type == "RECOVER" || type == "UPGRADE_RAMMER" || type == "UPGRADE_SCOUT" ||
      type == "UPGRADE_BOMBER" || type == "UPGRADE_SHIP" || type == "UPGRADE_BOAT";
}

bool macro_exp_is_resource_build_type(const std::string& type) {
  return type == "RESOURCE_GATHERING" || type == "BUILD" || type == "BURN_FOREST" ||
      type == "CLEAR_FOREST" || type == "GROW_FOREST" || type == "DESTROY";
}

bool macro_exp_is_diplomacy_type(const std::string& type) {
  return type == "BUILD_EMBASSY" || type == "PROPOSE_PEACE" || type == "ACCEPT_PEACE" ||
      type == "PROPOSE_TREATY" || type == "ACCEPT_TREATY" || type == "CANCEL_TREATY" ||
      type == "SEND_STARS";
}

bool macro_exp_belongs_to_stage(const NativeAction& action, MacroExpStage stage) {
  const std::string& type = action.type;
  switch (stage) {
    case MacroExpStage::Forced:
      return macro_exp_is_forced_type(type);
    case MacroExpStage::Tactical:
      return macro_exp_is_tactical_type(type);
    case MacroExpStage::Research:
      return type == "RESEARCH_TECH";
    case MacroExpStage::City:
      return type == "SPAWN";
    case MacroExpStage::ResourceBuild:
      return macro_exp_is_resource_build_type(type);
    case MacroExpStage::Road:
      return type == "BUILD_ROAD";
    case MacroExpStage::Diplomacy:
      return macro_exp_is_diplomacy_type(type);
    case MacroExpStage::Cleanup:
      return !macro_exp_is_forced_type(type) && !macro_exp_is_tactical_type(type) && type != "RESEARCH_TECH" &&
          type != "SPAWN" && !macro_exp_is_resource_build_type(type) && type != "BUILD_ROAD" &&
          !macro_exp_is_diplomacy_type(type) && type != "END_TURN";
    case MacroExpStage::EndTurn:
      return type == "END_TURN";
  }
  return false;
}

bool macro_exp_is_plausible_action(const NativeGameState& state, const NativeAction& action) {
  if (action.type == "END_TURN") return true;
  if (action.tribe_id >= 0 && action.tribe_id != state.active_player_id) return false;
  return true;
}

MacroExpKind macro_exp_kind_for_action(const NativeAction& action, MacroExpStage stage) {
  if (stage == MacroExpStage::Forced) return MacroExpKind::ForcedLevelUp;
  if (stage == MacroExpStage::Tactical) return MacroExpKind::Unit;
  if (stage == MacroExpStage::Research) return MacroExpKind::Research;
  if (stage == MacroExpStage::City) return MacroExpKind::City;
  if (stage == MacroExpStage::Road) return MacroExpKind::Road;
  if (stage == MacroExpStage::Diplomacy) return MacroExpKind::Diplomacy;
  if (stage == MacroExpStage::EndTurn) return MacroExpKind::EndTurn;
  if (action.type == "BUILD") return MacroExpKind::Build;
  if (action.type == "RESOURCE_GATHERING") return MacroExpKind::Resource;
  return MacroExpKind::Build;
}

int macro_exp_actor_id_for_action(const NativeAction& action, MacroExpStage stage) {
  if (stage == MacroExpStage::Tactical || action.unit_id > 0) return action.unit_id;
  if (stage == MacroExpStage::City || stage == MacroExpStage::ResourceBuild || action.city_id > 0) {
    return action.city_id;
  }
  return action.tribe_id;
}

std::string macro_exp_factor_key_for_action(const NativeAction& action, MacroExpStage stage, const NativeGameState& state) {
  std::ostringstream out;
  switch (stage) {
    case MacroExpStage::Forced:
      out << "forced:" << action.type << ":" << macro_exp_actor_id_for_action(action, stage);
      break;
    case MacroExpStage::Tactical:
      out << "unit:" << action.unit_id;
      break;
    case MacroExpStage::Research:
      out << "research:" << state.active_player_id;
      break;
    case MacroExpStage::City:
      out << "city:" << action.city_id;
      break;
    case MacroExpStage::ResourceBuild:
      out << "resource_build:" << action.city_id;
      break;
    case MacroExpStage::Road:
      out << "road:" << state.active_player_id;
      break;
    case MacroExpStage::Diplomacy:
      out << "diplomacy:" << state.active_player_id;
      break;
    case MacroExpStage::Cleanup:
      out << "cleanup:" << action.type << ":" << macro_exp_actor_id_for_action(action, stage);
      break;
    case MacroExpStage::EndTurn:
      out << "end_turn:" << state.active_player_id;
      break;
  }
  return out.str();
}

int macro_exp_cap_for_stage(MacroExpStage stage, const TurnMacroExpConfig& config) {
  switch (stage) {
    case MacroExpStage::Tactical: return config.max_unit_choices;
    case MacroExpStage::City: return config.max_city_choices;
    case MacroExpStage::ResourceBuild: return config.max_resource_choices;
    case MacroExpStage::Road: return config.max_road_choices;
    default: return config.max_choices_per_factor;
  }
}

double macro_exp_choice_score(
    const MacroExpChoiceStats& stats,
    double prior,
    int total_samples,
    double c,
    double prior_weight,
    double player_sign) {
  const double q = stats.visits > 0 ? stats.value_sum_root / static_cast<double>(stats.visits) : 0.0;
  const double u = c * std::sqrt(std::log(1.0 + static_cast<double>(total_samples)) / (1.0 + stats.visits));
  return player_sign * q + u + prior_weight * prior;
}

void macro_exp_record_choice(MacroExpTurnPlan& plan, const MacroExpChoice& choice) {
  MacroExpTurnPlanChoice selected;
  selected.choice_key = choice.choice_key;
  selected.macro_exp_action_signature = choice.macro_exp_action_signature;
  selected.stage = choice.stage;
  selected.kind = choice.kind;
  plan.choices.push_back(std::move(selected));
}

std::string macro_exp_commitment_signature(const NativeAction& action) {
  std::ostringstream out;
  out << action.type;
  if (action.unit_id > 0) out << ":u=" << action.unit_id;
  if (action.city_id > 0) out << ":c=" << action.city_id;
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

void macro_exp_record_execution(MacroExpTurnPlan& plan, const NativeAction& action) {
  const std::string signature = macro_exp_action_signature(action);
  plan.executed_macro_exp_action_signatures.push_back(signature);
  plan.commitment_signatures.push_back(macro_exp_commitment_signature(action));
  plan.executed_action_ids.push_back(action.id);
  if (plan.first_action_id.empty()) {
    plan.first_action_id = action.id;
    plan.first_macro_exp_action_signature = signature;
  }
}

std::vector<MacroExpStage> macro_exp_stage_order_for_state(const NativeGameState& state) {
  if (state.leveling_up) {
    return {MacroExpStage::Forced};
  }
  return {
      MacroExpStage::Forced,
      MacroExpStage::Tactical,
      MacroExpStage::Research,
      MacroExpStage::City,
      MacroExpStage::ResourceBuild,
      MacroExpStage::Road,
      MacroExpStage::Diplomacy,
      MacroExpStage::Cleanup,
      MacroExpStage::EndTurn,
  };
}

}  // namespace

struct TurnMacroExpMCTS::TurnEdge {
  MacroExpTurnPlan plan;
  int child_node_id = -1;
  int visits = 0;
  double value_sum_root = 0.0;
  double value_sum_actor = 0.0;
  double prior = 0.0;
  double log_confidence = -std::numeric_limits<double>::infinity();
  std::string diversity_key;
  std::vector<std::string> selection_reasons;
};

struct TurnMacroExpMCTS::TurnNode {
  int state_index = -1;
  bool terminal = false;
  bool expansion_exhausted = false;
  double value_estimate_root = 0.0;
  int visits = 0;
  std::vector<TurnEdge> edges;
  MacroExpStats macro_exp;
};

std::string macro_exp_action_signature(const NativeAction& action) {
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

std::string macro_exp_state_fingerprint(
    const NativeGameState& state,
    const std::vector<NativeAction>& actions) {
  std::ostringstream out;
  out << "board_size=" << state.board_size << "|root=" << state.root_player_id
      << "|active=" << state.active_player_id
      << "|tick=" << state.tick
      << "|terminal=" << state.terminal
      << "|leveling_up=" << state.leveling_up
      << "|can_end_turn=" << state.can_end_turn
      << "|winner=" << state.winner_id
      << "|tiles=";
  for (const NativeTile& tile : state.tiles) {
    out << tile.x << ',' << tile.y << ',' << tile.visible << ',' << tile.explored << ',' << tile.road << ','
        << tile.terrain << ',' << tile.resource << ',' << tile.building << ',' << std::max(0, tile.city_id) << ','
        << std::max(0, tile.unit_id)
        << '[';
    for (int id : tile.hidden_explored_by_tribes) out << id << ',';
    out << "];";
  }
  out << "|units=";
  for (const NativeUnit& unit : state.units) {
    out << unit.id << ',' << unit.tribe_id << ',' << unit.city_id << ',' << unit.x << ',' << unit.y << ','
        << unit.current_hp << ',' << unit.current_hp_exact << ',' << unit.max_hp << ',' << unit.kills << ','
        << unit.veteran << ',' << unit.hidden << ',' << unit.hidden_at_turn_start << ',' << unit.hidden_enemy_hint
        << ',' << unit.attack << ',' << unit.defence << ',' << unit.movement << ',' << unit.range << ',' << unit.cost
        << ',' << unit.type << ',' << unit.status << ';';
  }
  out << "|cities=";
  for (const NativeCity& city : state.cities) {
    out << city.id << ',' << city.tribe_id << ',' << city.x << ',' << city.y << ',' << city.level << ','
        << city.population << ',' << city.population_need << ',' << city.production << ',' << city.bound << ','
        << city.points_worth << ',' << city.capital << ',' << city.walls << ',' << city.infiltrated << '[';
    for (int id : city.unit_ids) out << id << ',';
    out << "]{";
    for (const NativeBuilding& building : city.buildings) {
      out << building.type << ',' << building.x << ',' << building.y << ',' << building.city_id << ','
          << building.owner_tribe_id << ',' << building.level << ',' << building.turns_to_score << ';';
    }
    out << "};";
  }
  out << "|tribes=";
  for (const NativeTribe& tribe : state.tribes) {
    out << tribe.id << ',' << tribe.stars << ',' << tribe.score << ',' << tribe.capital_id << ',' << tribe.kills << ','
        << tribe.pacifist_count << ',' << tribe.units_disabled_next_turn << ',' << tribe.result << ',' << tribe.tribe_type
        << "{tech=";
    for (const std::string& tech : tribe.researched_tech_ids) out << tech << ',';
    out << ";cities=";
    for (int id : tribe.city_ids) out << id << ',';
    out << ";extra=";
    for (int id : tribe.extra_unit_ids) out << id << ',';
    out << ";connected=";
    for (int id : tribe.connected_city_ids) out << id << ',';
    out << ";met=";
    for (int id : tribe.met_tribe_ids) out << id << ',';
    out << ";capitals=";
    for (int id : tribe.known_capital_tribe_ids) out << id << ',';
    out << ";lighthouses=";
    for (int id : tribe.discovered_lighthouses) out << id << ',';
    out << ";monuments=";
    for (const auto& monument : tribe.monuments) out << monument.first << ':' << monument.second << ',';
    out << "};";
  }
  out << "|relationships=";
  for (const auto& row : state.relationships) {
    out << '[';
    for (const std::string& value : row) out << value << ',';
    out << "]";
  }
  out << "|offers=";
  for (const auto& row : state.pending_offer_from) {
    out << '[';
    for (int value : row) out << value << ',';
    out << "]";
  }
  for (const auto& row : state.pending_offer_types) {
    out << '[';
    for (const std::string& value : row) out << value << ',';
    out << "]";
  }
  out
      << "|legal=";
  for (int index : state.legal_action_indexes) {
    if (index < 0 || index >= static_cast<int>(actions.size())) continue;
    out << macro_exp_action_signature(actions[index]) << ';';
  }
  return out.str();
}

std::string macro_exp_plan_diversity_key(const MacroExpTurnPlan& plan) {
  if (plan.executed_macro_exp_action_signatures.empty()) return {};
  std::ostringstream key;
  key << plan.executed_macro_exp_action_signatures.front() << "\n";
  std::set<std::string> continuation_commitments;
  for (size_t index = 1; index < plan.commitment_signatures.size(); ++index) {
    continuation_commitments.insert(plan.commitment_signatures[index]);
  }
  for (const std::string& commitment : continuation_commitments) {
    key << commitment << "\n";
  }
  return key.str();
}

void macro_exp_require_finite_evaluation(const StaticEvaluation& evaluation, const char* context) {
  if (!std::isfinite(evaluation.value)) {
    throw std::runtime_error(std::string("Turn-macro static evaluation returned a non-finite value in ") + context);
  }
  for (size_t i = 0; i < evaluation.priors.size(); ++i) {
    if (!std::isfinite(evaluation.priors[i])) {
      throw std::runtime_error(
          std::string("Turn-macro static evaluation returned a non-finite prior in ") + context +
          " at local action " + std::to_string(i));
    }
  }
}

int macro_exp_find_legal_action_by_signature(
    const NativeGameState& state,
    const std::vector<NativeAction>& actions,
    const std::string& signature) {
  if (signature.empty()) return -1;
  for (int action_index : state.legal_action_indexes) {
    if (action_index < 0 || action_index >= static_cast<int>(actions.size())) continue;
    if (macro_exp_action_signature(actions[action_index]) == signature) return action_index;
  }
  return -1;
}

bool macro_exp_is_end_turn_signature(const std::string& signature) {
  return signature == "END_TURN" || signature.rfind("END_TURN:", 0) == 0;
}

std::vector<MacroExpFactor> macro_exp_build_factors_for_stage(
    const NativeGameState& state,
    const std::vector<NativeAction>& actions,
    MacroExpStage stage,
    const TurnMacroExpConfig& config) {
  StaticEvaluation eval = evaluate_static_state(state, actions);
  macro_exp_require_finite_evaluation(eval, "factor construction");
  std::unordered_map<int, double> prior_by_global_index;
  for (size_t local = 0; local < state.legal_action_indexes.size() && local < eval.priors.size(); ++local) {
    prior_by_global_index[state.legal_action_indexes[local]] = eval.priors[local];
  }

  std::unordered_map<std::string, size_t> factor_offsets;
  std::vector<MacroExpFactor> factors;
  for (int global_index : state.legal_action_indexes) {
    if (global_index < 0 || global_index >= static_cast<int>(actions.size())) continue;
    const NativeAction& action = actions[global_index];
    if (!macro_exp_belongs_to_stage(action, stage)) continue;

    const std::string factor_key = macro_exp_factor_key_for_action(action, stage, state);
    auto offset_it = factor_offsets.find(factor_key);
    if (offset_it == factor_offsets.end()) {
      offset_it = factor_offsets.emplace(factor_key, factors.size()).first;
      MacroExpFactor factor;
      factor.factor_key = factor_key;
      factor.stage = stage;
      factor.kind = macro_exp_kind_for_action(action, stage);
      factors.push_back(std::move(factor));
    }

    MacroExpChoice choice;
    choice.macro_exp_action_signature = macro_exp_action_signature(action);
    choice.choice_key = std::string(macro_exp_stage_name(stage)) + "|" + factor_key + "|" + choice.macro_exp_action_signature;
    choice.initial_action_index = global_index;
    choice.prior = prior_by_global_index.count(global_index) ? prior_by_global_index[global_index] : 0.0;
    choice.stage = stage;
    choice.kind = macro_exp_kind_for_action(action, stage);
    choice.actor_id = macro_exp_actor_id_for_action(action, stage);
    factors[offset_it->second].choices.push_back(std::move(choice));
  }

  const bool optional = stage != MacroExpStage::Forced && stage != MacroExpStage::EndTurn;
  for (MacroExpFactor& factor : factors) {
    std::sort(factor.choices.begin(), factor.choices.end(), [](const MacroExpChoice& left, const MacroExpChoice& right) {
      return left.prior > right.prior;
    });
    const int executable_cap = std::max(1, macro_exp_cap_for_stage(stage, config));
    if (static_cast<int>(factor.choices.size()) > executable_cap) {
      factor.choices.resize(executable_cap);
    }
    if (optional) {
      MacroExpChoice none;
      none.choice_key = std::string(macro_exp_stage_name(stage)) + "|" + factor.factor_key + "|none";
      none.prior = 0.05;
      none.stage = stage;
      none.kind = factor.kind;
      factor.choices.push_back(std::move(none));
    }
    double total = 0.0;
    for (const MacroExpChoice& choice : factor.choices) total += std::max(0.0, choice.prior);
    if (total <= 0.0) {
      const double uniform = factor.choices.empty() ? 0.0 : 1.0 / static_cast<double>(factor.choices.size());
      for (MacroExpChoice& choice : factor.choices) choice.prior = uniform;
    } else {
      for (MacroExpChoice& choice : factor.choices) choice.prior = std::max(0.0, choice.prior) / total;
    }
  }
  return factors;
}

struct MacroExpInnerActionCandidate {
  int global_action_index = -1;
  int visits = 0;
  double q_utility = 0.0;
  double prior = 0.0;
  double score = 0.0;
  double log_confidence = -std::numeric_limits<double>::infinity();
  std::vector<std::string> action_signatures;
  std::vector<double> action_priors;
  std::vector<double> conditional_visit_shares;
};

struct MacroExpCandidateSelection {
  int candidate_index = -1;
  std::vector<std::string> reasons;
};

std::string macro_exp_function_class(const NativeAction& action) {
  if (macro_exp_is_forced_type(action.type)) return "forced";
  if (macro_exp_is_tactical_type(action.type)) return "tactical";
  if (action.type == "RESEARCH_TECH") return "research";
  if (action.type == "SPAWN") return "city";
  if (macro_exp_is_resource_build_type(action.type)) return "resource_build";
  if (action.type == "BUILD_ROAD") return "road";
  if (macro_exp_is_diplomacy_type(action.type)) return "diplomacy";
  if (action.type == "END_TURN") return "end_turn";
  return "cleanup";
}

std::string macro_exp_unit_dimension(const NativeAction& action) {
  return action.unit_id > 0 ? "unit:" + std::to_string(action.unit_id) : std::string();
}

std::string macro_exp_city_dimension(const NativeAction& action) {
  return action.city_id > 0 ? "city:" + std::to_string(action.city_id) : std::string();
}

std::string macro_exp_resource_dimension(const NativeAction& action) {
  if (!action.resource_type.empty()) return "resource:" + action.resource_type;
  if (!action.building_type.empty()) return "resource:building:" + action.building_type;
  if (!action.unit_type.empty()) return "resource:unit:" + action.unit_type;
  if (!action.tech.empty()) return "resource:tech:" + action.tech;
  return std::string();
}

std::vector<MacroExpCandidateSelection> macro_exp_select_diverse_candidates(
    const std::vector<MacroExpInnerActionCandidate>& candidates,
    const std::vector<NativeAction>& actions,
    const NativeGameState& state,
    int max_edges) {
  std::vector<int> eligible;
  for (int index = 0; index < static_cast<int>(candidates.size()); ++index) {
    const int action_index = candidates[index].global_action_index;
    if (action_index < 0 || action_index >= static_cast<int>(actions.size())) continue;
    if (!macro_exp_is_plausible_action(state, actions[action_index])) continue;
    eligible.push_back(index);
  }
  const int target = std::min(std::max(0, max_edges), static_cast<int>(eligible.size()));
  if (target <= 0) return {};

  auto ranking = [&](auto metric, auto tie_break) {
    std::vector<int> ranked = eligible;
    std::stable_sort(ranked.begin(), ranked.end(), [&](int left, int right) {
      const double left_metric = metric(candidates[left]);
      const double right_metric = metric(candidates[right]);
      if (left_metric != right_metric) return left_metric > right_metric;
      const double left_tie = tie_break(candidates[left]);
      const double right_tie = tie_break(candidates[right]);
      if (left_tie != right_tie) return left_tie > right_tie;
      return left < right;
    });
    return ranked;
  };

  const std::vector<int> visit_rank = ranking(
      [](const MacroExpInnerActionCandidate& candidate) { return static_cast<double>(candidate.visits); },
      [](const MacroExpInnerActionCandidate& candidate) { return candidate.q_utility; });
  const std::vector<int> value_rank = ranking(
      [](const MacroExpInnerActionCandidate& candidate) { return candidate.q_utility; },
      [](const MacroExpInnerActionCandidate& candidate) { return static_cast<double>(candidate.visits); });
  const std::vector<int> prior_rank = ranking(
      [](const MacroExpInnerActionCandidate& candidate) { return candidate.prior; },
      [](const MacroExpInnerActionCandidate& candidate) { return candidate.score; });

  std::unordered_map<int, std::set<std::string>> reasons;
  const int signal_width = std::min(target, static_cast<int>(eligible.size()));
  auto mark_signal = [&](const std::vector<int>& ranked, const char* reason) {
    for (int position = 0; position < signal_width; ++position) {
      reasons[ranked[position]].insert(reason);
    }
  };
  mark_signal(visit_rank, "visited");
  mark_signal(value_rank, "value");
  mark_signal(prior_rank, "prior");

  std::vector<bool> selected(candidates.size(), false);
  std::vector<int> selected_indices;
  selected_indices.reserve(target);
  auto add_anchor = [&](const std::vector<int>& ranked) {
    for (int index : ranked) {
      if (selected[index]) continue;
      selected[index] = true;
      selected_indices.push_back(index);
      return;
    }
  };
  add_anchor(visit_rank);
  if (static_cast<int>(selected_indices.size()) < target) add_anchor(value_rank);
  if (static_cast<int>(selected_indices.size()) < target) add_anchor(prior_rank);

  std::set<std::string> used_functions;
  std::set<std::string> used_units;
  std::set<std::string> used_cities;
  std::set<std::string> used_resources;
  auto record_dimensions = [&](int index) {
    const NativeAction& action = actions[candidates[index].global_action_index];
    used_functions.insert(macro_exp_function_class(action));
    const std::string unit = macro_exp_unit_dimension(action);
    const std::string city = macro_exp_city_dimension(action);
    const std::string resource = macro_exp_resource_dimension(action);
    if (!unit.empty()) used_units.insert(unit);
    if (!city.empty()) used_cities.insert(city);
    if (!resource.empty()) used_resources.insert(resource);
  };
  for (int index : selected_indices) record_dimensions(index);

  while (static_cast<int>(selected_indices.size()) < target) {
    int best_index = -1;
    int best_novelty = std::numeric_limits<int>::min();
    double best_score = -std::numeric_limits<double>::infinity();
    for (int index : eligible) {
      if (selected[index]) continue;
      const NativeAction& action = actions[candidates[index].global_action_index];
      const std::string function = macro_exp_function_class(action);
      const std::string unit = macro_exp_unit_dimension(action);
      const std::string city = macro_exp_city_dimension(action);
      const std::string resource = macro_exp_resource_dimension(action);
      int novelty = 0;
      if (!used_functions.count(function)) novelty += 1000;
      if (!unit.empty() && !used_units.count(unit)) novelty += 100;
      if (!city.empty() && !used_cities.count(city)) novelty += 100;
      if (!resource.empty() && !used_resources.count(resource)) novelty += 100;
      if (best_index < 0 || novelty > best_novelty ||
          (novelty == best_novelty && candidates[index].score > best_score) ||
          (novelty == best_novelty && candidates[index].score == best_score && index < best_index)) {
        best_index = index;
        best_novelty = novelty;
        best_score = candidates[index].score;
      }
    }
    if (best_index < 0) break;
    selected[best_index] = true;
    selected_indices.push_back(best_index);
    const NativeAction& action = actions[candidates[best_index].global_action_index];
    const std::string function = macro_exp_function_class(action);
    const std::string unit = macro_exp_unit_dimension(action);
    const std::string city = macro_exp_city_dimension(action);
    const std::string resource = macro_exp_resource_dimension(action);
    if (!used_functions.count(function)) reasons[best_index].insert("new_function");
    if (!unit.empty() && !used_units.count(unit)) reasons[best_index].insert("new_unit");
    if (!city.empty() && !used_cities.count(city)) reasons[best_index].insert("new_city");
    if (!resource.empty() && !used_resources.count(resource)) reasons[best_index].insert("new_resource");
    record_dimensions(best_index);
  }

  std::vector<MacroExpCandidateSelection> out;
  out.reserve(selected_indices.size());
  for (int index : selected_indices) {
    MacroExpCandidateSelection selection;
    selection.candidate_index = index;
    selection.reasons.assign(reasons[index].begin(), reasons[index].end());
    out.push_back(std::move(selection));
  }
  return out;
}

class MacroExpInnerPrimitiveMCTS {
 public:
  MacroExpInnerPrimitiveMCTS(
      NativeGameState root_state,
      std::vector<NativeAction>& actions,
      int root_player_id,
      int owner_player_id,
      int utility_player_id,
      int max_actions,
      double c_puct) :
      actions_(actions),
      root_player_id_(root_player_id),
      owner_player_id_(owner_player_id),
      utility_player_id_(utility_player_id),
      max_actions_(max_actions),
      c_puct_(c_puct) {
    make_node(std::move(root_state));
  }

  void run(int simulations) {
    for (int sim = 0; sim < simulations; ++sim) {
      run_one();
    }
  }

  std::vector<MacroExpInnerActionCandidate> root_candidates() const {
    std::vector<MacroExpInnerActionCandidate> out;
    if (nodes_.empty()) return out;
    std::vector<std::string> signatures;
    std::vector<double> priors;
    std::vector<double> visit_shares;
    std::function<void(int, double)> enumerate = [&](int node_id, double log_confidence) {
      if (node_id < 0 || node_id >= static_cast<int>(nodes_.size())) return;
      const InnerNode& node = nodes_[node_id];
      const NativeGameState& state = states_[node.state_index];
      int total_positive_visits = 0;
      for (int local = 0; local < static_cast<int>(state.legal_action_indexes.size()); ++local) {
        const int action_index = state.legal_action_indexes[local];
        if (action_index < 0 || action_index >= static_cast<int>(actions_.size()) ||
            !macro_exp_is_plausible_action(state, actions_[action_index])) continue;
        if (local < static_cast<int>(node.visits.size()) && node.visits[local] > 0) {
          total_positive_visits += node.visits[local];
        }
      }

      // A searched prefix with no positively visited continuation is maximal.
      // The outer completion step will perform forced actions and end the turn.
      if (node.terminal || state.terminal || total_positive_visits <= 0) {
        if (!signatures.empty()) {
          MacroExpInnerActionCandidate candidate;
          candidate.action_signatures = signatures;
          candidate.action_priors = priors;
          candidate.conditional_visit_shares = visit_shares;
          candidate.log_confidence = log_confidence;
          candidate.score = log_confidence;
          const NativeGameState& root_state = states_[nodes_[0].state_index];
          candidate.global_action_index = macro_exp_find_legal_action_by_signature(
              root_state, actions_, signatures.front());
          if (candidate.global_action_index >= 0) {
            for (size_t local = 0; local < root_state.legal_action_indexes.size(); ++local) {
              if (root_state.legal_action_indexes[local] != candidate.global_action_index) continue;
              candidate.visits = nodes_[0].visits[local];
              candidate.prior = nodes_[0].priors[local];
              candidate.q_utility = candidate.visits > 0
                  ? nodes_[0].value_sums_utility[local] / static_cast<double>(candidate.visits)
                  : nodes_[0].value_estimate_utility;
              break;
            }
            out.push_back(std::move(candidate));
          }
        }
        return;
      }

      for (int local = 0; local < static_cast<int>(state.legal_action_indexes.size()); ++local) {
        const int visits = local < static_cast<int>(node.visits.size()) ? node.visits[local] : 0;
        if (visits <= 0) continue;
        const int action_index = state.legal_action_indexes[local];
        if (action_index < 0 || action_index >= static_cast<int>(actions_.size()) ||
            !macro_exp_is_plausible_action(state, actions_[action_index])) continue;
        signatures.push_back(macro_exp_action_signature(actions_[action_index]));
        priors.push_back(local < static_cast<int>(node.priors.size()) ? node.priors[local] : 0.0);
        const double conditional_visit_share =
            static_cast<double>(visits) / static_cast<double>(total_positive_visits);
        visit_shares.push_back(conditional_visit_share);
        const double next_log_confidence = log_confidence + std::log(conditional_visit_share);
        const int child_id = local < static_cast<int>(node.child_node_ids.size())
            ? node.child_node_ids[local]
            : -1;
        if (child_id >= 0 && child_id < static_cast<int>(nodes_.size())) {
          enumerate(child_id, next_log_confidence);
        } else {
          MacroExpInnerActionCandidate candidate;
          candidate.action_signatures = signatures;
          candidate.action_priors = priors;
          candidate.conditional_visit_shares = visit_shares;
          candidate.log_confidence = next_log_confidence;
          candidate.score = next_log_confidence;
          const NativeGameState& root_state = states_[nodes_[0].state_index];
          candidate.global_action_index = macro_exp_find_legal_action_by_signature(
              root_state, actions_, signatures.front());
          if (candidate.global_action_index >= 0) {
            for (size_t root_local = 0; root_local < root_state.legal_action_indexes.size(); ++root_local) {
              if (root_state.legal_action_indexes[root_local] != candidate.global_action_index) continue;
              candidate.visits = nodes_[0].visits[root_local];
              candidate.prior = nodes_[0].priors[root_local];
              candidate.q_utility = candidate.visits > 0
                  ? nodes_[0].value_sums_utility[root_local] / static_cast<double>(candidate.visits)
                  : nodes_[0].value_estimate_utility;
              break;
            }
            out.push_back(std::move(candidate));
          }
        }
        signatures.pop_back();
        priors.pop_back();
        visit_shares.pop_back();
      }
    };
    enumerate(0, 0.0);
    std::sort(out.begin(), out.end(), [](const MacroExpInnerActionCandidate& left, const MacroExpInnerActionCandidate& right) {
      if (left.log_confidence != right.log_confidence) return left.log_confidence > right.log_confidence;
      return left.action_signatures < right.action_signatures;
    });
    return out;
  }

  int expanded_nodes() const {
    return std::max(0, static_cast<int>(nodes_.size()) - 1);
  }

 private:
  struct InnerNode {
    int state_index = -1;
    bool terminal = false;
    double value_estimate_utility = 0.0;
    int total_visits = 0;
    std::vector<double> priors;
    std::vector<int> visits;
    std::vector<double> value_sums_utility;
    std::vector<int> child_node_ids;
  };

  std::vector<NativeAction>& actions_;
  std::vector<NativeGameState> states_;
  std::vector<InnerNode> nodes_;
  int root_player_id_ = 0;
  int owner_player_id_ = 0;
  int utility_player_id_ = 0;
  int max_actions_ = 512;
  double c_puct_ = 1.5;

  double evaluate_utility(const NativeGameState& state) const {
    if (state.terminal) {
      if (state.terminal_value_known && utility_player_id_ == root_player_id_) {
        return state.terminal_value;
      }
      if (state.winner_id >= 0) {
        return state.winner_id == utility_player_id_ ? 1.0 : -1.0;
      }
      return 0.0;
    }
    NativeGameState utility_state = state;
    utility_state.active_player_id = utility_player_id_;
    StaticEvaluation evaluation = evaluate_static_state(utility_state, actions_);
    macro_exp_require_finite_evaluation(evaluation, "inner utility evaluation");
    return evaluation.value;
  }

  int make_node(NativeGameState state) {
    const int state_index = static_cast<int>(states_.size());
    const bool terminal = state.terminal || state.legal_action_indexes.empty();
    StaticEvaluation eval = evaluate_static_state(state, actions_);
    macro_exp_require_finite_evaluation(eval, "inner node construction");
    const double value_utility = !terminal && utility_player_id_ == state.active_player_id
        ? eval.value
        : evaluate_utility(state);
    const size_t action_count = state.legal_action_indexes.size();
    states_.push_back(std::move(state));
    const NativeGameState& stored_state = states_[state_index];
    InnerNode node;
    node.state_index = state_index;
    node.terminal = terminal;
    node.value_estimate_utility = value_utility;
    node.priors.assign(action_count, 0.0);
    int selectable_count = 0;
    for (size_t i = 0; i < action_count; ++i) {
      const int global_action_index = stored_state.legal_action_indexes[i];
      if (global_action_index < 0 || global_action_index >= static_cast<int>(actions_.size()) ||
          !macro_exp_is_plausible_action(stored_state, actions_[global_action_index])) {
        continue;
      }
      selectable_count += 1;
      if (i < eval.priors.size()) {
        node.priors[i] = std::max(0.0, eval.priors[i]);
      }
    }
    const double prior_total = std::accumulate(node.priors.begin(), node.priors.end(), 0.0);
    if (prior_total > 0.0) {
      for (double& prior : node.priors) prior /= prior_total;
    } else if (selectable_count > 0) {
      const double uniform = 1.0 / static_cast<double>(selectable_count);
      for (size_t i = 0; i < action_count; ++i) {
        const int global_action_index = stored_state.legal_action_indexes[i];
        if (global_action_index >= 0 && global_action_index < static_cast<int>(actions_.size()) &&
            macro_exp_is_plausible_action(stored_state, actions_[global_action_index])) {
          node.priors[i] = uniform;
        }
      }
    }
    node.visits.assign(action_count, 0);
    node.value_sums_utility.assign(action_count, 0.0);
    node.child_node_ids.assign(action_count, -1);
    const int node_id = static_cast<int>(nodes_.size());
    nodes_.push_back(std::move(node));
    return node_id;
  }

  int select_action(const InnerNode& node, const NativeGameState& state) const {
    const double sqrt_total = std::sqrt(1.0 + static_cast<double>(std::max(1, node.total_visits)));
    double best_score = -std::numeric_limits<double>::infinity();
    int best = -1;
    for (int local = 0; local < static_cast<int>(state.legal_action_indexes.size()); ++local) {
      const int global_action_index = state.legal_action_indexes[local];
      if (global_action_index < 0 || global_action_index >= static_cast<int>(actions_.size()) ||
          !macro_exp_is_plausible_action(state, actions_[global_action_index])) {
        continue;
      }
      const int visits = node.visits[local];
      const double q = visits > 0 ? node.value_sums_utility[local] / static_cast<double>(visits) : node.value_estimate_utility;
      const double u = c_puct_ * node.priors[local] * sqrt_total / (1.0 + static_cast<double>(visits));
      const double score = q + u;
      if (score > best_score) {
        best_score = score;
        best = local;
      }
    }
    return best;
  }

  void run_one() {
    if (nodes_.empty()) return;
    std::vector<int> path_nodes;
    std::vector<int> path_actions;
    path_nodes.reserve(32);
    path_actions.reserve(32);
    int node_id = 0;
    double leaf_value_utility = nodes_[0].value_estimate_utility;

    for (size_t depth = 0; depth <= nodes_.size(); ++depth) {
      const int state_index = nodes_[node_id].state_index;
      const bool node_terminal = nodes_[node_id].terminal;
      const NativeGameState& state = states_[state_index];
      if (node_terminal || state.terminal || state.legal_action_indexes.empty()) {
        leaf_value_utility = evaluate_utility(state);
        break;
      }
      const int local_action = select_action(nodes_[node_id], state);
      if (local_action < 0) {
        leaf_value_utility = evaluate_utility(state);
        break;
      }
      path_nodes.push_back(node_id);
      path_actions.push_back(local_action);
      const int child_node_id = nodes_[node_id].child_node_ids[local_action];
      if (child_node_id < 0) {
        NativeGameState child = apply_action_strict(
            state,
            actions_,
            state.legal_action_indexes[local_action],
            max_actions_);
        leaf_value_utility = evaluate_utility(child);

        // The inner search is only meant to choose the current turn owner's
        // primitive sequence.  Do not expand the next player's turn into the
        // same tree: doing so would select that player's actions using the
        // fixed utility_player_id_ and therefore model them as cooperative
        // with the current turn owner.
        if (!child.terminal && child.active_player_id != owner_player_id_) {
          break;
        }

        const int child_id = make_node(std::move(child));
        nodes_[node_id].child_node_ids[local_action] = child_id;
        break;
      }
      node_id = child_node_id;
    }

    for (size_t i = 0; i < path_nodes.size() && i < path_actions.size(); ++i) {
      InnerNode& node = nodes_[path_nodes[i]];
      const int action = path_actions[i];
      node.total_visits += 1;
      node.visits[action] += 1;
      node.value_sums_utility[action] += leaf_value_utility;
    }
  }
};

TurnMacroExpMCTS::TurnMacroExpMCTS(
    const py::dict& root_payload,
    const TurnMacroExpConfig& config,
    uint64_t seed) :
    config_(config),
    rng_(seed) {
  NativeRoot root = parse_root_payload(root_payload, config_.max_actions);
  actions_ = std::move(root.actions);
  root_player_id_ = root.state.root_player_id;
  root.state.terminal = root.state.terminal || root.state.legal_action_indexes.empty();
  make_turn_node(std::move(root.state));
}

int TurnMacroExpMCTS::make_turn_node(NativeGameState state) {
  const double value_estimate_root = evaluate_state_root_perspective(state);
  return make_turn_node_with_value(std::move(state), value_estimate_root);
}

int TurnMacroExpMCTS::make_turn_node_with_value(NativeGameState state, double value_estimate_root) {
  const int state_index = static_cast<int>(states_.size());
  const bool terminal = state.terminal || state.legal_action_indexes.empty();
  states_.push_back(std::move(state));
  TurnNode node;
  node.state_index = state_index;
  node.terminal = terminal;
  node.value_estimate_root = value_estimate_root;
  const int node_id = static_cast<int>(nodes_.size());
  nodes_.push_back(std::move(node));
  return node_id;
}

double TurnMacroExpMCTS::evaluate_state_for_player(const NativeGameState& state, int player_id) {
  if (state.terminal) {
    if (state.winner_id >= 0) {
      return state.winner_id == player_id ? 1.0 : -1.0;
    }
    return player_id == root_player_id_ && state.terminal_value_known
        ? state.terminal_value
        : 0.0;
  }
  static_eval_calls_ += 1;
  const auto started = Clock::now();
  NativeGameState utility_state = state;
  utility_state.active_player_id = player_id;
  StaticEvaluation eval = evaluate_static_state(utility_state, actions_);
  macro_exp_require_finite_evaluation(eval, "outer utility evaluation");
  if (config_.profile_json) {
    static_eval_ms_ += std::chrono::duration<double, std::milli>(Clock::now() - started).count();
  }
  return eval.value;
}

double TurnMacroExpMCTS::evaluate_state_root_perspective(const NativeGameState& state) {
  return evaluate_state_for_player(state, root_player_id_);
}

int TurnMacroExpMCTS::visible_edge_limit(const TurnNode& node) const {
  const int progressively_visible = config_.progressive_base +
      static_cast<int>(config_.progressive_scale * std::sqrt(std::max(0, node.visits)));
  return std::min(
      static_cast<int>(node.edges.size()),
      std::min(config_.max_new_edges_per_node, std::max(1, progressively_visible)));
}

int TurnMacroExpMCTS::select_existing_turn_edge(const TurnNode& node) const {
  if (node.edges.empty()) return -1;
  const NativeGameState& state = states_[node.state_index];
  const bool use_actor_value =
      config_.opponent_mode == TurnMacroExpOpponentMode::Maximalist &&
      state.active_player_id != root_player_id_;
  const double log_n = std::log(1.0 + static_cast<double>(std::max(1, node.visits)));
  double best_score = -std::numeric_limits<double>::infinity();
  int best = 0;
  const int selectable_edges = visible_edge_limit(node);
  for (int i = 0; i < selectable_edges; ++i) {
    const TurnEdge& edge = node.edges[i];
    const double value_sum = use_actor_value ? edge.value_sum_actor : edge.value_sum_root;
    const double q = edge.visits > 0 ? value_sum / static_cast<double>(edge.visits) : 0.0;
    const double u = config_.outer_c * std::sqrt(log_n / (1.0 + edge.visits));
    const double score = q + u + 0.10 * edge.prior;
    if (score > best_score) {
      best_score = score;
      best = i;
    }
  }
  return best;
}

TurnMacroExpMCTS::TurnEdge TurnMacroExpMCTS::sample_new_turn_edge(int node_id) {
  std::vector<TurnEdge> edges = generate_turn_edges(node_id, 1);
  return edges.empty() ? TurnEdge() : std::move(edges.front());
}

std::vector<TurnMacroExpMCTS::TurnEdge> TurnMacroExpMCTS::generate_turn_edges(int node_id, int max_edges) {
  const int state_index = nodes_[node_id].state_index;
  const NativeGameState root_state = states_[state_index];
  const int starting_player = root_state.active_player_id;
  std::unordered_set<std::string> used_plan_keys;
  std::unordered_set<std::string> used_diversity_keys;
  for (const TurnEdge& edge : nodes_[node_id].edges) {
    std::ostringstream key;
    for (const std::string& signature : edge.plan.executed_macro_exp_action_signatures) {
      key << signature << "\n";
    }
    used_plan_keys.insert(key.str());
    const std::string diversity_key = edge.diversity_key.empty()
        ? macro_exp_plan_diversity_key(edge.plan)
        : edge.diversity_key;
    if (!diversity_key.empty()) used_diversity_keys.insert(diversity_key);
  }
  std::vector<TurnEdge> out;
  if (max_edges <= 0 || root_state.terminal || root_state.legal_action_indexes.empty()) return out;
  int sampled_plans = 0;

  auto plan_key = [](const MacroExpTurnPlan& plan) {
    std::ostringstream key;
    for (const std::string& signature : plan.executed_macro_exp_action_signatures) {
      key << signature << "\n";
    }
    return key.str();
  };

  auto run_inner_candidates = [&](const NativeGameState& state) {
    const auto inner_started = Clock::now();
    std::vector<NativeAction> inner_actions = actions_;
    MacroExpInnerPrimitiveMCTS inner(
        state,
        inner_actions,
        root_player_id_,
        state.active_player_id,
        config_.opponent_mode == TurnMacroExpOpponentMode::Maximalist
            ? state.active_player_id
            : root_player_id_,
        config_.max_actions,
        config_.inner_c_puct);
    const int sims = std::max(1, config_.inner_simulations);
    inner.run(sims);
    if (config_.profile_json) {
      inner_search_ms_ += std::chrono::duration<double, std::milli>(Clock::now() - inner_started).count();
    }
    inner_searches_ += 1;
    inner_simulations_executed_ += sims;
    inner_nodes_expanded_ += inner.expanded_nodes();
    std::vector<MacroExpInnerActionCandidate> candidates = inner.root_candidates();
    for (MacroExpInnerActionCandidate& candidate : candidates) {
      if (candidate.global_action_index < 0 || candidate.global_action_index >= static_cast<int>(inner_actions.size())) {
        candidate.global_action_index = -1;
        continue;
      }
      const std::string signature = macro_exp_action_signature(inner_actions[candidate.global_action_index]);
      candidate.global_action_index = macro_exp_find_legal_action_by_signature(state, actions_, signature);
    }
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(), [](const MacroExpInnerActionCandidate& candidate) {
          return candidate.global_action_index < 0;
        }),
        candidates.end());
    return candidates;
  };

  auto timed_static_eval = [&](const NativeGameState& state) {
    static_eval_calls_ += 1;
    const auto started = Clock::now();
    StaticEvaluation eval = evaluate_static_state(state, actions_);
    macro_exp_require_finite_evaluation(eval, "outer static evaluation");
    if (config_.profile_json) {
      static_eval_ms_ += std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    }
    return eval;
  };

  auto static_best_action = [&](const NativeGameState& state) -> std::pair<int, double> {
    if (state.legal_action_indexes.empty()) return {-1, 0.0};
    int best_action = -1;
    double best_score = -std::numeric_limits<double>::infinity();
    double best_prior = 0.0;
    greedy_static_calls_ += 1;
    StaticEvaluation eval = timed_static_eval(state);
    struct GreedyCandidate {
      int action_index = -1;
      double prior = 0.0;
      bool is_end_turn = false;
    };
    std::vector<GreedyCandidate> greedy_candidates;
    greedy_candidates.reserve(state.legal_action_indexes.size());
    for (size_t local = 0; local < state.legal_action_indexes.size(); ++local) {
      const int action_index = state.legal_action_indexes[local];
      if (action_index < 0 || action_index >= static_cast<int>(actions_.size())) continue;
      if (!macro_exp_is_plausible_action(state, actions_[action_index])) continue;
      const double prior = local < eval.priors.size() ? eval.priors[local] : 0.0;
      const bool is_end_turn = macro_exp_is_end_turn_signature(macro_exp_action_signature(actions_[action_index]));
      greedy_candidates.push_back({action_index, prior, is_end_turn});
    }
    greedy_static_candidates_considered_ += static_cast<int>(greedy_candidates.size());
    std::sort(greedy_candidates.begin(), greedy_candidates.end(), [](const GreedyCandidate& left, const GreedyCandidate& right) {
      if (left.is_end_turn != right.is_end_turn) return right.is_end_turn;
      return left.prior > right.prior;
    });
    int child_evals_used = 0;
    const int child_eval_limit = std::max(1, config_.greedy_eval_top_k);
    const double base_value = config_.opponent_mode == TurnMacroExpOpponentMode::Maximalist
        ? evaluate_state_for_player(state, state.active_player_id)
        : evaluate_state_for_player(state, root_player_id_);
    for (const GreedyCandidate& candidate : greedy_candidates) {
      const int action_index = candidate.action_index;
      const double prior = candidate.prior;
      double value = base_value;
      if (!candidate.is_end_turn) {
        if (child_evals_used >= child_eval_limit) {
          greedy_static_child_eval_skips_ += 1;
          value = base_value;
        } else {
          child_evals_used += 1;
          greedy_static_child_evals_ += 1;
          const NativeGameState child = apply_action_strict(state, actions_, action_index, config_.max_actions);
          value = child.terminal
              ? (child.terminal_value_known ? child.terminal_value : 0.0)
              : (config_.opponent_mode == TurnMacroExpOpponentMode::Maximalist
                  ? evaluate_state_for_player(child, state.active_player_id)
                  : evaluate_state_for_player(child, root_player_id_));
        }
      }
      const double score = value + 0.05 * prior;
      if (score > best_score) {
        best_score = score;
        best_action = action_index;
        best_prior = prior;
      }
    }
    return {best_action, best_prior};
  };

  auto resolve_action_index = [&](const NativeGameState& state, int action_index) {
    if (action_index >= 0 && action_index < static_cast<int>(actions_.size())) {
      for (int legal_index : state.legal_action_indexes) {
        if (legal_index == action_index && macro_exp_is_plausible_action(state, actions_[action_index])) return action_index;
      }
      const std::string signature = macro_exp_action_signature(actions_[action_index]);
      const int resolved = macro_exp_find_legal_action_by_signature(state, actions_, signature);
      if (resolved >= 0 && macro_exp_is_plausible_action(state, actions_[resolved])) return resolved;
    }
    return -1;
  };

  auto resolve_action_signature = [&](const NativeGameState& state, const std::string& signature) {
    const int resolved = macro_exp_find_legal_action_by_signature(state, actions_, signature);
    if (resolved >= 0 && macro_exp_is_plausible_action(state, actions_[resolved])) return resolved;
    return -1;
  };

  auto record_selected_action = [&](MacroExpTurnPlan& plan, int action_index, double prior) {
    const NativeAction& action = actions_[action_index];
    MacroExpChoice choice;
    choice.choice_key = std::string("inner_mcts|") + macro_exp_action_signature(action);
    choice.macro_exp_action_signature = macro_exp_action_signature(action);
    choice.initial_action_index = action_index;
    choice.prior = prior;
    choice.stage = action.type == "END_TURN" ? MacroExpStage::EndTurn : MacroExpStage::Cleanup;
    choice.kind = action.type == "END_TURN" ? MacroExpKind::EndTurn : MacroExpKind::Build;
    macro_exp_record_choice(plan, choice);
    macro_exp_record_execution(plan, action);
    if (plan.first_action_root_index < 0) {
      for (size_t local = 0; local < root_state.legal_action_indexes.size(); ++local) {
        if (root_state.legal_action_indexes[local] == action_index) {
          plan.first_action_root_index = static_cast<int>(local);
          break;
        }
      }
    }
  };

  auto apply_recorded_action = [&](MacroExpTurnPlan& plan, NativeGameState& current, int action_index, double prior) {
    const int resolved_index = resolve_action_index(current, action_index);
    if (resolved_index < 0) return false;
    const std::string signature = macro_exp_action_signature(actions_[resolved_index]);
    const bool is_end_turn = macro_exp_is_end_turn_signature(signature);
    if (std::getenv("TRIBES_TURN_MACRO_TRACE") != nullptr) {
      std::cerr << "turn_macro_apply primitive=" << (plan.primitives_executed + 1)
                << " player=" << current.active_player_id
                << " legal=" << current.legal_action_indexes.size()
                << " action=" << signature << std::endl;
    }
    record_selected_action(plan, resolved_index, prior);
    const auto apply_started = Clock::now();
    current = apply_action_strict(current, actions_, resolved_index, config_.max_actions);
    if (std::getenv("TRIBES_TURN_MACRO_TRACE") != nullptr) {
      std::cerr << "turn_macro_apply_done primitive=" << (plan.primitives_executed + 1)
                << " player=" << current.active_player_id
                << " legal=" << current.legal_action_indexes.size()
                << " terminal=" << current.terminal
                << " boundary=" << (is_end_turn || current.active_player_id != starting_player)
                << std::endl;
    }
    if (config_.profile_json) {
      apply_action_ms_ += std::chrono::duration<double, std::milli>(Clock::now() - apply_started).count();
    }
    plan.primitives_executed += 1;
    primitive_actions_executed_ += 1;
    plan.prefix_state_fingerprints.push_back(macro_exp_state_fingerprint(current, actions_));
    if (is_end_turn || current.active_player_id != starting_player) {
      plan.reached_turn_boundary = true;
    }
    return true;
  };

  auto complete_plan = [&](const std::vector<std::string>& forced_action_signatures, const std::vector<double>& forced_priors) {
    NativeGameState current = root_state;
    MacroExpTurnPlan plan;
    plan.prefix_state_fingerprints.push_back(macro_exp_state_fingerprint(current, actions_));
    double prior_sum = 0.0;
    int prior_count = 0;
    bool tactical_used = false;
    bool stop_greedy_completion = true;

    for (size_t i = 0; i < forced_action_signatures.size(); ++i) {
      if (current.terminal || current.active_player_id != starting_player || plan.reached_turn_boundary) break;
      if (config_.max_primitives_per_turn > 0 &&
          plan.primitives_executed >= config_.max_primitives_per_turn) break;
      const double prior = i < forced_priors.size() ? forced_priors[i] : 0.0;
      const int resolved_index = resolve_action_signature(current, forced_action_signatures[i]);
      // These signatures are the inner MCTS trajectory itself.  They have
      // already been selected from successive legal child states, so do not
      // truncate the trajectory merely because it contains a tactical action.
      // The old guard made every sampled plan stop after its first primitive.
      if (!apply_recorded_action(plan, current, resolved_index, prior)) break;
      if (resolved_index >= 0 && macro_exp_is_tactical_type(actions_[resolved_index].type)) tactical_used = true;
      prior_sum += std::max(0.0, prior);
      prior_count += 1;
    }

    while (!stop_greedy_completion &&
           !current.terminal && current.active_player_id == starting_player && !plan.reached_turn_boundary &&
           (config_.max_primitives_per_turn <= 0 ||
            plan.primitives_executed < config_.max_primitives_per_turn)) {
      const auto [action_index, prior] = static_best_action(current);
      if (action_index < 0) break;
      if (tactical_used && macro_exp_is_tactical_type(actions_[action_index].type)) break;
      if (!apply_recorded_action(plan, current, action_index, prior)) break;
      if (macro_exp_is_tactical_type(actions_[action_index].type)) tactical_used = true;
      if (macro_exp_is_tactical_type(actions_[action_index].type) ||
          macro_exp_is_forced_type(actions_[action_index].type)) {
        stop_greedy_completion = true;
      }
      prior_sum += std::max(0.0, prior);
      prior_count += 1;
    }

    int forced_finalize_steps = 0;
    while (!current.terminal && current.active_player_id == starting_player && !plan.reached_turn_boundary) {
      if (forced_finalize_steps++ > 1024) {
        throw std::runtime_error("Turn-macro forced finalization exceeded its cycle-safety guard.");
      }
      const int end_index = macro_exp_find_legal_action_by_signature(
          current,
          actions_,
          "END_TURN:p=" + std::to_string(current.active_player_id));
      const int fallback_end_index = end_index >= 0
          ? end_index
          : macro_exp_find_legal_action_by_signature(current, actions_, "END_TURN");
      if (fallback_end_index >= 0) {
        if (apply_recorded_action(plan, current, fallback_end_index, 0.0)) {
          prior_count += 1;
        }
        break;
      }

      int forced_index = -1;
      for (int legal_index : current.legal_action_indexes) {
        if (legal_index < 0 || legal_index >= static_cast<int>(actions_.size())) continue;
        if (!macro_exp_is_plausible_action(current, actions_[legal_index])) continue;
        if (macro_exp_is_forced_type(actions_[legal_index].type)) {
          forced_index = legal_index;
          break;
        }
      }
      if (forced_index < 0) break;
      if (apply_recorded_action(plan, current, forced_index, 0.0)) {
        prior_count += 1;
      } else {
        break;
      }
    }

    if (!current.terminal && current.active_player_id == starting_player && !plan.reached_turn_boundary) {
      return TurnEdge();
    }

    plan.result_state = std::move(current);
    plan.terminal = plan.result_state.terminal;
    if (std::getenv("TRIBES_TURN_MACRO_TRACE") != nullptr) {
      std::cerr << "turn_macro_value_begin terminal=" << plan.terminal
                << " player=" << plan.result_state.active_player_id
                << " legal=" << plan.result_state.legal_action_indexes.size()
                << std::endl;
    }
    plan.value_root = evaluate_state_root_perspective(plan.result_state);
    plan.value_actor = starting_player == root_player_id_
        ? plan.value_root
        : evaluate_state_for_player(plan.result_state, starting_player);
    if (std::getenv("TRIBES_TURN_MACRO_TRACE") != nullptr) {
      std::cerr << "turn_macro_value_done value=" << plan.value_root << std::endl;
    }
    max_primitives_per_turn_sample_ = std::max(max_primitives_per_turn_sample_, plan.primitives_executed);
    sampled_plans += 1;

    TurnEdge edge;
    edge.plan = std::move(plan);
    edge.prior = prior_count > 0 ? prior_sum / static_cast<double>(prior_count) : 0.0;
    return edge;
  };

  auto accept_edge = [&](TurnEdge&& edge, const std::vector<std::string>& reasons) {
    if (edge.plan.executed_macro_exp_action_signatures.empty()) return false;
    const std::string exact_key = plan_key(edge.plan);
    if (exact_key.empty() || !used_plan_keys.insert(exact_key).second) {
      macro_exp_exact_duplicates_removed_ += 1;
      return false;
    }
    edge.diversity_key = macro_exp_plan_diversity_key(edge.plan);
    if (edge.diversity_key.empty() || !used_diversity_keys.insert(edge.diversity_key).second) {
      macro_exp_material_duplicates_removed_ += 1;
      return false;
    }
    edge.selection_reasons = reasons;
    return true;
  };

  auto build_confident_edges = [&](const std::vector<MacroExpInnerActionCandidate>& inner_candidates) {
    std::vector<TurnEdge> edges;
    edges.reserve(static_cast<size_t>(std::max(1, max_edges)));
    for (const MacroExpInnerActionCandidate& candidate : inner_candidates) {
      if (static_cast<int>(edges.size()) >= max_edges) break;
      if (candidate.global_action_index < 0 ||
          candidate.global_action_index >= static_cast<int>(actions_.size()) ||
          !macro_exp_is_plausible_action(root_state, actions_[candidate.global_action_index]) ||
          !std::isfinite(candidate.log_confidence)) {
        continue;
      }
      TurnEdge edge = complete_plan(candidate.action_signatures, candidate.action_priors);
      edge.log_confidence = candidate.log_confidence;
      if (accept_edge(std::move(edge), {"confidence"})) {
        edges.push_back(std::move(edge));
      }
    }
    if (!edges.empty()) {
      const double max_log_confidence = std::max_element(
          edges.begin(), edges.end(), [](const TurnEdge& left, const TurnEdge& right) {
            return left.log_confidence < right.log_confidence;
          })->log_confidence;
      double confidence_sum = 0.0;
      for (TurnEdge& edge : edges) {
        edge.prior = std::exp(edge.log_confidence - max_log_confidence);
        confidence_sum += edge.prior;
      }
      if (confidence_sum > 0.0 && std::isfinite(confidence_sum)) {
        for (TurnEdge& edge : edges) edge.prior /= confidence_sum;
      } else {
        const double uniform = 1.0 / static_cast<double>(edges.size());
        for (TurnEdge& edge : edges) edge.prior = uniform;
      }
    }
    return edges;
  };

  if (root_state.active_player_id != root_player_id_) {
    std::vector<MacroExpInnerActionCandidate> opponent_candidates = run_inner_candidates(root_state);
    if (opponent_candidates.empty()) {
      throw std::runtime_error(
          "Turn-macro inner search produced no plausible opponent action for active player " +
          std::to_string(root_state.active_player_id));
    }
    macro_exp_raw_candidates_ += static_cast<int>(opponent_candidates.size());
    std::vector<TurnEdge> opponent_plans = build_confident_edges(opponent_candidates);

    for (TurnEdge& edge : opponent_plans) {
      if (static_cast<int>(out.size()) >= max_edges) break;
      out.push_back(std::move(edge));
    }
    nodes_[node_id].macro_exp.samples += sampled_plans;
    return out;
  };

  std::vector<MacroExpInnerActionCandidate> root_candidates = run_inner_candidates(root_state);
  if (root_candidates.empty()) return out;

  macro_exp_raw_candidates_ += static_cast<int>(root_candidates.size());
  std::vector<TurnEdge> candidates = build_confident_edges(root_candidates);

  for (TurnEdge& edge : candidates) {
    if (static_cast<int>(out.size()) >= max_edges) break;
    out.push_back(std::move(edge));
  }
  nodes_[node_id].macro_exp.samples += sampled_plans;
  return out;
}

void TurnMacroExpMCTS::run(int simulations) {
  if (nodes_.empty()) return;
  const auto started = Clock::now();
  const bool trace = std::getenv("TRIBES_TURN_MACRO_TRACE") != nullptr;
  for (int sim = 0; sim < simulations; ++sim) {
    std::vector<int> path_node_ids;
    std::vector<int> path_edge_ids;
    int node_id = 0;
    int sim_turn_depth = 0;
    double leaf_value_root = nodes_[0].value_estimate_root;
    if (trace) {
      std::cerr << "turn_macro_sim_begin sim=" << sim << " nodes=" << nodes_.size()
                << " states=" << states_.size() << " actions=" << actions_.size() << std::endl;
    }

    for (;;) {
      const int state_index = nodes_[node_id].state_index;
      const bool node_terminal = nodes_[node_id].terminal;
      const NativeGameState& state = states_[state_index];
      if (trace) {
        std::cerr << "turn_macro_node sim=" << sim << " node=" << node_id
                  << " state=" << state_index
                  << " player=" << state.active_player_id
                  << " edges=" << nodes_[node_id].edges.size()
                  << " visits=" << nodes_[node_id].visits
                  << " exhausted=" << nodes_[node_id].expansion_exhausted
                  << " legal=" << state.legal_action_indexes.size() << std::endl;
      }
      if (node_terminal || state.terminal) {
        leaf_value_root = evaluate_state_root_perspective(state);
        break;
      }

      if (!nodes_[node_id].expansion_exhausted) {
        nodes_[node_id].expansion_exhausted = true;
        const int requested_edges = std::max(1, config_.max_new_edges_per_node);
        if (trace) {
          std::cerr << "turn_macro_expand sim=" << sim << " node=" << node_id
                    << " slots=" << requested_edges << std::endl;
        }
        std::vector<TurnEdge> new_edges = generate_turn_edges(node_id, requested_edges);
        if (trace) {
          std::cerr << "turn_macro_expand_done sim=" << sim << " node=" << node_id
                    << " new_edges=" << new_edges.size() << std::endl;
        }
        if (!new_edges.empty()) {
          const int edge_id = static_cast<int>(nodes_[node_id].edges.size());
          for (TurnEdge& edge : new_edges) {
            if (trace) {
              std::cerr << "turn_macro_edge_push_begin sim=" << sim << " node=" << node_id
                        << " existing_edges=" << nodes_[node_id].edges.size()
                        << " primitives=" << edge.plan.primitives_executed
                        << " terminal=" << edge.plan.terminal << std::endl;
            }
            nodes_[node_id].edges.push_back(std::move(edge));
            if (trace) {
              std::cerr << "turn_macro_edge_push_done sim=" << sim << " node=" << node_id
                        << " edges=" << nodes_[node_id].edges.size() << std::endl;
            }
          }
          path_node_ids.push_back(node_id);
          path_edge_ids.push_back(edge_id);
          sim_turn_depth += 1;
          leaf_value_root = nodes_[node_id].edges[edge_id].plan.value_root;
          break;
        }
        if (nodes_[node_id].edges.empty()) break;
      }

      const int edge_id = select_existing_turn_edge(nodes_[node_id]);
      if (edge_id < 0) break;
      if (trace) {
        std::cerr << "turn_macro_select sim=" << sim << " node=" << node_id
                  << " edge=" << edge_id
                  << " child=" << nodes_[node_id].edges[edge_id].child_node_id
                  << " terminal=" << nodes_[node_id].edges[edge_id].plan.terminal
                  << " primitives=" << nodes_[node_id].edges[edge_id].plan.primitives_executed
                  << std::endl;
      }
      path_node_ids.push_back(node_id);
      path_edge_ids.push_back(edge_id);
      sim_turn_depth += 1;
      const int child_node_id = nodes_[node_id].edges[edge_id].child_node_id;
      if (child_node_id < 0) {
        const double edge_value_root = nodes_[node_id].edges[edge_id].plan.value_root;
        const bool edge_terminal = nodes_[node_id].edges[edge_id].plan.terminal;
        leaf_value_root = edge_value_root;
        if (!edge_terminal) {
          NativeGameState child_state = nodes_[node_id].edges[edge_id].plan.result_state;
          const int new_child_node_id = make_turn_node_with_value(std::move(child_state), edge_value_root);
          nodes_[node_id].edges[edge_id].child_node_id = new_child_node_id;
        }
        break;
      }
      node_id = child_node_id;
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

void TurnMacroExpMCTS::backup_turn_path(
    const std::vector<int>& node_ids,
    const std::vector<int>& edge_ids,
    double value_root) {
  for (size_t i = 0; i < node_ids.size() && i < edge_ids.size(); ++i) {
    TurnNode& node = nodes_[node_ids[i]];
    TurnEdge& edge = node.edges[edge_ids[i]];
    node.visits += 1;
    edge.visits += 1;
    edge.value_sum_root += value_root;
    edge.value_sum_actor += edge.plan.value_actor;
  }
}

py::dict TurnMacroExpMCTS::result_py(double temperature, bool sample_action) {
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
    const double lq = lv > 0 ? value_by_id[left] / static_cast<double>(lv) : 0.0;
    const double rq = rv > 0 ? value_by_id[right] / static_cast<double>(rv) : 0.0;
    if (lv != rv) return lv > rv;
    if (lq != rq) return lq > rq;
    return left < right;
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

  const TurnEdge* selected_edge = nullptr;
  for (const TurnEdge& edge : root.edges) {
    if (edge.plan.first_action_id != selected_id) continue;
    if (selected_edge == nullptr || edge.visits > selected_edge->visits ||
        (edge.visits == selected_edge->visits && edge.plan.value_root > selected_edge->plan.value_root)) {
      selected_edge = &edge;
    }
  }

  response["actionId"] = selected_id.empty() ? py::none() : py::str(selected_id);
  response["rankedActionIds"] = ranked;
  if (selected_edge != nullptr && selected_edge->plan.executed_macro_exp_action_signatures.size() > 1 &&
      selected_edge->plan.prefix_state_fingerprints.size() ==
          selected_edge->plan.executed_macro_exp_action_signatures.size() + 1) {
    py::dict continuation;
    py::list signatures;
    py::list fingerprints;
    for (size_t index = 1; index < selected_edge->plan.executed_macro_exp_action_signatures.size(); ++index) {
      signatures.append(selected_edge->plan.executed_macro_exp_action_signatures[index]);
      fingerprints.append(selected_edge->plan.prefix_state_fingerprints[index]);
    }
    continuation["action_signatures"] = signatures;
    continuation["state_fingerprints"] = fingerprints;
    continuation["root_player_id"] = root_player_id_;
    continuation["reaches_turn_boundary"] = selected_edge->plan.reached_turn_boundary;
    response["_macro_continuation"] = continuation;
  }
  if (config_.profile_json) {
    response["_profile"] = profile_py();
  }
  return response;
}

py::dict TurnMacroExpMCTS::profile_py() const {
  py::dict profile;
  int edge_count = 0;
  int macro_exp_samples = 0;
  for (const TurnNode& node : nodes_) {
    edge_count += static_cast<int>(node.edges.size());
    macro_exp_samples += node.macro_exp.samples;
  }
  py::dict root_first_action_visits;
  py::list root_turn_plans;
  if (!nodes_.empty()) {
    for (const TurnEdge& edge : nodes_[0].edges) {
      py::dict plan;
      py::list signatures;
      for (const std::string& signature : edge.plan.executed_macro_exp_action_signatures) {
        signatures.append(signature);
      }
      plan["action_signatures"] = signatures;
      plan["first_action_id"] = edge.plan.first_action_id;
      plan["prior"] = edge.prior;
      plan["edge_prior"] = edge.prior;
      plan["log_confidence"] = edge.log_confidence;
      plan["value_root"] = edge.plan.value_root;
      plan["primitives_executed"] = edge.plan.primitives_executed;
      plan["diversity_key"] = edge.diversity_key;
      py::list selection_reasons;
      for (const std::string& reason : edge.selection_reasons) {
        selection_reasons.append(reason);
      }
      plan["selection_reasons"] = selection_reasons;
      root_turn_plans.append(plan);
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
  timing["inner_search_ms"] = inner_search_ms_;
  timing["apply_action_ms"] = apply_action_ms_;
  timing["static_eval_ms"] = static_eval_ms_;
  timing["backup_ms"] = backup_ms_;

  profile["search_mode"] = py::str("turn-macro-exp");
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
  profile["macro_exp_samples"] = macro_exp_samples;
  profile["inner_searches"] = inner_searches_;
  profile["inner_simulations"] = inner_simulations_executed_;
  profile["inner_nodes_expanded"] = inner_nodes_expanded_;
  profile["inner_simulations_per_search"] = inner_searches_ > 0
      ? static_cast<double>(inner_simulations_executed_) / static_cast<double>(inner_searches_)
      : 0.0;
  profile["macro_exp_raw_candidates"] = macro_exp_raw_candidates_;
  profile["macro_exp_visited_candidates"] = macro_exp_visited_candidates_;
  profile["macro_exp_value_candidates"] = macro_exp_value_candidates_;
  profile["macro_exp_prior_candidates"] = macro_exp_prior_candidates_;
  profile["macro_exp_exact_duplicates_removed"] = macro_exp_exact_duplicates_removed_;
  profile["macro_exp_material_duplicates_removed"] = macro_exp_material_duplicates_removed_;
  profile["static_eval_calls"] = static_eval_calls_;
  profile["greedy_static_calls"] = greedy_static_calls_;
  profile["greedy_static_candidates_considered"] = greedy_static_candidates_considered_;
  profile["greedy_static_child_evals"] = greedy_static_child_evals_;
  profile["greedy_static_child_eval_skips"] = greedy_static_child_eval_skips_;
  profile["greedy_static_child_eval_limit"] = config_.greedy_eval_top_k;
  profile["greedy_static_child_eval_skip_rate"] = greedy_static_child_evals_ + greedy_static_child_eval_skips_ > 0
      ? static_cast<double>(greedy_static_child_eval_skips_) /
            static_cast<double>(greedy_static_child_evals_ + greedy_static_child_eval_skips_)
      : 0.0;
  profile["primitive_actions_executed"] = primitive_actions_executed_;
  profile["avg_primitives_per_turn_sample"] = macro_exp_samples > 0
      ? static_cast<double>(primitive_actions_executed_) / static_cast<double>(macro_exp_samples)
      : 0.0;
  profile["max_primitives_per_turn_sample"] = max_primitives_per_turn_sample_;
  profile["fallback_used"] = false;
  profile["root_first_action_visits"] = root_first_action_visits;
  profile["root_turn_plans"] = root_turn_plans;
  profile["timing_ms"] = timing;
  return profile;
}

int TurnMacroExpMCTS::root_edge_count() const {
  return nodes_.empty() ? 0 : static_cast<int>(nodes_[0].edges.size());
}

}  // namespace tribes::native
