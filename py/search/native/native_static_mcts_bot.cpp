#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "third_party/nlohmann/json.hpp"

#ifndef TRIBES_NATIVE_MCTS_STANDALONE
#define TRIBES_NATIVE_MCTS_STANDALONE
#endif
#include "native_mcts.cpp"

using json = nlohmann::json;

namespace {

struct CliConfig {
  int simulations = 64;
  double wall_clock_seconds = 0.0;
  int top_k_actions = 64;
  int max_actions = 512;
  int batch_size = 64;
  double c_puct = 1.5;
  double dirichlet_alpha = 0.3;
  double dirichlet_epsilon = 0.25;
  double root_temperature = 1.0;
  bool sample_action = true;
  bool reuse_tree = false;
  bool progressive_widening = true;
  bool profile_json = false;
  uint64_t seed = 13;
  std::string static_eval_variant = "baseline";
};

py::object py_from_json(const json& value) {
  if (value.is_null()) return py::none();
  if (value.is_boolean()) return py::bool_(value.get<bool>());
  if (value.is_number_integer()) return py::int_(value.get<long long>());
  if (value.is_number_unsigned()) return py::int_(value.get<unsigned long long>());
  if (value.is_number_float()) return py::float_(value.get<double>());
  if (value.is_string()) return py::str(value.get<std::string>());
  if (value.is_array()) {
    py::list out;
    for (const json& item : value) out.append(py_from_json(item));
    return std::move(out);
  }
  if (value.is_object()) {
    py::dict out;
    for (auto it = value.begin(); it != value.end(); ++it) out[py::str(it.key())] = py_from_json(it.value());
    return std::move(out);
  }
  return py::none();
}

json json_from_py(py::handle value) {
  if (value.is_none()) return nullptr;
  if (py::isinstance<py::bool_>(value)) return py::cast<bool>(value);
  if (py::isinstance<py::int_>(value)) return py::cast<long long>(value);
  if (py::isinstance<py::float_>(value)) return py::cast<double>(value);
  if (py::isinstance<py::str>(value)) return py::cast<std::string>(py::str(value));
  if (py::isinstance<py::dict>(value)) {
    json out = json::object();
    py::dict dict = py::reinterpret_borrow<py::dict>(value);
    for (const auto& item : dict) out[py::cast<std::string>(py::str(item.first))] = json_from_py(item.second);
    return out;
  }
  if (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value)) {
    json out = json::array();
    py::list values = py::reinterpret_borrow<py::list>(value);
    for (const auto& item : values) out.push_back(json_from_py(item));
    return out;
  }
  return py::cast<std::string>(py::str(value));
}

int json_int(const json& object, const char* key, int fallback = 0) {
  if (!object.is_object() || !object.contains(key) || object[key].is_null()) return fallback;
  try {
    if (object[key].is_number_integer()) return object[key].get<int>();
    if (object[key].is_number()) return static_cast<int>(object[key].get<double>());
    if (object[key].is_string()) return std::stoi(object[key].get<std::string>());
  } catch (...) {
  }
  return fallback;
}

std::string json_string(const json& object, const char* key, const std::string& fallback = "") {
  if (!object.is_object() || !object.contains(key) || object[key].is_null()) return fallback;
  if (object[key].is_string()) return object[key].get<std::string>();
  return object[key].dump();
}

std::vector<json> json_actions(const json& payload, int max_actions) {
  std::vector<json> out;
  if (!payload.contains("actions") || !payload["actions"].is_array()) return out;
  int count = static_cast<int>(payload["actions"].size());
  if (max_actions >= 0) count = std::min(count, max_actions);
  out.reserve(count);
  for (int i = 0; i < count; ++i) {
    if (payload["actions"][i].is_object()) out.push_back(payload["actions"][i]);
  }
  return out;
}

std::string cli_action_type(const json& action) {
  std::string type = json_string(action, "type");
  if (type.empty()) type = json_string(action, "t");
  if (type.empty() && action.contains("payload") && action["payload"].is_object()) {
    type = json_string(action["payload"], "type");
    if (type.empty()) type = json_string(action["payload"], "t");
  }
  return type;
}

std::string cli_action_id(const json& action, int index) {
  std::string id = json_string(action, "id");
  if (id.empty()) id = "A" + std::to_string(json_int(action, "i", index));
  return id;
}

int cli_action_unit_id(const json& action) {
  int value = json_int(action, "unit_id", 0);
  if (value == 0) value = json_int(action, "u", 0);
  if (value == 0 && action.contains("payload") && action["payload"].is_object()) {
    value = json_int(action["payload"], "unit_id", json_int(action["payload"], "u", 0));
  }
  return value;
}

std::pair<int, int> cli_action_xy(const json& action) {
  const json* source = &action;
  if (action.contains("payload") && action["payload"].is_object()) source = &action["payload"];
  if (source->contains("destination") && (*source)["destination"].is_array() && (*source)["destination"].size() >= 2) {
    return {(*source)["destination"][0].get<int>(), (*source)["destination"][1].get<int>()};
  }
  return {json_int(*source, "x", 0), json_int(*source, "y", 0)};
}

bool cli_keep_root_type(const std::string& type) {
  static const std::unordered_set<std::string> keep = {
      "CAPTURE", "MAKE_VETERAN", "ATTACK", "CONVERT", "EXAMINE", "RESOURCE_GATHERING",
      "LEVEL_UP", "RESEARCH_TECH", "BUILD", "SPAWN", "RECOVER", "HEAL_OTHERS",
      "UPGRADE_SHIP", "UPGRADE_BOAT", "UPGRADE_RAMMER", "UPGRADE_SCOUT", "UPGRADE_BOMBER"};
  return keep.count(type) > 0;
}

std::vector<int> select_root_indexes(const std::vector<json>& actions, const std::vector<double>& priors, int top_k_actions) {
  std::vector<int> indexes(actions.size());
  for (int i = 0; i < static_cast<int>(actions.size()); ++i) indexes[i] = i;
  if (top_k_actions <= 0 || static_cast<int>(indexes.size()) <= top_k_actions) return indexes;

  std::vector<int> forced;
  std::unordered_set<int> selected;
  for (int i = 0; i < static_cast<int>(actions.size()); ++i) {
    if (cli_keep_root_type(cli_action_type(actions[i]))) {
      forced.push_back(i);
      selected.insert(i);
    }
  }
  const int budget = std::max(top_k_actions, static_cast<int>(selected.size()));
  std::sort(indexes.begin(), indexes.end(), [&](int left, int right) {
    const double l = left < static_cast<int>(priors.size()) ? priors[left] : 0.0;
    const double r = right < static_cast<int>(priors.size()) ? priors[right] : 0.0;
    return l > r;
  });
  std::unordered_set<std::string> seen_moves;
  for (int index : indexes) {
    if (static_cast<int>(selected.size()) >= budget) break;
    if (cli_action_type(actions[index]) == "MOVE") {
      auto [x, y] = cli_action_xy(actions[index]);
      const std::string sig = std::to_string(cli_action_unit_id(actions[index])) + ":" + std::to_string(x) + ":" + std::to_string(y);
      if (seen_moves.count(sig)) continue;
      seen_moves.insert(sig);
    }
    selected.insert(index);
  }

  std::vector<int> ordered = forced;
  for (int index : indexes) {
    if (selected.count(index) && std::find(forced.begin(), forced.end(), index) == forced.end()) ordered.push_back(index);
    if (static_cast<int>(ordered.size()) >= budget) break;
  }
  return ordered;
}

std::vector<double> coerce_priors(py::handle raw_priors, int action_count) {
  std::vector<double> priors;
  if (py::isinstance<py::list>(raw_priors) || py::isinstance<py::tuple>(raw_priors)) {
    py::list values = py::reinterpret_borrow<py::list>(raw_priors);
    for (py::handle item : values) {
      if (static_cast<int>(priors.size()) >= action_count) break;
      priors.push_back(py::cast<double>(item));
    }
  }
  if (static_cast<int>(priors.size()) != action_count) {
    throw std::runtime_error("Static eval prior/action mismatch.");
  }
  double total = 0.0;
  for (double prior : priors) total += std::max(0.0, prior);
  if (total <= 0.0) throw std::runtime_error("Static eval returned non-positive total prior mass.");
  for (double& prior : priors) prior = std::max(0.0, prior) / total;
  return priors;
}

json choose_action_with_native_tree(const json& message, const CliConfig& cfg, std::mt19937_64& rng) {
  std::vector<json> root_actions = json_actions(message, cfg.max_actions);
  if (root_actions.empty()) return json{{"actionId", nullptr}, {"rankedActionIds", json::array()}};

  py::dict root_payload = py::reinterpret_borrow<py::dict>(py_from_json(message));
  py::dict eval_payload;
  eval_payload["player_id"] = py::int_(json_int(message, "player_id", 0));
  eval_payload["observation"] = root_payload["observation"];
  eval_payload["actions"] = root_payload["actions"];
  py::dict root_eval = tribes::native::evaluate_static(eval_payload, cfg.max_actions);
  std::vector<double> all_priors = coerce_priors(root_eval["priors"], static_cast<int>(root_actions.size()));
  const double root_value = py::cast<double>(root_eval["value"]);

  std::vector<int> root_indexes = select_root_indexes(root_actions, all_priors, cfg.top_k_actions);
  std::vector<std::string> searched_ids;
  std::vector<double> priors;
  for (int index : root_indexes) {
    searched_ids.push_back(cli_action_id(root_actions[index], index));
    priors.push_back(all_priors[index]);
  }
  double total = 0.0;
  for (double prior : priors) total += std::max(0.0, prior);
  if (total <= 0.0) {
    std::fill(priors.begin(), priors.end(), 1.0 / std::max<size_t>(1, priors.size()));
  } else {
    for (double& prior : priors) prior = std::max(0.0, prior) / total;
  }

  NativeMCTS tree(
      root_payload,
      root_indexes,
      priors,
      root_value,
      message.value("is_terminal", message.value("terminal", false)),
      cfg.seed,
      cfg.max_actions,
      cfg.progressive_widening);
  tree.add_root_dirichlet_noise(cfg.dirichlet_alpha, cfg.dirichlet_epsilon);
  tree.reserve_tree_capacity(std::max(2, cfg.simulations + 1));

  const auto started = std::chrono::steady_clock::now();
  int expanded = 0;
  int completed_paths = 0;
  int depth_sum = 0;
  int max_selected_depth = 0;
  int turn_depth_sum = 0;
  int max_turn_depth = 0;
  while ((cfg.wall_clock_seconds > 0.0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() < cfg.wall_clock_seconds) ||
         (cfg.wall_clock_seconds <= 0.0 && expanded < cfg.simulations)) {
    const int frontier = cfg.wall_clock_seconds > 0.0 ? std::max(1, cfg.batch_size) : std::min(std::max(1, cfg.batch_size), std::max(1, cfg.simulations - expanded));
    py::tuple batch = tree.run_static_search_batch(frontier, cfg.c_puct);
    const int expanded_count = py::cast<int>(batch[0]);
    const int completed = py::cast<int>(batch[1]);
    py::tuple batch_stats = tree.last_batch_stats();
    expanded += std::max(0, expanded_count);
    completed_paths += std::max(0, completed);
    depth_sum += py::cast<int>(batch_stats[0]);
    max_selected_depth = std::max(max_selected_depth, py::cast<int>(batch_stats[1]));
    turn_depth_sum += py::cast<int>(batch_stats[2]);
    max_turn_depth = std::max(max_turn_depth, py::cast<int>(batch_stats[3]));
    if (completed <= 0 || (cfg.wall_clock_seconds <= 0.0 && expanded_count <= 0)) break;
  }
  const double search_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

  std::vector<double> probs = tree.root_visit_distribution_by_index(cfg.root_temperature);
  if (probs.empty()) probs = priors;
  int selected = 0;
  if (cfg.sample_action) {
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    selected = dist(rng);
  } else {
    for (int i = 1; i < static_cast<int>(probs.size()); ++i) {
      if (probs[i] > probs[selected]) selected = i;
    }
  }

  std::vector<int> rank(probs.size());
  for (int i = 0; i < static_cast<int>(rank.size()); ++i) rank[i] = i;
  std::sort(rank.begin(), rank.end(), [&](int left, int right) {
    return probs[left] > probs[right];
  });
  json ranked = json::array();
  std::unordered_set<std::string> seen;
  for (int index : rank) {
    if (index >= 0 && index < static_cast<int>(searched_ids.size())) {
      ranked.push_back(searched_ids[index]);
      seen.insert(searched_ids[index]);
    }
  }
  for (int i = 0; i < static_cast<int>(root_actions.size()); ++i) {
    const std::string id = cli_action_id(root_actions[i], i);
    if (!seen.count(id)) ranked.push_back(id);
  }
  const std::string selected_id = selected < static_cast<int>(searched_ids.size()) ? searched_ids[selected] : searched_ids.front();
  json response{{"actionId", selected_id}, {"rankedActionIds", ranked}};
  if (cfg.profile_json) {
    response["_profile"] = {
        {"elapsed_sec", search_elapsed},
        {"simulations", expanded},
        {"selected_paths", completed_paths},
        {"expanded_nodes", expanded},
        {"depth_sum", depth_sum},
        {"max_depth", max_selected_depth},
        {"turn_depth_sum", turn_depth_sum},
        {"max_turn_depth", max_turn_depth},
        {"node_count", tree.node_count()},
    };
  }
  return response;
}

void parse_args(int argc, char** argv, CliConfig& cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
      return argv[++i];
    };
    if (arg == "--simulations") cfg.simulations = std::stoi(next());
    else if (arg == "--wall-clock-per-action-seconds") cfg.wall_clock_seconds = std::stod(next());
    else if (arg == "--top-k-actions") cfg.top_k_actions = std::stoi(next());
    else if (arg == "--max-actions") cfg.max_actions = std::stoi(next());
    else if (arg == "--search-batch-size") cfg.batch_size = std::stoi(next());
    else if (arg == "--static-eval-variant") cfg.static_eval_variant = next();
    else if (arg == "--deterministic") {
      cfg.sample_action = false;
      cfg.root_temperature = 1e-6;
      cfg.dirichlet_epsilon = 0.0;
    } else if (arg == "--reuse-tree") {
      cfg.reuse_tree = true;
    } else if (arg == "--profile-json") {
      cfg.profile_json = true;
    } else if (arg == "--seed") {
      cfg.seed = static_cast<uint64_t>(std::stoull(next()));
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "native_static_mcts_bot.exe [--simulations N] [--wall-clock-per-action-seconds SEC]\n"
          << "  [--top-k-actions N] [--max-actions N] [--search-batch-size N]\n"
          << "  [--static-eval-variant baseline|experimental] [--deterministic] [--reuse-tree] [--seed N]\n";
      std::exit(0);
    }
  }
}

void set_static_eval_variant_env(const std::string& variant) {
#ifdef _WIN32
  _putenv_s("TRIBES_STATIC_EVAL_VARIANT", variant.c_str());
#else
  setenv("TRIBES_STATIC_EVAL_VARIANT", variant.c_str(), 1);
#endif
}

}  // namespace

int main(int argc, char** argv) {
  CliConfig cfg;
  try {
    parse_args(argc, argv, cfg);
    set_static_eval_variant_env(cfg.static_eval_variant);
    std::mt19937_64 rng(cfg.seed);
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      json message = json::parse(line, nullptr, false);
      if (message.is_discarded()) continue;
      const std::string type = message.value("type", "");
      if (type == "action_request") {
        std::cout << choose_action_with_native_tree(message, cfg, rng).dump() << std::endl;
      } else if (type == "game_over") {
        break;
      }
    }
  } catch (const std::exception& exc) {
    std::cerr << "[native_static_mcts_bot] error: " << exc.what() << std::endl;
    return 1;
  }
  return 0;
}
