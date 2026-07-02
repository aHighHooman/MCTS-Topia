#include <algorithm>
#include <chrono>
#include <cctype>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../py/search/native/third_party/nlohmann/json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

struct Args {
  fs::path config_path;
  std::vector<fs::path> records;
  std::vector<std::string> whitelist;
  std::vector<int> level_seed;
  std::vector<int> validation_level_seed;
  std::vector<std::string> population_overrides;
  std::vector<std::string> validation_opponents{"baseline=baseline"};
  std::vector<std::string> tribes{"Xin Xi", "Imperius"};
  fs::path output_dir = "debug-logs/analysis/static-value-result-tuning";
  std::string run_id;
  fs::path recording_proxy_exe = "out/native/static_recording_proxy_native.exe";
  fs::path native_tuner_exe = "out/native/static_value_result_tuner.exe";
  fs::path static_exe = "out/native/static_mcts_bot.exe";
  fs::path java_exe;
  fs::path tournament_config;
  std::string static_eval_variant = "baseline";
  std::string initial_overrides;
  std::string search_mode = "primitive";
  std::string run_mode = "PlayLG";
  std::string game_mode = "Capitals";
  std::string map_type = "Drylands";
  std::string map_size = "Tiny";
  int max_actions = 512;
  int steps = 2000;
  int seed = 0;
  int validation_games = 0;
  int validation_seed_start = 900000;
  int collect_games = 0;
  int collect_seed_start = 800000;
  int loop_iterations = 0;
  int loop_games = 0;
  int parallel_games = 8;
  int simulations = 1000;
  int batch_size = 64;
  int top_k_actions = 0;
  int max_turns_capitals = 40;
  int max_actions_per_turn = 0;
  int max_actions_per_game = 1024;
  int external_action_timeout_ms = 120000;
  int feature_threads = 0;
  double learning_rate = 0.05;
  double l2 = 0.01;
  double l1 = 0.0;
  double max_abs_delta = 0.75;
  double max_relative_delta = 0.5;
  double validation_fraction = 0.2;
  double collect_sample_rate = 1.0;
  std::unordered_map<std::string, double> feature_scales;
  bool preserve_sign = true;
  bool balance_seats = true;
  bool validation_balance_seats = true;
  bool write_match_logs = false;
  bool collect_deterministic = false;
  bool validation_deterministic = false;
  bool resume_loop = false;
  int resume_from_iteration = 0;
};

struct ValidationOpponent {
  std::string label;
  std::string variant;
  std::string overrides;
};

std::string read_text(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("failed to read " + path.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void write_text(const fs::path& path, const std::string& text) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("failed to write " + path.string());
  out << text;
}

json read_json(const fs::path& path) {
  if (!fs::exists(path)) return json::object();
  json value = json::parse(read_text(path), nullptr, true);
  return value.is_object() ? value : json::object();
}

json read_json_any(const fs::path& path, const json& fallback = json::object()) {
  if (!fs::exists(path)) return fallback;
  return json::parse(read_text(path), nullptr, true);
}

std::string as_string(const json& value, const std::string& fallback = "") {
  if (value.is_string()) return value.get<std::string>();
  if (value.is_null()) return fallback;
  return value.dump();
}

int as_int(const json& value, int fallback = 0) {
  try {
    if (value.is_number_integer()) return value.get<int>();
    if (value.is_number()) return static_cast<int>(value.get<double>());
    if (value.is_string()) return std::stoi(value.get<std::string>());
  } catch (...) {
  }
  return fallback;
}

double as_double(const json& value, double fallback = 0.0) {
  try {
    if (value.is_number()) return value.get<double>();
    if (value.is_string()) return std::stod(value.get<std::string>());
  } catch (...) {
  }
  return fallback;
}

bool as_bool(const json& value, bool fallback = false) {
  if (value.is_boolean()) return value.get<bool>();
  if (value.is_number()) return value.get<double>() != 0.0;
  if (value.is_string()) {
    std::string s = value.get<std::string>();
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !(s.empty() || s == "0" || s == "false" || s == "no" || s == "off");
  }
  return fallback;
}

std::vector<std::string> string_array(const json& value) {
  std::vector<std::string> out;
  if (!value.is_array()) return out;
  for (const json& item : value) out.push_back(as_string(item));
  return out;
}

std::vector<int> int_array(const json& value) {
  std::vector<int> out;
  if (!value.is_array()) return out;
  for (const json& item : value) out.push_back(as_int(item));
  return out;
}

void add_feature_scale(Args& args, const std::string& raw) {
  const size_t pos = raw.find('=');
  if (pos == std::string::npos || pos == 0) throw std::runtime_error("feature scale must be term=scale");
  args.feature_scales[raw.substr(0, pos)] = std::stod(raw.substr(pos + 1));
}

std::string safe_label(std::string value) {
  for (char& ch : value) {
    if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '-' && ch != '_') ch = '_';
  }
  return value.empty() ? "opponent" : value;
}

ValidationOpponent parse_validation_opponent(const std::string& raw) {
  ValidationOpponent out;
  const size_t eq = raw.find('=');
  if (eq == std::string::npos) {
    out.label = safe_label(raw);
    out.variant = raw;
    return out;
  }
  out.label = safe_label(raw.substr(0, eq));
  const std::string rest = raw.substr(eq + 1);
  const size_t pipe = rest.find('|');
  out.variant = pipe == std::string::npos ? rest : rest.substr(0, pipe);
  out.overrides = pipe == std::string::npos ? "" : rest.substr(pipe + 1);
  if (out.variant.empty()) out.variant = out.label;
  return out;
}

std::vector<ValidationOpponent> validation_opponents(const Args& args) {
  std::vector<ValidationOpponent> out;
  for (const std::string& raw : args.validation_opponents) out.push_back(parse_validation_opponent(raw));
  if (out.empty()) out.push_back({"baseline", "baseline", ""});
  return out;
}

fs::path repo_path(const fs::path& path) {
  return path.is_absolute() ? path : fs::current_path() / path;
}

std::string quote_arg(const std::string& raw) {
  if (raw.empty()) return "\"\"";
  const bool needs_quotes = raw.find_first_of(" \t\n\v\"") != std::string::npos;
  if (!needs_quotes) return raw;
  std::string out = "\"";
  size_t backslashes = 0;
  for (char ch : raw) {
    if (ch == '\\') {
      ++backslashes;
    } else if (ch == '"') {
      out.append(backslashes * 2 + 1, '\\');
      out.push_back('"');
      backslashes = 0;
    } else {
      out.append(backslashes, '\\');
      backslashes = 0;
      out.push_back(ch);
    }
  }
  out.append(backslashes * 2, '\\');
  out.push_back('"');
  return out;
}

std::string join_command(const std::vector<std::string>& args) {
  std::string command;
  for (const std::string& arg : args) {
    if (!command.empty()) command.push_back(' ');
    command += quote_arg(arg);
  }
  return command;
}

int run_command(const std::vector<std::string>& args) {
  const std::string command = join_command(args);
  std::cout << command << "\n";
  std::cout.flush();
  return std::system(command.c_str());
}

std::string java_executable(const Args& args) {
  if (!args.java_exe.empty()) return repo_path(args.java_exe).string();
  const char* java_home = std::getenv("JAVA_HOME");
  if (java_home) {
    fs::path candidate = fs::path(java_home) / "bin" / "java.exe";
    if (fs::exists(candidate)) return candidate.string();
  }
  return "java";
}

fs::path json_jar() {
  const char* root = std::getenv("TRIBES_GAME_ROOT");
  if (root) {
    fs::path candidate = fs::path(root) / "lib" / "json.jar";
    if (fs::exists(candidate)) return candidate;
  }
  fs::path sibling = fs::current_path().parent_path() / "TribesTopia" / "Tribes" / "lib" / "json.jar";
  if (fs::exists(sibling)) return sibling;
  return fs::current_path() / "lib" / "json.jar";
}

json tournament_base(const Args& args) {
  if (!args.tournament_config.empty()) return read_json(repo_path(args.tournament_config));
  return json::object();
}

std::vector<int> selected_seeds(const std::vector<int>& configured, int count, int fallback_start, int offset = 0) {
  if (count <= 0) return {};
  if (!configured.empty() && offset < static_cast<int>(configured.size())) {
    const int available = static_cast<int>(configured.size()) - offset;
    if (available >= count) return std::vector<int>(configured.begin() + offset, configured.begin() + offset + count);
  }
  std::vector<int> seeds;
  for (int i = 0; i < count; ++i) seeds.push_back(fallback_start + offset + i);
  return seeds;
}

std::map<std::string, std::string> parse_override_map(const std::string& overrides) {
  std::map<std::string, std::string> out;
  std::stringstream ss(overrides);
  std::string item;
  while (std::getline(ss, item, ',')) {
    const size_t pos = item.find('=');
    if (pos == std::string::npos || pos == 0) continue;
    out[item.substr(0, pos)] = item.substr(pos + 1);
  }
  return out;
}

std::string override_map_string(const std::map<std::string, std::string>& values) {
  std::string out;
  for (const auto& [key, value] : values) {
    if (!out.empty()) out.push_back(',');
    out += key;
    out.push_back('=');
    out += value;
  }
  return out;
}

std::string merge_overrides(const std::string& base, const std::string& updates) {
  std::map<std::string, std::string> merged = parse_override_map(base);
  std::map<std::string, std::string> changed = parse_override_map(updates);
  for (const auto& [key, value] : changed) merged[key] = value;
  return override_map_string(merged);
}

std::vector<std::string> proxy_command(
    const Args& args,
    const fs::path& record_dir,
    const std::string& branch,
    const std::string& overrides,
    int seed,
    bool deterministic,
    double sample_rate) {
  std::vector<std::string> cmd{
      repo_path(args.recording_proxy_exe).string(),
      "--record-dir", record_dir.string(),
      "--branch-name", branch,
      "--record-seed", std::to_string(seed),
      "--record-sample-rate", std::to_string(sample_rate),
      "--static-exe", repo_path(args.static_exe).string(),
      "--search-mode", args.search_mode,
      "--simulations", std::to_string(args.simulations),
      "--top-k-actions", std::to_string(args.top_k_actions),
      "--max-actions", std::to_string(args.max_actions),
      "--search-batch-size", std::to_string(args.batch_size),
      "--static-eval-variant", args.static_eval_variant,
      "--seed", std::to_string(seed),
  };
  if (!overrides.empty()) {
    cmd.push_back("--static-eval-weight-overrides");
    cmd.push_back(overrides);
  }
  if (deterministic) cmd.push_back("--deterministic");
  return cmd;
}

fs::path write_tournament_config(
    const Args& args,
    const fs::path& output_dir,
    const std::string& label,
    const std::vector<int>& seeds,
    const std::string& left_name,
    const std::string& left_tribe,
    const std::vector<std::string>& left_command,
    const std::string& right_name,
    const std::string& right_tribe,
    const std::vector<std::string>& right_command) {
  json base = tournament_base(args);
  json cfg = base;
  cfg["Game Mode"] = base.value("Game Mode", args.game_mode);
  cfg["Map Type"] = base.value("Map Type", args.map_type);
  cfg["Map Size"] = base.value("Map Size", args.map_size);
  cfg["Level Seeds"] = seeds;
  cfg["Game Count"] = seeds.size();
  cfg["Parallel Games"] = args.parallel_games;
  cfg["Max Actions Per Turn"] = args.max_actions_per_turn;
  cfg["Balance Seats"] = args.balance_seats;
  cfg["Concise"] = true;
  cfg["Write Match Logs"] = args.write_match_logs;
  cfg["External Log Dir"] = (output_dir / ("external_" + label)).string();
  cfg["Tournament Summary Log Path"] = (output_dir / ("summary_" + label + ".log")).string();
  cfg["Stats Report Path"] = (output_dir / ("stats_" + label)).string();
  cfg["Participants"] = json::array({
      {
          {"Type", "External"},
          {"Name", left_name},
          {"Tribe", left_tribe},
          {"External Action Timeout Ms", args.external_action_timeout_ms},
          {"External Command", left_command},
      },
      {
          {"Type", "External"},
          {"Name", right_name},
          {"Tribe", right_tribe},
          {"External Action Timeout Ms", args.external_action_timeout_ms},
          {"External Command", right_command},
      },
  });
  fs::path path = output_dir / ("tournament_" + label + ".json");
  write_text(path, cfg.dump(2));
  return path;
}

void run_tournament(const Args& args, const fs::path& config_path) {
  std::vector<std::string> command{
      java_executable(args),
      "-cp",
      (fs::current_path() / "out").string() + ";" + json_jar().string(),
      "Tournament",
      config_path.string(),
  };
  const int code = run_command(command);
  if (code != 0) throw std::runtime_error("Tournament failed with code " + std::to_string(code));
}

fs::path run_collection(const Args& args, const fs::path& output_dir, int seed_offset = 0, int collect_games_override = -1, const std::string* override_population = nullptr) {
  const int collect_games = collect_games_override >= 0 ? collect_games_override : args.collect_games;
  if (collect_games <= 0) return {};
  fs::path dir = output_dir / "collection" / "tournament";
  fs::create_directories(dir);
  std::vector<std::string> population = args.population_overrides.empty() ? std::vector<std::string>{""} : args.population_overrides;
  if (override_population != nullptr) population = {*override_population};
  const std::string left_overrides = population[0];
  const std::string right_overrides = population.size() > 1 ? population[1] : population[0];
  std::vector<int> seeds = selected_seeds(args.level_seed, collect_games, args.collect_seed_start, seed_offset);
  auto left = proxy_command(args, dir, "population0", left_overrides, 0, args.collect_deterministic, args.collect_sample_rate);
  auto right = proxy_command(args, dir, "population1", right_overrides, 1000, args.collect_deterministic, args.collect_sample_rate);
  fs::path config = write_tournament_config(
      args, dir, "collection", seeds, "Population0", args.tribes.empty() ? "Xin Xi" : args.tribes[0], left,
      "Population1", args.tribes.size() > 1 ? args.tribes[1] : "Imperius", right);
  run_tournament(args, config);
  return output_dir / "collection";
}

json run_native_fit(const Args& args, const fs::path& output_dir, const std::vector<fs::path>& records) {
  json cfg;
  cfg["records"] = json::array();
  for (const fs::path& record : records) cfg["records"].push_back(record.string());
  cfg["whitelist"] = args.whitelist;
  cfg["static_eval_variant"] = args.static_eval_variant;
  cfg["initial_overrides"] = args.initial_overrides;
  cfg["max_actions"] = args.max_actions;
  cfg["steps"] = args.steps;
  cfg["learning_rate"] = args.learning_rate;
  cfg["l2"] = args.l2;
  cfg["l1"] = args.l1;
  cfg["max_abs_delta"] = args.max_abs_delta;
  cfg["max_relative_delta"] = args.max_relative_delta;
  cfg["preserve_sign"] = args.preserve_sign;
  cfg["validation_fraction"] = args.validation_fraction;
  cfg["seed"] = args.seed;
  cfg["feature_threads"] = args.feature_threads;
  cfg["feature_scales"] = args.feature_scales;
  cfg["output_dir"] = output_dir.parent_path().string();
  cfg["run_id"] = output_dir.filename().string();
  fs::path config_path = output_dir / "native_fit_config.json";
  write_text(config_path, cfg.dump(2));
  const int code = run_command({repo_path(args.native_tuner_exe).string(), "--config", config_path.string()});
  if (code != 0) throw std::runtime_error("native fitter failed with code " + std::to_string(code));
  return read_json(output_dir / "candidate_weights.json");
}

json parse_tournament_score(const fs::path& summary_log, const std::string& player) {
  json result{{"summary_log", summary_log.string()}};
  if (!fs::exists(summary_log)) return result;
  const std::string text = read_text(summary_log);
  std::regex line_re(R"(\[N:(\d+)\];\[%:([0-9.]+)\];\[W:(\d+)\].*\[Player:([^\]]+)\])");
  std::smatch match;
  auto begin = std::sregex_iterator(text.begin(), text.end(), line_re);
  auto end = std::sregex_iterator();
  for (auto it = begin; it != end; ++it) {
    match = *it;
    if (match[4].str() == player) {
      result["games"] = std::stoi(match[1].str());
      result["score_percent"] = std::stod(match[2].str());
      result["wins"] = std::stoi(match[3].str());
      result["score_rate"] = std::stod(match[2].str()) / 100.0;
    }
  }
  return result;
}

json run_validation(const Args& args, const fs::path& output_dir, const std::string& candidate_overrides) {
  if (args.validation_games <= 0) return json{{"enabled", false}};
  Args validation_args = args;
  validation_args.balance_seats = args.validation_balance_seats;
  std::vector<int> seeds = !args.validation_level_seed.empty()
                               ? selected_seeds(args.validation_level_seed, args.validation_games, args.validation_seed_start)
                               : selected_seeds({}, args.validation_games, args.validation_seed_start);
  json result{{"enabled", true}, {"balance_seats", validation_args.balance_seats}, {"seed_count", args.validation_games}, {"seeds", seeds}};
  json opponent_scores = json::object();
  bool primary_set = false;
  for (const ValidationOpponent& opponent : validation_opponents(args)) {
    Args candidate_args = validation_args;
    candidate_args.static_eval_variant = args.static_eval_variant;
    Args opponent_args = validation_args;
    opponent_args.static_eval_variant = opponent.variant;
    const std::string label = safe_label(opponent.label);
    fs::path dir = output_dir / "validation" / label;
    fs::create_directories(dir);
    auto candidate = proxy_command(candidate_args, dir, "candidate", candidate_overrides, 0, args.validation_deterministic, 0.25);
    auto baseline = proxy_command(opponent_args, dir, label, opponent.overrides, 1000, args.validation_deterministic, 0.25);
    fs::path config = write_tournament_config(
        validation_args, dir, label, seeds, "Candidate", args.tribes.empty() ? "Xin Xi" : args.tribes[0], candidate,
        label, args.tribes.size() > 1 ? args.tribes[1] : "Imperius", baseline);
    run_tournament(validation_args, config);
    json score = parse_tournament_score(dir / ("summary_" + label + ".log"), "Candidate");
    const double score_rate = score.value("score_rate", 0.0);
    score["enabled"] = true;
    score["accepted"] = score_rate >= 0.5;
    score["opponent"] = label;
    score["opponent_variant"] = opponent.variant;
    score["balance_seats"] = validation_args.balance_seats;
    score["seed_count"] = args.validation_games;
    score["seeds"] = seeds;
    opponent_scores[label] = score;
    if (!primary_set) {
      result["accepted"] = score["accepted"];
      result["score_rate"] = score.value("score_rate", 0.0);
      result["score_percent"] = score.value("score_percent", 0.0);
      result["wins"] = score.value("wins", 0);
      result["games"] = score.value("games", 0);
      result["summary_log"] = score.value("summary_log", "");
      result["primary_opponent"] = label;
      primary_set = true;
    }
  }
  result["opponents"] = opponent_scores;
  if (!primary_set) result["accepted"] = true;
  return result;
}

json summary_for_result(const Args& args, const json& result, const json& validation, const fs::path& collection, const std::string& candidate_overrides) {
  return json{
      {"accepted", result.value("accepted", true)},
      {"records", result.value("records", 0)},
      {"examples", result.value("examples", 0)},
      {"collection_dir", collection.empty() ? "" : collection.string()},
      {"changed_override_env", candidate_overrides},
      {"feature_stats", result.value("feature_stats", json::array())},
      {"train_metrics", result.value("train_metrics", json::object())},
      {"validation_metrics", result.value("validation_metrics", json::object())},
      {"validation", validation},
      {"timings", result.value("timings", json::object())},
      {"native_tuner_exe", repo_path(args.native_tuner_exe).string()},
  };
}

json run_training_loop(const Args& args, const fs::path& output_dir) {
  const int iterations = std::max(1, args.loop_iterations);
  const int games_per_iteration = args.loop_games > 0 ? args.loop_games : (args.collect_games > 0 ? args.collect_games : 100);
  std::string current_overrides = args.initial_overrides;
  json iterations_json = json::array();
  int start_iteration = 0;

  if (args.resume_loop || args.resume_from_iteration > 0) {
    iterations_json = read_json_any(output_dir / "iterations.json", json::array());
    if (!iterations_json.is_array()) iterations_json = json::array();
    if (args.resume_from_iteration > 0) {
      start_iteration = std::max(0, std::min(iterations, args.resume_from_iteration - 1));
      if (static_cast<int>(iterations_json.size()) > start_iteration) {
        json trimmed = json::array();
        for (int i = 0; i < start_iteration; ++i) trimmed.push_back(iterations_json[static_cast<size_t>(i)]);
        iterations_json = std::move(trimmed);
      }
    } else {
      start_iteration = std::min(iterations, static_cast<int>(iterations_json.size()));
    }
    if (start_iteration > 0) {
      const json& last = iterations_json[static_cast<size_t>(start_iteration - 1)];
      current_overrides = as_string(last.value("merged_override_env", ""), current_overrides);
      if (current_overrides.empty()) {
        current_overrides = read_text(output_dir / "current_override_env");
        while (!current_overrides.empty() && (current_overrides.back() == '\n' || current_overrides.back() == '\r')) {
          current_overrides.pop_back();
        }
      }
    }
    for (int pending = start_iteration; pending < iterations; ++pending) {
      fs::path pending_dir = output_dir / ("iter_" + std::to_string(pending + 1));
      if (fs::exists(pending_dir)) fs::remove_all(pending_dir);
    }
    write_text(output_dir / "iterations.json", iterations_json.dump(2));
    write_text(output_dir / "current_override_env", current_overrides + "\n");
  }

  int next_seed_offset = 0;
  for (const json& item : iterations_json) {
    next_seed_offset += as_int(item.value("collect_games", 0), 0);
  }
  for (int iteration = start_iteration; iteration < iterations; ++iteration) {
    const std::string label = "iter_" + std::to_string(iteration + 1);
    fs::path iteration_dir = output_dir / label;
    fs::create_directories(iteration_dir);

    Args iter_args = args;
    iter_args.initial_overrides = current_overrides;
    iter_args.collect_games = games_per_iteration;
    iter_args.seed = args.seed + iteration;
    const int seed_offset = next_seed_offset;
    next_seed_offset += games_per_iteration;

    fs::path collection = run_collection(iter_args, iteration_dir, seed_offset, games_per_iteration, &current_overrides);
    std::vector<fs::path> records{collection};
    json result = run_native_fit(iter_args, iteration_dir, records);
    const std::string changed_overrides = result.value("changed_override_env", result.value("override_env", ""));
    const std::string next_overrides = merge_overrides(current_overrides, changed_overrides);
    json validation = run_validation(iter_args, iteration_dir, next_overrides);

    result["validation"] = validation;
    result["accepted"] = !validation.value("enabled", false) || validation.value("accepted", false);
    result["collection_dir"] = collection.string();
    result["whitelist"] = iter_args.whitelist;
    result["input_override_env"] = current_overrides;
    result["merged_override_env"] = next_overrides;
    write_text(iteration_dir / "candidate_weights.json", result.dump(2));
    write_text(iteration_dir / "summary.json", summary_for_result(iter_args, result, validation, collection, next_overrides).dump(2));

    iterations_json.push_back(json{
        {"iteration", iteration + 1},
        {"run_dir", iteration_dir.string()},
        {"seed_offset", seed_offset},
        {"collect_games", games_per_iteration},
        {"input_override_env", current_overrides},
        {"changed_override_env", changed_overrides},
        {"merged_override_env", next_overrides},
        {"train_metrics", result.value("train_metrics", json::object())},
        {"validation_metrics", result.value("validation_metrics", json::object())},
        {"validation", validation},
        {"timings", result.value("timings", json::object())},
    });

    current_overrides = next_overrides;
    write_text(output_dir / "iterations.json", iterations_json.dump(2));
    write_text(output_dir / "current_override_env", current_overrides + "\n");
  }

  json summary{
      {"mode", "selfplay_loop"},
      {"iterations", iterations_json},
      {"loop_iterations", iterations},
      {"loop_games", games_per_iteration},
      {"final_override_env", current_overrides},
      {"whitelist", args.whitelist},
      {"static_eval_variant", args.static_eval_variant},
  };
  write_text(output_dir / "summary.json", summary.dump(2));
  return summary;
}

void apply_config(Args& args, const json& cfg) {
  if (!cfg.is_object()) return;
  if (cfg.contains("records")) {
    args.records.clear();
    for (const json& item : cfg["records"]) args.records.emplace_back(as_string(item));
  }
  if (cfg.contains("whitelist")) args.whitelist = string_array(cfg["whitelist"]);
  if (cfg.contains("level_seed")) args.level_seed = int_array(cfg["level_seed"]);
  if (cfg.contains("validation_level_seed")) args.validation_level_seed = int_array(cfg["validation_level_seed"]);
  if (cfg.contains("population_overrides")) args.population_overrides = string_array(cfg["population_overrides"]);
  if (cfg.contains("validation_opponents")) {
    args.validation_opponents.clear();
    if (cfg["validation_opponents"].is_array()) {
      for (const json& item : cfg["validation_opponents"]) {
        if (item.is_object()) {
          const std::string label = as_string(item.value("label", item.value("variant", "opponent")));
          const std::string variant = as_string(item.value("variant", label));
          const std::string overrides = as_string(item.value("overrides", ""));
          args.validation_opponents.push_back(label + "=" + variant + (overrides.empty() ? "" : "|" + overrides));
        } else {
          args.validation_opponents.push_back(as_string(item));
        }
      }
    }
  }
  if (cfg.contains("tribes")) args.tribes = string_array(cfg["tribes"]);
  if (cfg.contains("output_dir")) args.output_dir = as_string(cfg["output_dir"]);
  if (cfg.contains("run_id")) args.run_id = as_string(cfg["run_id"]);
  if (cfg.contains("recording_proxy_exe")) args.recording_proxy_exe = as_string(cfg["recording_proxy_exe"]);
  if (cfg.contains("native_tuner_exe")) args.native_tuner_exe = as_string(cfg["native_tuner_exe"]);
  if (cfg.contains("static_exe")) args.static_exe = as_string(cfg["static_exe"]);
  if (cfg.contains("java_exe") && !cfg["java_exe"].is_null()) args.java_exe = as_string(cfg["java_exe"]);
  if (cfg.contains("tournament_config") && !cfg["tournament_config"].is_null()) args.tournament_config = as_string(cfg["tournament_config"]);
  if (cfg.contains("static_eval_variant")) args.static_eval_variant = as_string(cfg["static_eval_variant"]);
  if (cfg.contains("initial_overrides")) args.initial_overrides = as_string(cfg["initial_overrides"]);
  if (cfg.contains("search_mode")) args.search_mode = as_string(cfg["search_mode"]);
  if (cfg.contains("run_mode")) args.run_mode = as_string(cfg["run_mode"]);
  if (cfg.contains("game_mode")) args.game_mode = as_string(cfg["game_mode"]);
  if (cfg.contains("map_type")) args.map_type = as_string(cfg["map_type"]);
  if (cfg.contains("map_size")) args.map_size = as_string(cfg["map_size"]);
  if (cfg.contains("max_actions")) args.max_actions = as_int(cfg["max_actions"], args.max_actions);
  if (cfg.contains("steps")) args.steps = as_int(cfg["steps"], args.steps);
  if (cfg.contains("seed")) args.seed = as_int(cfg["seed"], args.seed);
  if (cfg.contains("validation_games")) args.validation_games = as_int(cfg["validation_games"], args.validation_games);
  if (cfg.contains("validation_seed_start")) args.validation_seed_start = as_int(cfg["validation_seed_start"], args.validation_seed_start);
  if (cfg.contains("collect_games")) args.collect_games = as_int(cfg["collect_games"], args.collect_games);
  if (cfg.contains("collect_seed_start")) args.collect_seed_start = as_int(cfg["collect_seed_start"], args.collect_seed_start);
  if (cfg.contains("loop_iterations")) args.loop_iterations = as_int(cfg["loop_iterations"], args.loop_iterations);
  if (cfg.contains("loop_games")) args.loop_games = as_int(cfg["loop_games"], args.loop_games);
  if (cfg.contains("parallel_games")) args.parallel_games = as_int(cfg["parallel_games"], args.parallel_games);
  if (cfg.contains("simulations")) args.simulations = as_int(cfg["simulations"], args.simulations);
  if (cfg.contains("batch_size")) args.batch_size = as_int(cfg["batch_size"], args.batch_size);
  if (cfg.contains("top_k_actions")) args.top_k_actions = as_int(cfg["top_k_actions"], args.top_k_actions);
  if (cfg.contains("max_turns_capitals")) args.max_turns_capitals = as_int(cfg["max_turns_capitals"], args.max_turns_capitals);
  if (cfg.contains("max_actions_per_turn")) args.max_actions_per_turn = as_int(cfg["max_actions_per_turn"], args.max_actions_per_turn);
  if (cfg.contains("max_actions_per_game")) args.max_actions_per_game = as_int(cfg["max_actions_per_game"], args.max_actions_per_game);
  if (cfg.contains("external_action_timeout_ms")) args.external_action_timeout_ms = as_int(cfg["external_action_timeout_ms"], args.external_action_timeout_ms);
  if (cfg.contains("feature_threads")) args.feature_threads = as_int(cfg["feature_threads"], args.feature_threads);
  if (cfg.contains("feature_scales") && cfg["feature_scales"].is_object()) {
    for (auto it = cfg["feature_scales"].begin(); it != cfg["feature_scales"].end(); ++it) {
      args.feature_scales[it.key()] = as_double(it.value(), 1.0);
    }
  }
  if (cfg.contains("learning_rate")) args.learning_rate = as_double(cfg["learning_rate"], args.learning_rate);
  if (cfg.contains("l2")) args.l2 = as_double(cfg["l2"], args.l2);
  if (cfg.contains("l1")) args.l1 = as_double(cfg["l1"], args.l1);
  if (cfg.contains("max_abs_delta")) args.max_abs_delta = as_double(cfg["max_abs_delta"], args.max_abs_delta);
  if (cfg.contains("max_relative_delta")) args.max_relative_delta = as_double(cfg["max_relative_delta"], args.max_relative_delta);
  if (cfg.contains("validation_fraction")) args.validation_fraction = as_double(cfg["validation_fraction"], args.validation_fraction);
  if (cfg.contains("collect_sample_rate")) args.collect_sample_rate = as_double(cfg["collect_sample_rate"], args.collect_sample_rate);
  if (cfg.contains("preserve_sign")) args.preserve_sign = as_bool(cfg["preserve_sign"], args.preserve_sign);
  if (cfg.contains("balance_seats")) args.balance_seats = as_bool(cfg["balance_seats"], args.balance_seats);
  if (cfg.contains("validation_balance_seats")) args.validation_balance_seats = as_bool(cfg["validation_balance_seats"], args.validation_balance_seats);
  if (cfg.contains("write_match_logs")) args.write_match_logs = as_bool(cfg["write_match_logs"], args.write_match_logs);
  if (cfg.contains("collect_deterministic")) args.collect_deterministic = as_bool(cfg["collect_deterministic"], args.collect_deterministic);
  if (cfg.contains("validation_deterministic")) args.validation_deterministic = as_bool(cfg["validation_deterministic"], args.validation_deterministic);
  if (cfg.contains("resume_from_iteration")) args.resume_from_iteration = as_int(cfg["resume_from_iteration"], args.resume_from_iteration);
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
      return argv[++i];
    };
    if (arg == "--config") {
      args.config_path = next();
      apply_config(args, read_json(repo_path(args.config_path)));
    } else if (arg == "--records") {
      args.records.emplace_back(next());
    } else if (arg == "--collect-games") {
      args.collect_games = std::stoi(next());
    } else if (arg == "--loop-iterations") {
      args.loop_iterations = std::stoi(next());
    } else if (arg == "--loop-games") {
      args.loop_games = std::stoi(next());
    } else if (arg == "--validation-games") {
      args.validation_games = std::stoi(next());
    } else if (arg == "--validation-seed-start") {
      args.validation_seed_start = std::stoi(next());
    } else if (arg == "--run-id") {
      args.run_id = next();
    } else if (arg == "--output-dir") {
      args.output_dir = next();
    } else if (arg == "--balance-seats") {
      args.balance_seats = true;
    } else if (arg == "--no-balance-seats") {
      args.balance_seats = false;
    } else if (arg == "--validation-balance-seats") {
      args.validation_balance_seats = true;
    } else if (arg == "--no-validation-balance-seats") {
      args.validation_balance_seats = false;
    } else if (arg == "--collect-deterministic") {
      args.collect_deterministic = true;
    } else if (arg == "--validation-deterministic") {
      args.validation_deterministic = true;
    } else if (arg == "--resume-loop") {
      args.resume_loop = true;
    } else if (arg == "--resume-from-iteration") {
      args.resume_from_iteration = std::stoi(next());
    } else if (arg == "--simulations") {
      args.simulations = std::stoi(next());
    } else if (arg == "--parallel-games") {
      args.parallel_games = std::stoi(next());
    } else if (arg == "--external-action-timeout-ms") {
      args.external_action_timeout_ms = std::stoi(next());
    } else if (arg == "--max-actions-per-turn") {
      args.max_actions_per_turn = std::stoi(next());
    } else if (arg == "--static-eval-variant") {
      args.static_eval_variant = next();
    } else if (arg == "--validation-opponent") {
      args.validation_opponents.push_back(next());
    } else if (arg == "--feature-scale") {
      add_feature_scale(args, next());
    } else if (arg == "--whitelist") {
      args.whitelist.push_back(next());
    } else if (arg == "--max-abs-delta") {
      args.max_abs_delta = std::stod(next());
    } else if (arg == "--max-relative-delta") {
      args.max_relative_delta = std::stod(next());
    } else if (arg == "--preserve-sign") {
      args.preserve_sign = true;
    } else if (arg == "--no-preserve-sign") {
      args.preserve_sign = false;
    } else if (arg == "--learning-rate") {
      args.learning_rate = std::stod(next());
    } else if (arg == "--l2") {
      args.l2 = std::stod(next());
    } else if (arg == "--steps") {
      args.steps = std::stoi(next());
    }
  }
  if (args.run_id.empty()) args.run_id = "native-driver-" + std::to_string(std::time(nullptr));
  if (args.whitelist.empty()) throw std::runtime_error("no whitelist terms configured");
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);
    fs::path output_dir = repo_path(args.output_dir) / args.run_id;
    fs::create_directories(output_dir);

    if (args.loop_iterations > 0) {
      run_training_loop(args, output_dir);
      std::cout << "wrote " << output_dir.string() << "\n";
      return 0;
    }

    std::vector<fs::path> records = args.records;
    fs::path collection = run_collection(args, output_dir);
    if (!collection.empty()) records.push_back(collection);
    if (records.empty()) throw std::runtime_error("no result records found; pass --records or set collect_games");

    json result = run_native_fit(args, output_dir, records);
    const std::string candidate_overrides = result.value("changed_override_env", result.value("override_env", ""));
    json validation = run_validation(args, output_dir, candidate_overrides);
    result["validation"] = validation;
    result["accepted"] = !validation.value("enabled", false) || validation.value("accepted", false);
    result["collection_dir"] = collection.empty() ? "" : collection.string();
    result["whitelist"] = args.whitelist;
    write_text(output_dir / "candidate_weights.json", result.dump(2));

    write_text(output_dir / "summary.json", summary_for_result(args, result, validation, collection, candidate_overrides).dump(2));
    std::cout << "wrote " << output_dir.string() << "\n";
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
}
