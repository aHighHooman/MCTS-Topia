#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../py/search/native/third_party/nlohmann/json.hpp"

#ifndef TRIBES_NATIVE_MCTS_STANDALONE
#define TRIBES_NATIVE_MCTS_STANDALONE
#endif
#include "../py/search/native/src/rules.hpp"
#include "../py/search/native/src/static_eval.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

constexpr double kValueScale = 200.0;

struct Args {
  std::vector<fs::path> records;
  std::vector<std::string> whitelist;
  std::string static_eval_variant = "baseline";
  std::string initial_overrides;
  int max_actions = 512;
  int steps = 2000;
  double learning_rate = 0.05;
  double l2 = 0.01;
  double l1 = 0.0;
  double max_abs_delta = 0.75;
  double max_relative_delta = 0.5;
  bool preserve_sign = true;
  int feature_threads = 0;
  double validation_fraction = 0.2;
  int seed = 0;
  std::unordered_map<std::string, double> feature_scales;
  fs::path output_dir = "debug-logs/analysis/static-value-result-tuning";
  std::string run_id;
};

using FeatureTuple = std::tuple<double, std::unordered_map<std::string, double>, std::unordered_map<std::string, double>>;

struct Example {
  double target = 0.0;
  double fixed_raw = 0.0;
  std::unordered_map<std::string, double> features;
  std::unordered_map<std::string, double> initial_weights;
  std::string game_key;
  double sample_weight = 1.0;
};

struct Metrics {
  double count = 0.0;
  double mse = 0.0;
  double mae = 0.0;
  double sign_accuracy = 0.0;
};

struct DenseExample {
  double target = 0.0;
  double fixed_raw = 0.0;
  double sample_weight = 1.0;
  std::vector<double> features;
};

struct DenseSplit {
  std::vector<DenseExample> train;
  std::vector<DenseExample> val;
  std::vector<std::string> terms;
  std::vector<double> initial;
  std::vector<double> actual_initial;
  std::vector<double> feature_scales;
  std::vector<std::pair<double, double>> bounds;
  double train_weight_sum = 1.0;
};

double as_double(const json& value, double fallback = 0.0) {
  try {
    if (value.is_number()) return value.get<double>();
    if (value.is_string()) return std::stod(value.get<std::string>());
  } catch (...) {
  }
  return fallback;
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

std::string as_string(const json& value, const std::string& fallback = "") {
  if (value.is_string()) return value.get<std::string>();
  if (value.is_null()) return fallback;
  return value.dump();
}

bool as_bool(const json& value, bool fallback = true) {
  if (value.is_boolean()) return value.get<bool>();
  if (value.is_number()) return value.get<double>() != 0.0;
  if (value.is_string()) {
    std::string s = value.get<std::string>();
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return !(s.empty() || s == "0" || s == "false" || s == "no" || s == "off");
  }
  return fallback;
}

double feature_scale_for(const Args& args, const std::string& term) {
  auto found = args.feature_scales.find(term);
  if (found == args.feature_scales.end() || std::abs(found->second) <= 1e-12 || !std::isfinite(found->second)) {
    return 1.0;
  }
  return found->second;
}

std::string read_text_file(const fs::path& path, const std::string& label) {
  std::string text;
#ifdef _WIN32
  std::wstring wide = fs::absolute(path).wstring();
  if (wide.rfind(LR"(\\?\)", 0) != 0) {
    if (wide.rfind(LR"(\\)", 0) == 0) {
      wide = LR"(\\?\UNC\)" + wide.substr(2);
    } else {
      wide = LR"(\\?\)" + wide;
    }
  }
  FILE* file = _wfopen(wide.c_str(), L"rb");
  if (!file) throw std::runtime_error("failed to open " + label + ": " + path.string());
  char buffer[1 << 15];
  while (true) {
    const size_t read = std::fread(buffer, 1, sizeof(buffer), file);
    if (read > 0) text.append(buffer, read);
    if (read < sizeof(buffer)) {
      if (std::ferror(file)) {
        std::fclose(file);
        throw std::runtime_error("failed to read " + label + ": " + path.string());
      }
      break;
    }
  }
  std::fclose(file);
#else
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("failed to open " + label + ": " + path.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  text = ss.str();
#endif
  return text;
}

json read_json_file(const fs::path& path) {
  const std::string text = read_text_file(path, "JSON file");
  json value = json::parse(text, nullptr, false);
  if (value.is_discarded()) throw std::runtime_error("failed to parse JSON file: " + path.string());
  return value;
}

void set_env_var(const std::string& name, const std::string& value) {
#ifdef _WIN32
  _putenv_s(name.c_str(), value.c_str());
#else
  setenv(name.c_str(), value.c_str(), 1);
#endif
}

std::string terminal_winner_string(const json& game_over) {
  for (const std::string key : {"winner_id", "winner"}) {
    if (!game_over.contains(key) || game_over[key].is_null()) continue;
    const std::string value = as_string(game_over[key]);
    std::string upper = value;
    std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
    if (upper.empty() || upper == "NONE" || upper == "DRAW" || upper == "NULL" || upper == "-1") continue;
    return value;
  }
  return "";
}

double terminal_target_for_row(const json& row) {
  const int player_id = as_int(row.value("player_id", 0), 0);
  if (row.contains("game_over") && row["game_over"].is_object()) {
    const json& game_over = row["game_over"];
    const std::string winner = terminal_winner_string(game_over);
    if (!winner.empty()) {
      return as_int(winner, -999999) == player_id ? 1.0 : -1.0;
    }
    const json* ranking = nullptr;
    if (game_over.contains("ranking") && game_over["ranking"].is_array()) ranking = &game_over["ranking"];
    if (game_over.contains("rank") && game_over["rank"].is_array()) ranking = &game_over["rank"];
    if (ranking != nullptr && !ranking->empty()) {
      const int first = as_int((*ranking)[0], -999999);
      if (player_id == first) return 1.0;
      for (const json& item : *ranking) {
        if (as_int(item, -999999) == player_id) return -1.0;
      }
    }
    const json* scores = nullptr;
    if (game_over.contains("final_scores") && game_over["final_scores"].is_array()) scores = &game_over["final_scores"];
    if (game_over.contains("scores") && game_over["scores"].is_array()) scores = &game_over["scores"];
    if (scores != nullptr && !scores->empty()) {
      double best = -1e300;
      double player_score = -1e300;
      bool found = false;
      for (const json& item : *scores) {
        if (!item.is_array() || item.size() < 2) continue;
        const int pid = as_int(item[0], -999999);
        const double score = as_double(item[1], 0.0);
        best = std::max(best, score);
        if (pid == player_id) {
          player_score = score;
          found = true;
        }
      }
      if (found) return player_score == best ? 1.0 : -1.0;
    }
  }
  return std::max(-1.0, std::min(1.0, as_double(row.value("target", 0.0), 0.0)));
}

std::string game_key_for_row(const json& row) {
  for (const std::string key : {"game_id", "game_seed", "level_seed", "seed", "source_file"}) {
    if (row.contains(key) && !row[key].is_null() && as_string(row[key]) != "") {
      return key + ":" + as_string(row[key]);
    }
  }
  return "payload:" + as_string(row.value("payload_hash", ""));
}

bool ends_with(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string replace_once(std::string value, const std::string& needle, const std::string& replacement) {
  const size_t pos = value.find(needle);
  if (pos != std::string::npos) value.replace(pos, needle.size(), replacement);
  return value;
}

std::string ownership_symmetric_group(const std::string& name, const std::string& parent) {
  if (name.find(".own.") != std::string::npos) return replace_once(name, ".own.", ".");
  if (name.find(".enemy.") != std::string::npos) return replace_once(name, ".enemy.", ".");
  if (ends_with(name, ".own") || ends_with(name, ".enemy") || ends_with(name, ".enemy_best")) {
    return parent.empty() ? name : parent;
  }
  if (ends_with(name, ".own_capacity") || ends_with(name, ".enemy_capacity")) {
    return parent.empty() ? name : parent;
  }
  return "";
}

bool is_ownership_grouped(const json& term, const std::string& group) {
  const std::string name = as_string(term.value("name", ""));
  const std::string parent = as_string(term.value("parent", ""));
  return !group.empty() && group != name && !ownership_symmetric_group(name, parent).empty();
}

std::string term_group(const json& term, const std::unordered_set<std::string>& whitelist) {
  const std::string parent = as_string(term.value("parent", ""));
  const std::string name = as_string(term.value("name", ""));
  if (whitelist.count("*")) {
    const std::string grouped = ownership_symmetric_group(name, parent);
    return grouped.empty() ? name : grouped;
  }
  if (!parent.empty() && whitelist.count(parent)) return parent;
  if (whitelist.count(name)) return name;
  return "";
}

double initial_weight_for_group(const json& term, const std::string& group) {
  const std::string parent = as_string(term.value("parent", ""));
  const double weight = as_double(term.value("weight", 0.0), 0.0);
  return parent == group || is_ownership_grouped(term, group) ? std::abs(weight) : weight;
}

FeatureTuple aggregate_features(const json& breakdown, const std::unordered_set<std::string>& whitelist) {
  std::unordered_map<std::string, double> features;
  std::unordered_map<std::string, double> initial;
  double tunable_raw = 0.0;
  if (!breakdown.is_object() || !breakdown.contains("terms") || !breakdown["terms"].is_array()) {
    return {0.0, features, initial};
  }
  for (const json& term : breakdown["terms"]) {
    if (!term.is_object()) continue;
    if (!as_bool(term.value("contributes", true), true) || !as_bool(term.value("tunable", true), true)) continue;
    const std::string group = term_group(term, whitelist);
    if (group.empty()) continue;
    const double feature_value = as_double(term.value("feature_value", 0.0), 0.0);
    const double weight = as_double(term.value("weight", 0.0), 0.0);
    const std::string parent = as_string(term.value("parent", ""));
    const double basis = parent == group || is_ownership_grouped(term, group) ? std::copysign(feature_value, weight) : feature_value;
    features[group] += basis;
    if (!initial.count(group)) initial[group] = initial_weight_for_group(term, group);
    tunable_raw += as_double(term.value("raw", 0.0), 0.0);
  }
  const double fixed_raw = as_double(breakdown.value("raw_total", 0.0), 0.0) - tunable_raw;
  return {fixed_raw, features, initial};
}

std::vector<json> load_record_rows(const std::vector<fs::path>& paths) {
  std::vector<json> rows;
  for (const fs::path& raw : paths) {
    fs::path path = raw;
    if (!path.is_absolute()) path = fs::current_path() / path;
    if (!fs::exists(path)) continue;
    std::vector<fs::path> files;
    if (fs::is_directory(path)) {
      for (const auto& entry : fs::recursive_directory_iterator(path)) {
        if (entry.is_regular_file()) {
          const std::string name = entry.path().filename().string();
          if (name.rfind("records_", 0) == 0 && entry.path().extension() == ".jsonl") files.push_back(entry.path());
        }
      }
      std::sort(files.begin(), files.end());
    } else {
      files.push_back(path);
    }
    for (const fs::path& file : files) {
      std::istringstream in(read_text_file(file, "record file"));
      std::string line;
      while (std::getline(in, line)) {
        if (line.empty()) continue;
        json row = json::parse(line, nullptr, false);
        if (row.is_object()) {
          row["source_file"] = file.string();
          rows.push_back(std::move(row));
        }
      }
    }
  }
  return rows;
}

std::vector<Example> build_examples(const std::vector<json>& rows, const Args& args) {
  std::unordered_set<std::string> whitelist(args.whitelist.begin(), args.whitelist.end());
  set_env_var("TRIBES_STATIC_EVAL_VARIANT", args.static_eval_variant);
  set_env_var("TRIBES_STATIC_EVAL_WEIGHT_OVERRIDES", args.initial_overrides);
  std::unordered_map<std::string, FeatureTuple> feature_cache;
  std::vector<std::pair<std::string, std::string>> payloads;
  payloads.reserve(rows.size());
  std::unordered_set<std::string> seen_payloads;
  seen_payloads.reserve(rows.size());
  for (const json& row : rows) {
    if (!row.contains("payload_path")) continue;
    const std::string payload_path = as_string(row["payload_path"]);
    if (payload_path.empty()) continue;
    const std::string payload_key = as_string(row.value("payload_hash", payload_path), payload_path);
    if (seen_payloads.insert(payload_key).second) payloads.emplace_back(payload_key, payload_path);
  }

  std::vector<FeatureTuple> computed(payloads.size());
  std::vector<bool> ok(payloads.size(), false);
  std::atomic<size_t> next_index{0};
  std::atomic<size_t> payload_errors{0};
  std::string first_error;
  std::mutex error_mutex;
  unsigned int hardware = std::thread::hardware_concurrency();
  int thread_count = args.feature_threads > 0 ? args.feature_threads : static_cast<int>(hardware == 0 ? 4 : hardware);
  thread_count = std::max(1, std::min(thread_count, static_cast<int>(payloads.empty() ? 1 : payloads.size())));
  thread_count = std::min(thread_count, 16);

  auto worker = [&]() {
    while (true) {
      const size_t index = next_index.fetch_add(1);
      if (index >= payloads.size()) break;
      try {
        json payload = read_json_file(payloads[index].second);
        py::dict py_payload(payload);
        json out = py::to_json(tribes::native::evaluate_static_breakdown(py_payload, args.max_actions));
        if (!out.is_object() || !out.contains("value_breakdown")) continue;
        FeatureTuple aggregate = aggregate_features(out["value_breakdown"], whitelist);
        if (std::get<1>(aggregate).empty()) continue;
        computed[index] = std::move(aggregate);
        ok[index] = true;
      } catch (const std::exception& exc) {
        payload_errors.fetch_add(1);
        std::lock_guard<std::mutex> lock(error_mutex);
        if (first_error.empty()) first_error = exc.what();
      }
    }
  };
  std::vector<std::thread> workers;
  workers.reserve(thread_count);
  for (int i = 0; i < thread_count; ++i) workers.emplace_back(worker);
  for (std::thread& thread : workers) thread.join();
  if (payload_errors.load() > 0) {
    std::cerr << "warning: skipped " << payload_errors.load() << " malformed/evaluation-failed payloads";
    if (!first_error.empty()) std::cerr << "; first error: " << first_error;
    std::cerr << "\n";
  }
  for (size_t i = 0; i < payloads.size(); ++i) {
    if (ok[i]) feature_cache.emplace(payloads[i].first, std::move(computed[i]));
  }

  std::vector<Example> examples;
  examples.reserve(rows.size());
  for (const json& row : rows) {
    if (!row.contains("payload_path")) continue;
    const std::string payload_path = as_string(row["payload_path"]);
    if (payload_path.empty()) continue;
    const std::string payload_key = as_string(row.value("payload_hash", payload_path), payload_path);
    auto found = feature_cache.find(payload_key);
    if (found == feature_cache.end()) continue;
    const auto& [fixed_raw, features, initial] = found->second;
    if (features.empty()) continue;
    Example example;
    example.target = terminal_target_for_row(row);
    example.fixed_raw = fixed_raw;
    example.features = features;
    example.initial_weights = initial;
    example.game_key = game_key_for_row(row);
    examples.push_back(std::move(example));
  }
  std::unordered_map<std::string, int> counts;
  for (const Example& example : examples) counts[example.game_key] += 1;
  for (Example& example : examples) example.sample_weight = 1.0 / std::max(1, counts[example.game_key]);
  return examples;
}

std::pair<double, double> bounds_for(double initial, const Args& args) {
  const double rel_cap = std::abs(initial) * std::max(0.0, args.max_relative_delta);
  const double cap = std::abs(initial) <= 1e-12 ? args.max_abs_delta : std::min(args.max_abs_delta, rel_cap);
  double lower = initial - cap;
  double upper = initial + cap;
  if (args.preserve_sign) {
    if (initial > 0.0) lower = std::max(0.0, lower);
    else if (initial < 0.0) upper = std::min(0.0, upper);
    else lower = std::max(0.0, lower);
  }
  if (lower > upper) lower = upper = initial;
  return {lower, upper};
}

double predict(const std::unordered_map<std::string, double>& weights, const Example& example) {
  double raw = example.fixed_raw;
  for (const auto& [term, value] : example.features) {
    auto found = weights.find(term);
    if (found != weights.end()) raw += found->second * value;
  }
  return std::tanh(raw / kValueScale);
}

double predict_dense(const std::vector<double>& weights, const DenseExample& example) {
  double raw = example.fixed_raw;
  const size_t n = std::min(weights.size(), example.features.size());
  for (size_t i = 0; i < n; ++i) raw += weights[i] * example.features[i];
  return std::tanh(raw / kValueScale);
}

Metrics metrics_for(const std::unordered_map<std::string, double>& weights, const std::vector<Example>& examples) {
  Metrics m;
  m.count = static_cast<double>(examples.size());
  if (examples.empty()) return m;
  double weight_sum = 0.0;
  for (const Example& example : examples) weight_sum += example.sample_weight;
  if (weight_sum <= 0.0) weight_sum = 1.0;
  double sign_ok = 0.0;
  for (const Example& example : examples) {
    const double pred = predict(weights, example);
    const double err = pred - example.target;
    m.mse += example.sample_weight * err * err;
    m.mae += example.sample_weight * std::abs(err);
    if ((pred >= 0.0 && example.target >= 0.0) || (pred < 0.0 && example.target < 0.0)) {
      sign_ok += example.sample_weight;
    }
  }
  m.mse /= weight_sum;
  m.mae /= weight_sum;
  m.sign_accuracy = sign_ok / weight_sum;
  return m;
}

Metrics metrics_for_dense(const std::vector<double>& weights, const std::vector<DenseExample>& examples) {
  Metrics m;
  m.count = static_cast<double>(examples.size());
  if (examples.empty()) return m;
  double weight_sum = 0.0;
  for (const DenseExample& example : examples) weight_sum += example.sample_weight;
  if (weight_sum <= 0.0) weight_sum = 1.0;
  double sign_ok = 0.0;
  for (const DenseExample& example : examples) {
    const double pred = predict_dense(weights, example);
    const double err = pred - example.target;
    m.mse += example.sample_weight * err * err;
    m.mae += example.sample_weight * std::abs(err);
    if ((pred >= 0.0 && example.target >= 0.0) || (pred < 0.0 && example.target < 0.0)) {
      sign_ok += example.sample_weight;
    }
  }
  m.mse /= weight_sum;
  m.mae /= weight_sum;
  m.sign_accuracy = sign_ok / weight_sum;
  return m;
}

json metrics_json(const Metrics& m) {
  return json{{"count", m.count}, {"mse", m.mse}, {"mae", m.mae}, {"sign_accuracy", m.sign_accuracy}};
}

json feature_stats_json(const DenseSplit& split) {
  json rows = json::array();
  for (size_t i = 0; i < split.terms.size(); ++i) {
    double sum = 0.0;
    double abs_sum = 0.0;
    double max_abs = 0.0;
    size_t nonzero = 0;
    size_t count = 0;
    auto accumulate = [&](const std::vector<DenseExample>& examples) {
      for (const DenseExample& example : examples) {
        if (i >= example.features.size()) continue;
        const double value = example.features[i];
        sum += value;
        abs_sum += std::abs(value);
        max_abs = std::max(max_abs, std::abs(value));
        if (std::abs(value) > 1e-12) ++nonzero;
        ++count;
      }
    };
    accumulate(split.train);
    accumulate(split.val);
    rows.push_back(json{
        {"term", split.terms[i]},
        {"count", count},
        {"nonzero", nonzero},
        {"zero", count >= nonzero ? count - nonzero : 0},
        {"mean", count == 0 ? 0.0 : sum / static_cast<double>(count)},
        {"abs_mean", count == 0 ? 0.0 : abs_sum / static_cast<double>(count)},
        {"max_abs", max_abs},
        {"initial", split.actual_initial[i]},
        {"optimizer_initial", split.initial[i]},
        {"feature_scale", split.feature_scales[i]},
        {"lower_bound", split.bounds[i].first * split.feature_scales[i]},
        {"upper_bound", split.bounds[i].second * split.feature_scales[i]}});
  }
  return rows;
}

std::string override_string(const std::vector<json>& changes) {
  std::ostringstream out;
  bool first = true;
  for (const json& row : changes) {
    if (!first) out << ",";
    first = false;
    out << row["term"].get<std::string>() << "=" << std::setprecision(12) << row["after"].get<double>();
  }
  return out.str();
}

DenseSplit make_dense_split(const std::vector<Example>& examples, const Args& args) {
  std::unordered_map<std::string, double> initial_map;
  for (const Example& example : examples) {
    for (const auto& [term, value] : example.initial_weights) {
      if (!initial_map.count(term)) initial_map[term] = value;
    }
  }

  DenseSplit split;
  split.terms.reserve(initial_map.size());
  for (const std::string& term : args.whitelist) {
    if (initial_map.count(term)) split.terms.push_back(term);
  }
  for (const auto& [term, _] : initial_map) {
    if (std::find(split.terms.begin(), split.terms.end(), term) == split.terms.end()) {
      split.terms.push_back(term);
    }
  }

  std::unordered_map<std::string, size_t> term_index;
  term_index.reserve(split.terms.size());
  for (size_t i = 0; i < split.terms.size(); ++i) {
    term_index[split.terms[i]] = i;
    const double actual_initial = initial_map[split.terms[i]];
    const double scale = feature_scale_for(args, split.terms[i]);
    const double optimizer_initial = actual_initial / scale;
    split.actual_initial.push_back(actual_initial);
    split.feature_scales.push_back(scale);
    split.initial.push_back(optimizer_initial);
    split.bounds.push_back(bounds_for(optimizer_initial, args));
  }

  std::unordered_map<std::string, std::vector<size_t>> by_game;
  for (size_t i = 0; i < examples.size(); ++i) by_game[examples[i].game_key].push_back(i);
  std::vector<std::string> keys;
  keys.reserve(by_game.size());
  for (const auto& [key, _] : by_game) keys.push_back(key);
  std::sort(keys.begin(), keys.end());
  std::mt19937 rng(static_cast<uint32_t>(args.seed));
  std::shuffle(keys.begin(), keys.end(), rng);
  const size_t val_count = static_cast<size_t>(std::llround(keys.size() * std::max(0.0, std::min(0.9, args.validation_fraction))));
  std::unordered_set<std::string> val_keys(keys.begin(), keys.begin() + std::min(val_count, keys.size()));

  split.train.reserve(examples.size());
  split.val.reserve(val_count == 0 ? 0 : examples.size());
  for (const Example& example : examples) {
    DenseExample dense;
    dense.target = example.target;
    dense.fixed_raw = example.fixed_raw;
    dense.sample_weight = example.sample_weight;
    dense.features.assign(split.terms.size(), 0.0);
    for (const auto& [term, value] : example.features) {
      auto found = term_index.find(term);
      if (found != term_index.end()) {
        const size_t index = found->second;
        dense.features[index] = value * split.feature_scales[index];
      }
    }
    (val_keys.count(example.game_key) ? split.val : split.train).push_back(std::move(dense));
  }
  if (split.train.empty() && !split.val.empty()) std::swap(split.train, split.val);
  split.train_weight_sum = 0.0;
  for (const DenseExample& example : split.train) split.train_weight_sum += example.sample_weight;
  if (split.train_weight_sum <= 0.0) split.train_weight_sum = 1.0;
  return split;
}

json optimize(const std::vector<Example>& examples, const Args& args) {
  if (examples.empty()) throw std::runtime_error("no training examples available");
  DenseSplit split = make_dense_split(examples, args);
  std::vector<double> weights = split.initial;

  json history = json::array();
  const int steps = std::max(1, args.steps);
  for (int step = 0; step < steps; ++step) {
    std::vector<double> grad(weights.size(), 0.0);
    for (size_t i = 0; i < weights.size(); ++i) {
      grad[i] = 2.0 * args.l2 * (weights[i] - split.initial[i]);
      if (args.l1 > 0.0) {
        const double delta = weights[i] - split.initial[i];
        if (delta > 0.0) grad[i] += args.l1;
        else if (delta < 0.0) grad[i] -= args.l1;
      }
    }
    for (const DenseExample& example : split.train) {
      double raw = example.fixed_raw;
      for (size_t i = 0; i < weights.size(); ++i) raw += weights[i] * example.features[i];
      const double pred = std::tanh(raw / kValueScale);
      const double scale = 2.0 * example.sample_weight * (pred - example.target) * (1.0 - pred * pred) / kValueScale / split.train_weight_sum;
      for (size_t i = 0; i < weights.size(); ++i) grad[i] += scale * example.features[i];
    }
    for (size_t i = 0; i < weights.size(); ++i) {
      double next = weights[i] - args.learning_rate * grad[i];
      if (!std::isfinite(next)) next = split.initial[i];
      const auto [lower, upper] = split.bounds[i];
      weights[i] = std::min(upper, std::max(lower, next));
    }
    if (step == 0 || step == steps - 1 || (step + 1) % std::max(1, steps / 10) == 0) {
      history.push_back(json{{"step", step + 1}, {"train", metrics_json(metrics_for_dense(weights, split.train))}, {"validation", metrics_json(metrics_for_dense(weights, split.val))}});
    }
  }

  std::unordered_map<std::string, double> weights_map;
  std::unordered_map<std::string, double> initial_map;
  json changes = json::array();
  for (size_t i = 0; i < split.terms.size(); ++i) {
    const std::string& term = split.terms[i];
    const double value = weights[i] * split.feature_scales[i];
    const double initial = split.actual_initial[i];
    const double lower = split.bounds[i].first * split.feature_scales[i];
    const double upper = split.bounds[i].second * split.feature_scales[i];
    weights_map[term] = value;
    initial_map[term] = initial;
    const double delta = value - initial;
    if (std::abs(delta) <= 1e-12) continue;
    changes.push_back(json{
        {"term", term},
        {"before", initial},
        {"after", value},
        {"delta", delta},
        {"feature_scale", split.feature_scales[i]},
        {"optimizer_before", split.initial[i]},
        {"optimizer_after", weights[i]},
        {"lower_bound", lower},
        {"upper_bound", upper}});
  }
  std::sort(changes.begin(), changes.end(), [](const json& a, const json& b) {
    return std::abs(a["delta"].get<double>()) > std::abs(b["delta"].get<double>());
  });
  json out;
  out["weights"] = weights_map;
  out["initial_weights"] = initial_map;
  out["changes"] = changes;
  out["feature_stats"] = feature_stats_json(split);
  out["changed_override_env"] = override_string(changes);
  out["train_metrics"] = metrics_json(metrics_for_dense(weights, split.train));
  out["validation_metrics"] = metrics_json(metrics_for_dense(weights, split.val));
  out["history"] = history;
  out["optimizer"] = {
      {"steps", args.steps},
      {"learning_rate", args.learning_rate},
      {"l2", args.l2},
      {"l1", args.l1},
      {"max_abs_delta", args.max_abs_delta},
      {"max_relative_delta", args.max_relative_delta},
      {"preserve_sign", args.preserve_sign},
      {"feature_threads", args.feature_threads},
      {"validation_fraction", args.validation_fraction},
      {"feature_scales", args.feature_scales},
      {"seed", args.seed}};
  return out;
}

void write_csv(const fs::path& path, const json& rows) {
  std::ofstream out(path);
  out << "term,before,after,delta,lower_bound,upper_bound\n";
  for (const json& row : rows) {
    out << row["term"].get<std::string>() << ","
        << row["before"].get<double>() << ","
        << row["after"].get<double>() << ","
        << row["delta"].get<double>() << ","
        << row["lower_bound"].get<double>() << ","
        << row["upper_bound"].get<double>() << "\n";
  }
}

void write_metrics_csv(const fs::path& path, const json& history) {
  std::ofstream out(path);
  out << "step,split,count,mse,mae,sign_accuracy\n";
  for (const json& item : history) {
    for (const std::string split : {"train", "validation"}) {
      const json& m = item[split];
      out << item["step"].get<int>() << "," << split << ","
          << m["count"].get<double>() << ","
          << m["mse"].get<double>() << ","
          << m["mae"].get<double>() << ","
          << m["sign_accuracy"].get<double>() << "\n";
    }
  }
}

void apply_config(Args& args, const json& cfg) {
  if (!cfg.is_object()) return;
  if (cfg.contains("records")) {
    args.records.clear();
    for (const json& item : cfg["records"]) args.records.emplace_back(as_string(item));
  }
  if (cfg.contains("whitelist")) {
    args.whitelist.clear();
    for (const json& item : cfg["whitelist"]) args.whitelist.push_back(as_string(item));
  }
  if (cfg.contains("static_eval_variant")) args.static_eval_variant = as_string(cfg["static_eval_variant"]);
  if (cfg.contains("initial_overrides")) args.initial_overrides = as_string(cfg["initial_overrides"]);
  if (cfg.contains("max_actions")) args.max_actions = as_int(cfg["max_actions"], args.max_actions);
  if (cfg.contains("steps")) args.steps = as_int(cfg["steps"], args.steps);
  if (cfg.contains("learning_rate")) args.learning_rate = as_double(cfg["learning_rate"], args.learning_rate);
  if (cfg.contains("l2")) args.l2 = as_double(cfg["l2"], args.l2);
  if (cfg.contains("l1")) args.l1 = as_double(cfg["l1"], args.l1);
  if (cfg.contains("max_abs_delta")) args.max_abs_delta = as_double(cfg["max_abs_delta"], args.max_abs_delta);
  if (cfg.contains("max_relative_delta")) args.max_relative_delta = as_double(cfg["max_relative_delta"], args.max_relative_delta);
  if (cfg.contains("preserve_sign")) args.preserve_sign = as_bool(cfg["preserve_sign"], args.preserve_sign);
  if (cfg.contains("feature_threads")) args.feature_threads = as_int(cfg["feature_threads"], args.feature_threads);
  if (cfg.contains("feature_scales") && cfg["feature_scales"].is_object()) {
    for (auto it = cfg["feature_scales"].begin(); it != cfg["feature_scales"].end(); ++it) {
      args.feature_scales[it.key()] = as_double(it.value(), 1.0);
    }
  }
  if (cfg.contains("validation_fraction")) args.validation_fraction = as_double(cfg["validation_fraction"], args.validation_fraction);
  if (cfg.contains("seed")) args.seed = as_int(cfg["seed"], args.seed);
  if (cfg.contains("output_dir")) args.output_dir = as_string(cfg["output_dir"]);
  if (cfg.contains("run_id")) args.run_id = as_string(cfg["run_id"]);
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
      apply_config(args, read_json_file(next()));
    } else if (arg == "--records") {
      args.records.emplace_back(next());
    } else if (arg == "--whitelist") {
      args.whitelist.push_back(next());
    } else if (arg == "--static-eval-variant") {
      args.static_eval_variant = next();
    } else if (arg == "--initial-overrides") {
      args.initial_overrides = next();
    } else if (arg == "--max-actions") {
      args.max_actions = std::stoi(next());
    } else if (arg == "--steps") {
      args.steps = std::stoi(next());
    } else if (arg == "--learning-rate") {
      args.learning_rate = std::stod(next());
    } else if (arg == "--l2") {
      args.l2 = std::stod(next());
    } else if (arg == "--output-dir") {
      args.output_dir = next();
    } else if (arg == "--run-id") {
      args.run_id = next();
    } else if (arg == "--no-preserve-sign") {
      args.preserve_sign = false;
    } else if (arg == "--preserve-sign") {
      args.preserve_sign = true;
    } else if (arg == "--feature-threads") {
      args.feature_threads = std::stoi(next());
    } else if (arg == "--feature-scale") {
      const std::string raw = next();
      const size_t pos = raw.find('=');
      if (pos == std::string::npos || pos == 0) throw std::runtime_error("feature scale must be term=scale");
      args.feature_scales[raw.substr(0, pos)] = std::stod(raw.substr(pos + 1));
    }
  }
  if (args.whitelist.empty()) throw std::runtime_error("no whitelist terms provided");
  if (args.run_id.empty()) args.run_id = "native-" + std::to_string(std::time(nullptr));
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto t0 = std::chrono::steady_clock::now();
    Args args = parse_args(argc, argv);
    fs::path output_dir = args.output_dir / args.run_id;
    fs::create_directories(output_dir);
    std::vector<json> rows = load_record_rows(args.records);
    const auto t_rows = std::chrono::steady_clock::now();
    if (rows.empty()) throw std::runtime_error("no result records found");
    std::vector<Example> examples = build_examples(rows, args);
    const auto t_examples = std::chrono::steady_clock::now();
    json result = optimize(examples, args);
    const auto t_optimize = std::chrono::steady_clock::now();
    auto seconds_between = [](auto start, auto end) {
      return std::chrono::duration<double>(end - start).count();
    };
    json timings = {
        {"load_records_sec", seconds_between(t0, t_rows)},
        {"build_examples_sec", seconds_between(t_rows, t_examples)},
        {"optimize_sec", seconds_between(t_examples, t_optimize)},
        {"total_sec", seconds_between(t0, t_optimize)}};
    result["records"] = rows.size();
    result["examples"] = examples.size();
    result["whitelist"] = args.whitelist;
    result["static_eval_variant"] = args.static_eval_variant;
    result["timings"] = timings;
    std::ofstream(output_dir / "candidate_weights.json") << result.dump(2) << "\n";
    write_csv(output_dir / "weight_changes.csv", result["changes"]);
    write_metrics_csv(output_dir / "training_metrics.csv", result["history"]);
    json summary = {
        {"records", rows.size()},
        {"examples", examples.size()},
        {"changed_override_env", result["changed_override_env"]},
        {"feature_stats", result["feature_stats"]},
        {"train_metrics", result["train_metrics"]},
        {"validation_metrics", result["validation_metrics"]},
        {"timings", timings}};
    std::ofstream(output_dir / "summary.json") << summary.dump(2) << "\n";
    std::cout << "wrote " << output_dir.string() << "\n";
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
}
