#pragma once

#ifdef TRIBES_NATIVE_MCTS_STANDALONE
#include "native_json_py.hpp"
#else
#include <pybind11/pybind11.h>
#endif

#include "native_rules.hpp"

#include <vector>

#ifndef TRIBES_NATIVE_MCTS_STANDALONE
namespace py = pybind11;
#endif

namespace tribes::native {

struct StaticEvaluation {
  std::vector<double> priors;
  double value = 0.0;
};

StaticEvaluation evaluate_static_state(
    const NativeGameState& state,
    const std::vector<NativeAction>& actions);
py::dict evaluate_static(const py::dict& payload, int max_actions);
py::list evaluate_static_batch(const py::list& payloads, int max_actions);
py::dict evaluate_static_breakdown(const py::dict& payload, int max_actions);
py::list evaluate_static_breakdown_batch(const py::list& payloads, int max_actions);
py::dict evaluate_action_breakdown(const py::dict& payload, const std::string& action_id, int max_actions);

}  // namespace tribes::native
