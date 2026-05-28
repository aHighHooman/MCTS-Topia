#pragma once

#include <pybind11/pybind11.h>

#include "native_rules.hpp"

#include <vector>

namespace py = pybind11;

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

}  // namespace tribes::native
