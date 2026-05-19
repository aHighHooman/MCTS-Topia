#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace tribes::native {

py::dict evaluate_static(const py::dict& payload, int max_actions);
py::list evaluate_static_batch(const py::list& payloads, int max_actions);

}  // namespace tribes::native
