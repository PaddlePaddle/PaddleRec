#pragma once

#include <pybind11/pybind11.h>

namespace paddle {
namespace pybind {
void BindExpandApi(pybind11::module *m);
}  // namespace pybind
}  // namespace paddle
