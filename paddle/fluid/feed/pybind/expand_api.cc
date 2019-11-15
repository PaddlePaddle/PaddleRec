#include "paddle/fluid/feed/pybind/expand_api.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/feed/src/common/dict_plugin.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {
using paddle::framework::DictPluginManager;
using paddle::framework::FeasignCacheDict;
 
void BindExpandDictPlugin(py::module *m);

void BindExpandApi(py::module *m) {
  BindExpandDictPlugin(m);
}

void BindExpandDictPlugin(py::module *m) {
  py::class_<FeasignCacheDict>(*m, "FeasignCacheDict")
      .def(py::init<>())
      .def(py::init<const FeasignCacheDict &>())
      .def("load", &FeasignCacheDict::Load);
  py::class_<DictPluginManager>(*m, "DictPluginManager")
      .def(py::init<>())
      .def_static("instance", &DictPluginManager::Instance)
      .def("load_dict", &DictPluginManager::LoadDict)
      .def("create_dict", &DictPluginManager::CreateDict);
}



}  // namespace pybind
}  // namespace paddle
