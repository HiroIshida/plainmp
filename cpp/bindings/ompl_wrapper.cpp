#include <pybind11/functional.h>
#include <optional>
#include "bindings.hpp"
#include "constraint.hpp"
#include "ompl/ompl_thin_wrap.hpp"

namespace plainmp::bindings {

void bind_ompl_wrapper_submodule(py::module& m) {
  auto ompl_m = m.def_submodule("ompl");
  ompl_m.def("set_random_seed", &setGlobalSeed);
  ompl_m.def("set_log_level_none", &setLogLevelNone);

  py::class_<OMPLPlanner>(ompl_m, "OMPLPlanner", py::module_local())
      .def(py::init<std::vector<double>&, std::vector<double>&,
                    cst::IneqConstraintBase::Ptr, size_t, std::vector<double>,
                    std::string, std::optional<double>>())
      .def("get_call_count", &OMPLPlanner::getCallCount)
      .def("solve", &OMPLPlanner::solve, py::arg("start"), py::arg("goal"),
           py::arg("simplify"), py::arg("timeout") = py::none(),
           py::arg("goal_sampler") = py::none(),
           py::arg("max_goal_sample_count") = py::none());

  py::class_<ERTConnectPlanner>(ompl_m, "ERTConnectPlanner", py::module_local())
      .def(
          py::init<std::vector<double>, std::vector<double>,
                   cst::IneqConstraintBase::Ptr, size_t, std::vector<double>>())
      .def("get_call_count", &OMPLPlanner::getCallCount)
      .def("solve", &ERTConnectPlanner::solve, py::arg("start"),
           py::arg("goal"), py::arg("simplify"),
           py::arg("timeout") = py::none(),
           py::arg("goal_sampler") = py::none(),
           py::arg("max_goal_sample_count") = py::none())
      .def("set_parameters", &ERTConnectPlanner::set_parameters)
      .def("set_heuristic", &ERTConnectPlanner::set_heuristic);
}

}  // namespace plainmp::bindings
