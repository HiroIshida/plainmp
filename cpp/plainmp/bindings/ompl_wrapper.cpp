/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include <pybind11/functional.h>
#include <optional>
#include "plainmp/bindings/bindings.hpp"
#include "plainmp/constraints/primitive.hpp"
#include "plainmp/ompl/ompl_thin_wrap.hpp"

namespace plainmp::bindings {

using namespace plainmp::ompl_wrapper;

void bind_ompl_wrapper_submodule(py::module& m) {
  auto ompl_m = m.def_submodule("ompl");
  ompl_m.def("set_random_seed", &setGlobalSeed);
  ompl_m.def("set_log_level_none", &setLogLevelNone);

  py::enum_<ValidatorConfig::Type>(ompl_m, "ValidatorType")
      .value("BOX", ValidatorConfig::Type::BOX)
      .value("EUCLIDEAN", ValidatorConfig::Type::EUCLIDEAN)
      .export_values();

  py::class_<ValidatorConfig>(ompl_m, "ValidatorConfig")
      .def(py::init<>())
      .def_readwrite("type", &ValidatorConfig::type)
      .def_readwrite("resolution", &ValidatorConfig::resolution)
      .def_readwrite("box_width", &ValidatorConfig::box_width);

  py::class_<OMPLPlanner>(ompl_m, "OMPLPlanner", py::module_local())
      .def(py::init<std::vector<double>&, std::vector<double>&,
                    constraint::IneqConstraintBase::Ptr, size_t,
                    ValidatorConfig, std::string, std::optional<double>>())
      .def("get_call_count", &OMPLPlanner::getCallCount)
      .def("solve", &OMPLPlanner::solve, py::arg("start"), py::arg("goal"),
           py::arg("simplify"), py::arg("timeout") = py::none(),
           py::arg("goal_sampler") = py::none(),
           py::arg("max_goal_sample_count") = py::none());

  py::class_<ERTConnectPlanner>(ompl_m, "ERTConnectPlanner", py::module_local())
      .def(py::init<std::vector<double>, std::vector<double>,
                    constraint::IneqConstraintBase::Ptr, size_t,
                    ValidatorConfig>())
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
