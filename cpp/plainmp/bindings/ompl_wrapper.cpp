/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include <optional>
#include "plainmp/bindings/bindings.hpp"
#include "plainmp/constraints/primitive.hpp"
#include "plainmp/ompl/ompl_thin_wrap.hpp"

namespace plainmp::bindings {

using namespace plainmp::ompl_wrapper;

void bind_ompl_wrapper_submodule(nb::module_& m) {
  auto ompl_m = m.def_submodule("ompl");
  ompl_m.def("set_random_seed", &setGlobalSeed);
  ompl_m.def("set_log_level_none", &setLogLevelNone);

  nb::enum_<ValidatorConfig::Type>(ompl_m, "ValidatorType")
      .value("BOX", ValidatorConfig::Type::BOX)
      .value("EUCLIDEAN", ValidatorConfig::Type::EUCLIDEAN)
      .export_values();

  nb::class_<ValidatorConfig>(ompl_m, "ValidatorConfig")
      .def(nb::init<>())
      .def_rw("type", &ValidatorConfig::type)
      .def_rw("resolution", &ValidatorConfig::resolution)
      .def_rw("box_width", &ValidatorConfig::box_width)
      .def_rw("enable_interval_pruning",
                     &ValidatorConfig::enable_interval_pruning);

  nb::enum_<RefineType>(ompl_m, "RefineType")
      .value("SHORTCUT", RefineType::SHORTCUT)
      .value("BSPLINE", RefineType::BSPLINE)
      .export_values();

  ompl_m.def("simplify", &simplify);

  nb::class_<OMPLPlanner>(ompl_m, "OMPLPlanner")
      .def(nb::init<std::vector<double>&, std::vector<double>&,
                    constraint::IneqConstraintBase::Ptr, size_t,
                    ValidatorConfig, std::string, std::optional<double>>(),
           nb::arg("lower_bound"), nb::arg("upper_bound"),
           nb::arg("constraint"), nb::arg("max_is_valid_call"),
           nb::arg("validator_config"), nb::arg("algorithm"),
           nb::arg("range").none())
      .def("get_call_count", &OMPLPlanner::getCallCount)
      .def("get_ns_internal", &OMPLPlanner::get_ns_internal)
      .def("solve", &OMPLPlanner::solve, nb::arg("start"),
           nb::arg("goal").none(), nb::arg("refine_seq"),
           nb::arg("timeout") = nb::none(),
           nb::arg("goal_sampler") = nb::none(),
           nb::arg("max_goal_sample_count") = nb::none());

  nb::class_<ERTConnectPlanner>(ompl_m, "ERTConnectPlanner")
      .def(nb::init<const std::vector<double>&, const std::vector<double>&,
                    constraint::IneqConstraintBase::Ptr, size_t,
                    const ValidatorConfig&>())
      .def("get_call_count", &OMPLPlanner::getCallCount)
      .def("get_ns_internal", &OMPLPlanner::get_ns_internal)
      .def("solve", &ERTConnectPlanner::solve, nb::arg("start"),
           nb::arg("goal").none(), nb::arg("refine_seq"),
           nb::arg("timeout") = nb::none(),
           nb::arg("goal_sampler") = nb::none(),
           nb::arg("max_goal_sample_count") = nb::none())
      .def("set_parameters", &ERTConnectPlanner::set_parameters)
      .def("set_heuristic", &ERTConnectPlanner::set_heuristic);
}

}  // namespace plainmp::bindings
