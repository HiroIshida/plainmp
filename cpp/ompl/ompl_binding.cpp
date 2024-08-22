#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ompl_thin_wrap.hpp"
#include "constraint.hpp"
#include "ompl_binding.hpp"

namespace py = pybind11;

void bind_ompl(py::module &m)
{
  auto ompl_m = m.def_submodule("ompl");
  ompl_m.def("set_random_seed", &setGlobalSeed);
  ompl_m.def("set_log_level_none", &setLogLevelNone);

  // py::enum_<ConstStateType>(m, "ConstStateType")
  //     .value("PROJECTION", ConstStateType::PROJECTION)
  //     .value("ATLAS", ConstStateType::ATLAS)
  //     .value("TANGENT", ConstStateType::TANGENT);

  // py::class_<ConstrainedPlanner>(m, "_ConstrainedPlanner")
  //     .def(py::init<const ConstFn&,
  //                   const ConstJacFn&,
  //                   std::vector<double>,
  //                   std::vector<double>,
  //                   std::function<bool(std::vector<double>)>,
  //                   size_t,
  //                   std::vector<double>,
  //                   std::string,
  //                   std::optional<double>,
  //                   ConstStateType>())
  //     .def("reset_is_valid", &ConstrainedPlanner::resetIsValid)
  //     .def("solve", &ConstrainedPlanner::solve);

  py::class_<OMPLPlanner>(ompl_m, "OMPLPlanner", py::module_local())
      .def(py::init<std::vector<double>&,
                    std::vector<double>&,
                    cst::IneqConstraintBase::Ptr,
                    size_t,
                    std::vector<double>,
                    std::string,
                    std::optional<double>>())
      .def("get_call_count", &OMPLPlanner::getCallCount)
      .def("solve", &OMPLPlanner::solve);

  // py::class_<LightningDBWrap>(m, "_LightningDB")
  //     .def(py::init<size_t>())
  //     .def("save", &LightningDBWrap::save)
  //     .def("load", &LightningDBWrap::load)
  //     .def("add_experience", &LightningDBWrap::addExperience)
  //     .def("get_experienced_paths", &LightningDBWrap::getExperiencedPaths)
  //     .def("get_experiences_count", &LightningDBWrap::getExperiencesCount);

  // py::class_<LightningPlanner>(m, "_LightningPlanner")
  //     .def(py::init<LightningDBWrap,
  //                   std::vector<double>,
  //                   std::vector<double>,
  //                   std::function<bool(std::vector<double>)>,
  //                   size_t,
  //                   std::vector<double>,
  //                   std::string,
  //                   std::optional<double>>())
  //     .def("reset_is_valid", &LightningPlanner::resetIsValid)
  //     .def("solve", &LightningPlanner::solve);

  // py::class_<LightningRepairPlanner>(m, "_LightningRepairPlanner")
  //     .def(py::init<std::vector<double>,
  //                   std::vector<double>,
  //                   std::function<bool(std::vector<double>)>,
  //                   size_t,
  //                   std::vector<double>,
  //                   std::string,
  //                   std::optional<double>>())
  //     .def("reset_is_valid", &LightningRepairPlanner::resetIsValid)
  //     .def("solve", &LightningRepairPlanner::solve)
  //     .def("set_heuristic", &LightningRepairPlanner::set_heuristic);

  py::class_<ERTConnectPlanner>(ompl_m, "ERTConnectPlanner", py::module_local())
      .def(py::init<std::vector<double>,
                    std::vector<double>,
                    cst::IneqConstraintBase::Ptr,
                    size_t,
                    std::vector<double>>())
      .def("get_call_count", &OMPLPlanner::getCallCount)
      .def("solve", &ERTConnectPlanner::solve)
      .def("set_parameters", &ERTConnectPlanner::set_parameters)
      .def("set_heuristic", &ERTConnectPlanner::set_heuristic);
}
