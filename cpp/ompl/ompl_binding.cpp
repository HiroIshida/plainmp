#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ompl_thin_wrap.hpp"
#include "constraint.hpp"
#include "ompl_binding.hpp"

namespace py = pybind11;

template <typename Scalar>
void bind_ompl(py::module &m)
{
  auto ompl_m = m.def_submodule("ompl");
  ompl_m.def("set_random_seed", &setGlobalSeed);
  ompl_m.def("set_log_level_none", &setLogLevelNone);

  py::class_<OMPLPlanner<Scalar>>(ompl_m, "OMPLPlanner", py::module_local())
      .def(py::init<std::vector<Scalar>&,
                    std::vector<Scalar>&,
                    typename cst::IneqConstraintBase<Scalar>::Ptr,
                    size_t,
                    std::vector<Scalar>,
                    std::string,
                    std::optional<Scalar>>())
      .def("get_call_count", &OMPLPlanner<Scalar>::getCallCount)
      .def("solve", &OMPLPlanner<Scalar>::solve);

  py::class_<ERTConnectPlanner<Scalar>>(ompl_m, "ERTConnectPlanner", py::module_local())
      .def(py::init<std::vector<Scalar>,
                    std::vector<Scalar>,
                    typename cst::IneqConstraintBase<Scalar>::Ptr,
                    size_t,
                    std::vector<Scalar>>())
      .def("get_call_count", &OMPLPlanner<Scalar>::getCallCount)
      .def("solve", &ERTConnectPlanner<Scalar>::solve)
      .def("set_parameters", &ERTConnectPlanner<Scalar>::set_parameters)
      .def("set_heuristic", &ERTConnectPlanner<Scalar>::set_heuristic);
}

template void bind_ompl<double>(py::module &m);
template void bind_ompl<float>(py::module &m);
