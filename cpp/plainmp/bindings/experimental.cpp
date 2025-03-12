#include "plainmp/bindings/bindings.hpp"
#include "plainmp/experimental/multigoal_rrt.hpp"

namespace plainmp::bindings {

void bind_experimental_submodule(py::module& m) {
  auto m_experimental = m.def_submodule("experimental");
  py::class_<experimental::MultiGoalRRT>(m_experimental, "MultiGoalRRT")
      .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&,
                    const Eigen::VectorXd&,
                    const plainmp::constraint::IneqConstraintBase::Ptr&,
                    size_t>())
      .def("is_reachable", &experimental::MultiGoalRRT::is_reachable)
      .def("is_reachable_batch",
           &experimental::MultiGoalRRT::is_reachable_batch)
      .def("get_debug_states", &experimental::MultiGoalRRT::get_debug_states)
      .def("get_debug_parents", &experimental::MultiGoalRRT::get_debug_parents);
}

}  // namespace plainmp::bindings
