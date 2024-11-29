#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace plainmp::bindings {

namespace py = pybind11;
void bind_kdtree_submodule(py::module& m);
void bind_primitive_submodule(py::module& m);
void bind_constraint_submodule(py::module& m);
void bind_kinematics_submodule(py::module& m);
void bind_ompl_wrapper_submodule(py::module& m);

}  // namespace plainmp::bindings
