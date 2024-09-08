#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace cst {
template <typename Scalar>
void bind_collision_constraints(py::module& m);
};
