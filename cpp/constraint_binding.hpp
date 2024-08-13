#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace cst {
void bind_collision_constraints(py::module& m);
};
