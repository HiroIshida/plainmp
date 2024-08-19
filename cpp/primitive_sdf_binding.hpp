#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace primitive_sdf {
namespace py = pybind11;
void bind_primitive_sdf(py::module& m);
}  // namespace primitive_sdf
