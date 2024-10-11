#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "primitive_sdf.hpp"

namespace py = pybind11;
void bind_kdtree(py::module& m);
void bind_primitive_sdf(py::module& m);
