#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename Scalar>
void bind_ompl(py::module &m);
