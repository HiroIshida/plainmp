#include <pybind11/pybind11.h>

namespace tinyfk {
namespace py = pybind11;
void bind_tinyfk(py::module& m);
}  // namespace tinyfk
