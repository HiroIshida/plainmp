#include "collision/kdtree.hpp"
#include <vector>
#include "bindings.hpp"

using namespace plainmp::collision;

namespace plainmp::bindings {
void bind_kdtree_submodule(py::module& m) {
  auto m_kdtree = m.def_submodule("kdtree");
  py::class_<KDTree>(m_kdtree, "KDTree")
      .def(py::init<const std::vector<Eigen::Vector3d>&>())
      .def("query", &KDTree::query)
      .def("sqdist", &KDTree::sqdist);
}
}  // namespace plainmp::bindings
