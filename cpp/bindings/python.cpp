#include "bindings/bindings.hpp"

namespace pb = plainmp::bindings;

PYBIND11_MODULE(_plainmp, m) {
  pb::bind_kdtree_submodule(m);
  pb::bind_primitive_submodule(m);
  pb::bind_constraint_submodule(m);
  pb::bind_kinematics_submodule(m);
  pb::bind_ompl_wrapper_submodule(m);
}
