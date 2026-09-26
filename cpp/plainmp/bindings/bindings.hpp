/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <stdexcept>

namespace plainmp::bindings {

namespace nb = nanobind;

inline Eigen::VectorXd vector_from_sequence(nb::sequence values) {
  Eigen::VectorXd vector(nb::len(values));
  for (size_t i = 0; i < nb::len(values); ++i) {
    vector[i] = nb::cast<double>(values[i]);
  }
  return vector;
}

inline Eigen::Vector3d vector3_from_sequence(nb::sequence values) {
  if (nb::len(values) != 3) {
    throw std::invalid_argument("expected a 3-element vector");
  }
  Eigen::Vector3d vector;
  for (size_t i = 0; i < 3; ++i) {
    vector[i] = nb::cast<double>(values[i]);
  }
  return vector;
}
void bind_kdtree_submodule(nb::module_& m);
void bind_primitive_submodule(nb::module_& m);
void bind_constraint_submodule(nb::module_& m);
void bind_kinematics_submodule(nb::module_& m);
void bind_ompl_wrapper_submodule(nb::module_& m);

}  // namespace plainmp::bindings
