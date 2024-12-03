/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

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
