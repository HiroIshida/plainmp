/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "plainmp/bindings/bindings.hpp"

namespace pb = plainmp::bindings;

PYBIND11_MODULE(_plainmp, m) {
  pb::bind_kdtree_submodule(m);
  pb::bind_primitive_submodule(m);
  pb::bind_constraint_submodule(m);
  pb::bind_kinematics_submodule(m);
  pb::bind_ompl_wrapper_submodule(m);
  pb::bind_experimental_submodule(m);
}
