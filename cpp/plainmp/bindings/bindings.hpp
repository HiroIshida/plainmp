/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace plainmp::bindings {

namespace py = pybind11;
void bind_kdtree_submodule(py::module& m);
void bind_primitive_submodule(py::module& m);
void bind_constraint_submodule(py::module& m);
void bind_kinematics_submodule(py::module& m);
void bind_ompl_wrapper_submodule(py::module& m);
void bind_experimental_submodule(py::module& m);

}  // namespace plainmp::bindings
