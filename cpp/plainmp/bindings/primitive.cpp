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
#include "plainmp/collision/primitive_sdf.hpp"

using namespace plainmp::collision;

namespace plainmp::bindings {

void bind_primitive_submodule(py::module& m) {
  auto m_psdf = m.def_submodule("primitive_sdf");
  py::class_<Pose>(m_psdf, "Pose", py::module_local())
      .def(py::init<const Eigen::Vector3d&, const Eigen::Matrix3d&>())
      .def_readonly("axis_aligned", &Pose::axis_aligned_)
      .def_readonly("z_axis_aligned", &Pose::z_axis_aligned_);
  py::class_<SDFBase, SDFBase::Ptr>(
      m_psdf, "SDFBase",
      py::module_local());  // user is not supposed to instantiate this class.
                            // This to tell pybind that this is a base class
  py::class_<PrimitiveSDFBase, PrimitiveSDFBase::Ptr, SDFBase>(
      m_psdf, "PrimitiveSDFBase",
      py::module_local());  // user is not supposed to instantiate this class.
                            // This to tell pybind that this is a base class
  py::class_<UnionSDF, UnionSDF::Ptr, SDFBase>(m_psdf, "UnionSDF",
                                               py::module_local())
      .def(py::init<std::vector<SDFBase::Ptr>>())
      .def("evaluate_batch", &UnionSDF::evaluate_batch)
      .def("evaluate", &UnionSDF::evaluate)
      .def("is_outside", &UnionSDF::is_outside);
  py::class_<GroundSDF, GroundSDF::Ptr, PrimitiveSDFBase>(m_psdf, "GroundSDF",
                                                          py::module_local())
      .def(py::init<double>())
      .def("evaluate_batch", &GroundSDF::evaluate_batch)
      .def("evaluate", &GroundSDF::evaluate)
      .def("is_outside", &GroundSDF::is_outside);
  py::class_<BoxSDF, BoxSDF::Ptr, PrimitiveSDFBase>(m_psdf, "BoxSDF",
                                                    py::module_local())
      .def(py::init<const Eigen::Vector3d&, const Pose&>())
      .def("evaluate_batch", &BoxSDF::evaluate_batch)
      .def("evaluate", &BoxSDF::evaluate)
      .def("is_outside", &BoxSDF::is_outside);
  py::class_<CylinderSDF, CylinderSDF::Ptr, PrimitiveSDFBase>(
      m_psdf, "CylinderSDF", py::module_local())
      .def(py::init<double, double, const Pose&>())
      .def("evaluate_batch", &CylinderSDF::evaluate_batch)
      .def("evaluate", &CylinderSDF::evaluate)
      .def("is_outside", &CylinderSDF::is_outside);
  py::class_<SphereSDF, SphereSDF::Ptr, PrimitiveSDFBase>(m_psdf, "SphereSDF",
                                                          py::module_local())
      .def(py::init<double, const Pose&>())
      .def("evaluate_batch", &SphereSDF::evaluate_batch)
      .def("evaluate", &SphereSDF::evaluate)
      .def("is_outside", &SphereSDF::is_outside);

  py::class_<CloudSDF, CloudSDF::Ptr, PrimitiveSDFBase>(m_psdf, "CloudSDF",
                                                        py::module_local())
      .def(py::init<std::vector<Eigen::Vector3d>&, double>())
      .def("evaluate_batch", &CloudSDF::evaluate_batch)
      .def("evaluate", &CloudSDF::evaluate)
      .def("is_outside", &CloudSDF::is_outside);
}

}  // namespace plainmp::bindings
