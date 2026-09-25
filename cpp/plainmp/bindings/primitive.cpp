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

void bind_primitive_submodule(nb::module_& m) {
  auto m_psdf = m.def_submodule("primitive_sdf");
  nb::class_<Pose>(m_psdf, "Pose")
      .def(nb::init<const Eigen::Vector3d&, const Eigen::Matrix3d&>(),
           nb::arg("position") = Eigen::Vector3d::Zero(),
           nb::arg("rotation") = Eigen::Matrix3d::Identity())
      .def(
          "__init__",
          [](Pose* self, nb::sequence position,
             const Eigen::Matrix3d& rotation) {
            new (self) Pose(vector3_from_sequence(position), rotation);
          },
          nb::arg("position"), nb::arg("rotation"))
      .def("__copy__", [](const Pose& self) { return Pose(self); })
      .def("__deepcopy__",
           [](const Pose& self, nb::dict) { return Pose(self); })
      .def("translate", &Pose::translate)
      .def("rotate_z", &Pose::rotate_z)
      .def_ro("axis_aligned", &Pose::axis_aligned_)
      .def_ro("z_axis_aligned", &Pose::z_axis_aligned_)
      .def_ro("position", &Pose::position_, nb::rv_policy::copy)
      .def_ro("rotation", &Pose::rot_, nb::rv_policy::copy);
  // Register the abstract base classes so derived SDFs retain their hierarchy.
  nb::class_<SDFBase>(m_psdf, "SDFBase");
  nb::class_<PrimitiveSDFBase, SDFBase>(m_psdf, "PrimitiveSDFBase");
  nb::class_<UnionSDF, SDFBase>(m_psdf, "UnionSDF")
      .def(nb::init<std::vector<SDFBase::Ptr>>())
      .def("clone", &UnionSDF::clone)
      .def("merge", &UnionSDF::merge, nb::arg("other"),
           nb::arg("clone") = false)
      .def("add", &UnionSDF::add, nb::arg("sdf"), nb::arg("clone") = false)
      .def("translate", &UnionSDF::translate)
      .def("rotate_z", &UnionSDF::rotate_z)
      .def("evaluate_batch", &UnionSDF::evaluate_batch)
      .def("evaluate", &UnionSDF::evaluate)
      .def("is_outside", &UnionSDF::is_outside);
  nb::class_<GroundSDF, PrimitiveSDFBase>(m_psdf, "GroundSDF")
      .def(nb::init<double>())
      .def("clone", &GroundSDF::clone)
      .def("translate", &GroundSDF::translate)
      .def("rotate_z", &GroundSDF::rotate_z)
      .def("evaluate_batch", &GroundSDF::evaluate_batch)
      .def("evaluate", &GroundSDF::evaluate)
      .def("is_outside", &GroundSDF::is_outside)
      .def_ro("lb", &GroundSDF::lb, nb::rv_policy::copy)
      .def_ro("ub", &GroundSDF::ub, nb::rv_policy::copy);
  nb::class_<BoxSDF, PrimitiveSDFBase>(m_psdf, "BoxSDF")
      .def(nb::init<const Eigen::Vector3d&, const Pose&>())
      .def("__init__",
           [](BoxSDF* self, nb::sequence width, const Pose& pose) {
             new (self) BoxSDF(vector3_from_sequence(width), pose);
           })
      .def("clone", &BoxSDF::clone)
      .def("translate", &BoxSDF::translate)
      .def("rotate_z", &BoxSDF::rotate_z)
      .def("evaluate_batch", &BoxSDF::evaluate_batch)
      .def("evaluate", &BoxSDF::evaluate)
      .def("is_outside", &BoxSDF::is_outside)
      .def_ro("lb", &BoxSDF::lb, nb::rv_policy::copy)
      .def_ro("ub", &BoxSDF::ub, nb::rv_policy::copy)
      .def_ro("pose", &BoxSDF::pose);
  nb::class_<CylinderSDF, PrimitiveSDFBase>(m_psdf, "CylinderSDF")
      .def(nb::init<double, double, const Pose&>())
      .def("clone", &CylinderSDF::clone)
      .def("translate", &CylinderSDF::translate)
      .def("rotate_z", &CylinderSDF::rotate_z)
      .def("evaluate_batch", &CylinderSDF::evaluate_batch)
      .def("evaluate", &CylinderSDF::evaluate)
      .def("is_outside", &CylinderSDF::is_outside)
      .def_ro("lb", &CylinderSDF::lb, nb::rv_policy::copy)
      .def_ro("ub", &CylinderSDF::ub, nb::rv_policy::copy)
      .def_ro("pose", &CylinderSDF::pose);
  nb::class_<SphereSDF, PrimitiveSDFBase>(m_psdf, "SphereSDF")
      .def(nb::init<double, const Pose&>())
      .def("clone", &SphereSDF::clone)
      .def("translate", &SphereSDF::translate)
      .def("rotate_z", &SphereSDF::rotate_z)
      .def("evaluate_batch", &SphereSDF::evaluate_batch)
      .def("evaluate", &SphereSDF::evaluate)
      .def("is_outside", &SphereSDF::is_outside)
      .def_ro("lb", &SphereSDF::lb, nb::rv_policy::copy)
      .def_ro("ub", &SphereSDF::ub, nb::rv_policy::copy)
      .def_ro("pose", &SphereSDF::pose);

  nb::class_<CloudSDF, PrimitiveSDFBase>(m_psdf, "CloudSDF")
      .def(nb::init<std::vector<Eigen::Vector3d>&, double>())
      .def("clone", &CloudSDF::clone)
      .def("translate", &CloudSDF::translate)
      .def("rotate_z", &CloudSDF::rotate_z)
      .def("evaluate_batch", &CloudSDF::evaluate_batch)
      .def("evaluate", &CloudSDF::evaluate)
      .def("is_outside", &CloudSDF::is_outside)
      .def_ro("lb", &CloudSDF::lb, nb::rv_policy::copy)
      .def_ro("ub", &CloudSDF::ub, nb::rv_policy::copy);
}

}  // namespace plainmp::bindings
