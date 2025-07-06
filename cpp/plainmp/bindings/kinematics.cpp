/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "plainmp/kinematics/kinematics.hpp"
#include "plainmp/bindings/bindings.hpp"
#include "plainmp/kinematics/kinematic_model_wrapper.hpp"

using namespace plainmp::kinematics;

namespace plainmp::bindings {

void bind_kinematics_submodule(py::module& m) {
  auto m_kin = m.def_submodule("kinematics");
  py::class_<urdf::Link, urdf::LinkSharedPtr>(m_kin, "Link")
      .def_readonly("name", &urdf::Link::name)
      .def_readonly("id", &urdf::Link::id);

  // parent class
  py::class_<KinematicModel<double>, std::shared_ptr<KinematicModel<double>>>(
      m_kin, "KinematicModel_cpp", py::module_local());

  // child "binding" class
  py::class_<_KinematicModel, std::shared_ptr<_KinematicModel>,
             KinematicModel<double>>(m_kin, "KinematicModel",
                                     py::module_local())
      .def(py::init<std::string&>())
      .def("add_new_link", &_KinematicModel::add_new_link_py)
      .def("debug_get_link_pose", &_KinematicModel::debug_get_link_pose)
      .def("set_joint_positions", &_KinematicModel::set_joint_angles,
           py::arg("joint_ids"), py::arg("positions"),
           py::arg("accurate") = true)
      .def("get_joint_positions", &_KinematicModel::get_joint_angles)
      .def("set_base_pose", &_KinematicModel::set_base_pose)
      .def("get_base_pose", &_KinematicModel::get_base_pose)
      .def("get_joint_position_limits",
           &_KinematicModel::get_joint_position_limits)
      .def("get_link_ids", &_KinematicModel::get_link_ids)
      .def("get_joint_ids", &_KinematicModel::get_joint_ids);

  py::enum_<BaseType>(m_kin, "BaseType")
      .value("FIXED", BaseType::FIXED)
      .value("FLOATING", BaseType::FLOATING)
      .value("PLANAR", BaseType::PLANAR);
}

}  // namespace plainmp::bindings
