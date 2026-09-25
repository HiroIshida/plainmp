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

void bind_kinematics_submodule(nb::module_& m) {
  auto m_kin = m.def_submodule("kinematics");
  nb::class_<urdf::Link>(m_kin, "Link")
      .def_ro("name", &urdf::Link::name)
      .def_ro("id", &urdf::Link::id);

  nb::enum_<BaseType>(m_kin, "BaseType")
      .value("FIXED", BaseType::FIXED)
      .value("FLOATING", BaseType::FLOATING)
      .value("PLANAR", BaseType::PLANAR);

  // parent class
  nb::class_<KinematicModel<double>>(m_kin, "KinematicModel_cpp");

  // child "binding" class
  nb::class_<utils::_KinematicModel, KinematicModel<double>>(m_kin,
                                                             "KinematicModel")
      .def(nb::init<std::string&>())
      .def("add_new_link",
           [](utils::_KinematicModel& self, const std::string& link_name,
              const std::string& parent_name, nb::sequence position,
              nb::sequence rpy, bool consider_rotation) {
             const Eigen::Vector3d position_vec =
                 vector3_from_sequence(position);
             const Eigen::Vector3d rpy_vec = vector3_from_sequence(rpy);
             return self.add_new_link_py(
                 link_name, parent_name,
                 {position_vec.x(), position_vec.y(), position_vec.z()},
                 {rpy_vec.x(), rpy_vec.y(), rpy_vec.z()}, consider_rotation);
           })
      .def("debug_get_link_pose", &utils::_KinematicModel::debug_get_link_pose)
      .def("set_joint_positions", &utils::_KinematicModel::set_joint_angles,
           nb::arg("joint_ids"), nb::arg("positions"),
           nb::arg("accurate") = true)
      .def(
          "set_joint_positions",
          [](utils::_KinematicModel& self, const std::vector<size_t>& joint_ids,
             nb::sequence positions, bool accurate) {
            Eigen::VectorXd vector = vector_from_sequence(positions);
            self.set_joint_angles(joint_ids, vector, accurate);
          },
          nb::arg("joint_ids"), nb::arg("positions"),
          nb::arg("accurate") = true)
      .def("get_joint_positions", &utils::_KinematicModel::get_joint_angles)
      .def("set_base_pose", &utils::_KinematicModel::set_base_pose)
      .def("set_base_pose",
           [](utils::_KinematicModel& self, nb::sequence values) {
             if (nb::len(values) != 7) {
               throw std::invalid_argument("expected a 7-element pose");
             }
             const Eigen::VectorXd vector = vector_from_sequence(values);
             self.set_base_pose(vector);
           })
      .def("get_base_pose", &utils::_KinematicModel::get_base_pose)
      .def("get_joint_position_limits",
           &utils::_KinematicModel::get_joint_position_limits)
      .def("get_gravity_term", &utils::_KinematicModel::get_gravity_term,
           nb::arg("joint_ids"), nb::arg("base_type") = BaseType::FIXED)
      .def("get_gravity_term2", &utils::_KinematicModel::get_gravity_term2,
           nb::arg("joint_ids"), nb::arg("positions"),
           nb::arg("base_type") = BaseType::FIXED, nb::arg("accurate") = true)
      .def("get_link_ids", &utils::_KinematicModel::get_link_ids)
      .def("get_joint_ids", &utils::_KinematicModel::get_joint_ids);
}

}  // namespace plainmp::bindings
