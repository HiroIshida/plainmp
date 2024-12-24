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

using namespace plainmp::kinematics;

namespace plainmp::bindings {

using Transform = KinematicModel<double>::Transform;
using Vector7d = Eigen::Matrix<double, 7, 1>;

Vector7d pose_to_vector(const Transform& pose) {
  Vector7d vec;
  vec << pose.trans().x(), pose.trans().y(), pose.trans().z(), pose.quat().x(),
      pose.quat().y(), pose.quat().z(), pose.quat().w();
  return vec;
}

Transform vector_to_pose(const Vector7d& pose_vec) {
  Eigen::Vector3d position;
  position << pose_vec[0], pose_vec[1], pose_vec[2];
  // NOTE: eigen uses wxyz order
  Eigen::Quaterniond orientation(pose_vec[6], pose_vec[3], pose_vec[4],
                                 pose_vec[5]);
  return Transform{orientation, position};
}

class _KinematicModel : public KinematicModel<double> {
  // a utility class for easy binding
 public:
  using KinematicModel::KinematicModel;
  size_t add_new_link_py(const std::string& link_name,
                         const std::string& parent_name,
                         const std::array<double, 3>& position,
                         const std::array<double, 3>& rpy,
                         bool consider_rotation) {
    size_t parent_id = get_link_ids({parent_name})[0];
    return KinematicModel::add_new_link(parent_id, position, rpy,
                                        consider_rotation, link_name);
  }

  Vector7d get_base_pose() {
    auto pose = KinematicModel::get_base_pose();
    return pose_to_vector(pose);
  }

  void set_base_pose(const Vector7d& pose_vec) {
    auto pose = vector_to_pose(pose_vec);
    KinematicModel::set_base_pose(pose);
  }

  Vector7d debug_get_link_pose(const std::string& link_name) {
    size_t link_id = get_link_ids({link_name})[0];  // slow
    auto pose = KinematicModel::get_link_pose(link_id);
    return pose_to_vector(pose);
  }
};

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
