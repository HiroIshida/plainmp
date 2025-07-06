/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "plainmp/kinematics/kinematic_model_wrapper.hpp"

namespace plainmp::kinematics::utils {

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

size_t _KinematicModel::add_new_link_py(const std::string& link_name,
                                        const std::string& parent_name,
                                        const std::array<double, 3>& position,
                                        const std::array<double, 3>& rpy,
                                        bool consider_rotation) {
  size_t parent_id = get_link_ids({parent_name})[0];
  return KinematicModel::add_new_link(parent_id, position, rpy,
                                      consider_rotation, link_name);
}

Vector7d _KinematicModel::get_base_pose() {
  auto pose = KinematicModel::get_base_pose();
  return pose_to_vector(pose);
}

void _KinematicModel::set_base_pose(const Vector7d& pose_vec) {
  auto pose = vector_to_pose(pose_vec);
  KinematicModel::set_base_pose(pose);
}

Vector7d _KinematicModel::debug_get_link_pose(const std::string& link_name) {
  size_t link_id = get_link_ids({link_name})[0];  // slow
  auto pose = KinematicModel::get_link_pose(link_id);
  return pose_to_vector(pose);
}

}  // namespace plainmp::kinematics::utils