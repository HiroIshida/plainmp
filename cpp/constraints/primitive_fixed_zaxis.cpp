/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "primitive_fixed_zaxis.hpp"

namespace plainmp::constraint {

FixedZAxisCst::FixedZAxisCst(
    std::shared_ptr<kin::KinematicModel<double>> kin,
    const std::vector<std::string>& control_joint_names,
    bool with_base,
    const std::string& link_name)
    : EqConstraintBase(kin, control_joint_names, with_base),
      link_id_(kin_->get_link_ids({link_name})[0]) {
  aux_link_ids_.clear();
  {
    auto pose = Transform::Identity();
    pose.trans().x() = 1;
    auto new_link_id = kin_->add_new_link(link_id_, pose, false);
    aux_link_ids_.push_back(new_link_id);
  }

  {
    auto pose = Transform::Identity();
    pose.trans().y() = 1;
    auto new_link_id = kin_->add_new_link(link_id_, pose, false);
    aux_link_ids_.push_back(new_link_id);
  }
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> FixedZAxisCst::evaluate_dirty() {
  const auto& pose_here = kin_->get_link_pose(link_id_);
  const auto& pose_plus1_x = kin_->get_link_pose(aux_link_ids_[0]);
  const auto& pose_plus1_y = kin_->get_link_pose(aux_link_ids_[1]);
  Eigen::VectorXd vals(2);
  double diff_plus1_x_z = pose_plus1_x.trans().z() - pose_here.trans().z();
  double diff_plus1_y_z = pose_plus1_y.trans().z() - pose_here.trans().z();
  vals << diff_plus1_x_z, diff_plus1_y_z;

  // jacobian
  Eigen::MatrixXd jac_here(3, q_dim());
  Eigen::MatrixXd jac_plus1_x(3, q_dim());
  Eigen::MatrixXd jac_plus1_y(3, q_dim());
  jac_here = kin_->get_jacobian(link_id_, control_joint_ids_,
                                kin::RotationType::IGNORE, with_base_);
  jac_plus1_x = kin_->get_jacobian(aux_link_ids_[0], control_joint_ids_,
                                   kin::RotationType::IGNORE, with_base_);
  jac_plus1_y = kin_->get_jacobian(aux_link_ids_[1], control_joint_ids_,
                                   kin::RotationType::IGNORE, with_base_);
  Eigen::MatrixXd jac(2, q_dim());
  jac.row(0) = jac_plus1_x.row(2) - jac_here.row(2);
  jac.row(1) = jac_plus1_y.row(2) - jac_here.row(2);
  return {vals, jac};
};

}  // namespace plainmp::constraint
