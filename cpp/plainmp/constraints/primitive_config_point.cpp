/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "plainmp/constraints/primitive_config_point.hpp"
#include "plainmp/kinematics/kinematics.hpp"

namespace plainmp::constraint {

ConfigPointCst::ConfigPointCst(
    std::shared_ptr<kin::KinematicModel<double>> kin,
    const std::vector<std::string>& control_joint_names,
    kin::BaseType base_type,
    const Eigen::VectorXd& q)
    : EqConstraintBase(kin, control_joint_names, base_type), q_(q) {
  size_t dof = control_joint_names.size() +
               (base_type == kin::BaseType::FLOATING) * 6 +
               (base_type == kin::BaseType::PLANAR) * 3;
  if (q.size() != dof) {
    throw std::runtime_error(
        "q must have the same size as the number of control joints");
  }
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> ConfigPointCst::evaluate_dirty() {
  size_t dof = q_dim();
  std::vector<double> q_now_joint_std =
      kin_->get_joint_angles(control_joint_ids_);

  Eigen::VectorXd q_now(dof);
  for (size_t i = 0; i < control_joint_ids_.size(); i++) {
    q_now[i] = q_now_joint_std[i];
  }

  if (base_type_ == kin::BaseType::FLOATING) {
    size_t head = control_joint_ids_.size();
    auto base_pose = kin_->get_base_pose();
    q_now(head) = base_pose.trans().x();
    q_now(head + 1) = base_pose.trans().y();
    q_now(head + 2) = base_pose.trans().z();
    auto base_rpy = base_pose.getRPY();
    q_now(head + 3) = base_rpy.x();
    q_now(head + 4) = base_rpy.y();
    q_now(head + 5) = base_rpy.z();
  }
  if (base_type_ == kin::BaseType::PLANAR) {
    size_t head = control_joint_ids_.size();
    auto base_pose = kin_->get_base_pose();
    q_now(head) = base_pose.trans().x();
    q_now(head + 1) = base_pose.trans().y();
    auto base_rpy = base_pose.getRPY();
    q_now(head + 2) = base_rpy.z();
  }
  return {q_now - q_, Eigen::MatrixXd::Identity(dof, dof)};
}

}  // namespace plainmp::constraint
