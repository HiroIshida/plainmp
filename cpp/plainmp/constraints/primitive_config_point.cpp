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
  Eigen::VectorXd vals(dof);
  Eigen::MatrixXd jac(dof, dof);
  evaluate_dirty_into(vals, jac);
  return {vals, jac};
}

void ConfigPointCst::evaluate_dirty_into(Eigen::Ref<Eigen::VectorXd> vals,
                                         Eigen::Ref<Eigen::MatrixXd> jac) {
  std::vector<double> q_now_joint_std =
      kin_->get_joint_angles(control_joint_ids_);

  for (size_t i = 0; i < control_joint_ids_.size(); i++) {
    vals[i] = q_now_joint_std[i];
  }

  if (base_type_ == kin::BaseType::FLOATING) {
    size_t head = control_joint_ids_.size();
    auto base_pose = kin_->get_base_pose();
    vals(head) = base_pose.trans().x();
    vals(head + 1) = base_pose.trans().y();
    vals(head + 2) = base_pose.trans().z();
    auto base_rpy = base_pose.getRPY();
    vals(head + 3) = base_rpy.x();
    vals(head + 4) = base_rpy.y();
    vals(head + 5) = base_rpy.z();
  }
  if (base_type_ == kin::BaseType::PLANAR) {
    size_t head = control_joint_ids_.size();
    auto base_pose = kin_->get_base_pose();
    vals(head) = base_pose.trans().x();
    vals(head + 1) = base_pose.trans().y();
    auto base_rpy = base_pose.getRPY();
    vals(head + 2) = base_rpy.z();
  }
  vals -= q_;
  jac.setIdentity();
}

}  // namespace plainmp::constraint
