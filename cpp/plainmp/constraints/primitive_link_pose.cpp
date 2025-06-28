/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "plainmp/constraints/primitive_link_pose.hpp"
#include "plainmp/kinematics/kinematics.hpp"

namespace plainmp::constraint {

LinkPoseCst::LinkPoseCst(std::shared_ptr<kin::KinematicModel<double>> kin,
                         const std::vector<std::string>& control_joint_names,
                         kin::BaseType base_type,
                         const std::vector<std::string>& link_names,
                         const std::vector<Eigen::VectorXd>& poses)
    : EqConstraintBase(kin, control_joint_names, base_type),
      link_ids_(kin_->get_link_ids(link_names)),
      poses_(poses) {
  for (auto& pose : poses_) {
    if (pose.size() != 3 && pose.size() != 6 && pose.size() != 7) {
      throw std::runtime_error("All poses must be 3 or 6 or 7 dimensional");
    }
  }
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> LinkPoseCst::evaluate_dirty() {
  Eigen::VectorXd vals(cst_dim());
  Eigen::MatrixXd jac(cst_dim(), q_dim());
  size_t head = 0;
  for (size_t i = 0; i < link_ids_.size(); i++) {
    const auto& pose = kin_->get_link_pose(link_ids_[i]);

    if (poses_[i].size() == 3) {
      vals.segment(head, 3) = pose.trans() - poses_[i];
      jac.block(head, 0, 3, q_dim()) =
          kin_->get_jacobian(link_ids_[i], control_joint_ids_,
                             kin::RotationType::IGNORE, base_type_);
      head += 3;
    } else if (poses_[i].size() == 6) {
      vals.segment(head, 3) = pose.trans() - poses_[i].head(3);
      vals.segment(head + 3, 3) = pose.getRPY() - poses_[i].tail(3);
      jac.block(head, 0, 6, q_dim()) = kin_->get_jacobian(
          link_ids_[i], control_joint_ids_, kin::RotationType::RPY, base_type_);
      head += 6;
    } else {
      vals.segment(head, 3) = pose.trans() - poses_[i].head(3);
      vals.segment(head + 3, 4) = pose.quat().coeffs() - poses_[i].tail(4);
      jac.block(head, 0, 7, q_dim()) =
          kin_->get_jacobian(link_ids_[i], control_joint_ids_,
                             kin::RotationType::XYZW, base_type_);
      head += 7;
    }
  }
  return {vals, jac};
}

size_t LinkPoseCst::cst_dim() const {
  size_t dim = 0;
  for (auto& pose : poses_) {
    dim += pose.size();
  }
  return dim;
}

std::vector<Eigen::VectorXd> LinkPoseCst::get_desired_poses() const {
  return poses_;
}

}  // namespace plainmp::constraint
