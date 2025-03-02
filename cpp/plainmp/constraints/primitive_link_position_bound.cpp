/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2025 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "plainmp/constraints/primitive_link_position_bound.hpp"

namespace plainmp::constraint {

namespace kin = plainmp::kinematics;

LinkPositionBoundCst::LinkPositionBoundCst(std::shared_ptr<kin::KinematicModel<double>> kin,
                                           const std::vector<std::string>& control_joint_names,
                                           kin::BaseType base_type,
                                           const std::string& link_name,
                                           size_t axis,
                                           const std::optional<double>& lb,
                                           const std::optional<double>& ub)
    : IneqConstraintBase(kin, control_joint_names, base_type),
      axis_(axis),
      lower_bound_(lb),
      upper_bound_(ub) {
  if (axis_ > 2) {
    throw std::runtime_error("Axis index must be 0 (x), 1 (y), or 2 (z).");
  }
  if (!lower_bound_.has_value() && !upper_bound_.has_value()) {
    throw std::runtime_error("At least one of lower or upper bound must be provided.");
  }
  auto link_ids = kin_->get_link_ids({link_name});
  if (link_ids.empty()) {
    throw std::runtime_error("Invalid link name: " + link_name);
  }
  link_id_ = link_ids[0];
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> LinkPositionBoundCst::evaluate_dirty() {
  const auto& pose = kin_->get_link_pose(link_id_);
  const auto& pos = pose.trans();

  size_t n_constraints =
      (lower_bound_.has_value() ? 1 : 0) + (upper_bound_.has_value() ? 1 : 0);
  Eigen::VectorXd vals(n_constraints);
  Eigen::MatrixXd jac(n_constraints, q_dim());

  Eigen::MatrixXd pos_jac = kin_->get_jacobian(
      link_id_, control_joint_ids_, kin::RotationType::IGNORE, base_type_);
  size_t row = 0;
  if (lower_bound_.has_value()) {
    vals(row) = pos[axis_] - lower_bound_.value();
    jac.row(row) = pos_jac.row(axis_);
    row++;
  }
  if (upper_bound_.has_value()) {
    vals(row) = upper_bound_.value() - pos[axis_];
    jac.row(row) = -pos_jac.row(axis_);
    row++;
  }
  return {vals, jac};
}

bool LinkPositionBoundCst::is_valid_dirty() {
  const auto& pose = kin_->get_link_pose(link_id_);
  const auto& position = pose.trans();
  if (lower_bound_.has_value() && position[axis_] < lower_bound_.value()) {
    return false;
  }
  if (upper_bound_.has_value() && position[axis_] > upper_bound_.value()) {
    return false;
  }
  return true;
}

size_t LinkPositionBoundCst::cst_dim() const {
  return (lower_bound_.has_value() ? 1 : 0) +
         (upper_bound_.has_value() ? 1 : 0);
}

std::string LinkPositionBoundCst::get_name() const {
  return "LinkPositionBoundCst";
}

}  // namespace plainmp::constraint
