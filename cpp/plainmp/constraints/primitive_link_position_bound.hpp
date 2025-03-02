/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2025 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <Eigen/Dense>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "plainmp/constraints/primitive.hpp"
#include "plainmp/kinematics/kinematics.hpp"

namespace plainmp::constraint {

namespace kin = plainmp::kinematics;

class LinkPositionBoundCst : public IneqConstraintBase {
 public:
  using Ptr = std::shared_ptr<LinkPositionBoundCst>;

  LinkPositionBoundCst(std::shared_ptr<kin::KinematicModel<double>> kin,
                       const std::vector<std::string>& control_joint_names,
                       kin::BaseType base_type,
                       const std::string& link_name,
                       size_t axis,
                       const std::optional<double>& lb,
                       const std::optional<double>& ub);

  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate_dirty() override;
  bool is_valid_dirty() override;
  size_t cst_dim() const override;
  std::string get_name() const override;

 private:
  size_t link_id_;
  size_t axis_;  // Axis index (0: x, 1: y, 2: z).
  std::optional<double> lower_bound_;
  std::optional<double> upper_bound_;
};

}  // namespace plainmp::constraint
