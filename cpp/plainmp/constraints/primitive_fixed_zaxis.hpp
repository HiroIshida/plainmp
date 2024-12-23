/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "plainmp/constraints/primitive.hpp"
#include "plainmp/kinematics/kinematics.hpp"

namespace plainmp::constraint {
class FixedZAxisCst : public EqConstraintBase {
 public:
  using Ptr = std::shared_ptr<FixedZAxisCst>;
  FixedZAxisCst(std::shared_ptr<kin::KinematicModel<double>> kin,
                const std::vector<std::string>& control_joint_names,
                kin::BaseType base_type,
                const std::string& link_name);

  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate_dirty() override;
  size_t cst_dim() const override { return 2; }
  std::string get_name() const override { return "FixedZAxisCst"; }

 private:
  size_t link_id_;
  std::vector<size_t> aux_link_ids_;
};

}  // namespace plainmp::constraint
