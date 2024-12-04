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

namespace plainmp::constraint {

class ConfigPointCst : public EqConstraintBase {
 public:
  using Ptr = std::shared_ptr<ConfigPointCst>;
  ConfigPointCst(std::shared_ptr<kin::KinematicModel<double>> kin,
                 const std::vector<std::string>& control_joint_names,
                 bool with_base,
                 const Eigen::VectorXd& q);
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate_dirty() override;
  size_t cst_dim() const { return q_.size(); }
  std::string get_name() const override { return "ConfigPointCst"; }

 private:
  Eigen::VectorXd q_;
};

}  // namespace plainmp::constraint
