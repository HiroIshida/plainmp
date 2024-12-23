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

class RelativePoseCst : public EqConstraintBase {
 public:
  using Ptr = std::shared_ptr<RelativePoseCst>;
  RelativePoseCst(std::shared_ptr<kin::KinematicModel<double>> kin,
                  const std::vector<std::string>& control_joint_names,
                  kin::BaseType base_type,
                  const std::string& link_name1,
                  const std::string& link_name2,
                  const Eigen::Vector3d& relative_pose);
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate_dirty() override;
  size_t cst_dim() const { return 7; }
  std::string get_name() const override { return "RelativePoseCst"; }

 private:
  size_t link_id2_;
  size_t dummy_link_id_;
  Eigen::Vector3d relative_pose_;
};

}  // namespace plainmp::constraint
