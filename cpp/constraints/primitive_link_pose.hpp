/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "primitive.hpp"

namespace plainmp::constraint {

class LinkPoseCst : public EqConstraintBase {
 public:
  using Ptr = std::shared_ptr<LinkPoseCst>;
  LinkPoseCst(std::shared_ptr<kin::KinematicModel<double>> kin,
              const std::vector<std::string>& control_joint_names,
              bool with_base,
              const std::vector<std::string>& link_names,
              const std::vector<Eigen::VectorXd>& poses);

  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate_dirty() override;
  size_t cst_dim() const;
  std::string get_name() const override { return "LinkPoseCst"; }

 private:
  std::vector<size_t> link_ids_;
  std::vector<Eigen::VectorXd> poses_;
};

}  // namespace plainmp::constraint
