/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "plainmp/collision/primitive_sdf.hpp"
#include "plainmp/constraints/primitive.hpp"
namespace plainmp::constraint {
struct AppliedForceSpec {
  std::string link_name;
  double force;  // currently only z-axis force (minus direction) is supported
};

class ComInPolytopeCst : public IneqConstraintBase {
 public:
  using Ptr = std::shared_ptr<ComInPolytopeCst>;
  ComInPolytopeCst(std::shared_ptr<kin::KinematicModel<double>> kin,
                   const std::vector<std::string>& control_joint_names,
                   kin::BaseType base_type,
                   plainmp::collision::BoxSDF::Ptr polytope_sdf,
                   const std::vector<AppliedForceSpec> applied_forces);
  bool is_valid_dirty() override;
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate_dirty() override;
  size_t cst_dim() const { return 1; }
  std::string get_name() const override { return "ComInPolytopeCst"; }

 private:
  plainmp::collision::BoxSDF::Ptr polytope_sdf_;
  std::vector<size_t> force_link_ids_;
  std::vector<double> applied_force_values_;
};

};  // namespace plainmp::constraint
