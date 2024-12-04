/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include <Eigen/Sparse>
#include <optional>
#include "plainmp/constraints/primitive.hpp"

namespace plainmp::constraint {

using SMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor>;

class SequentialCst {
 public:
  using Ptr = std::shared_ptr<SequentialCst>;
  SequentialCst(size_t T, size_t q_dim)
      : T_(T),
        cst_dim_(0),
        q_dim_(q_dim),
        constraints_seq_(T),
        fixed_points_(T, std::nullopt),
        finalized_(false),
        jac_(),
        msbox_width_(std::nullopt) {}
  void add_globally(const ConstraintBase::Ptr& constraint);
  void add_at(const ConstraintBase::Ptr& constraint, size_t t);
  void add_fixed_point_at(const Eigen::VectorXd& q, size_t t);
  void add_motion_step_box_constraint(const Eigen::VectorXd& box_width);
  void finalize();
  std::pair<Eigen::VectorXd, SMatrix> evaluate(const Eigen::VectorXd& x);
  inline size_t x_dim() const { return q_dim_ * T_; }
  inline size_t cst_dim() const { return cst_dim_; }
  std::string to_string() const;

 private:
  size_t T_;
  size_t cst_dim_;
  size_t q_dim_;
  std::vector<std::vector<ConstraintBase::Ptr>> constraints_seq_;
  std::vector<std::optional<Eigen::VectorXd>> fixed_points_;
  bool finalized_;
  SMatrix jac_;
  std::optional<Eigen::VectorXd> msbox_width_;
};

}  // namespace plainmp::constraint
