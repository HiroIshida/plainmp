/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "plainmp/constraints/sequential_constraint.hpp"

namespace plainmp::constraint {

void SequentialCst::add_globally(const ConstraintBase::Ptr& constraint) {
  for (size_t t = 0; t < T_; ++t) {
    this->add_at(constraint, t);
  }
}

void SequentialCst::add_at(const ConstraintBase::Ptr& constraint, size_t t) {
  if (t >= T_) {
    throw std::runtime_error("t is out of range");
  }
  finalized_ = false;
  constraints_seq_[t].push_back(constraint);
  cst_dim_ += constraint->cst_dim();
}

void SequentialCst::add_fixed_point_at(const Eigen::VectorXd& q, size_t t) {
  if (t >= T_) {
    throw std::runtime_error("t is out of range");
  }
  fixed_points_[t] = q;
  cst_dim_ += q.size();
}

void SequentialCst::add_motion_step_box_constraint(
    const Eigen::VectorXd& box_width) {
  for (size_t t = 0; t < T_ - 1; ++t) {
    cst_dim_ += box_width.size() * 2;  // 2 for lower and upper bounds
  }
  msbox_width_ = box_width;
}

void SequentialCst::finalize() {
  // remove constraints at time t if fixed_points are set
  for (size_t t = 0; t < T_; ++t) {
    if (fixed_points_[t].has_value()) {
      size_t cst_dim_at_t = 0;
      for (auto& constraint : constraints_seq_[t]) {
        cst_dim_at_t += constraint->cst_dim();
      }
      cst_dim_ -= cst_dim_at_t;
      constraints_seq_[t].clear();
    }
  }

  jac_ = SMatrix(cst_dim(), x_dim());
  finalized_ = true;
}

std::pair<Eigen::VectorXd, SMatrix> SequentialCst::evaluate(
    const Eigen::VectorXd& x) {
  Eigen::VectorXd c(cst_dim());
  size_t x_head = 0;
  size_t c_head = 0;
  for (size_t t = 0; t < T_; ++t) {
    const auto q = x.segment(x_head, q_dim_);
    // we assume that all the constraints share the same kinematic tree
    // thus updating one of the constraints propagates the update to all
    if (constraints_seq_[t].size() == 0) {
      continue;
    }

    constraints_seq_[t][0]->update_kintree(q);
    for (auto& constraint : constraints_seq_[t]) {
      auto [c_t, J_t] = constraint->evaluate(q);
      c.segment(c_head, constraint->cst_dim()) = c_t;
      // sparse matrix's block is read-only so..
      for (size_t i = 0; i < J_t.rows(); ++i) {
        for (size_t j = 0; j < J_t.cols(); ++j) {
          // 1e-17 to notifiy the sparsity pattern
          jac_.coeffRef(c_head + i, x_head + j) = J_t(i, j) + 1e-12;
        }
      }
      c_head += constraint->cst_dim();
    }
    x_head += q_dim_;
  }

  // evaluate fixed points
  for (size_t t = 0; t < T_; ++t) {
    if (fixed_points_[t].has_value()) {
      c.segment(c_head, q_dim_) =
          x.segment(t * q_dim_, q_dim_) - fixed_points_[t].value();
      for (size_t i = 0; i < q_dim_; ++i) {
        jac_.coeffRef(c_head + i, t * q_dim_ + i) = 1.0;
      }
      c_head += q_dim_;
    }
  }

  // evaluate msbox constraint. Note that msbox constraint is pairwise, and not
  // having kinematic tree, so we can evaluate it directly.
  if (msbox_width_.has_value()) {
    for (size_t t = 0; t < T_ - 1; ++t) {
      Eigen::VectorXd q1 = x.segment(t * q_dim_, q_dim_);
      Eigen::VectorXd q2 = x.segment((t + 1) * q_dim_, q_dim_);
      // ||q1 - q2|| <= msbox_width_ (element-wise)
      // equivalent to:
      // q1 - q2 <= msbox_width_ and q2 - q1 <= msbox_width_
      c.segment(c_head, q_dim_) = q1 - q2 + msbox_width_.value();
      c.segment(c_head + q_dim_, q_dim_) = q2 - q1 + msbox_width_.value();

      // fill in the sparse matrix
      for (size_t i = 0; i < q_dim_; ++i) {
        jac_.coeffRef(c_head + i, t * q_dim_ + i) = 1.0;
        jac_.coeffRef(c_head + i, (t + 1) * q_dim_ + i) = -1.0;
        jac_.coeffRef(c_head + q_dim_ + i, t * q_dim_ + i) = -1.0;
        jac_.coeffRef(c_head + q_dim_ + i, (t + 1) * q_dim_ + i) = 1.0;
      }
      c_head += q_dim_ * 2;
    }
  }
  return {c, jac_};
}

std::string SequentialCst::to_string() const {
  // clang-format off
  std::stringstream ss;
  ss << "Sequential constraint:" << std::endl;
  ss << "total dim: " << cst_dim() << std::endl;
  for (size_t t = 0; t < T_; ++t) {
    ss << "  - time " << t << std::endl;
    for (const auto& cst : constraints_seq_[t]) {
      ss << "    - " << cst->get_name() << ": " << cst->cst_dim() << std::endl;
    }
    if (fixed_points_[t].has_value()) {
      ss << "    - fixed point"
         << ": " << q_dim_ << std::endl;
    }
  }
  if (msbox_width_.has_value()) {
    ss << "  - motion step box constraint"
       << "  : " << q_dim_ * 2 * (T_ - 1) << std::endl;
  }
  // clang-format on
  return ss.str();
}

}  // namespace plainmp::constraint
