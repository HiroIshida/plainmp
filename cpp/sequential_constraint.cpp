#include "sequential_constraint.hpp"

namespace cst {

void SequentialCst::add_globally(const ConstraintBase::Ptr& constraint) {
  for (size_t t = 0; t < T_; ++t) {
    this->add_at(constraint, t);
  }
}

void SequentialCst::add_at(const ConstraintBase::Ptr& constraint, size_t t) {
  if (t >= T_) {
    throw std::runtime_error("t is out of range");
  }
  sparsity_pattern_determined_ = false;
  constraints_seq_[t].push_back(constraint);
  cst_dim_ += constraint->cst_dim();
}

void SequentialCst::add_fixed_point_at(const Eigen::VectorXd& q, size_t t) {
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

void SequentialCst::determine_sparsity_pattern() {
  jac_ = SMatrix(cst_dim(), x_dim());
  Eigen::VectorXd x = Eigen::VectorXd::Zero(q_dim() * T_);
  evaluate(x);
  sparsity_pattern_determined_ = true;
}

std::pair<Eigen::VectorXd, SMatrix> SequentialCst::evaluate(
    const Eigen::VectorXd& x) {
  size_t q_dim = this->q_dim();
  Eigen::VectorXd c(cst_dim());
  size_t x_head = 0;
  size_t c_head = 0;
  for (size_t t = 0; t < T_; ++t) {
    std::vector<double> q(x.segment(x_head, q_dim).data(),
                          x.segment(x_head, q_dim).data() + q_dim);
    // we assume that all the constraints share the same kinematic tree
    // thus updating one of the constraints propagates the update to all
    constraints_seq_[t][0]->update_kintree(q);
    for (auto& constraint : constraints_seq_[t]) {
      auto [c_t, J_t] = constraint->evaluate(q);
      c.segment(c_head, constraint->cst_dim()) = c_t;
      // sparse matrix's block is read-only so..
      for (size_t i = 0; i < J_t.rows(); ++i) {
        for (size_t j = 0; j < J_t.cols(); ++j) {
          jac_.coeffRef(c_head + i, x_head + j) = J_t(i, j);
        }
      }
      c_head += constraint->cst_dim();
    }
    x_head += q_dim;
  }

  // evaluate fixed points
  for (size_t t = 0; t < T_; ++t) {
    if (fixed_points_[t].has_value()) {
      c.segment(c_head, q_dim) =
          x.segment(t * q_dim, q_dim) - fixed_points_[t].value();
      for (size_t i = 0; i < q_dim; ++i) {
        jac_.coeffRef(c_head + i, t * q_dim + i) = 1.0;
      }
      c_head += q_dim;
    }
  }

  // evaluate msbox constraint. Note that msbox constraint is pairwise, and not
  // having kinematic tree, so we can evaluate it directly.
  for (size_t t = 0; t < T_ - 1; ++t) {
    Eigen::VectorXd q1 = x.segment(t * q_dim, q_dim);
    Eigen::VectorXd q2 = x.segment((t + 1) * q_dim, q_dim);
    // ||q1 - q2|| <= msbox_width_ (element-wise)
    // equivalent to:
    // q1 - q2 <= msbox_width_ and q2 - q1 <= msbox_width_
    c.segment(c_head, q_dim) = q1 - q2 + msbox_width_.value();
    c.segment(c_head + q_dim, q_dim) = q2 - q1 + msbox_width_.value();

    // fill in the sparse matrix
    for (size_t i = 0; i < q_dim; ++i) {
      jac_.coeffRef(c_head + i, t * q_dim + i) = 1.0;
      jac_.coeffRef(c_head + i, (t + 1) * q_dim + i) = -1.0;
      jac_.coeffRef(c_head + q_dim + i, t * q_dim + i) = -1.0;
      jac_.coeffRef(c_head + q_dim + i, (t + 1) * q_dim + i) = 1.0;
    }
    c_head += q_dim * 2;
  }
  return {c, jac_};
}

}  // namespace cst
