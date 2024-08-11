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

void SequentialCst::determine_sparsity_pattern() {
  jac_ = SMatrix(cst_dim(), x_dim());
  std::vector<double> x(q_dim() * T_, 0);
  evaluate(x);
  sparsity_pattern_determined_ = true;
}

std::pair<Eigen::VectorXd, SMatrix> SequentialCst::evaluate(
    const std::vector<double>& x) {
  Eigen::VectorXd c(cst_dim());
  size_t x_head = 0;
  size_t c_head = 0;
  for (size_t t = 0; t < T_; ++t) {
    std::vector<double> q(x.begin() + x_head, x.begin() + x_head + q_dim());
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
    x_head += q_dim();
  }
  return {c, jac_};
}

}  // namespace cst
