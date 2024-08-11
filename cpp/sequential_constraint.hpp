#include <Eigen/Sparse>
#include "constraint.hpp"

namespace cst {

using SMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor>;

class SequentialCst {
 public:
  using Ptr = std::shared_ptr<SequentialCst>;
  SequentialCst(size_t T)
      : T_(T),
        cst_dim_(0),
        constraints_seq_(T),
        sparsity_pattern_determined_(false),
        jac_() {}

  void add_globally(const ConstraintBase::Ptr& constraint);
  void add_at(const ConstraintBase::Ptr& constraint, size_t t);
  void determine_sparsity_pattern();
  std::pair<Eigen::VectorXd, SMatrix> evaluate(const std::vector<double>& x);
  inline size_t x_dim() const { return q_dim() * T_; }
  inline size_t q_dim() const { return constraints_seq_[0][0]->q_dim(); }
  inline size_t cst_dim() const { return cst_dim_; }

 private:
  size_t T_;
  size_t cst_dim_;
  std::vector<std::vector<ConstraintBase::Ptr>> constraints_seq_;
  bool sparsity_pattern_determined_;
  SMatrix jac_;
};

}  // namespace cst
