#include <Eigen/Sparse>
#include <optional>
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
  inline size_t x_dim() const { return q_dim() * T_; }
  inline size_t q_dim() const { return constraints_seq_[0][0]->q_dim(); }
  inline size_t cst_dim() const { return cst_dim_; }

 private:
  size_t T_;
  size_t cst_dim_;
  std::vector<std::vector<ConstraintBase::Ptr>> constraints_seq_;
  std::vector<std::optional<Eigen::VectorXd>> fixed_points_;
  bool finalized_;
  SMatrix jac_;
  std::optional<Eigen::VectorXd> msbox_width_;
};

}  // namespace cst
