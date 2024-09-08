#include <Eigen/Sparse>
#include <optional>
#include "constraint.hpp"

namespace cst {

template <typename Scalar>
class SequentialCst {
 public:
  using Ptr = std::shared_ptr<SequentialCst>;
  using SMatrix = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using Values = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using MatrixDynamic = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  SequentialCst(size_t T, size_t q_dim)
      : T_(T),
        cst_dim_(0),
        q_dim_(q_dim),
        constraints_seq_(T),
        fixed_points_(T, std::nullopt),
        finalized_(false),
        jac_(),
        msbox_width_(std::nullopt) {}
  void add_globally(const typename ConstraintBase<Scalar>::Ptr& constraint);
  void add_at(const typename ConstraintBase<Scalar>::Ptr& constraint, size_t t);
  void add_fixed_point_at(const Values& q, size_t t);
  void add_motion_step_box_constraint(const Values& box_width);
  void finalize();
  std::pair<Values, SMatrix> evaluate(const Values& x);
  inline size_t x_dim() const { return q_dim_ * T_; }
  inline size_t cst_dim() const { return cst_dim_; }
  std::string to_string() const;

 private:
  size_t T_;
  size_t cst_dim_;
  size_t q_dim_;
  std::vector<std::vector<typename ConstraintBase<Scalar>::Ptr>>
      constraints_seq_;
  std::vector<std::optional<Values>> fixed_points_;
  bool finalized_;
  SMatrix jac_;
  std::optional<Values> msbox_width_;
};

}  // namespace cst
