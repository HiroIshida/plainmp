#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <memory>
#include <sstream>
#include <utility>
#include "collision/primitive_sdf.hpp"
#include "constraint.hpp"
#include "kinematics/tinyfk.hpp"

namespace cst {

template <typename T, typename Scalar>
class CompositeConstraintBase {
 public:
  using Ptr = std::shared_ptr<CompositeConstraintBase>;
  using Values = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using MatrixDynamic = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  CompositeConstraintBase(std::vector<T> constraints)
      : constraints_(constraints) {
    // all constraints must have the same kinematic chain
    // otherwise, the jacobian will be wrong
    for (auto cst : constraints_) {
      if (cst->kin_ != constraints_.front()->kin_) {
        throw std::runtime_error(
            "All constraints must have the same kinematic chain");
      }
    }
  }

  void update_kintree(const std::vector<Scalar>& q, bool high_accuracy = true) {
    constraints_.front()->update_kintree(q, high_accuracy);
    for (auto& cst : constraints_) {
      cst->post_update_kintree();
    }
  }

  std::pair<Values, MatrixDynamic> evaluate(const std::vector<Scalar>& q) {
    this->update_kintree(q);

    size_t dim = this->cst_dim();
    Values vals(dim);
    MatrixDynamic jac(dim, q_dim());
    size_t head = 0;
    for (const auto& cst : constraints_) {
      size_t dim_local = cst->cst_dim();
      auto [vals_sub, jac_sub] = cst->evaluate_dirty();
      vals.segment(head, dim_local) = vals_sub;
      jac.block(head, 0, dim_local, q_dim()) = jac_sub;
      head += dim_local;
    }
    return {vals, jac};
  }

  size_t q_dim() const { return constraints_.front()->q_dim(); }

  size_t cst_dim() const {
    return std::accumulate(
        constraints_.begin(), constraints_.end(), 0,
        [](size_t sum, const T& cst) { return sum + cst->cst_dim(); });
  }

  std::string to_string() const {
    std::stringstream ss;
    ss << "Composite constraint:" << std::endl;
    ss << "total dim: " << cst_dim() << std::endl;
    for (const auto& cst : constraints_) {
      ss << "  - " << cst->get_name() << ": " << cst->cst_dim() << std::endl;
    }
    return ss.str();
  }

  std::vector<T> constraints_;
};

template <typename Scalar>
class EqCompositeCst
    : public CompositeConstraintBase<typename EqConstraintBase<Scalar>::Ptr,
                                     Scalar> {
 public:
  using Ptr = std::shared_ptr<EqCompositeCst>;
  using CompositeConstraintBase<typename EqConstraintBase<Scalar>::Ptr,
                                Scalar>::CompositeConstraintBase;
  size_t cst_dim() const;
  bool is_equality() const { return true; }
};

template <typename Scalar>
class IneqCompositeCst
    : public CompositeConstraintBase<typename IneqConstraintBase<Scalar>::Ptr,
                                     Scalar> {
 public:
  using Ptr = std::shared_ptr<IneqCompositeCst>;
  using CompositeConstraintBase<typename IneqConstraintBase<Scalar>::Ptr,
                                Scalar>::CompositeConstraintBase;
  bool is_valid(const std::vector<Scalar>& q) {
    this->update_kintree(q, false);
    for (const auto& cst : this->constraints_) {
      if (!cst->is_valid_dirty())
        return false;
    }
    return true;
  }
  size_t cst_dim() const;
  bool is_equality() const { return false; }
};

// explicit instantiation
template class CompositeConstraintBase<EqConstraintBase<double>::Ptr, double>;
template class CompositeConstraintBase<IneqConstraintBase<double>::Ptr, double>;

}  // namespace cst
