#include "primitive.hpp"

namespace cst {

class ConfigPointCst : public EqConstraintBase {
 public:
  using Ptr = std::shared_ptr<ConfigPointCst>;
  ConfigPointCst(std::shared_ptr<kin::KinematicModel<double>> kin,
                 const std::vector<std::string>& control_joint_names,
                 bool with_base,
                 const Eigen::VectorXd& q);
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate_dirty() override;
  size_t cst_dim() const { return q_.size(); }
  std::string get_name() const override { return "ConfigPointCst"; }

 private:
  Eigen::VectorXd q_;
};

}  // namespace cst
