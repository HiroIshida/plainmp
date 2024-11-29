#include "primitive.hpp"

namespace plainmp::constraint {
class FixedZAxisCst : public EqConstraintBase {
 public:
  using Ptr = std::shared_ptr<FixedZAxisCst>;
  FixedZAxisCst(std::shared_ptr<kin::KinematicModel<double>> kin,
                const std::vector<std::string>& control_joint_names,
                bool with_base,
                const std::string& link_name);

  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate_dirty() override;
  size_t cst_dim() const override { return 2; }
  std::string get_name() const override { return "FixedZAxisCst"; }

 private:
  size_t link_id_;
  std::vector<size_t> aux_link_ids_;
};

}  // namespace plainmp::constraint
