#include "primitive.hpp"
namespace constraint {
struct AppliedForceSpec {
  std::string link_name;
  double force;  // currently only z-axis force (minus direction) is supported
};

class ComInPolytopeCst : public IneqConstraintBase {
 public:
  using Ptr = std::shared_ptr<ComInPolytopeCst>;
  ComInPolytopeCst(std::shared_ptr<kin::KinematicModel<double>> kin,
                   const std::vector<std::string>& control_joint_names,
                   bool with_base,
                   BoxSDF::Ptr polytope_sdf,
                   const std::vector<AppliedForceSpec> applied_forces);
  bool is_valid_dirty() override;
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate_dirty() override;
  size_t cst_dim() const { return 1; }
  std::string get_name() const override { return "ComInPolytopeCst"; }

 private:
  BoxSDF::Ptr polytope_sdf_;
  std::vector<size_t> force_link_ids_;
  std::vector<double> applied_force_values_;
};

};  // namespace constraint
