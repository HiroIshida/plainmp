#include "primitive_com_in_polytope.hpp"

namespace plainmp::constraint {
ComInPolytopeCst::ComInPolytopeCst(
    std::shared_ptr<kin::KinematicModel<double>> kin,
    const std::vector<std::string>& control_joint_names,
    bool with_base,
    plainmp::collision::BoxSDF::Ptr polytope_sdf,
    const std::vector<AppliedForceSpec> applied_forces)
    : IneqConstraintBase(kin, control_joint_names, with_base),
      polytope_sdf_(polytope_sdf) {
  auto w = polytope_sdf_->get_width();
  w[2] = 1000;  // adhoc to represent infinite height
  polytope_sdf_->set_width(w);

  auto force_link_names = std::vector<std::string>();
  for (auto& force : applied_forces) {
    force_link_names.push_back(force.link_name);
    applied_force_values_.push_back(force.force);
  }
  force_link_ids_ = kin_->get_link_ids(force_link_names);
}

bool ComInPolytopeCst::is_valid_dirty() {
  // COPIED from evaluate() >> START
  auto com = kin_->get_com();
  if (force_link_ids_.size() > 0) {
    double vertical_force_sum = 1.0;  // 1.0 for normalized self
    for (size_t j = 0; j < force_link_ids_.size(); ++j) {
      double force = applied_force_values_[j] / kin_->total_mass_;
      vertical_force_sum += force;
      const auto& pose = kin_->get_link_pose(force_link_ids_[j]);
      com += force * pose.trans();
    }
    com /= vertical_force_sum;
  }
  // COPIED from evaluate() >> END
  return polytope_sdf_->evaluate(com) < 0;
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> ComInPolytopeCst::evaluate_dirty() {
  Eigen::VectorXd vals(cst_dim());
  Eigen::MatrixXd jac(cst_dim(), q_dim());

  auto com = kin_->get_com();
  auto com_jaco = kin_->get_com_jacobian(control_joint_ids_, q_dim());
  if (force_link_ids_.size() > 0) {
    double vertical_force_sum = 1.0;  // 1.0 for normalized self
    for (size_t j = 0; j < force_link_ids_.size(); ++j) {
      double force = applied_force_values_[j] / kin_->total_mass_;
      vertical_force_sum += force;
      const auto& pose = kin_->get_link_pose(force_link_ids_[j]);
      com += force * pose.trans();

      com_jaco += kin_->get_jacobian(force_link_ids_[j], control_joint_ids_,
                                     kin::RotationType::IGNORE, with_base_) *
                  force;
    }
    double inv = 1.0 / vertical_force_sum;
    com *= inv;
    com_jaco *= inv;
  }
  double val = -polytope_sdf_->evaluate(com);
  vals[0] = val;

  Eigen::Vector3d grad;
  for (size_t i = 0; i < 3; i++) {
    Eigen::Vector3d perturbed_com = com;
    perturbed_com[i] += 1e-6;
    double val_perturbed = -polytope_sdf_->evaluate(perturbed_com);
    grad[i] = (val_perturbed - val) / 1e-6;
  }
  jac.row(0) = com_jaco.transpose() * grad;

  return {vals, jac};
};

}  // namespace plainmp::constraint
