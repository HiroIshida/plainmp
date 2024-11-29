#include "primitive_config_point.hpp"

namespace constraint {

ConfigPointCst::ConfigPointCst(
    std::shared_ptr<kin::KinematicModel<double>> kin,
    const std::vector<std::string>& control_joint_names,
    bool with_base,
    const Eigen::VectorXd& q)
    : EqConstraintBase(kin, control_joint_names, with_base), q_(q) {
  size_t dof = control_joint_names.size() + (with_base ? 6 : 0);
  if (q.size() != dof) {
    throw std::runtime_error(
        "q must have the same size as the number of control joints");
  }
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> ConfigPointCst::evaluate_dirty() {
  size_t dof = q_dim();
  std::vector<double> q_now_joint_std =
      kin_->get_joint_angles(control_joint_ids_);

  Eigen::VectorXd q_now(dof);
  for (size_t i = 0; i < control_joint_ids_.size(); i++) {
    q_now[i] = q_now_joint_std[i];
  }
  if (with_base_) {
    size_t head = control_joint_ids_.size();
    auto base_pose = kin_->get_base_pose();
    q_now(head) = base_pose.trans().x();
    q_now(head + 1) = base_pose.trans().y();
    q_now(head + 2) = base_pose.trans().z();
    auto base_rpy = base_pose.getRPY();
    q_now(head + 3) = base_rpy.x();
    q_now(head + 4) = base_rpy.y();
    q_now(head + 5) = base_rpy.z();
  }
  return {q_now - q_, Eigen::MatrixXd::Identity(dof, dof)};
}

}  // namespace constraint
