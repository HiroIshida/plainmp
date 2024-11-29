#include "primitive_relative_pose.hpp"

namespace cst {

RelativePoseCst::RelativePoseCst(
    std::shared_ptr<kin::KinematicModel<double>> kin,
    const std::vector<std::string>& control_joint_names,
    bool with_base,
    const std::string& link_name1,
    const std::string& link_name2,
    const Eigen::Vector3d& relative_pose)
    : EqConstraintBase(kin, control_joint_names, with_base),
      link_id2_(kin_->get_link_ids({link_name2})[0]),
      relative_pose_(relative_pose) {
  // TODO: because name is hard-coded, we cannot create two RelativePoseCst...
  auto pose = Transform::Identity();
  pose.trans() = relative_pose;
  size_t link_id1_ = kin_->get_link_ids({link_name1})[0];
  dummy_link_id_ = kin_->add_new_link(link_id1_, pose, true);
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> RelativePoseCst::evaluate_dirty() {
  Eigen::VectorXd vals(cst_dim());
  Eigen::MatrixXd jac(cst_dim(), q_dim());
  const auto& pose_dummy = kin_->get_link_pose(dummy_link_id_);
  const auto& pose2 = kin_->get_link_pose(link_id2_);
  vals.head(3) = pose_dummy.trans() - pose2.trans();
  vals.segment(3, 4) = pose_dummy.quat().coeffs() - pose2.quat().coeffs();
  jac = kin_->get_jacobian(dummy_link_id_, control_joint_ids_,
                           kin::RotationType::XYZW, with_base_) -
        kin_->get_jacobian(link_id2_, control_joint_ids_,
                           kin::RotationType::XYZW, with_base_);
  return {vals, jac};
}

}  // namespace cst
