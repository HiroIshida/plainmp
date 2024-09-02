#include "tinyfk.hpp"
#include "urdf_model/pose.h"
#include <Eigen/Dense>
#include <cmath>
#include <stack>
#include <stdexcept>

namespace tinyfk {

Eigen::Vector3d rpy_derivative(const Eigen::Vector3d &rpy, const Eigen::Vector3d &axis) {
  Eigen::Vector3d drpy_dt;
  double a2 = -rpy.y();
  double a3 = -rpy.z();
  drpy_dt.x() = cos(a3) / cos(a2) * axis.x ()- sin(a3) / cos(a2) * axis.y();
  drpy_dt.y() = sin(a3) * axis.x() + cos(a3) * axis.y();
  drpy_dt.z() = -cos(a3) * sin(a2) / cos(a2) * axis.x() +
              sin(a3) * sin(a2) / cos(a2) * axis.y() + axis.z();
  return drpy_dt;
}

Eigen::Quaterniond q_derivative(const Eigen::Quaterniond &q, const Eigen::Vector3d &omega) {
  const double dxdt = 0.5 * (omega.z() * q.y() - omega.y() * q.z() + omega.x() * q.w());
  const double dydt = 0.5 * (-omega.z() * q.x() + 0 * q.y() + omega.x() * q.z() + omega.y() * q.w());
  const double dzdt = 0.5 * (omega.y() * q.x() - omega.x() * q.y() + 0 * q.z() + omega.z() * q.w());
  const double dwdt = 0.5 * (-omega.x() * q.x() - omega.y() * q.y() - omega.z() * q.z() + 0 * q.w());
  return Eigen::Quaterniond(-dwdt, dxdt, dydt, dzdt);
}

void KinematicModel::build_cache_until(size_t link_id) const
{
  if(links_[link_id]->consider_rotation) {
    this->build_cache_until_inner(link_id);
  } else {
    // TODO: we should remove this!
    auto hlink = links_[link_id];
    auto plink = hlink->getParent();
    auto pjoint = hlink->parent_joint;
    if(!transform_cache_.is_cached(plink->id)) {
      build_cache_until_inner(plink->id);
    }
    Transform& tf_rlink_to_plink = transform_cache_.data_[plink->id];
    auto&& rotmat = tf_rlink_to_plink.quat().toRotationMatrix();
    Eigen::Vector3d&& pos = tf_rlink_to_plink.trans() + rotmat * pjoint->parent_to_joint_origin_transform.trans();
    // HACK: we want to update the only position part
    // thus, we commented out the private: and directly access the data
    transform_cache_.cache_predicate_vector_[link_id] = true;
    transform_cache_.data_[link_id].trans() = std::move(pos);
  }
}

void KinematicModel::build_cache_until_inner(size_t hlink_id) const {
  std::array<size_t, 64> id_stack_like;  // 64 is enough for almost all cases
  size_t idx = 0;
  while(!transform_cache_.is_cached(hlink_id)) {
    id_stack_like[idx++] = hlink_id;
    hlink_id = link_parent_link_ids_[hlink_id];
  }

  Transform tf_rlink_to_plink = transform_cache_.data_[hlink_id];
  while(idx > 0) {
    size_t hid = id_stack_like[--idx];
    Transform tf_rlink_to_hlink = tf_rlink_to_plink.quat_identity_sensitive_mul(tf_plink_to_hlink_cache_[hid]);
    transform_cache_.set_cache(hid, tf_rlink_to_hlink);
    tf_rlink_to_plink = std::move(tf_rlink_to_hlink);
  }
}

Eigen::MatrixXd
KinematicModel::get_jacobian(size_t elink_id,
                             const std::vector<size_t> &joint_ids,
                             RotationType rot_type, bool with_base) {
  const size_t dim_jacobi = 3 + (rot_type == RotationType::RPY) * 3 +
                            (rot_type == RotationType::XYZW) * 4;
  const int dim_dof = joint_ids.size() + (with_base ? 6 : 0);

  const auto& tf_rlink_to_elink = get_link_pose(elink_id);
  auto &epos = tf_rlink_to_elink.trans();
  auto &erot = tf_rlink_to_elink.quat();

  Eigen::Vector3d erpy;
  Eigen::Quaterniond erot_inverse;
  if (rot_type == RotationType::RPY) {
    erpy = tf_rlink_to_elink.getRPY();
  }
  if (rot_type == RotationType::XYZW) {
    erot_inverse = erot.inverse();
  }

  // Jacobian computation
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(dim_jacobi, dim_dof);

  for (size_t i = 0; i < joint_ids.size(); i++) {
    int jid = joint_ids[i];
    if (rptable_.isRelevant(elink_id, jid)) {
      const urdf::JointSharedPtr &hjoint = joints_[jid];
      size_t type = hjoint->type;
      urdf::LinkSharedPtr clink =
          hjoint->getChildLink(); // rotation of clink and hlink is same. so
                                  // clink is ok.

      const auto& tf_rlink_to_clink = get_link_pose(clink->id);

      auto &crot = tf_rlink_to_clink.quat();
      auto &&world_axis = crot * hjoint->axis; // axis w.r.t root link
      Eigen::Vector3d dpos;
      if (type == urdf::Joint::PRISMATIC) {
        dpos = world_axis;
      } else { // revolute or continuous
        auto &cpos = tf_rlink_to_clink.trans();
        auto vec_clink_to_elink = epos - cpos;
        dpos = world_axis.cross(vec_clink_to_elink);
      }
      jacobian.block<3, 1>(0, i) = dpos;
      if (type == urdf::Joint::PRISMATIC) {
        // jacobian for rotation is all zero
      } else {

        if (rot_type == RotationType::RPY) { // (compute rpy jacobian)
          auto drpy_dt = rpy_derivative(erpy, world_axis);
          jacobian.block<3, 1>(3, i) = drpy_dt;
        }

        if (rot_type == RotationType::XYZW) { // (compute quat jacobian)
          auto dq_dt = q_derivative(erot_inverse, world_axis);
          jacobian.block<4, 1>(3, i) = dq_dt.coeffs();
        }
      }
    }
  }

  Transform tf_rlink_to_blink, tf_blink_to_rlink, tf_blink_to_elink;
  Eigen::Vector3d rpy_rlink_to_blink;
  if (with_base) {
    tf_rlink_to_blink = get_link_pose(root_link_id_);
    tf_blink_to_rlink = tf_rlink_to_blink.getInverse();
    rpy_rlink_to_blink = tf_rlink_to_blink.getRPY();
    tf_blink_to_elink = tf_blink_to_rlink * tf_rlink_to_elink;
  }

  if (with_base) {
    const size_t n_joint = joint_ids.size();
    jacobian(0, n_joint + 0) = 1.0;
    jacobian(1, n_joint + 1) = 1.0;
    jacobian(2, n_joint + 2) = 1.0;

    // we resort to numerical method to base pose jacobian (just because I don't
    // have time)
    // TODO(HiroIshida): compute using analytical method.
    constexpr double eps = 1e-7;
    for (size_t rpy_idx = 0; rpy_idx < 3; rpy_idx++) {
      const size_t idx_col = n_joint + 3 + rpy_idx;

      auto rpy_tweaked = rpy_rlink_to_blink;
      rpy_tweaked[rpy_idx] += eps;

      Transform tf_rlink_to_blink_tweaked = tf_rlink_to_blink;
      tf_rlink_to_blink_tweaked.setQuaternionFromRPY(rpy_tweaked);
      Transform tf_rlink_to_elink_tweaked = tf_rlink_to_blink_tweaked * tf_blink_to_elink;
      auto pose_out = tf_rlink_to_elink_tweaked;

      const auto pos_diff = pose_out.trans() - tf_rlink_to_elink.trans();
      jacobian.block<3, 1>(0, idx_col) = pos_diff / eps;
      if (rot_type == RotationType::RPY) {
        auto erpy_tweaked = pose_out.getRPY();
        jacobian.block<3, 1>(3, idx_col) = (erpy_tweaked - erpy) / eps;
      }
      if (rot_type == RotationType::XYZW) {
        // jacobian.block<4, 1>(3, idx_col) = (pose_out.q.coeffs() - erot).toEigen() / eps;
        jacobian.block<4, 1>(3, idx_col) = (pose_out.quat().coeffs() - erot.coeffs()) / eps;
      }
    }
  }
  return jacobian;
}


Eigen::MatrixXd KinematicModel::get_attached_point_jacobian(
        size_t plink_id,
        Eigen::Vector3d apoint_global_pos,
        const std::vector<size_t>& joint_ids,
        bool with_base){
  const int dim_dof = joint_ids.size() + (with_base ? 6 : 0);
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(3, dim_dof);

  // NOTE: the following logic is copied from get_jacobian()
  for (size_t i = 0; i < joint_ids.size(); i++) {
    int jid = joint_ids[i];
    if (rptable_.isRelevant(plink_id, jid)) {
      const urdf::JointSharedPtr &hjoint = joints_[jid];
      size_t type = hjoint->type;
      urdf::LinkSharedPtr clink =
          hjoint->getChildLink(); // rotation of clink and hlink is same. so
                                  // clink is ok.
      const auto& tf_rlink_to_clink = get_link_pose(clink->id);
      auto &crot = tf_rlink_to_clink.quat();
      auto &&world_axis = crot * hjoint->axis; // axis w.r.t root link

      Eigen::Vector3d dpos;
      if (type == urdf::Joint::PRISMATIC) {
        dpos = world_axis;
      } else { // revolute or continuous
        auto &cpos = tf_rlink_to_clink.trans();
        auto vec_clink_to_elink = apoint_global_pos - cpos;
        dpos = world_axis.cross(vec_clink_to_elink);
      }
      jacobian.block<3, 1>(0, i) = dpos;
    }
  }

  // NOTE: the following logic is copied from get_jacobian()
  Transform tf_rlink_to_elink = Transform::Identity();
  tf_rlink_to_elink.trans() = apoint_global_pos;
  Transform tf_rlink_to_blink, tf_blink_to_rlink, tf_blink_to_elink;
  Eigen::Vector3d rpy_rlink_to_blink;
  if (with_base) {
    tf_rlink_to_blink = get_link_pose(root_link_id_);
    tf_blink_to_rlink = tf_rlink_to_blink.getInverse();
    rpy_rlink_to_blink = tf_rlink_to_blink.getRPY();
    tf_blink_to_elink = tf_blink_to_rlink * tf_rlink_to_elink;
  }

  if (with_base) {
    const size_t n_joint = joint_ids.size();
    jacobian(0, n_joint + 0) = 1.0;
    jacobian(1, n_joint + 1) = 1.0;
    jacobian(2, n_joint + 2) = 1.0;

    // we resort to numerical method to base pose jacobian (just because I don't
    // have time)
    // TODO(HiroIshida): compute using analytical method.
    constexpr double eps = 1e-7;
    for (size_t rpy_idx = 0; rpy_idx < 3; rpy_idx++) {
      const size_t idx_col = n_joint + 3 + rpy_idx;

      auto rpy_tweaked = rpy_rlink_to_blink;
      rpy_tweaked[rpy_idx] += eps;

      Transform tf_rlink_to_blink_tweaked = tf_rlink_to_blink;
      tf_rlink_to_blink_tweaked.setQuaternionFromRPY(rpy_tweaked);
      Transform tf_rlink_to_elink_tweaked = tf_rlink_to_blink_tweaked * tf_blink_to_elink;
      auto pose_out = tf_rlink_to_elink_tweaked;

      const auto pos_diff = pose_out.trans() - tf_rlink_to_elink.trans();
      jacobian.block<3, 1>(0, idx_col) = pos_diff / eps;
    }
  }
  return jacobian;
}

Eigen::Vector3d KinematicModel::get_com() {
  Eigen::Vector3d com_average = Eigen::Vector3d::Zero();
  double mass_total = 0.0;
  for (const auto &link : com_dummy_links_) {
    mass_total += link->inertial->mass;
    const auto& tf_base_to_com = get_link_pose(link->id);
    com_average += link->inertial->mass * tf_base_to_com.trans();
  }
  com_average /= mass_total;
  return com_average;
}

Eigen::MatrixXd
KinematicModel::get_com_jacobian(const std::vector<size_t> &joint_ids,
                                 bool with_base) {
  constexpr size_t jac_rank = 3;
  const size_t dim_dof = joint_ids.size() + with_base * 6;
  Eigen::MatrixXd jac_average = Eigen::MatrixXd::Zero(jac_rank, dim_dof);
  double mass_total = 0.0;
  for (const auto &com_link : com_dummy_links_) {
    mass_total += com_link->inertial->mass;
    auto jac = this->get_jacobian(com_link->id, joint_ids, RotationType::IGNORE,
                                  with_base);
    jac_average += com_link->inertial->mass * jac;
  }
  jac_average /= mass_total;
  return jac_average;
}

}; // namespace tinyfk
