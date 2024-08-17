#include "tinyfk.hpp"
#include "urdf_model/pose.h"
#include <Eigen/Dense>
#include <cmath>

namespace tinyfk {

Vector3 rpy_derivative(const Vector3 &rpy, const Vector3 &axis) {
  Vector3 drpy_dt;
  double a2 = -rpy.y;
  double a3 = -rpy.z;
  drpy_dt.x = cos(a3) / cos(a2) * axis.x - sin(a3) / cos(a2) * axis.y;
  drpy_dt.y = sin(a3) * axis.x + cos(a3) * axis.y;
  drpy_dt.z = -cos(a3) * sin(a2) / cos(a2) * axis.x +
              sin(a3) * sin(a2) / cos(a2) * axis.y + axis.z;
  return drpy_dt;
}

Rotation q_derivative(const Rotation &q, const Vector3 &omega) {
  const double dxdt =
      0.5 * (0 * q.x + omega.z * q.y - omega.y * q.z + omega.x * q.w);
  const double dydt =
      0.5 * (-omega.z * q.x + 0 * q.y + omega.x * q.z + omega.y * q.w);
  const double dzdt =
      0.5 * (omega.y * q.x - omega.x * q.y + 0 * q.z + omega.z * q.w);
  const double dwdt =
      0.5 * (-omega.x * q.x - omega.y * q.y - omega.z * q.z + 0 * q.w);
  return Rotation(dxdt, dydt, dzdt, -dwdt); // TODO: why minus????
}

void KinematicModel::get_link_pose(size_t link_id,
                                   Transform &out_tf_rlink_to_elink) const {
  Transform const *pose_ptr = transform_cache_.get_cache(link_id);
  if (pose_ptr) {
    out_tf_rlink_to_elink = *pose_ptr;
    return;
  }
  this->get_link_pose_inner(link_id, out_tf_rlink_to_elink);
}

void KinematicModel::get_link_pose_inner(
    size_t link_id, Transform &out_tf_rlink_to_elink) const {
  urdf::LinkSharedPtr hlink = links_[link_id];

  Transform tf_rlink_to_blink = base_pose_;

  transform_stack_.reset();
  while (true) {

    urdf::LinkSharedPtr plink = hlink->getParent();
    if (plink == nullptr) {
      break;
    } // hit the root link

    Transform const *tf_rlink_to_blink_ptr =
        transform_cache_.get_cache(hlink->id);
    if (tf_rlink_to_blink_ptr) {
      tf_rlink_to_blink = *tf_rlink_to_blink_ptr;
      break;
    }

    Transform tf_plink_to_hlink;
    { // compute tf_plink_to_hlink
      const urdf::JointSharedPtr &pjoint = hlink->parent_joint;
      const Transform &tf_plink_to_pjoint =
          pjoint->parent_to_joint_origin_transform;

      if (pjoint->type == urdf::Joint::FIXED) {
        tf_plink_to_hlink = tf_plink_to_pjoint;
      } else {
        double angle = joint_angles_[pjoint->id];
        Transform tf_pjoint_to_hlink = pjoint->transform(angle);
        tf_plink_to_hlink =
            pose_transform(tf_plink_to_pjoint, tf_pjoint_to_hlink);
      }
    }

    // update
    transform_stack_.push(LinkIdAndTransform{
        hlink->id, std::move(tf_plink_to_hlink)}); // TODO(HiroIshida): move?
    hlink = plink;
  }

  Transform tf_rlink_to_plink = std::move(tf_rlink_to_blink);
  while (!transform_stack_.empty()) {

    const auto &pose_id_pair = transform_stack_.top();
    const Transform &tf_plink_to_hlink = pose_id_pair.pose;
    const size_t hid = pose_id_pair.id;
    transform_stack_.pop();
    Transform tf_rlink_to_hlink =
        pose_transform(tf_rlink_to_plink, tf_plink_to_hlink);
    transform_cache_.set_cache(hid, tf_rlink_to_hlink);
    tf_rlink_to_plink = std::move(tf_rlink_to_hlink);
  }
  out_tf_rlink_to_elink = std::move(tf_rlink_to_plink);
}

Eigen::MatrixXd
KinematicModel::get_jacobian(size_t elink_id,
                             const std::vector<size_t> &joint_ids,
                             RotationType rot_type, bool with_base) {
  const size_t dim_jacobi = 3 + (rot_type == RotationType::RPY) * 3 +
                            (rot_type == RotationType::XYZW) * 4;
  const int dim_dof = joint_ids.size() + (with_base ? 6 : 0);

  // compute values shared through the loop
  Transform tf_rlink_to_elink;
  this->get_link_pose(elink_id, tf_rlink_to_elink);
  Vector3 &epos = tf_rlink_to_elink.position;
  Rotation &erot = tf_rlink_to_elink.rotation;

  Vector3 erpy;
  Rotation erot_inverse;
  if (rot_type == RotationType::RPY) {
    erpy = erot.getRPY();
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
      if (type == urdf::Joint::FIXED) {
        assert(type != urdf::Joint::FIXED && "fixed type is not accepted");
      }
      urdf::LinkSharedPtr clink =
          hjoint->getChildLink(); // rotation of clink and hlink is same. so
                                  // clink is ok.

      Transform tf_rlink_to_clink;
      this->get_link_pose(clink->id, tf_rlink_to_clink);

      Rotation &crot = tf_rlink_to_clink.rotation;
      Vector3 &&world_axis = crot * hjoint->axis; // axis w.r.t root link
      Vector3 dpos;
      if (type == urdf::Joint::PRISMATIC) {
        dpos = world_axis;
      } else { // revolute or continuous
        Vector3 &cpos = tf_rlink_to_clink.position;
        Vector3 vec_clink_to_elink = {epos.x - cpos.x, epos.y - cpos.y,
                                      epos.z - cpos.z};
        cross_product(world_axis, vec_clink_to_elink, dpos);
      }
      jacobian(0, i) = dpos.x;
      jacobian(1, i) = dpos.y;
      jacobian(2, i) = dpos.z;
      if (type == urdf::Joint::PRISMATIC) {
        // jacobian for rotation is all zero
      } else {

        if (rot_type == RotationType::RPY) { // (compute rpy jacobian)
          Vector3 drpy_dt = rpy_derivative(erpy, world_axis);
          jacobian(3, i) = drpy_dt.x;
          jacobian(4, i) = drpy_dt.y;
          jacobian(5, i) = drpy_dt.z;
        }

        if (rot_type == RotationType::XYZW) { // (compute quat jacobian)
          Rotation dq_dt = q_derivative(erot_inverse, world_axis);
          jacobian(3, i) = dq_dt.x;
          jacobian(4, i) = dq_dt.y;
          jacobian(5, i) = dq_dt.z;
          jacobian(6, i) = dq_dt.w;
        }
      }
    }
  }

  Transform tf_rlink_to_blink, tf_blink_to_rlink, tf_blink_to_elink;
  Vector3 rpy_rlink_to_blink;
  if (with_base) {
    this->get_link_pose(this->root_link_id_, tf_rlink_to_blink);
    tf_blink_to_rlink = tf_rlink_to_blink.inverse();
    rpy_rlink_to_blink = tf_rlink_to_blink.rotation.getRPY();
    tf_blink_to_elink = pose_transform(tf_blink_to_rlink, tf_rlink_to_elink);
  }

  if (with_base) {
    const size_t dim_dof = joint_ids.size();

    jacobian(0, dim_dof + 0) = 1.0;
    jacobian(1, dim_dof + 1) = 1.0;
    jacobian(2, dim_dof + 2) = 1.0;

    // we resort to numerical method to base pose jacobian (just because I don't
    // have time)
    // TODO(HiroIshida): compute using analytical method.
    constexpr double eps = 1e-7;
    for (size_t rpy_idx = 0; rpy_idx < 3; rpy_idx++) {
      const size_t idx_col = dim_dof + 3 + rpy_idx;

      auto rpy_tweaked = rpy_rlink_to_blink;
      rpy_tweaked[rpy_idx] += eps;

      Transform tf_rlink_to_blink_tweaked = tf_rlink_to_blink;
      tf_rlink_to_blink_tweaked.rotation.setFromRPY(
          rpy_tweaked.x, rpy_tweaked.y, rpy_tweaked.z);
      Transform tf_rlink_to_elink_tweaked =
          pose_transform(tf_rlink_to_blink_tweaked, tf_blink_to_elink);
      auto pose_out = tf_rlink_to_elink_tweaked;

      const auto pos_diff = pose_out.position - tf_rlink_to_elink.position;
      jacobian(0, idx_col) = pos_diff.x / eps;
      jacobian(1, idx_col) = pos_diff.y / eps;
      jacobian(2, idx_col) = pos_diff.z / eps;
      if (rot_type == RotationType::RPY) {
        auto erpy_tweaked = pose_out.rotation.getRPY();
        const auto erpy_diff = erpy_tweaked - erpy;
        jacobian(3, idx_col) = erpy_diff.x / eps;
        jacobian(4, idx_col) = erpy_diff.y / eps;
        jacobian(5, idx_col) = erpy_diff.z / eps;
      }
      if (rot_type == RotationType::XYZW) {
        jacobian(3, idx_col) = (pose_out.rotation.x - erot.x) / eps;
        jacobian(4, idx_col) = (pose_out.rotation.y - erot.y) / eps;
        jacobian(5, idx_col) = (pose_out.rotation.z - erot.z) / eps;
        jacobian(6, idx_col) = (pose_out.rotation.w - erot.w) / eps;
      }
    }
  }
  return jacobian;
}

Vector3 KinematicModel::get_com() {
  Vector3 com_average;
  double mass_total = 0.0;
  Transform tf_base_to_com;
  for (const auto &link : com_dummy_links_) {
    mass_total += link->inertial->mass;
    this->get_link_pose(link->id, tf_base_to_com);
    com_average.x += link->inertial->mass * tf_base_to_com.position.x;
    com_average.y += link->inertial->mass * tf_base_to_com.position.y;
    com_average.z += link->inertial->mass * tf_base_to_com.position.z;
  }

  com_average.x = com_average.x / mass_total;
  com_average.y = com_average.y / mass_total;
  com_average.z = com_average.z / mass_total;
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

Eigen::Matrix3d KinematicModel::get_total_inertia_matrix() {
  auto com = this->get_com();

  Eigen::Matrix3d Imat_total = Eigen::Matrix3d::Zero();
  for (const auto &link : com_dummy_links_) {
    const auto inertial = link->inertial;
    if (inertial != nullptr) {
      double mass = inertial->mass;
      double ixx = inertial->ixx;
      double iyy = inertial->iyy;
      double izz = inertial->izz;
      double ixy = inertial->ixy;
      double ixz = inertial->ixz;
      double iyz = inertial->iyz;
      Eigen::Matrix3d Imat;
      Imat << ixx, ixy, ixz, ixy, iyy, iyz, ixz, iyz, izz;
      size_t link_id = link->id;

      Transform tf_base_to_link;
      this->get_link_pose(link_id, tf_base_to_link);
      const auto &trans = tf_base_to_link.position;
      Eigen::Vector3d vec;
      vec << trans.x - com.x, trans.y - com.y, trans.z - com.z;
      const auto &rot = tf_base_to_link.rotation;
      double xy2 = 2 * (rot.x * rot.y);
      double xz2 = 2 * (rot.x * rot.z);
      double xw2 = 2 * (rot.x * rot.w);
      double yz2 = 2 * (rot.y * rot.z);
      double yw2 = 2 * (rot.y * rot.w);
      double zw2 = 2 * (rot.z * rot.w);
      double xx2 = 2 * (rot.x * rot.x);
      double yy2 = 2 * (rot.y * rot.y);
      double zz2 = 2 * (rot.z * rot.z);

      Eigen::Matrix3d R;
      R << 1 - yy2 - zz2, xy2 - zw2, xz2 + yw2, xy2 + zw2, 1 - xx2 - zz2,
          yz2 - xw2, xz2 - yw2, yz2 + xw2, 1 - xx2 - yy2;

      Eigen::Matrix3d trans_term =
          mass * (vec.norm() * vec.norm() * Eigen::Matrix3d::Identity() -
                  vec * vec.transpose());
      Imat_total += (R * Imat * R.transpose() + trans_term);
    }
  }
  return Imat_total;
}

}; // namespace tinyfk
