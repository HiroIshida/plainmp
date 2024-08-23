#include "tinyfk.hpp"
#include "urdf_model/pose.h"
#include <Eigen/Dense>
#include <cmath>
#include <stack>

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
  if(transform_cache_.is_cached(link_id)) {
    out_tf_rlink_to_elink = transform_cache_.data_[link_id];
    return;
  }
  this->get_link_pose_cache_not_found(link_id, out_tf_rlink_to_elink);
}

void KinematicModel::get_link_pose_cache_not_found(size_t link_id, Transform &out_tf_rlink_to_elink) const
{
  if(links_[link_id]->consider_rotation) {
    this->get_link_pose_inner(link_id, out_tf_rlink_to_elink);
  } else {
    auto hlink = links_[link_id];
    auto plink = hlink->getParent();
    auto pjoint = hlink->parent_joint;
    Transform tf_rlink_to_plink;
    if(transform_cache_.is_cached(plink->id)) {
      tf_rlink_to_plink = transform_cache_.data_[plink->id];
    } else {
      get_link_pose_inner(plink->id, tf_rlink_to_plink);
    }
    out_tf_rlink_to_elink.t = tf_rlink_to_plink.t + tf_rlink_to_plink.q * pjoint->parent_to_joint_origin_transform.t;

    // HACK: we want to update the only position part
    // thus, we commented out the private: and directly access the data
    transform_cache_.cache_predicate_vector_[link_id] = true;
    transform_cache_.data_[link_id].t = std::move(out_tf_rlink_to_elink.t);
  }
}

void KinematicModel::get_link_pose_inner(
    size_t link_id, Transform &out_tf_rlink_to_elink) const {
  urdf::LinkSharedPtr hlink = links_[link_id];

  Transform tf_rlink_to_blink = base_pose_;

  link_id_stack_.reset();
  while (true) {

    size_t hlink_id = hlink->id;
    urdf::LinkSharedPtr plink = hlink->getParent();
    if (plink == nullptr) {
      break;
    } // hit the root link

    if(transform_cache_.is_cached(hlink_id)) {
      tf_rlink_to_blink = transform_cache_.data_[hlink_id];
      break;
    }

    link_id_stack_.push(hlink_id);
    hlink = plink;
  }

  Transform tf_rlink_to_plink = std::move(tf_rlink_to_blink);
  while(!link_id_stack_.empty()) {
    size_t hid = link_id_stack_.top();
    link_id_stack_.pop();
    auto& tf_plink_to_hlink = tf_plink_to_hlink_cache_[hid];
    Transform tf_rlink_to_hlink = tf_rlink_to_plink * tf_plink_to_hlink;
    transform_cache_.set_cache(hid, tf_rlink_to_hlink);
    tf_rlink_to_plink = std::move(tf_rlink_to_hlink);
  }
  out_tf_rlink_to_elink = std::move(tf_rlink_to_plink);
}

void KinematicModel::update_tree() {
  // transform_stack2_.reset();
  // transform_stack2_.push(std::make_pair(links_[root_link_id_], base_pose_));
  // while(!transform_stack2_.empty()) {
  //   auto [hlink, tf_b2h] = transform_stack2_.top();
  //   transform_stack2_.pop();
  //   transform_cache_.set_cache(hlink->id, tf_b2h);
  //   for (size_t i = 0; i < hlink->child_joints.size(); i++) {
  //     auto cjoint = hlink->child_joints[i];
  //     auto clink = hlink->child_links[i];
  //     if(cjoint->type == urdf::Joint::FIXED) {
  //       auto& tf_h2c = cjoint->parent_to_joint_origin_transform;
  //       auto&& tf_b2c = tf_b2h * tf_h2c;
  //       transform_stack2_.push({clink, tf_b2c});
  //     } else {
  //       auto angle = joint_angles_[cjoint->id];
  //       auto& tf_h2cj = cjoint->parent_to_joint_origin_transform;
  //       auto&& tf_cj2c = cjoint->transform(angle);
  //       auto&& tf_h2c = tf_h2cj * tf_cj2c;
  //       auto&& tf_b2c = tf_b2h * tf_h2c;
  //       transform_stack2_.push({clink, tf_b2c});
  //     }
  //   }
  // }
}

Eigen::MatrixXd
KinematicModel::get_jacobian(size_t elink_id,
                             const std::vector<size_t> &joint_ids,
                             RotationType rot_type, bool with_base) {
  // const size_t dim_jacobi = 3 + (rot_type == RotationType::RPY) * 3 +
  //                           (rot_type == RotationType::XYZW) * 4;
  // const int dim_dof = joint_ids.size() + (with_base ? 6 : 0);

  // // compute values shared through the loop
  // Transform tf_rlink_to_elink;
  // this->get_link_pose(elink_id, tf_rlink_to_elink);
  // Vector3 &epos = tf_rlink_to_elink.position;
  // Rotation &erot = tf_rlink_to_elink.rotation;

  // Vector3 erpy;
  // Rotation erot_inverse;
  // if (rot_type == RotationType::RPY) {
  //   erpy = erot.getRPY();
  // }
  // if (rot_type == RotationType::XYZW) {
  //   erot_inverse = erot.inverse();
  // }

  // // Jacobian computation
  // Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(dim_jacobi, dim_dof);

  // for (size_t i = 0; i < joint_ids.size(); i++) {
  //   int jid = joint_ids[i];
  //   if (rptable_.isRelevant(elink_id, jid)) {
  //     const urdf::JointSharedPtr &hjoint = joints_[jid];
  //     size_t type = hjoint->type;
  //     if (type == urdf::Joint::FIXED) {
  //       assert(type != urdf::Joint::FIXED && "fixed type is not accepted");
  //     }
  //     urdf::LinkSharedPtr clink =
  //         hjoint->getChildLink(); // rotation of clink and hlink is same. so
  //                                 // clink is ok.

  //     Transform tf_rlink_to_clink;
  //     this->get_link_pose(clink->id, tf_rlink_to_clink);

  //     Rotation &crot = tf_rlink_to_clink.rotation;
  //     Vector3 &&world_axis = crot * hjoint->axis; // axis w.r.t root link
  //     Vector3 dpos;
  //     if (type == urdf::Joint::PRISMATIC) {
  //       dpos = world_axis;
  //     } else { // revolute or continuous
  //       Vector3 &cpos = tf_rlink_to_clink.position;
  //       Vector3 vec_clink_to_elink = {epos.x - cpos.x, epos.y - cpos.y,
  //                                     epos.z - cpos.z};
  //       cross_product(world_axis, vec_clink_to_elink, dpos);
  //     }
  //     jacobian(0, i) = dpos.x;
  //     jacobian(1, i) = dpos.y;
  //     jacobian(2, i) = dpos.z;
  //     if (type == urdf::Joint::PRISMATIC) {
  //       // jacobian for rotation is all zero
  //     } else {

  //       if (rot_type == RotationType::RPY) { // (compute rpy jacobian)
  //         Vector3 drpy_dt = rpy_derivative(erpy, world_axis);
  //         jacobian(3, i) = drpy_dt.x;
  //         jacobian(4, i) = drpy_dt.y;
  //         jacobian(5, i) = drpy_dt.z;
  //       }

  //       if (rot_type == RotationType::XYZW) { // (compute quat jacobian)
  //         Rotation dq_dt = q_derivative(erot_inverse, world_axis);
  //         jacobian(3, i) = dq_dt.x;
  //         jacobian(4, i) = dq_dt.y;
  //         jacobian(5, i) = dq_dt.z;
  //         jacobian(6, i) = dq_dt.w;
  //       }
  //     }
  //   }
  // }

  // Transform tf_rlink_to_blink, tf_blink_to_rlink, tf_blink_to_elink;
  // Vector3 rpy_rlink_to_blink;
  // if (with_base) {
  //   this->get_link_pose(this->root_link_id_, tf_rlink_to_blink);
  //   tf_blink_to_rlink = tf_rlink_to_blink.inverse();
  //   rpy_rlink_to_blink = tf_rlink_to_blink.rotation.getRPY();
  //   tf_blink_to_elink = pose_transform(tf_blink_to_rlink, tf_rlink_to_elink);
  // }

  // if (with_base) {
  //   const size_t dim_dof = joint_ids.size();

  //   jacobian(0, dim_dof + 0) = 1.0;
  //   jacobian(1, dim_dof + 1) = 1.0;
  //   jacobian(2, dim_dof + 2) = 1.0;

  //   // we resort to numerical method to base pose jacobian (just because I don't
  //   // have time)
  //   // TODO(HiroIshida): compute using analytical method.
  //   constexpr double eps = 1e-7;
  //   for (size_t rpy_idx = 0; rpy_idx < 3; rpy_idx++) {
  //     const size_t idx_col = dim_dof + 3 + rpy_idx;

  //     auto rpy_tweaked = rpy_rlink_to_blink;
  //     rpy_tweaked[rpy_idx] += eps;

  //     Transform tf_rlink_to_blink_tweaked = tf_rlink_to_blink;
  //     tf_rlink_to_blink_tweaked.rotation.setFromRPY(
  //         rpy_tweaked.x, rpy_tweaked.y, rpy_tweaked.z);
  //     Transform tf_rlink_to_elink_tweaked =
  //         pose_transform(tf_rlink_to_blink_tweaked, tf_blink_to_elink);
  //     auto pose_out = tf_rlink_to_elink_tweaked;

  //     const auto pos_diff = pose_out.position - tf_rlink_to_elink.position;
  //     jacobian(0, idx_col) = pos_diff.x / eps;
  //     jacobian(1, idx_col) = pos_diff.y / eps;
  //     jacobian(2, idx_col) = pos_diff.z / eps;
  //     if (rot_type == RotationType::RPY) {
  //       auto erpy_tweaked = pose_out.rotation.getRPY();
  //       const auto erpy_diff = erpy_tweaked - erpy;
  //       jacobian(3, idx_col) = erpy_diff.x / eps;
  //       jacobian(4, idx_col) = erpy_diff.y / eps;
  //       jacobian(5, idx_col) = erpy_diff.z / eps;
  //     }
  //     if (rot_type == RotationType::XYZW) {
  //       jacobian(3, idx_col) = (pose_out.rotation.x - erot.x) / eps;
  //       jacobian(4, idx_col) = (pose_out.rotation.y - erot.y) / eps;
  //       jacobian(5, idx_col) = (pose_out.rotation.z - erot.z) / eps;
  //       jacobian(6, idx_col) = (pose_out.rotation.w - erot.w) / eps;
  //     }
  //   }
  // }
  auto jacobian = Eigen::MatrixXd(3, 3);
  return jacobian;
}

Eigen::Vector3d KinematicModel::get_com() {
  Eigen::Vector3d com_average;
  double mass_total = 0.0;
  Transform tf_base_to_com;
  for (const auto &link : com_dummy_links_) {
    mass_total += link->inertial->mass;
    this->get_link_pose(link->id, tf_base_to_com);
    com_average += link->inertial->mass * tf_base_to_com.t;
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

Eigen::Matrix3d KinematicModel::get_total_inertia_matrix() {
  auto com = this->get_com();

  Eigen::Matrix3d Imat_total = Eigen::Matrix3d::Zero();
  throw std::runtime_error("Not implemented");
  // for (const auto &link : com_dummy_links_) {
  //   const auto inertial = link->inertial;
  //   if (inertial != nullptr) {
  //     double mass = inertial->mass;
  //     double ixx = inertial->ixx;
  //     double iyy = inertial->iyy;
  //     double izz = inertial->izz;
  //     double ixy = inertial->ixy;
  //     double ixz = inertial->ixz;
  //     double iyz = inertial->iyz;
  //     Eigen::Matrix3d Imat;
  //     Imat << ixx, ixy, ixz, ixy, iyy, iyz, ixz, iyz, izz;
  //     size_t link_id = link->id;

  //     Transform tf_base_to_link;
  //     this->get_link_pose(link_id, tf_base_to_link);
  //     const auto &trans = tf_base_to_link.position;
  //     Eigen::Vector3d vec;
  //     vec << trans.x - com.x, trans.y - com.y, trans.z - com.z;
  //     const auto &rot = tf_base_to_link.rotation;
  //     double xy2 = 2 * (rot.x * rot.y);
  //     double xz2 = 2 * (rot.x * rot.z);
  //     double xw2 = 2 * (rot.x * rot.w);
  //     double yz2 = 2 * (rot.y * rot.z);
  //     double yw2 = 2 * (rot.y * rot.w);
  //     double zw2 = 2 * (rot.z * rot.w);
  //     double xx2 = 2 * (rot.x * rot.x);
  //     double yy2 = 2 * (rot.y * rot.y);
  //     double zz2 = 2 * (rot.z * rot.z);

  //     Eigen::Matrix3d R;
  //     R << 1 - yy2 - zz2, xy2 - zw2, xz2 + yw2, xy2 + zw2, 1 - xx2 - zz2,
  //         yz2 - xw2, xz2 - yw2, yz2 + xw2, 1 - xx2 - yy2;

  //     Eigen::Matrix3d trans_term =
  //         mass * (vec.norm() * vec.norm() * Eigen::Matrix3d::Identity() -
  //                 vec * vec.transpose());
  //     Imat_total += (R * Imat * R.transpose() + trans_term);
  //   }
  // }
  return Imat_total;
}

}; // namespace tinyfk