/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include <Eigen/Dense>
#include <cmath>
#include <stack>
#include <stdexcept>
#include "plainmp/kinematics/kinematics.hpp"
#include "urdf_model/pose.h"

namespace plainmp::kinematics {

template <typename Scalar>
Eigen::Matrix<Scalar, 3, 1> rpy_derivative(
    const Eigen::Matrix<Scalar, 3, 1>& rpy,
    const Eigen::Matrix<Scalar, 3, 1>& axis) {
  Eigen::Matrix<Scalar, 3, 1> drpy_dt;
  Scalar a2 = -rpy.y();
  Scalar a3 = -rpy.z();
  drpy_dt.x() = cos(a3) / cos(a2) * axis.x() - sin(a3) / cos(a2) * axis.y();
  drpy_dt.y() = sin(a3) * axis.x() + cos(a3) * axis.y();
  drpy_dt.z() = -cos(a3) * sin(a2) / cos(a2) * axis.x() +
                sin(a3) * sin(a2) / cos(a2) * axis.y() + axis.z();
  return drpy_dt;
}

template <typename Scalar>
Eigen::Quaternion<Scalar> q_derivative(
    const Eigen::Quaternion<Scalar>& q,
    const Eigen::Matrix<Scalar, 3, 1>& omega) {
  const Scalar dxdt =
      0.5 * (omega.z() * q.y() - omega.y() * q.z() + omega.x() * q.w());
  const Scalar dydt = 0.5 * (-omega.z() * q.x() + 0 * q.y() +
                             omega.x() * q.z() + omega.y() * q.w());
  const Scalar dzdt = 0.5 * (omega.y() * q.x() - omega.x() * q.y() + 0 * q.z() +
                             omega.z() * q.w());
  const Scalar dwdt = 0.5 * (-omega.x() * q.x() - omega.y() * q.y() -
                             omega.z() * q.z() + 0 * q.w());
  return Eigen::Quaternion<Scalar>(-dwdt, dxdt, dydt, dzdt);
}

template <typename Scalar>
void compute_approx_sin_cos(Scalar x, Scalar& s, Scalar& c) {
  // Approximate sin(x) = x - x^3/3! + x^5/5! - x^7/7! + x^9/9!
  // Approximate cos(x) = 1 - x^2/2! + x^4/4! - x^6/6! + x^8/8!
  constexpr auto coeff2 = 1.0 / (1.0 * 2.0);
  constexpr auto coeff3 = 1.0 / (1.0 * 2.0 * 3.0);
  constexpr auto coeff4 = 1.0 / (1.0 * 2.0 * 3.0 * 4.0);
  constexpr auto coeff5 = 1.0 / (1.0 * 2.0 * 3.0 * 4.0 * 5.0);
  constexpr auto coeff6 = 1.0 / (1.0 * 2.0 * 3.0 * 4.0 * 5.0 * 6.0);
  constexpr auto coeff7 = 1.0 / (1.0 * 2.0 * 3.0 * 4.0 * 5.0 * 6.0 * 7.0);
  constexpr auto coeff8 = 1.0 / (1.0 * 2.0 * 3.0 * 4.0 * 5.0 * 6.0 * 7.0 * 8.0);
  constexpr auto coeff9 =
      1.0 / (1.0 * 2.0 * 3.0 * 4.0 * 5.0 * 6.0 * 7.0 * 8.0 * 9.0);
  constexpr auto half_pi = M_PI * 0.5;
  constexpr auto one_dev_2pi = 1.0 / (2 * M_PI);
  auto cos_sign = 1.0;
  if (x > half_pi || x < -half_pi) {
    if (x > M_PI || x < -M_PI) {
      x = x - 2 * M_PI * std::floor(x * one_dev_2pi + 0.5);
    }
    if (x < -half_pi) {
      x = -x - M_PI;
      cos_sign = -1.0;
    } else if (x > half_pi) {
      x = -x + M_PI;
      cos_sign = -1.0;
    } else {
    }
  }
  auto xx = x * x;
  auto xxxx = xx * xx;
  auto xxxxxx = xxxx * xx;
  auto xxxxxxxx = xxxx * xxxx;
  s = x *
      (1 - xx * coeff3 + xxxx * coeff5 - xxxxxx * coeff7 + xxxxxxxx * coeff9);
  c = cos_sign *
      (1 - xx * coeff2 + xxxx * coeff4 - xxxxxx * coeff6 + xxxxxxxx * coeff8);
}

template <typename Scalar>
void KinematicModel<Scalar>::set_joint_angles(
    const std::vector<size_t>& joint_ids,
    const Vector& joint_angles,
    bool high_accuracy) {
  if (high_accuracy) {
    if (all_joint_orientation_identity_) {
      this->set_joint_angles_impl<false, true>(joint_ids, joint_angles);
    } else {
      this->set_joint_angles_impl<false, false>(joint_ids, joint_angles);
    }
  } else {
    if (all_joint_orientation_identity_) {
      this->set_joint_angles_impl<true, true>(joint_ids, joint_angles);
    } else {
      this->set_joint_angles_impl<true, false>(joint_ids, joint_angles);
    }
  }
  clear_cache();
}

template <typename Scalar>
template <bool approx, bool all_quat_identity>
void KinematicModel<Scalar>::set_joint_angles_impl(
    const std::vector<size_t>& joint_ids,
    const Vector& joint_angles) {
  Quat tf_pjoint_to_hlink_quat;  // pre-allocate

  for (size_t i = 0; i < joint_ids.size(); i++) {
    auto joint_id = joint_ids[i];
    joint_angles_[joint_id] = joint_angles[i];
    auto& tf_plink_to_hlink =
        tf_plink_to_hlink_cache_[joint_child_link_ids_[joint_id]];
    auto& tf_plink_to_pjoint_trans = joint_positions_[joint_id];
    if (joint_types_[joint_id] != urdf::Joint::PRISMATIC) {
      auto x = joint_angles[i] * 0.5;

      // Here we will compute the multiplication of two transformation
      // tf_plink_to_hlink = tf_plink_to_pjoint * tf_pjoint_to_hlink
      // without instantiating the transformation object because
      // 1) dont want to instantiate the object
      // 2) the tf_pjoint_to_hlink does not have translation

      if constexpr (approx) {
        Scalar s, c;
        compute_approx_sin_cos<Scalar>(x, s, c);
        tf_pjoint_to_hlink_quat.coeffs() << s * joint_axes_[joint_id], c;
      } else {
        tf_pjoint_to_hlink_quat.coeffs() << sin(x) * joint_axes_[joint_id],
            cos(x);
      }
      const auto& tf_plink_to_pjoint_quat = joint_orientations_[joint_id];

      if constexpr (all_quat_identity) {
        tf_plink_to_hlink.quat() = tf_pjoint_to_hlink_quat;
      } else {
        tf_plink_to_hlink.quat() =
            joint_orientation_identity_flags_[joint_id]
                ? tf_pjoint_to_hlink_quat
                : tf_plink_to_pjoint_quat * tf_pjoint_to_hlink_quat;
      }
      tf_plink_to_hlink.trans() = tf_plink_to_pjoint_trans;
      tf_plink_to_hlink.is_quat_identity_ = false;
    } else {
      Vector3&& trans = joint_axes_[joint_id] * joint_angles[i];
      tf_plink_to_hlink.trans() = tf_plink_to_pjoint_trans + trans;
      tf_plink_to_hlink.quat().setIdentity();
      tf_plink_to_hlink.is_quat_identity_ = true;
    }
  }
}

template <typename Scalar>
void KinematicModel<Scalar>::build_cache_until(size_t link_id) const {
  if (link_consider_rotation_[link_id]) {
    this->build_cache_until_inner(link_id);
  } else {
    // TODO: we should remove this!
    auto plink_id = link_parent_link_ids_[link_id];
    if (!transform_cache_.is_cached(plink_id)) {
      build_cache_until_inner(plink_id);
    }
    Transform& tf_rlink_to_plink = transform_cache_.data_[plink_id];
    auto&& rotmat = tf_rlink_to_plink.quat().toRotationMatrix();
    auto& plink_to_hlink_trans = tf_plink_to_hlink_cache_[link_id].trans();
    Vector3&& pos = tf_rlink_to_plink.trans() + rotmat * plink_to_hlink_trans;
    // HACK: we want to update the only position part
    // thus, we commented out the private: and directly access the data
    transform_cache_.cache_predicate_vector_[link_id] = true;
    transform_cache_.data_[link_id].trans() = std::move(pos);
  }
}

template <typename Scalar>
void KinematicModel<Scalar>::build_cache_until_inner(size_t hlink_id) const {
  std::array<size_t, 64> id_stack_like;  // 64 is enough for almost all cases
  size_t idx = 0;
  while (!transform_cache_.is_cached(hlink_id)) {
    id_stack_like[idx++] = hlink_id;
    hlink_id = link_parent_link_ids_[hlink_id];
  }

  // >> ORIGINAL EQUIVALENT CODE
  // Transform tf_rlink_to_plink = transform_cache_.data_[hlink_id];
  // while(idx > 0) {
  //   size_t hid = id_stack_like[--idx];
  //   Transform tf_rlink_to_hlink =
  //   tf_rlink_to_plink.quat_identity_sensitive_mul(tf_plink_to_hlink_cache_[hid]);
  //   transform_cache_.set_cache(hid, tf_rlink_to_hlink);
  //   tf_rlink_to_plink = std::move(tf_rlink_to_hlink);
  // }
  // << ORIGINAL EQUIVALENT CODE

  // >> LESS STACK ALLOCATION CODE
  size_t hid = hlink_id;
  while (idx > 0) {
    size_t hid_prev = hid;
    hid = id_stack_like[--idx];
    transform_cache_.data_[hid_prev].quat_identity_sensitive_mult_and_assign(
        tf_plink_to_hlink_cache_[hid], transform_cache_.data_[hid]);
    transform_cache_.cache_predicate_vector_[hid] = true;
  }
}

template <typename Scalar>
typename KinematicModel<Scalar>::MatrixDynamic
KinematicModel<Scalar>::get_jacobian(size_t elink_id,
                                     const std::vector<size_t>& joint_ids,
                                     RotationType rot_type,
                                     BaseType base_type) {
  const size_t dim_jacobi = 3 + (rot_type == RotationType::RPY) * 3 +
                            (rot_type == RotationType::XYZW) * 4;
  const size_t dim_dof = joint_ids.size() +
                         (base_type == BaseType::FLOATING) * 6 +
                         (base_type == BaseType::PLANAR) * 3;

  const auto& tf_rlink_to_elink = get_link_pose(elink_id);
  auto& epos = tf_rlink_to_elink.trans();
  auto& erot = tf_rlink_to_elink.quat();

  Vector3 erpy;
  Quat erot_inverse;
  if (rot_type == RotationType::RPY) {
    erpy = tf_rlink_to_elink.getRPY();
  }
  if (rot_type == RotationType::XYZW) {
    erot_inverse = erot.inverse();
  }

  // Jacobian computation
  MatrixDynamic jacobian = MatrixDynamic::Zero(dim_jacobi, dim_dof);

  for (size_t i = 0; i < joint_ids.size(); i++) {
    int jid = joint_ids[i];
    if (rptable_.isRelevant(elink_id, jid)) {
      auto jtype = joint_types_[jid];
      auto clink_id = joint_child_link_ids_[jid];
      auto& joint_axis = joint_axes_[jid];
      const auto& tf_rlink_to_clink = get_link_pose(clink_id);

      auto& crot = tf_rlink_to_clink.quat();
      auto&& world_axis = crot * joint_axis;  // axis w.r.t root link
      Vector3 dpos;
      if (jtype == urdf::Joint::PRISMATIC) {
        dpos = world_axis;
      } else {  // revolute or continuous
        auto& cpos = tf_rlink_to_clink.trans();
        auto vec_clink_to_elink = epos - cpos;
        dpos = world_axis.cross(vec_clink_to_elink);
      }
      jacobian.template block<3, 1>(0, i) = dpos;
      if (jtype == urdf::Joint::PRISMATIC) {
        // jacobian for rotation is all zero
      } else {
        if (rot_type == RotationType::RPY) {  // (compute rpy jacobian)
          auto drpy_dt = rpy_derivative<Scalar>(erpy, world_axis);
          jacobian.template block<3, 1>(3, i) = drpy_dt;
        }

        if (rot_type == RotationType::XYZW) {  // (compute quat jacobian)
          auto dq_dt = q_derivative<Scalar>(erot_inverse, world_axis);
          jacobian.template block<4, 1>(3, i) = dq_dt.coeffs();
        }
      }
    }
  }

  Transform tf_rlink_to_blink, tf_blink_to_rlink, tf_blink_to_elink;
  Vector3 rpy_rlink_to_blink;
  if (base_type != BaseType::FIXED) {
    tf_rlink_to_blink = get_link_pose(root_link_id_);
    tf_blink_to_rlink = tf_rlink_to_blink.getInverse();
    rpy_rlink_to_blink = tf_rlink_to_blink.getRPY();
    tf_blink_to_elink = tf_blink_to_rlink * tf_rlink_to_elink;
  }

  constexpr Scalar eps = 1e-7;
  if (base_type == BaseType::FLOATING) {
    const size_t n_joint = joint_ids.size();
    jacobian(0, n_joint + 0) = 1.0;
    jacobian(1, n_joint + 1) = 1.0;
    jacobian(2, n_joint + 2) = 1.0;

    // we resort to numerical method to base pose jacobian (just because I don't
    // have time)
    // TODO(HiroIshida): compute using analytical method.
    for (size_t rpy_idx = 0; rpy_idx < 3; rpy_idx++) {
      const size_t idx_col = n_joint + 3 + rpy_idx;

      auto rpy_tweaked = rpy_rlink_to_blink;
      rpy_tweaked[rpy_idx] += eps;

      Transform tf_rlink_to_blink_tweaked = tf_rlink_to_blink;
      tf_rlink_to_blink_tweaked.setQuaternionFromRPY(rpy_tweaked);
      Transform tf_rlink_to_elink_tweaked =
          tf_rlink_to_blink_tweaked * tf_blink_to_elink;
      auto pose_out = tf_rlink_to_elink_tweaked;

      const auto pos_diff = pose_out.trans() - tf_rlink_to_elink.trans();
      jacobian.template block<3, 1>(0, idx_col) = pos_diff / eps;
      if (rot_type == RotationType::RPY) {
        auto erpy_tweaked = pose_out.getRPY();
        jacobian.template block<3, 1>(3, idx_col) = (erpy_tweaked - erpy) / eps;
      }
      if (rot_type == RotationType::XYZW) {
        jacobian.template block<4, 1>(3, idx_col) =
            (pose_out.quat().coeffs() - erot.coeffs()) / eps;
      }
    }
  } else if (base_type == BaseType::PLANAR) {
    const size_t n_joint = joint_ids.size();
    jacobian(0, n_joint + 0) = 1.0;
    jacobian(1, n_joint + 1) = 1.0;

    auto rpy_tweaked = rpy_rlink_to_blink;
    rpy_tweaked[2] += eps;

    Transform tf_rlink_to_blink_tweaked = tf_rlink_to_blink;
    tf_rlink_to_blink_tweaked.setQuaternionFromRPY(rpy_tweaked);
    Transform tf_rlink_to_elink_tweaked =
        tf_rlink_to_blink_tweaked * tf_blink_to_elink;
    auto pose_out = tf_rlink_to_elink_tweaked;

    const auto pos_diff = pose_out.trans() - tf_rlink_to_elink.trans();
    jacobian.template block<3, 1>(0, n_joint + 2) = pos_diff / eps;
    if (rot_type == RotationType::RPY) {
      auto erpy_tweaked = pose_out.getRPY();
      jacobian.template block<3, 1>(3, n_joint + 2) =
          (erpy_tweaked - erpy) / eps;
    }
    if (rot_type == RotationType::XYZW) {
      jacobian.template block<4, 1>(3, n_joint + 2) =
          (pose_out.quat().coeffs() - erot.coeffs()) / eps;
    }
  }
  return jacobian;
}

template <typename Scalar>
typename KinematicModel<Scalar>::MatrixDynamic
KinematicModel<Scalar>::get_attached_point_jacobian(
    size_t plink_id,
    Vector3 apoint_global_pos,
    const std::vector<size_t>& joint_ids,
    BaseType base_type) {
  const size_t dim_dof = joint_ids.size() +
                         (base_type == BaseType::FLOATING) * 6 +
                         (base_type == BaseType::PLANAR) * 3;
  MatrixDynamic jacobian = MatrixDynamic::Zero(3, dim_dof);

  // NOTE: the following logic is copied from get_jacobian()
  for (size_t i = 0; i < joint_ids.size(); i++) {
    int jid = joint_ids[i];
    if (rptable_.isRelevant(plink_id, jid)) {
      auto jtype = joint_types_[jid];
      auto clink_id = joint_child_link_ids_[jid];
      auto& joint_axis = joint_axes_[jid];
      const auto& tf_rlink_to_clink = get_link_pose(clink_id);
      auto& crot = tf_rlink_to_clink.quat();
      auto&& world_axis = crot * joint_axis;  // axis w.r.t root link

      Vector3 dpos;
      if (jtype == urdf::Joint::PRISMATIC) {
        dpos = world_axis;
      } else {  // revolute or continuous
        auto& cpos = tf_rlink_to_clink.trans();
        auto vec_clink_to_elink = apoint_global_pos - cpos;
        dpos = world_axis.cross(vec_clink_to_elink);
      }
      jacobian.template block<3, 1>(0, i) = dpos;
    }
  }

  // NOTE: the following logic is copied from get_jacobian()
  Transform tf_rlink_to_elink = Transform::Identity();
  tf_rlink_to_elink.trans() = apoint_global_pos;
  Transform tf_rlink_to_blink, tf_blink_to_rlink, tf_blink_to_elink;
  Vector3 rpy_rlink_to_blink;
  if (base_type != BaseType::FIXED) {
    tf_rlink_to_blink = get_link_pose(root_link_id_);
    tf_blink_to_rlink = tf_rlink_to_blink.getInverse();
    rpy_rlink_to_blink = tf_rlink_to_blink.getRPY();
    tf_blink_to_elink = tf_blink_to_rlink * tf_rlink_to_elink;
  }

  constexpr Scalar eps = 1e-7;
  if (base_type == BaseType::FLOATING) {
    const size_t n_joint = joint_ids.size();
    jacobian(0, n_joint + 0) = 1.0;
    jacobian(1, n_joint + 1) = 1.0;
    jacobian(2, n_joint + 2) = 1.0;

    for (size_t rpy_idx = 0; rpy_idx < 3; rpy_idx++) {
      const size_t idx_col = n_joint + 3 + rpy_idx;

      auto rpy_tweaked = rpy_rlink_to_blink;
      rpy_tweaked[rpy_idx] += eps;

      Transform tf_rlink_to_blink_tweaked = tf_rlink_to_blink;
      tf_rlink_to_blink_tweaked.setQuaternionFromRPY(rpy_tweaked);
      Transform tf_rlink_to_elink_tweaked =
          tf_rlink_to_blink_tweaked * tf_blink_to_elink;
      auto pose_out = tf_rlink_to_elink_tweaked;

      const auto pos_diff = pose_out.trans() - tf_rlink_to_elink.trans();
      jacobian.template block<3, 1>(0, idx_col) = pos_diff / eps;
    }
  } else if (base_type == BaseType::PLANAR) {
    const size_t n_joint = joint_ids.size();
    jacobian(0, n_joint + 0) = 1.0;
    jacobian(1, n_joint + 1) = 1.0;

    auto rpy_tweaked = rpy_rlink_to_blink;
    rpy_tweaked[2] += eps;

    Transform tf_rlink_to_blink_tweaked = tf_rlink_to_blink;
    tf_rlink_to_blink_tweaked.setQuaternionFromRPY(rpy_tweaked);
    Transform tf_rlink_to_elink_tweaked =
        tf_rlink_to_blink_tweaked * tf_blink_to_elink;
    auto pose_out = tf_rlink_to_elink_tweaked;

    const auto pos_diff = pose_out.trans() - tf_rlink_to_elink.trans();
    jacobian.template block<3, 1>(0, n_joint + 2) = pos_diff / eps;
  }
  return jacobian;
}

template <typename Scalar>
typename KinematicModel<Scalar>::Vector3 KinematicModel<Scalar>::get_com() {
  Vector3 com_average = Vector3::Zero();
  Scalar mass_total = 0.0;
  for (size_t iter = 0; iter < com_link_ids_.size(); iter++) {
    const auto& tf_base_to_link = get_link_pose(com_link_ids_[iter]);
    const Vector3&& tf_base_to_com_trans =
        tf_base_to_link.trans() +
        tf_base_to_link.quat().toRotationMatrix() * com_local_positions_[iter];
    com_average += link_masses_[iter] * tf_base_to_com_trans;
    mass_total += link_masses_[iter];
  }
  com_average /= mass_total;
  return com_average;
}

template <typename Scalar>
typename KinematicModel<Scalar>::MatrixDynamic
KinematicModel<Scalar>::get_com_jacobian(const std::vector<size_t>& joint_ids,
                                         BaseType base_type) {
  constexpr size_t jac_rank = 3;
  const size_t dim_dof = joint_ids.size() +
                         (base_type == BaseType::FLOATING) * 6 +
                         (base_type == BaseType::PLANAR) * 3;
  MatrixDynamic jac_average = MatrixDynamic::Zero(jac_rank, dim_dof);
  Scalar mass_total = 0.0;
  for (size_t iter = 0; iter < com_link_ids_.size(); iter++) {
    const auto& tf_base_to_link = get_link_pose(com_link_ids_[iter]);
    const Vector3&& tf_base_to_com_trans =
        tf_base_to_link.trans() +
        tf_base_to_link.quat().toRotationMatrix() * com_local_positions_[iter];
    auto jac = this->get_attached_point_jacobian(
        com_link_ids_[iter], tf_base_to_com_trans, joint_ids, base_type);
    mass_total += link_masses_[iter];
    jac_average += link_masses_[iter] * jac;
  }
  jac_average /= mass_total;
  return jac_average;
}

template class KinematicModel<double>;
template class KinematicModel<float>;

};  // namespace plainmp::kinematics
