#include "constraint.hpp"
#include <pybind11/stl.h>
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace cst {

template <typename Scalar>
std::pair<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>,
          Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
LinkPoseCst<Scalar>::evaluate_dirty() {
  Values vals(cst_dim());
  MatrixDynamic jac(cst_dim(), this->q_dim());
  size_t head = 0;
  for (size_t i = 0; i < link_ids_.size(); i++) {
    const auto& pose = this->kin_->get_link_pose(link_ids_[i]);

    if (poses_[i].size() == 3) {
      vals.segment(head, 3) = pose.trans() - poses_[i];
      jac.block(head, 0, 3, this->q_dim()) = this->kin_->get_jacobian(
          link_ids_[i], this->control_joint_ids_, tinyfk::RotationType::IGNORE,
          this->with_base_);
      head += 3;
    } else if (poses_[i].size() == 6) {
      vals.segment(head, 3) = pose.trans() - poses_[i].head(3);
      vals.segment(head + 3, 3) = pose.getRPY() - poses_[i].tail(3);
      jac.block(head, 0, 6, this->q_dim()) =
          this->kin_->get_jacobian(link_ids_[i], this->control_joint_ids_,
                                   tinyfk::RotationType::RPY, this->with_base_);
      head += 6;
    } else {
      vals.segment(head, 3) = pose.trans() - poses_[i].head(3);
      vals.segment(head + 3, 4) = pose.quat().coeffs() - poses_[i].tail(4);
      jac.block(head, 0, 7, this->q_dim()) = this->kin_->get_jacobian(
          link_ids_[i], this->control_joint_ids_, tinyfk::RotationType::XYZW,
          this->with_base_);
      head += 7;
    }
  }
  return {vals, jac};
}

template <typename Scalar>
std::pair<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>,
          Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
RelativePoseCst<Scalar>::evaluate_dirty() {
  Values vals(cst_dim());
  MatrixDynamic jac(cst_dim(), this->q_dim());
  const auto& pose_dummy = this->kin_->get_link_pose(dummy_link_id_);
  const auto& pose2 = this->kin_->get_link_pose(link_id2_);
  vals.head(3) = pose_dummy.trans() - pose2.trans();
  vals.segment(3, 4) = pose_dummy.quat().coeffs() - pose2.quat().coeffs();
  jac = this->kin_->get_jacobian(dummy_link_id_, this->control_joint_ids_,
                                 tinyfk::RotationType::XYZW, this->with_base_) -
        this->kin_->get_jacobian(link_id2_, this->control_joint_ids_,
                                 tinyfk::RotationType::XYZW, this->with_base_);
  return {vals, jac};
}

template <typename Scalar>
FixedZAxisCst<Scalar>::FixedZAxisCst(
    std::shared_ptr<tinyfk::KinematicModel<Scalar>> kin,
    const std::vector<std::string>& control_joint_names,
    bool with_base,
    const std::string& link_name)
    : EqConstraintBase<Scalar>(kin, control_joint_names, with_base),
      link_id_(this->kin_->get_link_ids({link_name})[0]) {
  aux_link_ids_.clear();
  {
    auto pose = Transform::Identity();
    pose.trans().x() = 1;
    auto new_link_id = this->kin_->add_new_link(link_id_, pose, false);
    aux_link_ids_.push_back(new_link_id);
  }

  {
    auto pose = Transform::Identity();
    pose.trans().y() = 1;
    auto new_link_id = this->kin_->add_new_link(link_id_, pose, false);
    aux_link_ids_.push_back(new_link_id);
  }
}

template <typename Scalar>
std::pair<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>,
          Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
FixedZAxisCst<Scalar>::evaluate_dirty() {
  const auto& pose_here = this->kin_->get_link_pose(link_id_);
  const auto& pose_plus1_x = this->kin_->get_link_pose(aux_link_ids_[0]);
  const auto& pose_plus1_y = this->kin_->get_link_pose(aux_link_ids_[1]);
  Eigen::Matrix<Scalar, 2, 1> vals(2);
  Scalar diff_plus1_x_z = pose_plus1_x.trans().z() - pose_here.trans().z();
  Scalar diff_plus1_y_z = pose_plus1_y.trans().z() - pose_here.trans().z();
  vals << diff_plus1_x_z, diff_plus1_y_z;

  // jacobian
  MatrixDynamic jac_here(3, this->q_dim());
  MatrixDynamic jac_plus1_x(3, this->q_dim());
  MatrixDynamic jac_plus1_y(3, this->q_dim());
  jac_here =
      this->kin_->get_jacobian(link_id_, this->control_joint_ids_,
                               tinyfk::RotationType::IGNORE, this->with_base_);
  jac_plus1_x =
      this->kin_->get_jacobian(aux_link_ids_[0], this->control_joint_ids_,
                               tinyfk::RotationType::IGNORE, this->with_base_);
  jac_plus1_y =
      this->kin_->get_jacobian(aux_link_ids_[1], this->control_joint_ids_,
                               tinyfk::RotationType::IGNORE, this->with_base_);
  MatrixDynamic jac(2, this->q_dim());
  jac.row(0) = jac_plus1_x.row(2) - jac_here.row(2);
  jac.row(1) = jac_plus1_y.row(2) - jac_here.row(2);
  return {vals, jac};
};

template <typename Scalar>
SphereCollisionCst<Scalar>::SphereCollisionCst(
    std::shared_ptr<tinyfk::KinematicModel<Scalar>> kin,
    const std::vector<std::string>& control_joint_names,
    bool with_base,
    const std::vector<SphereAttachmentSpec<Scalar>>& sphere_specs,
    const std::vector<std::pair<std::string, std::string>>& selcol_group_pairs,
    std::optional<typename SDFBase<Scalar>::Ptr> fixed_sdf)
    : IneqConstraintBase<Scalar>(kin, control_joint_names, with_base),
      fixed_sdf_(fixed_sdf == std::nullopt ? nullptr : *fixed_sdf) {
  for (size_t i = 0; i < sphere_specs.size(); i++) {
    auto& spec = sphere_specs[i];
    auto parent_id = this->kin_->get_link_ids({spec.parent_link_name})[0];
    Point group_center = {0.0, 0.0, 0.0};
    for (size_t j = 0; j < spec.relative_positions.cols(); j++) {
      group_center += spec.relative_positions.col(j);
    }
    group_center /= spec.relative_positions.cols();

    Scalar max_dist = 0.0;
    for (size_t j = 0; j < spec.relative_positions.cols(); j++) {
      Scalar dist = (spec.relative_positions.col(j) - group_center).norm() +
                    spec.radii[j];
      if (dist > max_dist) {
        max_dist = dist;
      }
    }
    Scalar group_radius = max_dist;
    Points sphere_position_cache(3, spec.radii.size());
    sphere_groups_.push_back({spec.parent_link_name, parent_id, spec.radii,
                              group_radius, spec.ignore_collision,
                              spec.relative_positions, group_center,
                              Point::Zero(), true, Point::Zero(), true,
                              sphere_position_cache, true});
  }

  for (const auto& pair : selcol_group_pairs) {
    auto group1 = std::find_if(sphere_groups_.begin(), sphere_groups_.end(),
                               [&pair](const auto& group) {
                                 return group.parent_link_name == pair.first;
                               });
    auto group2 = std::find_if(sphere_groups_.begin(), sphere_groups_.end(),
                               [&pair](const auto& group) {
                                 return group.parent_link_name == pair.second;
                               });
    if (group1 == sphere_groups_.end() || group2 == sphere_groups_.end()) {
      throw std::runtime_error(
          "(cpp) Invalid pair of link names for self collision");
    }
    selcol_group_id_pairs_.push_back(
        {group1 - sphere_groups_.begin(), group2 - sphere_groups_.begin()});
  }
  set_all_sdfs();
}

template <typename Scalar>
bool SphereCollisionCst<Scalar>::is_valid_dirty() {
  if (all_sdfs_cache_.size() == 0) {
    throw std::runtime_error("(cpp) No SDFs are set");
  }
  if (!check_ext_collision()) {
    return false;
  }
  return check_self_collision();
}

template <typename Scalar>
bool SphereCollisionCst<Scalar>::check_ext_collision() {
  for (auto& group : sphere_groups_) {
    if (group.ignore_collision) {
      continue;
    }
    if (group.is_group_sphere_position_dirty) {
      group.create_group_sphere_position_cache(this->kin_);
    }

    for (auto& sdf : all_sdfs_cache_) {
      if (!sdf->is_outside_aabb(group.group_sphere_position_cache,
                                group.group_radius)) {
        if (!sdf->is_outside(group.group_sphere_position_cache,
                             group.group_radius)) {
          // now narrow phase collision checking
          if (group.is_sphere_positions_dirty) {
            group.create_sphere_position_cache(this->kin_);
          }
          for (size_t i = 0; i < group.radii.size(); i++) {
            if (!sdf->is_outside_aabb(group.sphere_positions_cache.col(i),
                                      group.radii[i])) {
              if (!sdf->is_outside(group.sphere_positions_cache.col(i),
                                   group.radii[i])) {
                return false;
              }
            }
          }
        }
      }
    }
  }
  return true;
}

template <typename Scalar>
bool SphereCollisionCst<Scalar>::check_self_collision() {
  for (auto& group_id_pair : selcol_group_id_pairs_) {
    auto& group1 = sphere_groups_[group_id_pair.first];
    auto& group2 = sphere_groups_[group_id_pair.second];
    if (group1.is_group_sphere_position_dirty) {
      group1.create_group_sphere_position_cache(this->kin_);
    }
    if (group2.is_group_sphere_position_dirty) {
      group2.create_group_sphere_position_cache(this->kin_);
    }

    Scalar outer_sqdist = (group1.group_sphere_position_cache -
                           group2.group_sphere_position_cache)
                              .squaredNorm();
    Scalar outer_r_sum = group1.group_radius + group2.group_radius;
    if (outer_sqdist > outer_r_sum * outer_r_sum) {
      continue;
    }

    if (group1.is_sphere_positions_dirty) {
      group1.create_sphere_position_cache(this->kin_);
    }
    if (group2.is_sphere_positions_dirty) {
      group2.create_sphere_position_cache(this->kin_);
    }

    // check if the inner volumes are colliding
    for (size_t i = 0; i < group1.radii.size(); i++) {
      for (size_t j = 0; j < group2.radii.size(); j++) {
        Scalar sqdist = (group1.sphere_positions_cache.col(i) -
                         group2.sphere_positions_cache.col(j))
                            .squaredNorm();
        Scalar r_sum = group1.radii[i] + group2.radii[j];
        if (sqdist < r_sum * r_sum) {
          return false;
        }
      }
    }
  }
  return true;
}

template <typename Scalar>
std::pair<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>,
          Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
SphereCollisionCst<Scalar>::evaluate_dirty() {
  if (all_sdfs_cache_.size() == 0) {
    throw std::runtime_error("(cpp) No SDFs are set");
  }

  // collision vs outers
  Values grad_in_cspace_other(this->q_dim());
  Scalar min_val_other = cutoff_dist_;
  std::optional<size_t> min_sphere_idx = std::nullopt;
  std::optional<size_t> min_group_idx = std::nullopt;
  std::optional<size_t> min_sdf_idx = std::nullopt;
  {
    for (size_t i = 0; i < sphere_groups_.size(); i++) {
      auto& group = sphere_groups_[i];
      if (group.ignore_collision) {
        continue;
      }

      // filter out groups that are not colliding with margin of cutoff
      if (group.is_group_sphere_position_dirty) {
        group.create_group_sphere_position_cache(this->kin_);
      }
      for (size_t j = 0; j < all_sdfs_cache_.size(); j++) {
        auto& sdf = all_sdfs_cache_[j];
        if (!sdf->is_outside_aabb(group.group_sphere_position_cache,
                                  group.group_radius + cutoff_dist_)) {
          if (!sdf->is_outside(group.group_sphere_position_cache,
                               group.group_radius + cutoff_dist_)) {
            // if broad collision with sdf-j detected
            if (group.is_sphere_positions_dirty) {
              group.create_sphere_position_cache(this->kin_);
            }
            for (size_t k = 0; k < group.radii.size(); k++) {
              auto sphere_center = group.sphere_positions_cache.col(k);
              if (sdf->is_outside_aabb(sphere_center,
                                       group.radii[k] + cutoff_dist_)) {
                continue;
              }
              Scalar val = sdf->evaluate(sphere_center) - group.radii[k];
              if (val < min_val_other) {
                min_val_other = val;
                min_group_idx = i;
                min_sdf_idx = j;
                min_sphere_idx = k;
              }
            }
          }
        }
      }
    }

    if (min_sphere_idx == std::nullopt) {
      // cutoff case
      grad_in_cspace_other.setConstant(0.);
    } else {
      // HACK: we already know that the sphere position cache is
      // already created
      const auto& min_sphere_trans =
          sphere_groups_[*min_group_idx].sphere_positions_cache.col(
              *min_sphere_idx);
      Scalar r = sphere_groups_[*min_group_idx].radii[*min_sphere_idx];
      Vector3 grad;
      for (size_t i = 0; i < 3; i++) {
        Vector3 perturbed_center = min_sphere_trans;
        perturbed_center[i] += 1e-6;
        Scalar val =
            all_sdfs_cache_[*min_sdf_idx]->evaluate(perturbed_center) - r;
        grad[i] = (val - min_val_other) / 1e-6;
      }
      auto&& sphere_jac = this->kin_->get_attached_point_jacobian(
          sphere_groups_[*min_group_idx].parent_link_id, min_sphere_trans,
          this->control_joint_ids_, this->with_base_);
      grad_in_cspace_other = sphere_jac.transpose() * grad;
    }
  }
  if (selcol_group_id_pairs_.size() == 0) {
    MatrixDynamic jac(1, grad_in_cspace_other.size());
    jac.row(0) = grad_in_cspace_other;
    return {Values::Constant(1, min_val_other), jac};
  } else {
    // collision vs inners (self collision)
    std::optional<std::array<size_t, 4>> min_pairs =
        std::nullopt;  // (group_i, sphere_i, group_j, sphere_j)
    Scalar dist_min = cutoff_dist_;
    for (auto& group_id_pair : selcol_group_id_pairs_) {
      auto& group1 = sphere_groups_[group_id_pair.first];
      auto& group2 = sphere_groups_[group_id_pair.second];

      if (group1.is_group_sphere_position_dirty) {
        group1.create_group_sphere_position_cache(this->kin_);
      }
      if (group2.is_group_sphere_position_dirty) {
        group2.create_group_sphere_position_cache(this->kin_);
      }

      Scalar outer_sqdist = (group1.group_sphere_position_cache -
                             group2.group_sphere_position_cache)
                                .squaredNorm();
      Scalar outer_r_sum_with_margin =
          group1.group_radius + group2.group_radius + cutoff_dist_;
      if (outer_sqdist > outer_r_sum_with_margin * outer_r_sum_with_margin) {
        continue;
      }

      // narrow phase
      if (group1.is_sphere_positions_dirty) {
        group1.create_sphere_position_cache(this->kin_);
      }
      if (group2.is_sphere_positions_dirty) {
        group2.create_sphere_position_cache(this->kin_);
      }

      for (size_t i = 0; i < group1.radii.size(); i++) {
        for (size_t j = 0; j < group2.radii.size(); j++) {
          const auto& sphere1_center = group1.sphere_positions_cache.col(i);
          const auto& sphere2_center = group2.sphere_positions_cache.col(j);
          Scalar dist = (sphere1_center - sphere2_center).norm() -
                        (group1.radii[i] + group2.radii[j]);
          if (dist < dist_min) {
            dist_min = dist;
            min_pairs = {group_id_pair.first, i, group_id_pair.second, j};
          }
        }
      }
    }

    // compute gradient
    if (min_pairs == std::nullopt) {
      MatrixDynamic jac(2, grad_in_cspace_other.size());
      jac.row(0) = grad_in_cspace_other;
      jac.row(1).setConstant(0.);
      return {Eigen::Matrix<Scalar, 2, 1>(min_val_other, dist_min), jac};
    } else {
      // HACK: we know that in the non-gradient evaluation the cache
      // is already created
      auto& group1 = sphere_groups_[min_pairs->at(0)];
      const auto& center1 = group1.sphere_positions_cache.col(min_pairs->at(1));
      auto& group2 = sphere_groups_[min_pairs->at(2)];
      const auto& center2 = group2.sphere_positions_cache.col(min_pairs->at(3));

      Vector3 center_diff = center1 - center2;
      auto&& jac1 = this->kin_->get_attached_point_jacobian(
          group1.parent_link_id, center1, this->control_joint_ids_,
          this->with_base_);
      auto&& jac2 = this->kin_->get_attached_point_jacobian(
          group2.parent_link_id, center2, this->control_joint_ids_,
          this->with_base_);
      Scalar norminv = 1.0 / center_diff.norm();
      Values&& grad_in_cspace_self =
          norminv * center_diff.transpose() * (jac1 - jac2);
      MatrixDynamic jac(2, grad_in_cspace_other.size());
      jac.row(0) = grad_in_cspace_other;
      jac.row(1) = grad_in_cspace_self;
      return {Eigen::Matrix<Scalar, 2, 1>(min_val_other, dist_min), jac};
    }
  }
}

template <typename Scalar>
std::vector<std::pair<Eigen::Matrix<Scalar, 3, 1>, Scalar>>
SphereCollisionCst<Scalar>::get_group_spheres() const {
  std::vector<std::pair<Eigen::Matrix<Scalar, 3, 1>, Scalar>> spheres;
  // for (auto& sphere_group : sphere_groups_) {
  //   const auto& pose =
  //   this->kin_->get_link_pose(sphere_group.group_sphere_id);
  //   spheres.push_back({pose.trans(), sphere_group.group_radius});
  // }
  return spheres;
}

template <typename Scalar>
std::vector<std::pair<Eigen::Matrix<Scalar, 3, 1>, Scalar>>
SphereCollisionCst<Scalar>::get_all_spheres() const {
  std::vector<std::pair<Eigen::Matrix<Scalar, 3, 1>, Scalar>> spheres;
  // for (auto& sphere_group : sphere_groups_) {
  //   for (size_t i = 0; i < sphere_group.sphere_ids.size(); i++) {
  //     const auto& pose =
  //     this->kin_->get_link_pose(sphere_group.sphere_ids[i]);
  //     spheres.push_back({pose.trans(), sphere_group.radii[i]});
  //   }
  // }
  return spheres;
}

template <typename Scalar>
void SphereCollisionCst<Scalar>::set_all_sdfs() {
  all_sdfs_cache_.clear();
  if (fixed_sdf_ != nullptr) {
    set_all_sdfs_inner(fixed_sdf_);
  }
  if (sdf_ != nullptr) {
    set_all_sdfs_inner(sdf_);
  }
}

template <typename Scalar>
void SphereCollisionCst<Scalar>::set_all_sdfs_inner(
    typename SDFBase<Scalar>::Ptr sdf) {
  if (sdf->get_type() == SDFType::UNION) {
    for (auto& sub_sdf :
         std::static_pointer_cast<UnionSDF<Scalar>>(sdf)->sdfs_) {
      set_all_sdfs_inner(sub_sdf);
    }
  } else {
    auto primitive_sdf =
        std::static_pointer_cast<PrimitiveSDFBase<Scalar>>(sdf);
    all_sdfs_cache_.push_back(primitive_sdf);
  }
}

template <typename Scalar>
bool ComInPolytopeCst<Scalar>::is_valid_dirty() {
  // COPIED from evaluate() >> START
  auto com = this->kin_->get_com();
  if (force_link_ids_.size() > 0) {
    Scalar vertical_force_sum = 1.0;  // 1.0 for normalized self
    for (size_t j = 0; j < force_link_ids_.size(); ++j) {
      Scalar force = applied_force_values_[j] / this->kin_->total_mass_;
      vertical_force_sum += force;
      const auto& pose = this->kin_->get_link_pose(force_link_ids_[j]);
      com += force * pose.trans();
    }
    com /= vertical_force_sum;
  }
  // COPIED from evaluate() >> END
  return polytope_sdf_->evaluate(com) < 0;
}

template <typename Scalar>
std::pair<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>,
          Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
ComInPolytopeCst<Scalar>::evaluate_dirty() {
  Values vals(cst_dim());
  MatrixDynamic jac(cst_dim(), this->q_dim());

  auto com = this->kin_->get_com();
  auto com_jaco =
      this->kin_->get_com_jacobian(this->control_joint_ids_, this->q_dim());
  if (force_link_ids_.size() > 0) {
    Scalar vertical_force_sum = 1.0;  // 1.0 for normalized self
    for (size_t j = 0; j < force_link_ids_.size(); ++j) {
      Scalar force = applied_force_values_[j] / this->kin_->total_mass_;
      vertical_force_sum += force;
      const auto& pose = this->kin_->get_link_pose(force_link_ids_[j]);
      com += force * pose.trans();

      com_jaco += this->kin_->get_jacobian(
                      force_link_ids_[j], this->control_joint_ids_,
                      tinyfk::RotationType::IGNORE, this->with_base_) *
                  force;
    }
    Scalar inv = 1.0 / vertical_force_sum;
    com *= inv;
    com_jaco *= inv;
  }
  Scalar val = -polytope_sdf_->evaluate(com);
  vals[0] = val;

  Vector3 grad;
  for (size_t i = 0; i < 3; i++) {
    Vector3 perturbed_com = com;
    perturbed_com[i] += 1e-6;
    Scalar val_perturbed = -polytope_sdf_->evaluate(perturbed_com);
    grad[i] = (val_perturbed - val) / 1e-6;
  }
  jac.row(0) = com_jaco.transpose() * grad;

  return {vals, jac};
};

// explicit instantiation all the classes
template class ConstraintBase<double>;
template class EqConstraintBase<double>;
template class IneqConstraintBase<double>;
template class ConfigPointCst<double>;
template class LinkPoseCst<double>;
template class RelativePoseCst<double>;
template class FixedZAxisCst<double>;
template class SphereCollisionCst<double>;
template class ComInPolytopeCst<double>;

template class ConstraintBase<float>;
template class EqConstraintBase<float>;
template class IneqConstraintBase<float>;
template class ConfigPointCst<float>;
template class LinkPoseCst<float>;
template class RelativePoseCst<float>;
template class FixedZAxisCst<float>;
template class SphereCollisionCst<float>;
template class ComInPolytopeCst<float>;

}  // namespace cst
