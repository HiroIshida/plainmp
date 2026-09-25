/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2026 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#include <typeinfo>

#include "plainmp/constraints/primitive_sphere_collision.hpp"

namespace plainmp::constraint {

namespace {
// Return the available clearance, or +infinity when the requested margin is
// already certified. Retain squared distances until a smaller interval must be
// computed; plane/face cases need no square root at all.
double analytic_clearance(const collision::PrimitiveSDFBase& sdf,
                          const Eigen::Vector3d& point, double radius,
                          double required, int kind) {
  const double target = radius + required;
  const double infinity = std::numeric_limits<double>::infinity();
  if (kind == collision::BOX) {
    const auto& box = static_cast<const collision::BoxSDF&>(sdf);
    Eigen::Vector3d p = point - box.pose.position_;
    if (!box.pose.axis_aligned_) p = (box.pose.rot_inv_ * p).eval();
    const Eigen::Vector3d d = p.cwiseAbs() - box.get_width() * .5;
    if (d.maxCoeff() > target) return infinity;
    double squared = 0, single = 0;
    unsigned count = 0;
    for (int k = 0; k < 3; ++k) {
      if (d[k] > 0) {
        squared += d[k] * d[k];
        single = d[k];
        ++count;
      }
    }
    if (count == 0) return -radius;
    if (count == 1) return single - radius;
    if (squared > target * target) return infinity;
    return std::sqrt(squared) - radius;
  }
  if (kind == collision::CYLINDER) {
    const auto& cylinder = static_cast<const collision::CylinderSDF&>(sdf);
    Eigen::Vector3d p = point - cylinder.pose.position_;
    if (!cylinder.pose.z_axis_aligned_) p = (cylinder.pose.rot_inv_ * p).eval();
    const double z = std::abs(p.z()) - cylinder.get_half_height();
    if (z > target) return infinity;
    const double squared = p.x() * p.x() + p.y() * p.y();
    const double r = cylinder.get_radius();
    const double expanded = r + target;
    if (squared > expanded * expanded) return infinity;
    if (squared <= r * r) return std::max(z, 0.0) - radius;
    const double radial = std::sqrt(squared) - r;
    if (z <= 0) return radial - radius;
    const double corner = radial * radial + z * z;
    if (corner > target * target) return infinity;
    return std::sqrt(corner) - radius;
  }
  if (kind == collision::SPHERE) {
    const auto& sphere = static_cast<const collision::SphereSDF&>(sdf);
    const double squared = (point - sphere.pose.position_).squaredNorm();
    const double r = sphere.get_radius();
    const double expanded = r + target;
    if (squared > expanded * expanded) return infinity;
    return std::sqrt(squared) - r - radius;
  }
  return sdf.evaluate(point) - radius;
}
}  // namespace

struct ScalarMotionBounds {
  bool supported = false;
  size_t links, joints;
  std::vector<std::vector<double>> group_speed, pair_speed;
  std::vector<double> error, group_margin, pair_margin;
  std::vector<double> absolute_delta;
  std::vector<size_t> uncontrolled_joints;
  std::vector<int> kinds;
  double maximum_rate = 0, minimum_rate = 0, rounding = 0;

  ScalarMotionBounds(const kin::KinematicModel<double>& model,
                     const std::vector<size_t>& controls,
                     const std::vector<SphereGroup>& groups,
                     const std::vector<std::pair<size_t, size_t>>& pairs)
      : links(model.link_parent_link_ids_.size()),
        joints(controls.size()),
        group_speed(groups.size(), std::vector<double>(joints)),
        pair_speed(pairs.size(), std::vector<double>(joints)),
        error(groups.size()),
        group_margin(groups.size()),
        pair_margin(pairs.size()),
        absolute_delta(joints) {
    for (bool rotate : model.link_consider_rotation_)
      if (!rotate) return;
    std::vector<int> incoming(links, -1),
        controlled(model.joint_types_.size(), -1);
    for (size_t j = 0; j < controls.size(); ++j) controlled[controls[j]] = j;
    for (size_t j = 0; j < model.joint_types_.size(); ++j) {
      incoming[model.joint_child_link_ids_[j]] = j;
      if (controlled[j] < 0) uncontrolled_joints.push_back(j);
      if (!(std::abs(model.joint_axes_[j].squaredNorm() - 1) <= 1e-12) ||
          !(std::abs(model.joint_orientations_[j].squaredNorm() - 1) <= 1e-12))
        return;
    }
    for (const auto& f : model.tf_plink_to_hlink_cache_)
      if (!f.trans().allFinite() || !f.quat().coeffs().allFinite()) return;
    std::vector<std::vector<bool>> relevant(groups.size(),
                                            std::vector<bool>(joints));
    for (size_t g = 0; g < groups.size(); ++g) {
      double reach = 0;
      for (size_t i = 0; i < groups[g].radii.size(); ++i) {
        // The legacy box predicate special-cases radii below 1e-6 and is
        // not equivalent to a Euclidean distance predicate in that range.
        if (!std::isfinite(groups[g].radii[i]) || groups[g].radii[i] < 1e-6 ||
            !groups[g].sphere_relative_positions.col(i).allFinite())
          return;
        reach =
            std::max(reach, groups[g].sphere_relative_positions.col(i).norm());
      }
      size_t link = groups[g].parent_link_id, depth = 0;
      while (link != model.root_link_id_) {
        if (++depth > 64) return;
        const int j = incoming[link];
        if (j >= 0) {
          const int c = controlled[j];
          if (c >= 0) {
            relevant[g][c] = true;
            group_speed[g][c] = model.joint_types_[j] == urdf::Joint::PRISMATIC
                                    ? model.joint_axes_[j].norm()
                                    : reach;
          }
          reach += model.joint_positions_[j].norm();
          if (model.joint_types_[j] == urdf::Joint::PRISMATIC) {
            const auto limits = model.joint_position_limits_[j];
            const double bound =
                std::max(std::abs(limits.first), std::abs(limits.second));
            if (!std::isfinite(bound)) return;
            reach += model.joint_axes_[j].norm() * bound;
          }
        } else {
          const auto& f = model.tf_plink_to_hlink_cache_[link];
          if (std::abs(f.quat().squaredNorm() - 1) > 1e-12) return;
          reach += f.trans().norm();
        }
        link = model.link_parent_link_ids_[link];
      }
      // Taylor sin/cos errors on [-pi/2,pi/2] give joint quaternion error
      // <4e-5, including normalized-origin rounding. Products accumulate to e.
      // For Eigen's unnormalized quaternion rotation, ||R(q)-R(u)|| <= 4e+2e^2.
      const double e = std::pow(1.00004, depth) - 1;
      error[g] = (4 * e + 2 * e * e) * reach + 1e-10 * (1 + reach);
      if (!std::isfinite(error[g])) return;
    }
    for (size_t p = 0; p < pairs.size(); ++p) {
      const auto [a, b] = pairs[p];
      for (size_t j = 0; j < joints; ++j)
        if (relevant[a][j] != relevant[b][j])
          pair_speed[p][j] = group_speed[a][j] + group_speed[b][j];
    }
    supported = true;
  }
};

bool SphereCollisionCst::prepare_motion_certificate(const VectorInput& start,
                                                    const VectorInput& end,
                                                    double rate_radius) {
  motion_certificate_prepared_ = false;
  if (base_type_ != kin::BaseType::FIXED ||
      typeid(*this) != typeid(SphereCollisionCst) || start.size() != q_dim() ||
      end.size() != q_dim() || !std::isfinite(rate_radius) ||
      rate_radius <= 0 || !kin_->base_pose_.trans().allFinite() ||
      !(std::abs(kin_->base_pose_.quat().squaredNorm() - 1) <= 1e-12))
    return false;
  double scene_scale = kin_->base_pose_.trans().cwiseAbs().maxCoeff();
  for (const auto& sdf : all_sdfs_cache_) {
    const auto& type = typeid(*sdf);
    if (type != typeid(collision::BoxSDF) &&
        type != typeid(collision::SphereSDF) &&
        type != typeid(collision::CylinderSDF) &&
        type != typeid(collision::GroundSDF))
      return false;
    if (type == typeid(collision::BoxSDF) ||
        type == typeid(collision::CylinderSDF)) {
      const auto& pose =
          static_cast<const collision::TransformableSDFBase&>(*sdf).pose;
      if (!pose.position_.allFinite() || !pose.rot_.allFinite() ||
          (pose.rot_.transpose() * pose.rot_ - Eigen::Matrix3d::Identity())
                  .cwiseAbs()
                  .maxCoeff() > 1e-12)
        return false;
      // Box's z-aligned predicate uses world z; require its distance function
      // to describe the same geometry, including after rotate_z().
      if (type == typeid(collision::BoxSDF) && pose.z_axis_aligned_ &&
          (pose.rot_(0, 2) != 0 || pose.rot_(1, 2) != 0 ||
           pose.rot_(2, 0) != 0 || pose.rot_(2, 1) != 0 ||
           pose.rot_(2, 2) != 1))
        return false;
      scene_scale = std::max(scene_scale, pose.position_.cwiseAbs().maxCoeff());
    } else if (type == typeid(collision::SphereSDF)) {
      const auto& p =
          static_cast<const collision::SphereSDF&>(*sdf).pose.position_;
      if (!p.allFinite()) return false;
      scene_scale = std::max(scene_scale, p.cwiseAbs().maxCoeff());
    }
  }
  if (!motion_bounds_ ||
      motion_bounds_->links != kin_->link_parent_link_ids_.size())
    motion_bounds_ = std::make_shared<ScalarMotionBounds>(
        *kin_, control_joint_ids_, sphere_groups_, selcol_group_id_pairs_);
  auto& bounds = *motion_bounds_;
  if (!bounds.supported) return false;
  if (bounds.kinds.size() != all_sdfs_cache_.size()) {
    bounds.kinds.clear();
    for (const auto& sdf : all_sdfs_cache_) {
      bool valid = true;
      if (sdf->get_type() == collision::BOX) {
        const auto width =
            static_cast<const collision::BoxSDF&>(*sdf).get_width();
        valid = width.allFinite() && width.minCoeff() >= 0;
      } else if (sdf->get_type() == collision::CYLINDER) {
        const auto& cylinder = static_cast<const collision::CylinderSDF&>(*sdf);
        valid = std::isfinite(cylinder.get_radius()) &&
                cylinder.get_radius() >= 0 &&
                std::isfinite(cylinder.get_half_height()) &&
                cylinder.get_half_height() >= 0;
      } else if (sdf->get_type() == collision::SPHERE) {
        const double radius =
            static_cast<const collision::SphereSDF&>(*sdf).get_radius();
        valid = std::isfinite(radius) && radius >= 0;
      } else {
        valid = std::isfinite(sdf->evaluate(Eigen::Vector3d::Zero()));
      }
      if (!valid) {
        bounds.supported = false;
        return false;
      }
      bounds.kinds.push_back(sdf->get_type());
    }
  }
  for (size_t j : bounds.uncontrolled_joints) {
    const double value = kin_->joint_angles_[j];
    if (!(std::abs(value) <= 1000)) return false;
    if (kin_->joint_types_[j] == urdf::Joint::PRISMATIC) {
      const auto limits = kin_->joint_position_limits_[j];
      if (value < limits.first || value > limits.second) return false;
    }
  }
  for (size_t c = 0; c < q_dim(); ++c) {
    if (!(std::abs(start[c]) <= 1000 && std::abs(end[c]) <= 1000)) return false;
    bounds.absolute_delta[c] = std::abs(end[c] - start[c]);
    const size_t j = control_joint_ids_[c];
    if (kin_->joint_types_[j] == urdf::Joint::PRISMATIC) {
      const auto limits = kin_->joint_position_limits_[j];
      if (std::min(start[c], end[c]) < limits.first ||
          std::max(start[c], end[c]) > limits.second)
        return false;
    }
  }
  const double round_guard = 1e-10 * (1 + scene_scale);
  bounds.maximum_rate = rate_radius;
  bounds.minimum_rate = rate_radius / 6.0;
  bounds.rounding = round_guard;
  for (size_t g = 0; g < sphere_groups_.size(); ++g) {
    double margin = 0;
    for (size_t j = 0; j < q_dim(); ++j)
      margin += bounds.group_speed[g][j] * bounds.absolute_delta[j];
    bounds.group_margin[g] = margin;
  }
  for (size_t p = 0; p < selcol_group_id_pairs_.size(); ++p) {
    double margin = 0;
    for (size_t j = 0; j < q_dim(); ++j)
      margin += bounds.pair_speed[p][j] * bounds.absolute_delta[j];
    bounds.pair_margin[p] = margin;
  }
  motion_certificate_prepared_ = true;
  return true;
}

bool SphereCollisionCst::is_valid_with_motion_certificate(
    const VectorInput& q, double& certified_radius) {
  update_kintree(q, false);
  post_update_kintree();
  certified_radius = motion_bounds_ ? motion_bounds_->maximum_rate : 0;
  return check_motion_envelope(certified_radius);
}

bool SphereCollisionCst::check_motion_envelope(double& rate) {
  if (!motion_certificate_prepared_ || !motion_bounds_ ||
      !motion_bounds_->supported) {
    rate = 0;
    return is_valid_dirty();
  }
  const auto& bounds = *motion_bounds_;
  for (size_t g = 0; g < sphere_groups_.size(); ++g) {
    auto& group = sphere_groups_[g];
    if (group.only_self_collision || all_sdfs_cache_.empty()) continue;
    group.create_group_sphere_position_cache(kin_);
    const double speed = bounds.group_margin[g];
    const double error = 2 * bounds.error[g] + bounds.rounding;
    for (size_t o = 0; o < all_sdfs_cache_.size(); ++o) {
      const auto& sdf = all_sdfs_cache_[o];
      const double outer = group.group_radius + speed * rate + error;
      if (sdf->is_outside_aabb(group.group_sphere_position_cache, outer) ||
          sdf->is_outside(group.group_sphere_position_cache, outer))
        continue;
      group.create_sphere_position_cache(kin_);
      for (size_t i = 0; i < group.radii.size(); ++i) {
        const double radius = group.radii[i] + speed * rate + error;
        const auto point = group.sphere_positions_cache.col(i);
        if (!sdf->is_outside_aabb(point, radius)) {
          const double clearance =
              analytic_clearance(*sdf, point, group.radii[i],
                                 speed * rate + error, bounds.kinds[o]) -
              error;
          if (clearance > speed * rate) continue;
          if (!(clearance > 0) || speed == 0) {
            rate = 0;
            return finish_point_check(g, o, i, 0, 0, 0);
          }
          rate = std::min(rate, clearance / speed);
          if (!(rate > bounds.minimum_rate)) {
            rate = 0;
            return finish_point_check(g, o, i, 0, 0, 0);
          }
        }
      }
    }
  }
  for (size_t p = 0; p < selcol_group_id_pairs_.size(); ++p) {
    const auto [a, b] = selcol_group_id_pairs_[p];
    auto& x = sphere_groups_[a];
    auto& y = sphere_groups_[b];
    x.create_group_sphere_position_cache(kin_);
    y.create_group_sphere_position_cache(kin_);
    const double speed = bounds.pair_margin[p];
    const double error =
        2 * (bounds.error[a] + bounds.error[b]) + 2 * bounds.rounding;
    const double outer = x.group_radius + y.group_radius + speed * rate + error;
    const double distance_sq =
        (x.group_sphere_position_cache - y.group_sphere_position_cache)
            .squaredNorm();
    if (distance_sq > outer * outer) continue;
    x.create_sphere_position_cache(kin_);
    y.create_sphere_position_cache(kin_);
    for (size_t i = 0; i < x.radii.size(); ++i)
      for (size_t j = 0; j < y.radii.size(); ++j) {
        const double radius = x.radii[i] + y.radii[j] + speed * rate + error;
        const double squared =
            (x.sphere_positions_cache.col(i) - y.sphere_positions_cache.col(j))
                .squaredNorm();
        if (squared <= radius * radius) {
          const double clearance =
              std::sqrt(squared) - (x.radii[i] + y.radii[j]) - error;
          if (!(clearance > 0) || speed == 0) {
            rate = 0;
            return finish_point_check(sphere_groups_.size(), 0, 0, p, i, j);
          }
          rate = std::min(rate, clearance / speed);
          if (!(rate > bounds.minimum_rate)) {
            rate = 0;
            return finish_point_check(sphere_groups_.size(), 0, 0, p, i, j);
          }
        }
      }
  }
  return true;
}

bool SphereCollisionCst::finish_point_check(size_t start_group,
                                            size_t start_sdf,
                                            size_t start_sphere,
                                            size_t start_pair, size_t start_row,
                                            size_t start_column) {
  // Earlier pairs were already certified free. Resume at the unresolved pair
  // instead of restarting the original point query and traversing them again.
  for (size_t g = start_group; g < sphere_groups_.size(); ++g) {
    auto& group = sphere_groups_[g];
    if (group.only_self_collision) continue;
    group.create_group_sphere_position_cache(kin_);
    for (size_t o = g == start_group ? start_sdf : 0;
         o < all_sdfs_cache_.size(); ++o) {
      const auto& sdf = *all_sdfs_cache_[o];
      if (sdf.is_outside_aabb(group.group_sphere_position_cache,
                              group.group_radius) ||
          sdf.is_outside(group.group_sphere_position_cache, group.group_radius))
        continue;
      group.create_sphere_position_cache(kin_);
      const size_t first =
          g == start_group && o == start_sdf ? start_sphere : 0;
      for (size_t i = first; i < group.radii.size(); ++i) {
        const auto point = group.sphere_positions_cache.col(i);
        if (!sdf.is_outside_aabb(point, group.radii[i]) &&
            !sdf.is_outside(point, group.radii[i]))
          return false;
      }
    }
  }
  for (size_t p = start_pair; p < selcol_group_id_pairs_.size(); ++p) {
    const auto [a, b] = selcol_group_id_pairs_[p];
    auto& x = sphere_groups_[a];
    auto& y = sphere_groups_[b];
    x.create_group_sphere_position_cache(kin_);
    y.create_group_sphere_position_cache(kin_);
    const double outer = x.group_radius + y.group_radius;
    if ((x.group_sphere_position_cache - y.group_sphere_position_cache)
            .squaredNorm() > outer * outer)
      continue;
    x.create_sphere_position_cache(kin_);
    y.create_sphere_position_cache(kin_);
    for (size_t i = p == start_pair ? start_row : 0; i < x.radii.size(); ++i)
      for (size_t j = p == start_pair && i == start_row ? start_column : 0;
           j < y.radii.size(); ++j) {
        const double radius = x.radii[i] + y.radii[j];
        if ((x.sphere_positions_cache.col(i) - y.sphere_positions_cache.col(j))
                .squaredNorm() < radius * radius)
          return false;
      }
  }
  return true;
}
}  // namespace plainmp::constraint
