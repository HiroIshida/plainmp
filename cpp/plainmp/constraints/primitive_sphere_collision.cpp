/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "plainmp/constraints/primitive_sphere_collision.hpp"
#include <cmath>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif
#include <numeric>
#include <type_traits>
#include <unordered_set>
#include "plainmp/kinematics/kinematics.hpp"

namespace plainmp::constraint {

/* NOTE: The create_group_sphere_position_cache function must be called before
 * create_sphere_position_cache to optimize performance.
 *
 * create_group_sphere_position_cache updates the rot_mat_cache, which is also
 * needed by create_sphere_position_cache. By enforcing this call order, we can
 * avoid redundant checks of rot_mat_cache in create_sphere_position_cache,
 * since we know it has already been updated.
 */
void SphereGroup::create_group_sphere_position_cache(
    const std::shared_ptr<kin::KinematicModel<double>>& kin) {
  if (!is_group_sphere_position_dirty) {
    return;
  }
  auto plink_pose = kin->get_link_pose(parent_link_id);
  this->rot_mat_cache = plink_pose.quat().toRotationMatrix();
  this->group_sphere_position_cache =
      this->rot_mat_cache * group_sphere_relative_position + plink_pose.trans();
  this->is_group_sphere_position_dirty = false;
}

void SphereGroup::create_sphere_position_cache(
    const std::shared_ptr<kin::KinematicModel<double>>& kin) {
  if (!is_sphere_positions_dirty) {
    return;
  }
  auto& plink_trans = kin->transform_cache_.data_[parent_link_id].trans();

  // NOTE: the for-loop below is faster than batch operation using Colwise
  for (int i = 0; i < sphere_positions_cache.cols(); i++) {
    sphere_positions_cache.col(i) =
        this->rot_mat_cache * sphere_relative_positions.col(i) + plink_trans;
  }
  this->is_sphere_positions_dirty = false;
}

void SphereGroup::initialize_radius_cache() {
  max_sphere_radius = -1.0;
  uniform_radii = false;
  if (radii.size() > 0 && radii.allFinite() && radii.minCoeff() >= 0.0) {
    max_sphere_radius = radii.maxCoeff();
    uniform_radii = (radii.array() == max_sphere_radius).all();
  }
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
// Keep xyz in three lanes of one vector, so the transform, reductions and
// validity checks do not compete for separate scalar registers. AVX is
// selected at runtime; the build's baseline ISA and Eigen settings stay intact.
template <bool Uniform>
__attribute__((target("avx"))) static void create_bounds_avx(
    SphereGroup& group,
    const Eigen::Vector3d& translation) {
  const auto& rotation = group.rot_mat_cache;
  const auto& relative = group.sphere_relative_positions;
  const auto& radii = group.radii;
  // Intrinsic stores may alias Eigen's owning objects in the compiler's
  // analysis. Capture their pointers and size before writing any positions,
  // so the loop does not reload those fields after every store.
  double* const positions = group.sphere_positions_cache.data();
  const double* const relative_data = relative.data();
  const double* const radius_data = radii.data();
  const Eigen::Index count = radii.size();
  const __m256d rx =
      _mm256_set_pd(0.0, rotation(2, 0), rotation(1, 0), rotation(0, 0));
  const __m256d ry =
      _mm256_set_pd(0.0, rotation(2, 1), rotation(1, 1), rotation(0, 1));
  const __m256d rz =
      _mm256_set_pd(0.0, rotation(2, 2), rotation(1, 2), rotation(0, 2));
  const __m256d t =
      _mm256_set_pd(0.0, translation[2], translation[1], translation[0]);
  __m256d lower = _mm256_set1_pd(std::numeric_limits<double>::infinity());
  __m256d upper = _mm256_set1_pd(-std::numeric_limits<double>::infinity());
  __m256d sum = _mm256_setzero_pd();
  for (Eigen::Index i = 0; i < count; ++i) {
    const double* p = relative_data + 3 * i;
    const __m256d px = _mm256_broadcast_sd(p);
    const __m256d py = _mm256_broadcast_sd(p + 1);
    const __m256d pz = _mm256_broadcast_sd(p + 2);
    // Match Eigen's scalar product order: (Ry*y + Rz*z) + Rx*x + t.
    // Store exactly three doubles; the unused lane must not overwrite the
    // next sphere, or access beyond the final column.
    const __m256d position = _mm256_add_pd(
        _mm256_add_pd(
            _mm256_add_pd(_mm256_mul_pd(ry, py), _mm256_mul_pd(rz, pz)),
            _mm256_mul_pd(rx, px)),
        t);
    _mm_storeu_pd(positions + 3 * i, _mm256_castpd256_pd128(position));
    _mm_store_sd(positions + 3 * i + 2, _mm256_extractf128_pd(position, 1));
    // NaN/Inf must survive later min/max operations. A finite-input
    // overflow also disables pruning, conservatively.
    sum = _mm256_add_pd(sum, position);
    if constexpr (Uniform) {
      lower = _mm256_min_pd(lower, position);
      upper = _mm256_max_pd(upper, position);
    } else {
      const __m256d radius = _mm256_broadcast_sd(radius_data + i);
      lower = _mm256_min_pd(lower, _mm256_sub_pd(position, radius));
      upper = _mm256_max_pd(upper, _mm256_add_pd(position, radius));
    }
  }
  if constexpr (Uniform) {
    const __m256d radius = _mm256_broadcast_sd(radius_data);
    lower = _mm256_sub_pd(lower, radius);
    upper = _mm256_add_pd(upper, radius);
  }
  // Use the same outward margin as the scalar path, including cancellation.
  const __m256d sign = _mm256_set1_pd(-0.0);
  const __m256d magnitude = _mm256_max_pd(_mm256_andnot_pd(sign, lower),
                                          _mm256_andnot_pd(sign, upper));
  const __m256d rounding = _mm256_add_pd(
      _mm256_mul_pd(
          _mm256_set1_pd(4.0 * std::numeric_limits<double>::epsilon()),
          magnitude),
      _mm256_set1_pd(std::numeric_limits<double>::denorm_min()));
  lower = _mm256_sub_pd(lower, rounding);
  upper = _mm256_add_pd(upper, rounding);
  const __m256d inf = _mm256_set1_pd(std::numeric_limits<double>::infinity());
  const __m256d finite = _mm256_and_pd(
      _mm256_cmp_pd(_mm256_andnot_pd(sign, sum), inf, _CMP_LT_OQ),
      _mm256_and_pd(
          _mm256_cmp_pd(_mm256_andnot_pd(sign, lower), inf, _CMP_LT_OQ),
          _mm256_cmp_pd(_mm256_andnot_pd(sign, upper), inf, _CMP_LT_OQ)));
  _mm_storeu_pd(group.aabb_lb_cache.data(), _mm256_castpd256_pd128(lower));
  _mm_store_sd(group.aabb_lb_cache.data() + 2, _mm256_extractf128_pd(lower, 1));
  _mm_storeu_pd(group.aabb_ub_cache.data(), _mm256_castpd256_pd128(upper));
  _mm_store_sd(group.aabb_ub_cache.data() + 2, _mm256_extractf128_pd(upper, 1));
  group.is_sphere_positions_dirty = false;
  group.is_aabb_valid = (_mm256_movemask_pd(finite) & 7) == 7;
}
#endif

void SphereGroup::create_aabb_cache(
    const std::shared_ptr<kin::KinematicModel<double>>& kin) {
  if (!is_aabb_dirty) {
    return;
  }
  is_aabb_dirty = false;
  is_aabb_valid = false;
  if (max_sphere_radius < 0.0) {
    create_sphere_position_cache(kin);
    return;
  }

  Eigen::Vector3d lower, upper;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  if (is_sphere_positions_dirty && __builtin_cpu_supports("avx")) {
    const auto& trans = kin->transform_cache_.data_[parent_link_id].trans();
    if (uniform_radii) {
      create_bounds_avx<true>(*this, trans);
    } else {
      create_bounds_avx<false>(*this, trans);
    }
    return;
  } else
#endif
  {
    // Keep the reductions scalar and materialize Eigen bounds only once below.
    const double inf = std::numeric_limits<double>::infinity();
    double lx = inf, ly = inf, lz = inf;
    double ux = -inf, uy = -inf, uz = -inf;
    // A non-finite center poisons this sum permanently, even if min/max
    // discard its NaNs. Check once after the loop instead of branching on
    // every coordinate. Overflow of otherwise finite centers also falls back.
    double coordinate_sum = 0.0;
    const auto extend_bounds = [&](const Eigen::Vector3d& p, Eigen::Index i,
                                   auto uniform) {
      coordinate_sum += p.sum();
      // These specializations keep the radius branch outside the hot loop.
      if constexpr (decltype(uniform)::value) {
        lx = std::min(p.x(), lx);
        ly = std::min(p.y(), ly);
        lz = std::min(p.z(), lz);
        ux = std::max(p.x(), ux);
        uy = std::max(p.y(), uy);
        uz = std::max(p.z(), uz);
      } else {
        const double r = radii[i];
        lx = std::min(p.x() - r, lx);
        ly = std::min(p.y() - r, ly);
        lz = std::min(p.z() - r, lz);
        ux = std::max(p.x() + r, ux);
        uy = std::max(p.y() + r, uy);
        uz = std::max(p.z() + r, uz);
      }
    };
    const auto create_bounds = [&](auto uniform) {
      if (is_sphere_positions_dirty) {
        const auto& trans = kin->transform_cache_.data_[parent_link_id].trans();
        for (Eigen::Index i = 0; i < radii.size(); ++i) {
          const Eigen::Vector3d position =
              rot_mat_cache * sphere_relative_positions.col(i) + trans;
          sphere_positions_cache.col(i) = position;
          extend_bounds(position, i, uniform);
        }
        is_sphere_positions_dirty = false;
      } else {
        for (Eigen::Index i = 0; i < radii.size(); ++i) {
          extend_bounds(sphere_positions_cache.col(i), i, uniform);
        }
      }
    };
    // For a common radius, monotonicity of subtraction/addition lets us
    // inflate once after reducing the centers, without loosening the bounds.
    if (uniform_radii) {
      create_bounds(std::true_type{});
      lx -= max_sphere_radius;
      ly -= max_sphere_radius;
      lz -= max_sphere_radius;
      ux += max_sphere_radius;
      uy += max_sphere_radius;
      uz += max_sphere_radius;
    } else {
      create_bounds(std::false_type{});
    }
    lower = Eigen::Vector3d(lx, ly, lz);
    upper = Eigen::Vector3d(ux, uy, uz);
    if (!std::isfinite(coordinate_sum)) {
      return;
    }
  }

  // Allow for both p +/- r above and the reversed object_bound +/- r
  // comparison in the original sphere broad phase. Scale by BOTH endpoints:
  // nextafter(p + r) alone is insufficient when p and r nearly cancel.
  const Eigen::Vector3d rounding =
      (4.0 * std::numeric_limits<double>::epsilon() *
       lower.cwiseAbs().cwiseMax(upper.cwiseAbs()).array()) +
      std::numeric_limits<double>::denorm_min();
  aabb_lb_cache = lower - rounding;
  aabb_ub_cache = upper + rounding;
  is_aabb_valid = aabb_lb_cache.allFinite() && aabb_ub_cache.allFinite();
}

std::vector<size_t> reorder_spheres_for_informative_collision_check(
    const Eigen::Matrix3Xd& positions) {
  std::vector<size_t> visited;
  std::unordered_set<size_t> unvisited;
  unvisited.reserve(positions.cols());
  for (size_t i = 0; i < positions.cols(); i++) {
    unvisited.insert(i);
  }

  auto first_unvisited = *unvisited.begin();  // anything is fine
  visited.push_back(first_unvisited);
  unvisited.erase(first_unvisited);

  for (size_t i = 0; i < positions.cols() - 1; i++) {
    double max_dist = 0.0;
    std::unordered_set<size_t>::iterator it_max_idx = unvisited.begin();

    for (auto it_unvisited = unvisited.begin(); it_unvisited != unvisited.end();
         it_unvisited++) {
      double min_dist_to_visited = std::numeric_limits<double>::max();
      size_t idx_unvisited = *it_unvisited;
      for (size_t idx_visited : visited) {
        double dist =
            (positions.col(idx_unvisited) - positions.col(idx_visited)).norm();
        if (dist < min_dist_to_visited) {
          min_dist_to_visited = dist;
        }
      }
      if (min_dist_to_visited > max_dist) {
        max_dist = min_dist_to_visited;
        it_max_idx = it_unvisited;
      }
    }

    visited.push_back(*it_max_idx);
    unvisited.erase(it_max_idx);
  }

  if (visited.size() != positions.cols()) {
    throw std::runtime_error(
        "(cpp) Not all spheres are visited. This should not happen.");
  }
  return visited;
}

void SphereGroup::max_distance_reorder() {
  const auto informative_order =
      reorder_spheres_for_informative_collision_check(
          sphere_relative_positions);
  Eigen::Matrix3Xd reordered_positions(3, sphere_relative_positions.cols());
  Eigen::VectorXd reordered_radii(sphere_relative_positions.cols());

  for (size_t i = 0; i < sphere_relative_positions.cols(); i++) {
    reordered_positions.col(i) =
        sphere_relative_positions.col(informative_order[i]);
    reordered_radii[i] = radii[informative_order[i]];
  }

  // update
  sphere_relative_positions = reordered_positions;
  radii = reordered_radii;
  clear_cache();
};

SphereCollisionCst::SphereCollisionCst(
    std::shared_ptr<kin::KinematicModel<double>> kin,
    const std::vector<std::string>& control_joint_names,
    kin::BaseType base_type,
    const std::vector<SphereAttachmentSpec>& sphere_specs,
    const std::vector<std::pair<std::string, std::string>>& selcol_group_pairs,
    std::optional<plainmp::collision::SDFBase::Ptr> fixed_sdf,
    bool reorder_spheres)
    : IneqConstraintBase(kin, control_joint_names, base_type),
      fixed_sdf_(fixed_sdf == std::nullopt ? nullptr : *fixed_sdf) {
  for (size_t i = 0; i < sphere_specs.size(); i++) {
    auto& spec = sphere_specs[i];
    auto parent_id = kin_->get_link_ids({spec.parent_link_name})[0];
    Eigen::Vector3d group_center = {0.0, 0.0, 0.0};
    for (size_t j = 0; j < spec.relative_positions.cols(); j++) {
      group_center += spec.relative_positions.col(j);
    }
    group_center /= spec.relative_positions.cols();

    double max_dist = 0.0;
    for (size_t j = 0; j < spec.relative_positions.cols(); j++) {
      double dist = (spec.relative_positions.col(j) - group_center).norm() +
                    spec.radii[j];
      if (dist > max_dist) {
        max_dist = dist;
      }
    }
    double group_radius = max_dist;
    Eigen::Matrix3Xd sphere_position_cache(3, spec.radii.size());
    sphere_groups_.push_back({spec.parent_link_name, parent_id, spec.radii,
                              group_radius, spec.only_self_collision,
                              spec.relative_positions, group_center,
                              Eigen::Matrix3d::Zero(), Eigen::Vector3d::Zero(),
                              sphere_position_cache});
    sphere_groups_.back().initialize_radius_cache();

    // 2024/12/16: This optimization is likely improve the performance of
    // collision checking and probably never make it worse. However, currently,
    // the performance improvement is not significant or visible.
    if (reorder_spheres) {
      for (auto& group : sphere_groups_) {
        group.max_distance_reorder();
      }
    }
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

bool SphereCollisionCst::is_valid_dirty() {
  if (ext_colliision_enabled() && !check_ext_collision()) {
    return false;
  }
  if (self_collision_enabled()) {
    return check_self_collision();
  }
  return true;
}

bool SphereCollisionCst::check_ext_collision() {
  for (auto& group : sphere_groups_) {
    if (group.only_self_collision) {
      continue;
    }
    group.create_group_sphere_position_cache(kin_);

    // check against all SDFs
    for (auto& sdf : all_sdfs_cache_) {
      if (group.is_aabb_valid
              ? sdf->is_outside_aabb(group.aabb_lb_cache, group.aabb_ub_cache)
              : sdf->is_outside_aabb(group.group_sphere_position_cache,
                                     group.group_radius)) {
        continue;
      }
      if (sdf->is_outside(group.group_sphere_position_cache,
                          group.group_radius)) {
        continue;
      }
      if (group.is_aabb_dirty) {
        group.create_aabb_cache(kin_);
        if (group.is_aabb_valid &&
            sdf->is_outside_aabb(group.aabb_lb_cache, group.aabb_ub_cache)) {
          continue;
        }
      }
      // Positions were prepared along with the AABB, even if its bounds
      // were non-finite. Keep the original order and early collision exit.
      for (size_t i = 0; i < group.radii.size(); i++) {
        if (!sdf->is_outside_aabb(group.sphere_positions_cache.col(i),
                                  group.radii[i]) &&
            !sdf->is_outside(group.sphere_positions_cache.col(i),
                             group.radii[i])) {
          return false;
        }
      }
    }
  }
  return true;
}

bool SphereCollisionCst::check_self_collision() {
  for (auto& group_id_pair : selcol_group_id_pairs_) {
    auto& group1 = sphere_groups_[group_id_pair.first];
    auto& group2 = sphere_groups_[group_id_pair.second];
    group1.create_group_sphere_position_cache(kin_);
    group2.create_group_sphere_position_cache(kin_);

    double outer_sqdist = (group1.group_sphere_position_cache -
                           group2.group_sphere_position_cache)
                              .squaredNorm();
    double outer_r_sum = group1.group_radius + group2.group_radius;
    if (outer_sqdist > outer_r_sum * outer_r_sum) {
      continue;
    }

    group1.create_sphere_position_cache(kin_);
    group2.create_sphere_position_cache(kin_);

    // check if the inner volumes are colliding
    for (size_t i = 0; i < group1.radii.size(); i++) {
      for (size_t j = 0; j < group2.radii.size(); j++) {
        double sqdist = (group1.sphere_positions_cache.col(i) -
                         group2.sphere_positions_cache.col(j))
                            .squaredNorm();
        double r_sum = group1.radii[i] + group2.radii[j];
        if (sqdist < r_sum * r_sum) {
          return false;
        }
      }
    }
  }
  return true;
}

double SphereCollisionCst::evaluate_ext_collision(
    Eigen::Ref<Eigen::RowVectorXd, 0, Eigen::InnerStride<Eigen::Dynamic>>
        grad_out) {
  double min_val_other = cutoff_dist_;
  std::optional<size_t> min_sphere_idx = std::nullopt;
  std::optional<size_t> min_group_idx = std::nullopt;
  std::optional<size_t> min_sdf_idx = std::nullopt;
  for (size_t i = 0; i < sphere_groups_.size(); i++) {
    auto& group = sphere_groups_[i];
    if (group.only_self_collision) {
      continue;
    }

    // filter out groups that are not colliding with margin of cutoff
    group.create_group_sphere_position_cache(kin_);
    for (size_t j = 0; j < all_sdfs_cache_.size(); j++) {
      auto& sdf = all_sdfs_cache_[j];
      if (!sdf->is_outside_aabb(group.group_sphere_position_cache,
                                group.group_radius + cutoff_dist_)) {
        if (!sdf->is_outside(group.group_sphere_position_cache,
                             group.group_radius + cutoff_dist_)) {
          // if broad collision with sdf-j detected
          group.create_sphere_position_cache(kin_);
          for (size_t k = 0; k < group.radii.size(); k++) {
            auto sphere_center = group.sphere_positions_cache.col(k);
            if (sdf->is_outside_aabb(sphere_center,
                                     group.radii[k] + cutoff_dist_)) {
              continue;
            }
            double val = sdf->evaluate(sphere_center) - group.radii[k];
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
    grad_out.setConstant(0.);
    return min_val_other;
  } else {
    // HACK: we already know that the sphere position cache is
    // already created
    const auto& min_sphere_trans =
        sphere_groups_[*min_group_idx].sphere_positions_cache.col(
            *min_sphere_idx);
    double r = sphere_groups_[*min_group_idx].radii[*min_sphere_idx];
    Eigen::Vector3d grad;
    for (size_t i = 0; i < 3; i++) {
      Eigen::Vector3d perturbed_center = min_sphere_trans;
      perturbed_center[i] += 1e-6;
      double val =
          all_sdfs_cache_[*min_sdf_idx]->evaluate(perturbed_center) - r;
      grad[i] = (val - min_val_other) / 1e-6;
    }
    auto&& sphere_jac = kin_->get_attached_point_jacobian(
        sphere_groups_[*min_group_idx].parent_link_id, min_sphere_trans,
        control_joint_ids_, base_type_);
    grad_out = sphere_jac.transpose() * grad;
    return min_val_other;
  }
}

double SphereCollisionCst::evaluate_self_collision(
    Eigen::Ref<Eigen::RowVectorXd, 0, Eigen::InnerStride<Eigen::Dynamic>>
        grad) {
  std::optional<std::array<size_t, 4>> min_pairs =
      std::nullopt;  // (group_i, sphere_i, group_j, sphere_j)
  double dist_min = cutoff_dist_;
  for (auto& group_id_pair : selcol_group_id_pairs_) {
    auto& group1 = sphere_groups_[group_id_pair.first];
    auto& group2 = sphere_groups_[group_id_pair.second];

    group1.create_group_sphere_position_cache(kin_);
    group2.create_group_sphere_position_cache(kin_);

    double outer_sqdist = (group1.group_sphere_position_cache -
                           group2.group_sphere_position_cache)
                              .squaredNorm();
    double outer_r_sum_with_margin =
        group1.group_radius + group2.group_radius + cutoff_dist_;
    if (outer_sqdist > outer_r_sum_with_margin * outer_r_sum_with_margin) {
      continue;
    }

    // narrow phase
    group1.create_sphere_position_cache(kin_);
    group2.create_sphere_position_cache(kin_);

    for (size_t i = 0; i < group1.radii.size(); i++) {
      for (size_t j = 0; j < group2.radii.size(); j++) {
        const auto& sphere1_center = group1.sphere_positions_cache.col(i);
        const auto& sphere2_center = group2.sphere_positions_cache.col(j);
        double dist = (sphere1_center - sphere2_center).norm() -
                      (group1.radii[i] + group2.radii[j]);
        if (dist < dist_min) {
          dist_min = dist;
          min_pairs = {group_id_pair.first, i, group_id_pair.second, j};
        }
      }
    }
  }

  if (min_pairs == std::nullopt) {
    grad.setConstant(0.);
    return dist_min;
  } else {
    // HACK: we know that in the non-gradient evaluation the cache
    // is already created
    auto& group1 = sphere_groups_[min_pairs->at(0)];
    const auto& center1 = group1.sphere_positions_cache.col(min_pairs->at(1));
    auto& group2 = sphere_groups_[min_pairs->at(2)];
    const auto& center2 = group2.sphere_positions_cache.col(min_pairs->at(3));

    Eigen::Vector3d center_diff = center1 - center2;
    auto&& jac1 = kin_->get_attached_point_jacobian(
        group1.parent_link_id, center1, control_joint_ids_, base_type_);
    auto&& jac2 = kin_->get_attached_point_jacobian(
        group2.parent_link_id, center2, control_joint_ids_, base_type_);
    double norminv = 1.0 / center_diff.norm();
    grad = norminv * center_diff.transpose() * (jac1 - jac2);
    return dist_min;
  }
}

void SphereCollisionCst::evaluate_dirty_into(Eigen::Ref<Eigen::VectorXd> vals,
                                             Eigen::Ref<Eigen::MatrixXd> jac) {
  size_t head = 0;
  if (ext_colliision_enabled()) {
    vals[head] = evaluate_ext_collision(jac.row(head));
    head++;
  }
  if (self_collision_enabled()) {
    vals[head] = evaluate_self_collision(jac.row(head));
  }
}

std::vector<std::pair<Eigen::Vector3d, double>>
SphereCollisionCst::get_group_spheres() {
  std::vector<std::pair<Eigen::Vector3d, double>> spheres;
  for (auto& sphere_group : sphere_groups_) {
    sphere_group.create_group_sphere_position_cache(kin_);
    spheres.push_back(
        {sphere_group.group_sphere_position_cache, sphere_group.group_radius});
  }
  return spheres;
}

std::vector<std::pair<Eigen::Vector3d, double>>
SphereCollisionCst::get_all_spheres() {
  std::vector<std::pair<Eigen::Vector3d, double>> spheres;
  for (auto& sphere_group : sphere_groups_) {
    sphere_group.create_group_sphere_position_cache(kin_);
    sphere_group.create_sphere_position_cache(kin_);
    for (size_t i = 0; i < sphere_group.radii.size(); i++) {
      spheres.push_back(
          {sphere_group.sphere_positions_cache.col(i), sphere_group.radii[i]});
    }
  }
  return spheres;
}

void SphereCollisionCst::set_all_sdfs() {
  all_sdfs_cache_.clear();
  if (fixed_sdf_ != nullptr) {
    set_all_sdfs_inner(fixed_sdf_);
  }
  if (sdf_ != nullptr) {
    set_all_sdfs_inner(sdf_);
  }
}

void SphereCollisionCst::set_all_sdfs_inner(
    plainmp::collision::SDFBase::Ptr sdf) {
  if (sdf->get_type() == plainmp::collision::SDFType::UNION) {
    for (auto& sub_sdf :
         std::static_pointer_cast<plainmp::collision::UnionSDF>(sdf)->sdfs_) {
      set_all_sdfs_inner(sub_sdf);
    }
  } else {
    auto primitive_sdf =
        std::static_pointer_cast<plainmp::collision::PrimitiveSDFBase>(sdf);
    all_sdfs_cache_.push_back(primitive_sdf);
  }
}

}  // namespace plainmp::constraint
