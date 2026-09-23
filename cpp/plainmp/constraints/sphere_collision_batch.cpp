/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/* Four-state dispatch. This translation unit uses the portable target ISA. */
#include "plainmp/constraints/primitive_sphere_collision.hpp"
#include <typeinfo>
namespace plainmp::constraint {
void SphereCollisionCst::update_batch_sdf_support() {
  // Dynamic types cannot change without replacing an SDF. Classify once when
  // set_all_sdfs() rebuilds the flattened obstacle list.
  batch_sdfs_supported_ = true;
  for (const auto &sdf : all_sdfs_cache_) {
    const auto &t = typeid(*sdf);
    if (t != typeid(collision::BoxSDF) && t != typeid(collision::SphereSDF) &&
        t != typeid(collision::CylinderSDF) &&
        t != typeid(collision::GroundSDF)) {
      batch_sdfs_supported_ = false;
      break;
    }
  }
}
bool SphereCollisionCst::batch_supported() const {
#ifdef PLAINMP_HAS_AVX2_COLLISION
  if (!__builtin_cpu_supports("avx2"))
    return false;
  if (base_type_ != kin::BaseType::FIXED || !batch_sdfs_supported_ ||
      typeid(*this) != typeid(SphereCollisionCst))
    return false;
  return kin_->all_links_consider_rotation_;
#else
  return false;
#endif
}

unsigned SphereCollisionCst::is_valid_batch(const double *const *states,
                                            size_t count) {
  if (count == 0 || count > 4)
    throw std::invalid_argument("Batch size must be 1..4");
#ifdef PLAINMP_HAS_AVX2_COLLISION
  if (count > 1 && batch_supported()) {
    const unsigned mask = is_valid_batch_avx2(states, count);
    restore_batch_state_avx2(states[count - 1], count - 1);
    post_update_kintree();
    return mask;
  }
#endif
  unsigned valid = 0;
  for (size_t k = 0; k < count; ++k)
    if (is_valid(Eigen::Map<const Eigen::VectorXd>(states[k], q_dim())))
      valid |= 1u << k;
  return valid;
}
size_t SphereCollisionCst::first_invalid_batch(const double *const *states,
                                               size_t count) {
  if (count == 0 || count > 4)
    throw std::invalid_argument("Batch size must be 1..4");
#ifdef PLAINMP_HAS_AVX2_COLLISION
  if (count > 1 && batch_supported()) {
    const unsigned mask = is_valid_batch_avx2(states, count);
    size_t first = 0;
    while (first < count && (mask & (1u << first)))
      ++first;
    const size_t last = std::min(first, count - 1);
    restore_batch_state_avx2(states[last], last);
    post_update_kintree();
    return first;
  }
#endif
  for (size_t k = 0; k < count; ++k)
    if (!is_valid(Eigen::Map<const Eigen::VectorXd>(states[k], q_dim())))
      return k;
  return count;
}
} // namespace plainmp::constraint
