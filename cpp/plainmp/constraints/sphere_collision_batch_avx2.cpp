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
#include <immintrin.h>
#include <typeinfo>

#define PLAINMP_BATCH_NAMESPACE wide_avx2
#define PLAINMP_BATCH_WORKSPACE BatchCollisionWorkspace4
#define PLAINMP_BATCH_MEMBER batch_workspace_4_
#define PLAINMP_BATCH_CHECK is_valid_batch_avx2
#define PLAINMP_BATCH_RESTORE restore_batch_state_avx2

namespace plainmp::constraint::wide_avx2 {
struct V {
  __m256d v;
  V() = default;
  V(double x) : v(_mm256_set1_pd(x)) {}
  V(__m256d x) : v(x) {}
  // Keep aggregate Frame/Mat copies as full-width vector stores. GCC's
  // generic tuning can split trivial aggregate copies into 128-bit stores,
  // which cannot forward to the immediately following 256-bit FK loads.
  V &operator=(const V &other) {
    v = other.v;
    return *this;
  }
};
inline V operator+(V a, V b) { return _mm256_add_pd(a.v, b.v); }
inline V operator-(V a, V b) { return _mm256_sub_pd(a.v, b.v); }
inline V operator*(V a, V b) { return _mm256_mul_pd(a.v, b.v); }
inline V negate(V a) { return _mm256_xor_pd(a.v, _mm256_set1_pd(-0.0)); }
inline V abs(V a) { return _mm256_andnot_pd(_mm256_set1_pd(-0.0), a.v); }
inline V gt(V a, V b) { return _mm256_cmp_pd(a.v, b.v, _CMP_GT_OQ); }
inline V lt(V a, V b) { return _mm256_cmp_pd(a.v, b.v, _CMP_LT_OQ); }
inline V either(V a, V b) { return _mm256_or_pd(a.v, b.v); }
inline V both(V a, V b) { return _mm256_and_pd(a.v, b.v); }
inline unsigned bits(V a) { return _mm256_movemask_pd(a.v); }
inline V select(V mask, V a, V b) { return _mm256_blendv_pd(b.v, a.v, mask.v); }
inline V positive(V a) { return _mm256_max_pd(a.v, _mm256_setzero_pd()); }
constexpr size_t width = 4;
constexpr size_t wide_alignment = 32;
inline V floor(V x) { return _mm256_floor_pd(x.v); }
inline V sqrt(V x) { return _mm256_sqrt_pd(x.v); }
inline void store(double *p, V x) { _mm256_store_pd(p, x.v); }
inline V load_joint(const double *const *states, size_t count, size_t joint) {
  return _mm256_set_pd(states[std::min(size_t(3), count - 1)][joint],
                       states[std::min(size_t(2), count - 1)][joint],
                       states[std::min(size_t(1), count - 1)][joint],
                       states[std::min(size_t(0), count - 1)][joint]);
}
} // namespace plainmp::constraint::wide_avx2

#include "sphere_collision_batch_impl.hpp"
