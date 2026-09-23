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

#define PLAINMP_BATCH_NAMESPACE wide_avx512
#define PLAINMP_BATCH_WORKSPACE BatchCollisionWorkspace8
#define PLAINMP_BATCH_MEMBER batch_workspace_8_
#define PLAINMP_BATCH_CHECK is_valid_batch_avx512
#define PLAINMP_BATCH_RESTORE restore_batch_state_avx512

namespace plainmp::constraint::wide_avx512 {
struct V {
  __m512d v;
  V() = default;
  V(double x) : v(_mm512_set1_pd(x)) {}
  V(__m512d x) : v(x) {}
  // Keep aggregate Frame/Mat copies as full-width vector stores. GCC's
  // generic tuning can split trivial aggregate copies into 128-bit stores,
  // which cannot forward to the immediately following 512-bit FK loads.
  V &operator=(const V &other) {
    v = other.v;
    return *this;
  }
};
inline V operator+(V a, V b) { return _mm512_add_pd(a.v, b.v); }
inline V operator-(V a, V b) { return _mm512_sub_pd(a.v, b.v); }
inline V operator*(V a, V b) { return _mm512_mul_pd(a.v, b.v); }
inline V negate(V a) { return _mm512_xor_pd(a.v, _mm512_set1_pd(-0.0)); }
inline V abs(V a) { return _mm512_andnot_pd(_mm512_set1_pd(-0.0), a.v); }
using Mask = __mmask8;
inline Mask gt(V a, V b) { return _mm512_cmp_pd_mask(a.v, b.v, _CMP_GT_OQ); }
inline Mask lt(V a, V b) { return _mm512_cmp_pd_mask(a.v, b.v, _CMP_LT_OQ); }
inline Mask either(Mask a, Mask b) { return a | b; }
inline Mask both(Mask a, Mask b) { return a & b; }
inline unsigned bits(Mask a) { return a; }
inline V select(Mask mask, V a, V b) {
  return _mm512_mask_blend_pd(mask, b.v, a.v);
}
inline V positive(V a) { return _mm512_max_pd(a.v, _mm512_setzero_pd()); }
constexpr size_t width = 8;
constexpr size_t wide_alignment = 64;
inline V floor(V x) { return _mm512_floor_pd(x.v); }
inline V sqrt(V x) { return _mm512_sqrt_pd(x.v); }
inline void store(double *p, V x) { _mm512_store_pd(p, x.v); }
inline V load_joint(const double *const *states, size_t count, size_t joint) {
  return _mm512_set_pd(states[std::min(size_t(7), count - 1)][joint],
                       states[std::min(size_t(6), count - 1)][joint],
                       states[std::min(size_t(5), count - 1)][joint],
                       states[std::min(size_t(4), count - 1)][joint],
                       states[std::min(size_t(3), count - 1)][joint],
                       states[std::min(size_t(2), count - 1)][joint],
                       states[std::min(size_t(1), count - 1)][joint],
                       states[std::min(size_t(0), count - 1)][joint]);
}
} // namespace plainmp::constraint::wide_avx512

#include "sphere_collision_batch_impl.hpp"
