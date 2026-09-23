/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

// Four-state collision kernel. Each SIMD lane is an independent
// configuration. Preserve the scalar Eigen operation order; no FMA/fast-math.
#include "plainmp/constraints/primitive_sphere_collision.hpp"
#include <immintrin.h>
#include <typeinfo>

namespace plainmp::constraint {
namespace wide {
struct V {
  __m256d v;
  V() = default;
  V(double x) : v(_mm256_set1_pd(x)) {}
  V(__m256d x) : v(x) {}
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
struct Vec {
  V x, y, z;
};
struct Quat {
  V x, y, z, w;
};
struct Frame {
  Quat q;
  Vec p;
  bool identity = false;
};
struct Mat {
  Vec row[3];
};
inline Vec broadcast(const Eigen::Vector3d &p) { return {p.x(), p.y(), p.z()}; }
inline Quat broadcast(const Eigen::Quaterniond &q) {
  return {q.x(), q.y(), q.z(), q.w()};
}
inline Frame broadcast(const kin::QuatTrans<double> &f) {
  return {broadcast(f.quat()), broadcast(f.trans()), f.is_quat_identity_};
}
inline Vec operator+(const Vec &a, const Vec &b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
inline Vec operator-(const Vec &a, const Vec &b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
inline Vec operator*(V a, const Vec &b) { return {a * b.x, a * b.y, a * b.z}; }
inline V dot(const Vec &a, const Vec &b) {
  return a.x * b.x + (a.y * b.y + a.z * b.z);
}
inline V norm2(const Vec &a) { return dot(a, a); }
inline Vec cross(const Vec &a, const Vec &b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline Quat mul(const Quat &a, const Quat &b) {
  return {a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
          a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
          a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x,
          a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}
inline Vec rotate(const Quat &q, const Vec &v) {
  Vec qv{q.x, q.y, q.z};
  Vec uv = cross(qv, v);
  uv = uv + uv;
  return (v + q.w * uv) + cross(qv, uv);
}
inline Frame mul(const Frame &a, const Frame &b) {
  return {b.identity ? a.q : mul(a.q, b.q), a.p + rotate(a.q, b.p), false};
}
inline Mat matrix(const Quat &q) {
  V tx = V(2) * q.x, ty = V(2) * q.y, tz = V(2) * q.z;
  V twx = tx * q.w, twy = ty * q.w, twz = tz * q.w;
  V txx = tx * q.x, txy = ty * q.x, txz = tz * q.x;
  V tyy = ty * q.y, tyz = tz * q.y, tzz = tz * q.z;
  return {{{V(1) - (tyy + tzz), txy - twz, txz + twy},
           {txy + twz, V(1) - (txx + tzz), tyz - twx},
           {txz - twy, tyz + twx, V(1) - (txx + tyy)}}};
}
inline Vec transform(const Mat &r, const Vec &p, const Vec &t) {
  return {dot(r.row[0], p) + t.x, dot(r.row[1], p) + t.y,
          dot(r.row[2], p) + t.z};
}
inline void sincos(V x, V &s, V &c) {
  V outside = either(gt(x, M_PI), lt(x, -M_PI));
  if (bits(outside)) {
    V reduced = x - V(2 * M_PI) *
                        V(_mm256_floor_pd((x * V(1.0 / (2 * M_PI)) + V(.5)).v));
    x = select(outside, reduced, x);
  }
  V low = lt(x, -M_PI * .5), high = gt(x, M_PI * .5), negative = negate(x);
  x = select(low, negative - V(M_PI), x);
  x = select(high, negative + V(M_PI), x);
  V sign = select(either(low, high), -1., 1.);
  V xx = x * x, xxxx = xx * xx, xxxxxx = xxxx * xx, xxxxxxxx = xxxx * xxxx;
  s = x * (V(1) - xx * V(1. / 6) + xxxx * V(1. / 120) - xxxxxx * V(1. / 5040) +
           xxxxxxxx * V(1. / 362880));
  c = sign * (V(1) - xx * V(.5) + xxxx * V(1. / 24) - xxxxxx * V(1. / 720) +
              xxxxxxxx * V(1. / 40320));
}
inline unsigned outside_aabb(const Vec &p, double radius,
                             const collision::PrimitiveSDFBase &sdf) {
  V out = either(lt(p.x, sdf.lb.x() - radius), gt(p.x, sdf.ub.x() + radius));
  out = either(
      out, either(lt(p.y, sdf.lb.y() - radius), gt(p.y, sdf.ub.y() + radius)));
  out = either(
      out, either(lt(p.z, sdf.lb.z() - radius), gt(p.z, sdf.ub.z() + radius)));
  return bits(out);
}
inline Vec relative(const Vec &p, const collision::Pose &pose) {
  return p - broadcast(pose.position_);
}
inline V projection(const Vec &p, const collision::Pose &pose, int axis) {
  return p.x * V(pose.rot_(0, axis)) +
         (p.y * V(pose.rot_(1, axis)) + p.z * V(pose.rot_(2, axis)));
}
inline unsigned outside_shape(const Vec &p, double r,
                              const collision::PrimitiveSDFBase &sdf, int kind,
                              unsigned active) {
  if (kind == collision::BOX) {
    const auto &b = static_cast<const collision::BoxSDF &>(sdf);
    const auto half = b.get_width() * .5;
    Vec d = relative(p, b.pose);
    V x, y, z;
    if (b.pose.axis_aligned_) {
      x = abs(d.x) - V(half.x());
      y = abs(d.y) - V(half.y());
      z = abs(d.z) - V(half.z());
    } else {
      x = abs(projection(d, b.pose, 0)) - V(half.x());
      y = abs(projection(d, b.pose, 1)) - V(half.y());
      z = abs(b.pose.z_axis_aligned_ ? d.z : projection(d, b.pose, 2)) -
          V(half.z());
    }
    V out = either(either(gt(x, r), gt(y, r)), gt(z, r));
    if (r >= 1e-6) {
      x = positive(x);
      y = positive(y);
      z = positive(z);
      out = either(out, gt((x * x + y * y) + z * z, r * r));
    }
    return bits(out);
  }
  if (kind == collision::SPHERE) {
    const auto &s = static_cast<const collision::SphereSDF &>(sdf);
    const double rr = r < 1e-6 ? s.get_radius() : s.get_radius() + r;
    return bits(gt(norm2(relative(p, s.pose)), rr * rr));
  }
  if (kind == collision::CYLINDER) {
    const auto &s = static_cast<const collision::CylinderSDF &>(sdf);
    Vec d = relative(p, s.pose);
    V x, y, z;
    if (s.pose.z_axis_aligned_) {
      x = abs(d.x);
      y = abs(d.y);
      z = abs(d.z) - V(s.get_half_height());
    } else {
      x = abs(projection(d, s.pose, 0));
      y = abs(projection(d, s.pose, 1));
      z = abs(projection(d, s.pose, 2)) - V(s.get_half_height());
    }
    V dist = x * x + y * y;
    const double rr = s.get_radius(), rsq = rr * rr;
    V out = gt(z, r);
    if (r < 1e-6)
      return bits(either(out, gt(dist, rsq)));
    out = either(out, gt(dist, (rr + r) * (rr + r)));
    V edge = both(gt(z, 0), gt(dist, rsq));
    if (bits(edge) & active & ~bits(out)) {
      V radial = V(_mm256_sqrt_pd(dist.v)) - V(rr);
      out = either(out, both(edge, gt(z * z + radial * radial, r * r)));
    }
    return bits(out);
  }
  // Ground and custom primitives use the original virtual predicate.
  alignas(32) double x[4], y[4], z[4];
  _mm256_store_pd(x, p.x.v);
  _mm256_store_pd(y, p.y.v);
  _mm256_store_pd(z, p.z.v);
  unsigned out = 0;
  for (unsigned k = 0; k < 4; ++k)
    if ((active & (1u << k)) &&
        sdf.is_outside(Eigen::Vector3d(x[k], y[k], z[k]), r))
      out |= 1u << k;
  return out;
}
} // namespace wide

struct BatchCollisionWorkspace {
  struct Group {
    wide::Mat rotation;
    wide::Vec center;
    std::vector<wide::Vec> spheres;
    bool center_ready = false, spheres_ready = false;
  };
  std::vector<wide::Frame> local, world;
  std::vector<unsigned char> local_ready, world_ready;
  std::vector<size_t> stack;
  std::vector<Group> groups;
  std::vector<int> kinds;
  BatchCollisionWorkspace(
      size_t links, const std::vector<SphereGroup> &specs,
      const std::vector<collision::PrimitiveSDFBase::Ptr> &sdfs)
      : local(links), world(links), local_ready(links), world_ready(links),
        stack(links), groups(specs.size()) {
    for (size_t i = 0; i < specs.size(); ++i)
      groups[i].spheres.resize(specs[i].radii.size());
    for (const auto &sdf : sdfs) {
      const auto &t = typeid(*sdf);
      kinds.push_back(t == typeid(collision::BoxSDF)      ? collision::BOX
                      : t == typeid(collision::SphereSDF) ? collision::SPHERE
                      : t == typeid(collision::CylinderSDF)
                          ? collision::CYLINDER
                          : -1);
    }
  }
  const wide::Frame &pose(size_t id, const kin::KinematicModel<double> &kin) {
    if (world_ready[id])
      return world[id];
    size_t n = 0, ancestor = id;
    while (!world_ready[ancestor]) {
      stack[n++] = ancestor;
      ancestor = kin.link_parent_link_ids_[ancestor];
    }
    while (n) {
      const size_t child = stack[--n];
      const auto local_pose =
          local_ready[child]
              ? local[child]
              : wide::broadcast(kin.tf_plink_to_hlink_cache_[child]);
      world[child] = wide::mul(world[ancestor], local_pose);
      world_ready[child] = 1;
      ancestor = child;
    }
    return world[id];
  }
  void group_center(size_t i, const SphereGroup &spec,
                    const kin::KinematicModel<double> &kin) {
    auto &g = groups[i];
    if (g.center_ready)
      return;
    const auto &f = pose(spec.parent_link_id, kin);
    g.rotation = wide::matrix(f.q);
    g.center = wide::transform(
        g.rotation, wide::broadcast(spec.group_sphere_relative_position), f.p);
    g.center_ready = true;
  }
  void group_spheres(size_t i, const SphereGroup &spec,
                     const kin::KinematicModel<double> &kin) {
    auto &g = groups[i];
    if (g.spheres_ready)
      return;
    const auto &t = world[spec.parent_link_id].p;
    for (size_t j = 0; j < g.spheres.size(); ++j)
      g.spheres[j] = wide::transform(
          g.rotation, wide::broadcast(spec.sphere_relative_positions.col(j)),
          t);
    g.spheres_ready = true;
  }
};

unsigned SphereCollisionCst::is_valid_batch_avx2(const double *const *states,
                                                 size_t count) {
  if (!batch_workspace_ ||
      batch_workspace_->world.size() != kin_->link_parent_link_ids_.size())
    batch_workspace_ = std::make_shared<BatchCollisionWorkspace>(
        kin_->link_parent_link_ids_.size(), sphere_groups_, all_sdfs_cache_);
  auto &w = *batch_workspace_;
  std::fill(w.local_ready.begin(), w.local_ready.end(), 0);
  std::fill(w.world_ready.begin(), w.world_ready.end(), 0);
  for (auto &g : w.groups) {
    g.center_ready = false;
    g.spheres_ready = false;
  }
  w.world[kin_->root_link_id_] = wide::broadcast(kin_->get_base_pose());
  w.world_ready[kin_->root_link_id_] = 1;
  for (size_t i = 0; i < control_joint_ids_.size(); ++i) {
    const size_t joint = control_joint_ids_[i],
                 link = kin_->joint_child_link_ids_[joint];
    auto &q = w.local[link];
    wide::V angle(_mm256_set_pd(states[std::min(size_t(3), count - 1)][i],
                                states[std::min(size_t(2), count - 1)][i],
                                states[1][i], states[0][i]));
    if (kin_->joint_types_[joint] == urdf::Joint::PRISMATIC) {
      q.p = wide::broadcast(kin_->joint_positions_[joint]) +
            angle * wide::broadcast(kin_->joint_axes_[joint]);
      q.q = {0., 0., 0., 1.};
      q.identity = true;
    } else {
      wide::V s, c;
      wide::sincos(angle * wide::V(.5), s, c);
      const auto axis = wide::broadcast(kin_->joint_axes_[joint]);
      q.q = {s * axis.x, s * axis.y, s * axis.z, c};
      if (!kin_->joint_orientation_identity_flags_[joint])
        q.q = wide::mul(wide::broadcast(kin_->joint_orientations_[joint]), q.q);
      q.p = wide::broadcast(kin_->joint_positions_[joint]);
      q.identity = false;
    }
    w.local_ready[link] = 1;
  }
  unsigned active = (1u << count) - 1;
  for (size_t g = 0;
       g < sphere_groups_.size() && active && !all_sdfs_cache_.empty(); ++g) {
    const auto &spec = sphere_groups_[g];
    if (spec.only_self_collision)
      continue;
    w.group_center(g, spec, *kin_);
    auto &group = w.groups[g];
    for (size_t o = 0; o < all_sdfs_cache_.size() && active; ++o) {
      const auto &sdf = *all_sdfs_cache_[o];
      unsigned narrow =
          active & ~wide::outside_aabb(group.center, spec.group_radius, sdf);
      if (!narrow)
        continue;
      narrow &= ~wide::outside_shape(group.center, spec.group_radius, sdf,
                                     w.kinds[o], narrow);
      if (!narrow)
        continue;
      w.group_spheres(g, spec, *kin_);
      for (size_t i = 0; i < group.spheres.size() && narrow; ++i) {
        const double r = spec.radii[i];
        const auto &p = group.spheres[i];
        unsigned inside = narrow & ~wide::outside_aabb(p, r, sdf);
        if (!inside)
          continue;
        inside &= ~wide::outside_shape(p, r, sdf, w.kinds[o], inside);
        active &= ~inside;
        narrow &= ~inside;
      }
    }
  }
  for (const auto &pair : selcol_group_id_pairs_) {
    if (!active)
      break;
    const size_t a = pair.first, b = pair.second;
    const auto &as = sphere_groups_[a];
    const auto &bs = sphere_groups_[b];
    w.group_center(a, as, *kin_);
    w.group_center(b, bs, *kin_);
    auto &ag = w.groups[a];
    auto &bg = w.groups[b];
    const double r = as.group_radius + bs.group_radius;
    unsigned narrow =
        active &
        ~wide::bits(wide::gt(wide::norm2(ag.center - bg.center), r * r));
    if (!narrow)
      continue;
    w.group_spheres(a, as, *kin_);
    w.group_spheres(b, bs, *kin_);
    for (size_t i = 0; i < ag.spheres.size() && narrow; ++i) {
      for (size_t j = 0; j < bg.spheres.size() && narrow; ++j) {
        const double rr = as.radii[i] + bs.radii[j];
        unsigned inside =
            narrow & wide::bits(wide::lt(
                         wide::norm2(ag.spheres[i] - bg.spheres[j]), rr * rr));
        active &= ~inside;
        narrow &= ~inside;
      }
    }
  }
  return active;
}
} // namespace plainmp::constraint
