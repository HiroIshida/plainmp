/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

// Shared arithmetic and traversal for the separately compiled SIMD kernels.
// Each translation unit supplies its vector operations and distinct workspace
// and method names. Do not include this header from portable callers.
namespace plainmp::constraint {
namespace PLAINMP_BATCH_NAMESPACE {
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
  auto outside = either(gt(x, M_PI), lt(x, -M_PI));
  if (bits(outside)) {
    V reduced = x - V(2 * M_PI) * floor(x * V(1.0 / (2 * M_PI)) + V(.5));
    x = select(outside, reduced, x);
  }
  auto low = lt(x, -M_PI * .5), high = gt(x, M_PI * .5);
  V negative = negate(x);
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
  auto out = either(lt(p.x, sdf.lb.x() - radius), gt(p.x, sdf.ub.x() + radius));
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
    auto out = either(either(gt(x, r), gt(y, r)), gt(z, r));
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
    auto out = gt(z, r);
    if (r < 1e-6)
      return bits(either(out, gt(dist, rsq)));
    out = either(out, gt(dist, (rr + r) * (rr + r)));
    auto edge = both(gt(z, 0), gt(dist, rsq));
    if (bits(edge) & active & ~bits(out)) {
      V radial = sqrt(dist) - V(rr);
      out = either(out, both(edge, gt(z * z + radial * radial, r * r)));
    }
    return bits(out);
  }
  // Ground and custom primitives use the original virtual predicate.
  alignas(wide_alignment) double x[width], y[width], z[width];
  store(x, p.x);
  store(y, p.y);
  store(z, p.z);
  unsigned out = 0;
  for (unsigned k = 0; k < width; ++k)
    if ((active & (1u << k)) &&
        sdf.is_outside(Eigen::Vector3d(x[k], y[k], z[k]), r))
      out |= 1u << k;
  return out;
}
} // namespace PLAINMP_BATCH_NAMESPACE
namespace wide = PLAINMP_BATCH_NAMESPACE;

struct PLAINMP_BATCH_WORKSPACE {
  struct Group {
    wide::Vec center;
    std::vector<wide::Vec> spheres;
  };
  // Local slots cover controlled joints only; world slots follow the FK
  // visit order instead of the robot's sparse link IDs.
  // A revolute joint's translation is configuration-independent. Retain only
  // its quaternion; broadcast the current model's origin at the point of use.
  // For prismatic joints only x is used, holding the displacement per lane.
  std::vector<wide::Quat> local;
  std::vector<wide::Frame> world;
  std::vector<size_t> control_joint_ids;
  struct Step {
    size_t link, parent;
    int local;
  };
  // Reset readiness in one small contiguous write, without touching each
  // group's geometry cache lines. Bits 0/1 mean center/spheres are ready.
  std::vector<unsigned char> group_ready;
  std::vector<Step> pose_order;
  std::vector<size_t> pose_end;
  size_t poses_ready = 0;
  std::vector<Group> groups;
  std::vector<int> kinds;
  PLAINMP_BATCH_WORKSPACE(
      const kin::KinematicModel<double> &kin,
      const std::vector<size_t> &control_joints,
      const std::vector<SphereGroup> &specs,
      const std::vector<std::pair<size_t, size_t>> &self_pairs,
      const std::vector<collision::PrimitiveSDFBase::Ptr> &sdfs)
      : local(control_joints.size()), control_joint_ids(control_joints),
        group_ready(specs.size()),
        pose_end(kin.link_parent_link_ids_.size(), 0), groups(specs.size()) {
    std::vector<int> local_index(pose_end.size(), -1);
    for (size_t i = 0; i < control_joints.size(); ++i)
      local_index[kin.joint_child_link_ids_[control_joints[i]]] = i;
    // Group visits have a fixed order: external checks, then self-collision
    // pairs. Flatten their ancestor paths once. Each query evaluates only the
    // prefix needed so far, retaining the original early exits.
    std::vector<unsigned char> seen(pose_end.size(), 0);
    seen[kin.root_link_id_] = 1;
    std::vector<size_t> path;
    auto append = [&](size_t group) {
      size_t id = specs[group].parent_link_id;
      path.clear();
      while (!seen[id]) {
        path.push_back(id);
        id = kin.link_parent_link_ids_[id];
      }
      for (auto it = path.rbegin(); it != path.rend(); ++it) {
        pose_order.push_back(
            {*it, pose_end[kin.link_parent_link_ids_[*it]], local_index[*it]});
        pose_end[*it] = pose_order.size();
        seen[*it] = 1;
      }
    };
    if (!sdfs.empty())
      for (size_t i = 0; i < specs.size(); ++i)
        if (!specs[i].only_self_collision)
          append(i);
    for (const auto &pair : self_pairs) {
      append(pair.first);
      append(pair.second);
    }
    world.resize(pose_order.size() + 1);
    for (size_t i = 0; i < specs.size(); ++i)
      groups[i].spheres.resize(specs[i].radii.size());
    for (const auto &sdf : sdfs) {
      const auto &t = typeid(*sdf);
      kinds.push_back(t == typeid(collision::BoxSDF)
                          ? collision::BOX
                          : t == typeid(collision::SphereSDF)
                                ? collision::SPHERE
                                : t == typeid(collision::CylinderSDF)
                                      ? collision::CYLINDER
                                      : -1);
    }
  }
  wide::Frame local_pose(size_t i,
                         const kin::KinematicModel<double> &kin) const {
    const size_t joint = control_joint_ids[i];
    if (kin.joint_types_[joint] == urdf::Joint::PRISMATIC) {
      const auto angle = local[i].x;
      return {{0., 0., 0., 1.},
              wide::broadcast(kin.joint_positions_[joint]) +
                  angle * wide::broadcast(kin.joint_axes_[joint]),
              true};
    }
    return {local[i], wide::broadcast(kin.joint_positions_[joint]), false};
  }
  const wide::Frame &pose(size_t id, const kin::KinematicModel<double> &kin) {
    while (poses_ready < pose_end[id]) {
      const auto &step = pose_order[poses_ready];
      const auto local_pose =
          step.local >= 0
              ? this->local_pose(step.local, kin)
              : wide::broadcast(kin.tf_plink_to_hlink_cache_[step.link]);
      world[poses_ready + 1] = wide::mul(world[step.parent], local_pose);
      ++poses_ready;
    }
    return world[pose_end[id]];
  }
  void group_center(size_t i, const SphereGroup &spec,
                    const kin::KinematicModel<double> &kin) {
    auto &g = groups[i];
    if (group_ready[i] & 1)
      return;
    const auto &f = pose(spec.parent_link_id, kin);
    // Persist only the center. Saving all nine rotation vectors for every
    // broad-phase group increases the working set; narrow-phase groups
    // reconstruct the same matrix from the cached pose when needed.
    const auto rotation = wide::matrix(f.q);
    g.center = wide::transform(
        rotation, wide::broadcast(spec.group_sphere_relative_position), f.p);
    group_ready[i] |= 1;
  }
  void group_spheres(size_t i, const SphereGroup &spec,
                     const kin::KinematicModel<double> &kin) {
    auto &g = groups[i];
    if (group_ready[i] & 2)
      return;
    const auto &f = world[pose_end[spec.parent_link_id]];
    const auto rotation = wide::matrix(f.q);
    const auto &t = f.p;
    for (size_t j = 0; j < g.spheres.size(); ++j)
      g.spheres[j] = wide::transform(
          rotation, wide::broadcast(spec.sphere_relative_positions.col(j)), t);
    group_ready[i] |= 2;
  }
};

void SphereCollisionCst::PLAINMP_BATCH_RESTORE(const double *state,
                                               size_t lane) {
  // Reuse the computed joint quaternion/displacement for the selected lane.
  // Fixed joint translations come directly from the current scalar model.
  auto extract = [lane](wide::V value) {
    alignas(wide::wide_alignment) double lanes[wide::width];
    wide::store(lanes, value);
    return lanes[lane];
  };
  for (size_t i = 0; i < control_joint_ids_.size(); ++i) {
    const size_t joint = control_joint_ids_[i];
    const size_t link = kin_->joint_child_link_ids_[joint];
    auto &target = kin_->tf_plink_to_hlink_cache_[link];
    kin_->joint_angles_[joint] = state[i];
    if (kin_->joint_types_[joint] == urdf::Joint::PRISMATIC) {
      target.quat().setIdentity();
      target.trans() =
          kin_->joint_positions_[joint] +
          kin_->joint_axes_[joint] * extract(PLAINMP_BATCH_MEMBER->local[i].x);
      target.is_quat_identity_ = true;
    } else {
      const auto &q = PLAINMP_BATCH_MEMBER->local[i];
      target.quat() = Eigen::Quaterniond(extract(q.w), extract(q.x),
                                         extract(q.y), extract(q.z));
      target.trans() = kin_->joint_positions_[joint];
      target.is_quat_identity_ = false;
    }
  }
  kin_->clear_cache();
}

unsigned SphereCollisionCst::PLAINMP_BATCH_CHECK(const double *const *states,
                                                 size_t count) {
  if (!PLAINMP_BATCH_MEMBER ||
      PLAINMP_BATCH_MEMBER->pose_end.size() !=
          kin_->link_parent_link_ids_.size() ||
      PLAINMP_BATCH_MEMBER->control_joint_ids != control_joint_ids_)
    PLAINMP_BATCH_MEMBER = std::make_shared<PLAINMP_BATCH_WORKSPACE>(
        *kin_, control_joint_ids_, sphere_groups_, selcol_group_id_pairs_,
        all_sdfs_cache_);
  auto &w = *PLAINMP_BATCH_MEMBER;
  std::fill(w.group_ready.begin(), w.group_ready.end(), 0);
  w.poses_ready = 0;
  w.world[0] = wide::broadcast(kin_->get_base_pose());
  for (size_t i = 0; i < control_joint_ids_.size(); ++i) {
    const size_t joint = control_joint_ids_[i];
    auto &q = w.local[i];
    const auto angle = wide::load_joint(states, count, i);
    if (kin_->joint_types_[joint] == urdf::Joint::PRISMATIC) {
      // A prismatic joint needs only its displacement, in the first slot.
      q.x = angle;
      continue;
    }
    wide::V s, c;
    wide::sincos(angle * wide::V(.5), s, c);
    const auto axis = wide::broadcast(kin_->joint_axes_[joint]);
    q = {s * axis.x, s * axis.y, s * axis.z, c};
    if (!kin_->joint_orientation_identity_flags_[joint])
      q = wide::mul(wide::broadcast(kin_->joint_orientations_[joint]), q);
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
