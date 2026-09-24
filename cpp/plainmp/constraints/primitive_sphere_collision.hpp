/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <array>
#include "plainmp/collision/primitive_sdf.hpp"
#include "plainmp/constraints/primitive.hpp"

namespace plainmp::constraint {

struct ScalarMotionBounds;

struct SphereAttachmentSpec {
  std::string parent_link_name;
  Eigen::Matrix3Xd relative_positions;
  Eigen::VectorXd radii;
  bool only_self_collision;
};

struct SphereGroup {
  std::string parent_link_name;
  size_t parent_link_id;
  Eigen::VectorXd radii;
  double group_radius;
  bool only_self_collision;
  Eigen::Matrix3Xd sphere_relative_positions;
  Eigen::Vector3d group_sphere_relative_position;
  // rot mat cache (NOTE: see comment in primitive_sphere_collision.cpp)
  Eigen::Matrix3d rot_mat_cache;

  // group sphere position
  Eigen::Vector3d group_sphere_position_cache;
  bool is_group_sphere_position_dirty;

  // sphere positions cache
  Eigen::Matrix3Xd sphere_positions_cache;
  bool is_sphere_positions_dirty;

  void max_distance_reorder();

  inline void clear_cache() {
    is_group_sphere_position_dirty = true;
    is_sphere_positions_dirty = true;
  }

  void create_group_sphere_position_cache(
      const std::shared_ptr<kin::KinematicModel<double>>& kin);
  void create_sphere_position_cache(
      const std::shared_ptr<kin::KinematicModel<double>>& kin);
};

class SphereCollisionCst : public IneqConstraintBase {
 public:
  using Ptr = std::shared_ptr<SphereCollisionCst>;
  SphereCollisionCst(
      std::shared_ptr<kin::KinematicModel<double>> kin,
      const std::vector<std::string>& control_joint_names,
      kin::BaseType base_type,
      const std::vector<SphereAttachmentSpec>& sphere_specs,
      const std::vector<std::pair<std::string, std::string>>& selcol_pairs,
      std::optional<plainmp::collision::SDFBase::Ptr> fixed_sdf,
      bool reorder_spheres = true);

  void post_update_kintree() override {
    for (auto& group : sphere_groups_) {
      group.clear_cache();
    }
  }

  void set_sdf(const plainmp::collision::SDFBase::Ptr& sdf) {
    sdf_ = sdf;
    set_all_sdfs();
  }

  plainmp::collision::SDFBase::Ptr get_sdf() const { return sdf_; }

  // Certificates survive configuration changes, but require static point clouds.
  // set_sdf() also invalidates them, including when reusing the same SDF pointer.
  void reset_clearance_cache();

  bool is_valid_dirty() override;
  bool check_ext_collision();
  bool check_self_collision();
  // Experimental scalar interval certificates. The anchor q must lie on the
  // prepared segment; robot structure, uncontrolled joints, base, and SDFs must
  // stay unchanged until the last certificate query for that segment.
  // The returned radius is in the segment's normalized interpolation parameter.
  bool prepare_motion_certificate(const VectorInput& start, const VectorInput& end,
                                  double rate_radius);
  bool is_valid_with_motion_certificate(const VectorInput& q, double& certified_radius);
  void note_certified_skip(size_t count) { motion_certificate_stats_[3] += count; }
  std::array<size_t, 4> motion_certificate_stats() const { return motion_certificate_stats_; }
  void reset_motion_certificate_stats() { motion_certificate_stats_.fill(0); }
  double motion_certificate_steps() const { return motion_certificate_steps_; }
  void set_motion_certificate_steps(double steps) {
    if (!std::isfinite(steps) || steps <= 0 || steps > 64)
      throw std::invalid_argument("Certificate radius must be in (0,64]");
    motion_certificate_steps_ = steps;
    motion_certificate_prepared_ = false;
  }
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate_dirty() override;
  // retrun double and take block of eigen matrix
  double evaluate_ext_collision(
      Eigen::Block<Eigen::MatrixXd, 1, Eigen::Dynamic> grad);
  double evaluate_self_collision(
      Eigen::Block<Eigen::MatrixXd, 1, Eigen::Dynamic> grad);

  inline bool ext_colliision_enabled() const {
    // NOTE: anchored primitives for self collision is also considered
    // as a part of external collision
    return (all_sdfs_cache_.size() > 0);
  }
  inline bool self_collision_enabled() const {
    return (selcol_group_id_pairs_.size() > 0);
  }
  inline size_t cst_dim() const override {
    return (ext_colliision_enabled() ? 1 : 0) +
           (self_collision_enabled() ? 1 : 0);
  }
  std::string get_name() const override { return "SphereCollisionCst"; }
  std::vector<std::pair<Eigen::Vector3d, double>> get_group_spheres();
  std::vector<std::pair<Eigen::Vector3d, double>> get_all_spheres();

 private:
  struct ClearanceCertificate {
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    // Negative values certify overlap of the bounding sphere only. A negative
    // certificate never rejects a configuration without testing its spheres.
    double signed_margin_sq = 0.0;
  };
  bool check_ext_collision_with_cloud_cache();
  bool check_motion_envelope(double& radius);
  bool finish_point_check(size_t group, size_t sdf, size_t sphere,
                          size_t pair, size_t row, size_t column);
  std::shared_ptr<ScalarMotionBounds> motion_bounds_;
  std::array<size_t, 4> motion_certificate_stats_{};
  double motion_certificate_steps_ = 6;
  bool motion_certificate_prepared_ = false;
  void initialize_clearance_cache();
  void set_all_sdfs();
  void set_all_sdfs_inner(plainmp::collision::SDFBase::Ptr sdf);

  std::vector<SphereGroup> sphere_groups_;
  std::vector<std::pair<size_t, size_t>> selcol_group_id_pairs_;
  plainmp::collision::SDFBase::Ptr fixed_sdf_;
  plainmp::collision::SDFBase::Ptr sdf_;  // set later by user
  std::vector<plainmp::collision::PrimitiveSDFBase::Ptr> all_sdfs_cache_;
  std::vector<int> cloud_sdf_ids_;
  std::vector<size_t> sphere_offsets_;
  std::vector<ClearanceCertificate> clearance_cache_;
  size_t cloud_sdf_count_ = 0;
  double cutoff_dist_ = 0.1;
};

}  // namespace plainmp::constraint
