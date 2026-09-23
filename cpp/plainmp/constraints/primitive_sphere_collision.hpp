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

#include "plainmp/collision/primitive_sdf.hpp"
#include "plainmp/constraints/primitive.hpp"

namespace plainmp::constraint {

struct BatchCollisionWorkspace4;
struct BatchCollisionWorkspace8;

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

  // Bit k reports validity of states[k], for one to eight contiguous q vectors.
  // The kinematic state after this call corresponds to the last input.
  unsigned is_valid_batch(const double* const* states, size_t count);
  // Returns count when all states are valid, otherwise the first invalid index.
  // Later SIMD lanes may be evaluated, but the visible state ends at that index.
  size_t first_invalid_batch(const double* const* states, size_t count);
  bool batch_supported() const;
  // Preferred width for motion validation; portable callers may still pass 1..8.
  size_t batch_size() const;
  bool is_valid_dirty() override;
  bool check_ext_collision();
  bool check_self_collision();
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
  unsigned is_valid_batch_avx2(const double* const* states, size_t count);
  void restore_batch_state_avx2(const double *state, size_t lane);
  unsigned is_valid_batch_avx512(const double* const* states, size_t count);
  void restore_batch_state_avx512(const double *state, size_t lane);
  void update_batch_sdf_support();
  bool batch_sdfs_supported_ = false;
  std::shared_ptr<BatchCollisionWorkspace4> batch_workspace_4_;
  std::shared_ptr<BatchCollisionWorkspace8> batch_workspace_8_;
  struct ClearanceCertificate {
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    // Negative values certify overlap of the bounding sphere only. A negative
    // certificate never rejects a configuration without testing its spheres.
    double signed_margin_sq = 0.0;
  };
  bool check_ext_collision_with_cloud_cache();
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
