/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "plainmp/collision/primitive_sdf.hpp"
#include "plainmp/constraints/primitive.hpp"

namespace plainmp::constraint {

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
      bool with_base,
      const std::vector<SphereAttachmentSpec>& sphere_specs,
      const std::vector<std::pair<std::string, std::string>>& selcol_pairs,
      std::optional<plainmp::collision::SDFBase::Ptr> fixed_sdf);

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

  bool is_valid_dirty() override;
  bool check_ext_collision();
  bool check_self_collision();
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate_dirty() override;
  // retrun double and take block of eigen matrix
  double evaluate_ext_collision(
      Eigen::Block<Eigen::MatrixXd, 1, Eigen::Dynamic> grad);
  double evaluate_self_collision(
      Eigen::Block<Eigen::MatrixXd, 1, Eigen::Dynamic> grad);

  size_t cst_dim() const override {
    if (selcol_group_id_pairs_.size() == 0) {
      return 1;
    } else {
      return 2;
    }
  }
  std::string get_name() const override { return "SphereCollisionCst"; }
  std::vector<std::pair<Eigen::Vector3d, double>> get_group_spheres();
  std::vector<std::pair<Eigen::Vector3d, double>> get_all_spheres();

 private:
  void set_all_sdfs();
  void set_all_sdfs_inner(plainmp::collision::SDFBase::Ptr sdf);

  std::vector<SphereGroup> sphere_groups_;
  std::vector<std::pair<size_t, size_t>> selcol_group_id_pairs_;
  plainmp::collision::SDFBase::Ptr fixed_sdf_;
  plainmp::collision::SDFBase::Ptr sdf_;  // set later by user
  std::vector<plainmp::collision::PrimitiveSDFBase::Ptr> all_sdfs_cache_;
  double cutoff_dist_ = 0.1;
};

}  // namespace plainmp::constraint
