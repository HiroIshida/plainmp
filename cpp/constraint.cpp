#include "constraint.hpp"
#include <pybind11/stl.h>
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace cst {

std::pair<Eigen::VectorXd, Eigen::MatrixXd> LinkPoseCst::evaluate_dirty() {
  Eigen::VectorXd vals(cst_dim());
  Eigen::MatrixXd jac(cst_dim(), q_dim());
  tinyfk::Transform pose;
  size_t head = 0;
  for (size_t i = 0; i < link_ids_.size(); i++) {
    kin_->get_link_pose(link_ids_[i], pose);
    if (poses_[i].size() == 3) {
      vals.segment(head, 3) = pose.trans() - poses_[i];
      jac.block(head, 0, 3, q_dim()) =
          kin_->get_jacobian(link_ids_[i], control_joint_ids_,
                             tinyfk::RotationType::IGNORE, with_base_);
      head += 3;
    } else if (poses_[i].size() == 6) {
      vals.segment(head, 3) = pose.trans() - poses_[i].head(3);
      vals.segment(head + 3, 3) = pose.getRPY() - poses_[i].tail(3);
      jac.block(head, 0, 6, q_dim()) =
          kin_->get_jacobian(link_ids_[i], control_joint_ids_,
                             tinyfk::RotationType::RPY, with_base_);
      head += 6;
    } else {
      vals.segment(head, 3) = pose.trans() - poses_[i].head(3);
      vals.segment(head + 3, 4) = pose.quat().coeffs() - poses_[i].tail(4);
      jac.block(head, 0, 7, q_dim()) =
          kin_->get_jacobian(link_ids_[i], control_joint_ids_,
                             tinyfk::RotationType::XYZW, with_base_);
      head += 7;
    }
  }
  return {vals, jac};
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> RelativePoseCst::evaluate_dirty() {
  Eigen::VectorXd vals(cst_dim());
  Eigen::MatrixXd jac(cst_dim(), q_dim());
  tinyfk::Transform pose_dummy, pose2;
  kin_->get_link_pose(dummy_link_id_, pose_dummy);
  kin_->get_link_pose(link_id2_, pose2);
  vals.head(3) = pose_dummy.trans() - pose2.trans();
  vals.segment(3, 4) = pose_dummy.quat().coeffs() - pose2.quat().coeffs();
  jac = kin_->get_jacobian(dummy_link_id_, control_joint_ids_,
                           tinyfk::RotationType::XYZW, with_base_) -
        kin_->get_jacobian(link_id2_, control_joint_ids_,
                           tinyfk::RotationType::XYZW, with_base_);
  return {vals, jac};
}

FixedZAxisCst::FixedZAxisCst(
    std::shared_ptr<tinyfk::KinematicModel> kin,
    const std::vector<std::string>& control_joint_names,
    bool with_base,
    const std::string& link_name)
    : EqConstraintBase(kin, control_joint_names, with_base),
      link_id_(kin_->get_link_ids({link_name})[0]) {
  aux_link_ids_.clear();
  {
    auto pose = tinyfk::Transform::Identity();
    pose.trans().x() = 1;
    auto new_link = kin_->add_new_link(link_id_, pose, false);
    aux_link_ids_.push_back(new_link->id);
  }

  {
    auto pose = tinyfk::Transform::Identity();
    pose.trans().y() = 1;
    auto new_link = kin_->add_new_link(link_id_, pose, false);
    aux_link_ids_.push_back(new_link->id);
  }
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> FixedZAxisCst::evaluate_dirty() {
  tinyfk::Transform pose_here, pose_plus1_x, pose_plus1_y;
  kin_->get_link_pose(link_id_, pose_here);
  kin_->get_link_pose(aux_link_ids_[0], pose_plus1_x);
  kin_->get_link_pose(aux_link_ids_[1], pose_plus1_y);
  Eigen::VectorXd vals(2);
  double diff_plus1_x_z = pose_plus1_x.trans().z() - pose_here.trans().z();
  double diff_plus1_y_z = pose_plus1_y.trans().z() - pose_here.trans().z();
  vals << diff_plus1_x_z, diff_plus1_y_z;

  // jacobian
  Eigen::MatrixXd jac_here(3, q_dim());
  Eigen::MatrixXd jac_plus1_x(3, q_dim());
  Eigen::MatrixXd jac_plus1_y(3, q_dim());
  jac_here = kin_->get_jacobian(link_id_, control_joint_ids_,
                                tinyfk::RotationType::IGNORE, with_base_);
  jac_plus1_x = kin_->get_jacobian(aux_link_ids_[0], control_joint_ids_,
                                   tinyfk::RotationType::IGNORE, with_base_);
  jac_plus1_y = kin_->get_jacobian(aux_link_ids_[1], control_joint_ids_,
                                   tinyfk::RotationType::IGNORE, with_base_);
  Eigen::MatrixXd jac(2, q_dim());
  jac.row(0) = jac_plus1_x.row(2) - jac_here.row(2);
  jac.row(1) = jac_plus1_y.row(2) - jac_here.row(2);
  return {vals, jac};
};

SphereCollisionCst::SphereCollisionCst(
    std::shared_ptr<tinyfk::KinematicModel> kin,
    const std::vector<std::string>& control_joint_names,
    bool with_base,
    const std::vector<SphereAttachmentSpec>& sphere_specs,
    const std::vector<std::pair<std::string, std::string>>& selcol_group_pairs,
    std::optional<SDFBase::Ptr> fixed_sdf)
    : IneqConstraintBase(kin, control_joint_names, with_base),
      fixed_sdf_(fixed_sdf == std::nullopt ? nullptr : *fixed_sdf) {
  for (size_t i = 0; i < sphere_specs.size(); i++) {
    auto& spec = sphere_specs[i];

    auto parent_id = kin_->get_link_ids({spec.parent_link_name})[0];
    Eigen::Vector3d group_center = {0.0, 0.0, 0.0};
    std::vector<size_t> sphere_ids;
    for (size_t j = 0; j < spec.relative_positions.cols(); j++) {
      Eigen::Vector3d relpos = spec.relative_positions.col(j);

      auto new_link =
          kin_->add_new_link(parent_id, {relpos.x(), relpos.y(), relpos.z()},
                             {0.0, 0.0, 0.0}, false);
      sphere_ids.push_back(new_link->id);
      group_center += relpos;
    }
    group_center /= spec.relative_positions.cols();

    auto group_sphere_link = kin_->add_new_link(
        parent_id, {group_center.x(), group_center.y(), group_center.z()},
        {0.0, 0.0, 0.0}, false);

    double max_dist = 0.0;
    for (size_t j = 0; j < spec.relative_positions.cols(); j++) {
      double dist = (spec.relative_positions.col(j) - group_center).norm() +
                    spec.radii[j];
      if (dist > max_dist) {
        max_dist = dist;
      }
    }
    double group_radius = max_dist;
    sphere_groups_.push_back({spec.parent_link_name, sphere_ids, spec.radii,
                              group_sphere_link->id, group_radius,
                              spec.ignore_collision});
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
  if (all_sdfs_cache_.size() == 0) {
    throw std::runtime_error("(cpp) No SDFs are set");
  }
  if (!check_ext_collision()) {
    return false;
  }
  return check_self_collision();
}

bool SphereCollisionCst::check_ext_collision() {
  for (auto& group : sphere_groups_) {
    if (group.ignore_collision) {
      continue;
    }

    tinyfk::Transform group_center;
    kin_->get_link_pose(group.group_sphere_id, group_center);
    bool broad_collision = false;
    for (auto& sdf : all_sdfs_cache_) {
      if (!sdf->is_outside(group_center.trans(), group.group_radius)) {
        broad_collision = true;
        break;
      }
    }
    if (!broad_collision) {
      continue;
    }

    for (auto& sdf : all_sdfs_cache_) {
      for (size_t i = 0; i < group.sphere_ids.size(); i++) {
        tinyfk::Transform center;
        kin_->get_link_pose(group.sphere_ids[i], center);
        if (!sdf->is_outside(center.trans(), group.radii[i])) {
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

    tinyfk::Transform group1_center, group2_center;
    kin_->get_link_pose(group1.group_sphere_id, group1_center);
    kin_->get_link_pose(group2.group_sphere_id, group2_center);
    double outer_sqdist =
        (group1_center.trans() - group2_center.trans()).squaredNorm();
    double outer_r_sum = group1.group_radius + group2.group_radius;
    if (outer_sqdist > outer_r_sum * outer_r_sum) {
      continue;
    }

    // check if the inner volumes are colliding
    for (size_t i = 0; i < group1.sphere_ids.size(); i++) {
      for (size_t j = 0; j < group2.sphere_ids.size(); j++) {
        tinyfk::Transform sphere1, sphere2;
        kin_->get_link_pose(group1.sphere_ids[i], sphere1);
        kin_->get_link_pose(group2.sphere_ids[j], sphere2);
        double sqdist = (sphere1.trans() - sphere2.trans()).squaredNorm();
        double r_sum = group1.radii[i] + group2.radii[j];
        if (sqdist < r_sum * r_sum) {
          return false;
        }
      }
    }
  }
  return true;
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd>
SphereCollisionCst::evaluate_dirty() {
  if (all_sdfs_cache_.size() == 0) {
    throw std::runtime_error("(cpp) No SDFs are set");
  }

  // collision vs outers
  Eigen::VectorXd grad_in_cspace_other(q_dim());
  double min_val_other = cutoff_dist_;
  std::optional<size_t> min_sphere_idx = std::nullopt;
  std::optional<size_t> min_group_idx = std::nullopt;
  std::optional<size_t> min_sdf_idx = std::nullopt;
  {
    for (size_t i = 0; i < sphere_groups_.size(); i++) {
      auto& group = sphere_groups_[i];
      if (group.ignore_collision) {
        continue;
      }

      // filter out groups that are not colliding with margin of cutoff
      tinyfk::Transform group_center;
      kin_->get_link_pose(group.group_sphere_id, group_center);
      bool broad_collision = false;
      for (auto& sdf : all_sdfs_cache_) {
        if (!sdf->is_outside(group_center.trans(),
                             group.group_radius + cutoff_dist_)) {
          broad_collision = true;
          break;
        }
      }
      if (broad_collision) {
        // compute min_val_other
        for (size_t j = 0; j < all_sdfs_cache_.size(); j++) {
          auto& sdf = all_sdfs_cache_[j];
          for (size_t k = 0; k < group.sphere_ids.size(); k++) {
            tinyfk::Transform sphere;
            kin_->get_link_pose(group.sphere_ids[k], sphere);
            double val = sdf->evaluate(sphere.trans()) - group.radii[k];
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

    if (min_sphere_idx == std::nullopt) {
      // cutoff case
      grad_in_cspace_other.setConstant(0.);
    } else {
      size_t min_sphere_id =
          sphere_groups_[*min_group_idx].sphere_ids[*min_sphere_idx];
      tinyfk::Transform min_sphere;
      kin_->get_link_pose(min_sphere_id, min_sphere);
      double r = sphere_groups_[*min_group_idx].radii[*min_sphere_idx];
      Eigen::Vector3d grad;
      for (size_t i = 0; i < 3; i++) {
        Eigen::Vector3d perturbed_center = min_sphere.trans();
        perturbed_center[i] += 1e-6;
        double val =
            all_sdfs_cache_[*min_sdf_idx]->evaluate(perturbed_center) - r;
        grad[i] = (val - min_val_other) / 1e-6;
      }
      auto sphere_jac =
          kin_->get_jacobian(min_sphere_id, control_joint_ids_,
                             tinyfk::RotationType::IGNORE, with_base_);
      grad_in_cspace_other = sphere_jac.transpose() * grad;
    }
  }
  if (selcol_group_id_pairs_.size() == 0) {
    Eigen::MatrixXd jac(1, grad_in_cspace_other.size());
    jac.row(0) = grad_in_cspace_other;
    return {Eigen::VectorXd::Constant(1, min_val_other), jac};
  } else {
    // collision vs inners (self collision)
    std::optional<std::array<size_t, 4>> min_pairs =
        std::nullopt;  // (group_i, sphere_i, group_j, sphere_j)
    double dist_min = cutoff_dist_;
    for (auto& group_id_pair : selcol_group_id_pairs_) {
      auto& group1 = sphere_groups_[group_id_pair.first];
      auto& group2 = sphere_groups_[group_id_pair.second];

      tinyfk::Transform group1_center, group2_center;
      kin_->get_link_pose(group1.group_sphere_id, group1_center);
      kin_->get_link_pose(group2.group_sphere_id, group2_center);
      double outer_sqdist =
          (group1_center.trans() - group2_center.trans()).squaredNorm();
      double outer_r_sum_with_margin =
          group1.group_radius + group2.group_radius + cutoff_dist_;
      if (outer_sqdist > outer_r_sum_with_margin * outer_r_sum_with_margin) {
        continue;
      }
      for (size_t i = 0; i < group1.sphere_ids.size(); i++) {
        for (size_t j = 0; j < group2.sphere_ids.size(); j++) {
          tinyfk::Transform group1_center, group2_center;
          kin_->get_link_pose(group1.sphere_ids[i], group1_center);
          kin_->get_link_pose(group2.sphere_ids[j], group2_center);
          double dist = (group1_center.trans() - group2_center.trans()).norm() -
                        (group1.radii[i] + group2.radii[j]);
          if (dist < dist_min) {
            dist_min = dist;
            min_pairs = {group_id_pair.first, i, group_id_pair.second, j};
          }
        }
      }
    }

    // compute gradient
    if (min_pairs == std::nullopt) {
      Eigen::MatrixXd jac(2, grad_in_cspace_other.size());
      jac.row(0) = grad_in_cspace_other;
      jac.row(1).setConstant(0.);
      return {Eigen::Vector2d(min_val_other, dist_min), jac};
    } else {
      auto& group1 = sphere_groups_[min_pairs->at(0)];
      auto& group2 = sphere_groups_[min_pairs->at(2)];
      auto& sphere1 = group1.sphere_ids[min_pairs->at(1)];
      auto& sphere2 = group2.sphere_ids[min_pairs->at(3)];
      tinyfk::Transform center1, center2;
      kin_->get_link_pose(sphere1, center1);
      kin_->get_link_pose(sphere2, center2);
      Eigen::Vector3d center_diff = center1.trans() - center2.trans();
      Eigen::MatrixXd&& jac1 =
          kin_->get_jacobian(sphere1, control_joint_ids_,
                             tinyfk::RotationType::IGNORE, with_base_);
      Eigen::MatrixXd&& jac2 =
          kin_->get_jacobian(sphere2, control_joint_ids_,
                             tinyfk::RotationType::IGNORE, with_base_);
      double norminv = 1.0 / center_diff.norm();
      Eigen::VectorXd&& grad_in_cspace_self =
          norminv * center_diff.transpose() * (jac1 - jac2);
      Eigen::MatrixXd jac(2, grad_in_cspace_other.size());
      jac.row(0) = grad_in_cspace_other;
      jac.row(1) = grad_in_cspace_self;
      return {Eigen::Vector2d(min_val_other, dist_min), jac};
    }
  }
}

std::vector<std::pair<Eigen::Vector3d, double>>
SphereCollisionCst::get_group_spheres() const {
  std::vector<std::pair<Eigen::Vector3d, double>> spheres;
  for (auto& sphere_group : sphere_groups_) {
    tinyfk::Transform pose;
    kin_->get_link_pose(sphere_group.group_sphere_id, pose);
    spheres.push_back({pose.trans(), sphere_group.group_radius});
  }
  return spheres;
}

std::vector<std::pair<Eigen::Vector3d, double>>
SphereCollisionCst::get_all_spheres() const {
  std::vector<std::pair<Eigen::Vector3d, double>> spheres;
  for (auto& sphere_group : sphere_groups_) {
    for (size_t i = 0; i < sphere_group.sphere_ids.size(); i++) {
      tinyfk::Transform pose;
      kin_->get_link_pose(sphere_group.sphere_ids[i], pose);
      spheres.push_back({pose.trans(), sphere_group.radii[i]});
    }
  }
  return spheres;
}

void SphereCollisionCst::set_all_sdfs() {
  all_sdfs_cache_.clear();
  if (fixed_sdf_ != nullptr) {
    all_sdfs_cache_.push_back(fixed_sdf_);
  }
  if (sdf_ != nullptr) {
    all_sdfs_cache_.push_back(sdf_);
  }
}

bool ComInPolytopeCst::is_valid_dirty() {
  // COPIED from evaluate() >> START
  auto com = kin_->get_com();
  if (force_link_ids_.size() > 0) {
    double vertical_force_sum = 1.0;  // 1.0 for normalized self
    for (size_t j = 0; j < force_link_ids_.size(); ++j) {
      double force = applied_force_values_[j] / kin_->total_mass_;
      vertical_force_sum += force;
      tinyfk::Transform pose;
      kin_->get_link_pose(force_link_ids_[j], pose);
      com += force * pose.trans();
    }
    com /= vertical_force_sum;
  }
  // COPIED from evaluate() >> END
  return polytope_sdf_->evaluate(com) < 0;
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> ComInPolytopeCst::evaluate_dirty() {
  Eigen::VectorXd vals(cst_dim());
  Eigen::MatrixXd jac(cst_dim(), q_dim());

  auto com = kin_->get_com();
  auto com_jaco = kin_->get_com_jacobian(control_joint_ids_, q_dim());
  if (force_link_ids_.size() > 0) {
    double vertical_force_sum = 1.0;  // 1.0 for normalized self
    for (size_t j = 0; j < force_link_ids_.size(); ++j) {
      double force = applied_force_values_[j] / kin_->total_mass_;
      vertical_force_sum += force;
      tinyfk::Transform pose;
      kin_->get_link_pose(force_link_ids_[j], pose);
      com += force * pose.trans();

      com_jaco += kin_->get_jacobian(force_link_ids_[j], control_joint_ids_,
                                     tinyfk::RotationType::IGNORE, with_base_) *
                  force;
    }
    double inv = 1.0 / vertical_force_sum;
    com *= inv;
    com_jaco *= inv;
  }
  double val = -polytope_sdf_->evaluate(com);
  vals[0] = val;

  Eigen::Vector3d grad;
  for (size_t i = 0; i < 3; i++) {
    Eigen::Vector3d perturbed_com = com;
    perturbed_com[i] += 1e-6;
    double val_perturbed = -polytope_sdf_->evaluate(perturbed_com);
    grad[i] = (val_perturbed - val) / 1e-6;
  }
  jac.row(0) = com_jaco.transpose() * grad;

  return {vals, jac};
};

}  // namespace cst
