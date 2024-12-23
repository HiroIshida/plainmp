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

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <memory>
#include <optional>
#include <utility>
#include "plainmp/kinematics/kinematics.hpp"

namespace plainmp::constraint {

namespace kin = plainmp::kinematics;

class ConstraintBase {
 public:
  using Ptr = std::shared_ptr<ConstraintBase>;
  using Transform = kin::KinematicModel<double>::Transform;
  ConstraintBase(std::shared_ptr<kin::KinematicModel<double>> kin,
                 const std::vector<std::string>& control_joint_names,
                 kin::BaseType base_type)
      : kin_(kin),
        control_joint_ids_(kin->get_joint_ids(control_joint_names)),
        base_type_(base_type) {}

  void update_kintree(const Eigen::VectorXd& q, bool high_accuracy = true) {
    if (base_type_ != kin::BaseType::FIXED) {
      const size_t n_joint = control_joint_ids_.size();
      const auto q_joint = q.head(n_joint);
      kin_->set_joint_angles(control_joint_ids_, q_joint);
      Transform pose;
      if (base_type_ == kin::BaseType::FLOATING) {
        size_t head = control_joint_ids_.size();
        pose.trans().x() = q[n_joint];
        pose.trans().y() = q[n_joint + 1];
        pose.trans().z() = q[n_joint + 2];
        pose.setQuaternionFromRPY(q[n_joint + 3], q[n_joint + 4],
                                  q[n_joint + 5]);
      }
      if (base_type_ == kin::BaseType::PLANAR) {
        pose.trans().x() = q[n_joint];
        pose.trans().y() = q[n_joint + 1];
        pose.trans().z() = 0.0;
        pose.setQuaternionFromRPY(0, 0, q[n_joint + 2]);
      }
      kin_->set_base_pose(pose);
    } else {
      kin_->set_joint_angles(control_joint_ids_, q, high_accuracy);
    }
  }

  // Intentionally not put this into update_kintree, considering that
  // this will be called from composite constraint by for-loop
  virtual void post_update_kintree() {}

  inline size_t q_dim() const {
    return control_joint_ids_.size() +
           (base_type_ == kin::BaseType::FLOATING) * 6 +
           (base_type_ == kin::BaseType::PLANAR) * 3;
  }

  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate(
      const Eigen::VectorXd& q) {
    update_kintree(q);
    post_update_kintree();
    return evaluate_dirty();
  }

  virtual std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate_dirty() = 0;
  virtual size_t cst_dim() const = 0;
  virtual std::string get_name() const = 0;
  virtual bool is_equality() const = 0;
  virtual ~ConstraintBase() = default;

 public:
  // want to make these protected, but will be used in CompositeConstraintBase
  // making this friend is also an option, but it's too complicated
  std::shared_ptr<kin::KinematicModel<double>> kin_;

 protected:
  std::vector<size_t> control_joint_ids_;
  kin::BaseType base_type_;
};

class EqConstraintBase : public ConstraintBase {
 public:
  using Ptr = std::shared_ptr<EqConstraintBase>;
  using ConstraintBase::ConstraintBase;
  bool is_equality() const override { return true; }
};

class IneqConstraintBase : public ConstraintBase {
 public:
  using Ptr = std::shared_ptr<IneqConstraintBase>;
  using ConstraintBase::ConstraintBase;
  bool is_valid(const Eigen::VectorXd& q) {
    update_kintree(q, false);
    post_update_kintree();
    return is_valid_dirty();
  }
  bool is_equality() const override { return false; }
  virtual bool is_valid_dirty() = 0;
};

};  // namespace plainmp::constraint
