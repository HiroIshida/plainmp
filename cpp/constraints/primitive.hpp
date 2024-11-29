#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <memory>
#include <optional>
#include <utility>
#include "collision/primitive_sdf.hpp"
#include "kinematics/kinematics.hpp"

namespace constraint {

namespace kin = kinematics;

class ConstraintBase {
 public:
  using Ptr = std::shared_ptr<ConstraintBase>;
  using Transform = kin::KinematicModel<double>::Transform;
  ConstraintBase(std::shared_ptr<kin::KinematicModel<double>> kin,
                 const std::vector<std::string>& control_joint_names,
                 bool with_base)
      : kin_(kin),
        control_joint_ids_(kin->get_joint_ids(control_joint_names)),
        with_base_(with_base) {}

  void update_kintree(const std::vector<double>& q, bool high_accuracy = true) {
    if (with_base_) {
      std::vector<double> q_head(control_joint_ids_.size());
      std::copy(q.begin(), q.begin() + control_joint_ids_.size(),
                q_head.begin());
      kin_->set_joint_angles(control_joint_ids_, q_head);
      Transform pose;
      size_t head = control_joint_ids_.size();
      pose.trans().x() = q[head];
      pose.trans().y() = q[head + 1];
      pose.trans().z() = q[head + 2];
      pose.setQuaternionFromRPY(q[head + 3], q[head + 4], q[head + 5]);
      kin_->set_base_pose(pose);
    } else {
      kin_->set_joint_angles(control_joint_ids_, q, high_accuracy);
    }
  }

  // Intentionally not put this into update_kintree, considering that
  // this will be called from composite constraint by for-loop
  virtual void post_update_kintree() {}

  inline size_t q_dim() const {
    return control_joint_ids_.size() + (with_base_ ? 6 : 0);
  }

  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate(
      const std::vector<double>& q) {
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
  bool with_base_;
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
  bool is_valid(const std::vector<double>& q) {
    update_kintree(q, false);
    post_update_kintree();
    return is_valid_dirty();
  }
  bool is_equality() const override { return false; }
  virtual bool is_valid_dirty() = 0;
};

};  // namespace constraint
