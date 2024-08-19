#ifndef CONSTRAINT_HPP
#define CONSTRAINT_HPP

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <memory>
#include <optional>
#include <utility>
#include "collision/primitive_sdf.hpp"
#include "kinematics/tinyfk.hpp"

namespace cst {

using namespace primitive_sdf;

class ConstraintBase {
 public:
  using Ptr = std::shared_ptr<ConstraintBase>;
  ConstraintBase(std::shared_ptr<tinyfk::KinematicModel> kin,
                 const std::vector<std::string>& control_joint_names,
                 bool with_base)
      : kin_(kin),
        control_joint_ids_(kin->get_joint_ids(control_joint_names)),
        with_base_(with_base) {}

  void update_kintree(const std::vector<double>& q) {
    if (with_base_) {
      std::vector<double> q_head(control_joint_ids_.size());
      std::copy(q.begin(), q.begin() + control_joint_ids_.size(),
                q_head.begin());
      kin_->set_joint_angles(control_joint_ids_, q_head);
      tinyfk::Transform pose;
      size_t head = control_joint_ids_.size();
      pose.position.x = q[head];
      pose.position.y = q[head + 1];
      pose.position.z = q[head + 2];
      pose.rotation.setFromRPY(q[head + 3], q[head + 4], q[head + 5]);
      kin_->set_base_pose(pose);
    } else {
      kin_->set_joint_angles(control_joint_ids_, q);
    }
  }

  inline size_t q_dim() const {
    return control_joint_ids_.size() + (with_base_ ? 6 : 0);
  }

  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate(
      const std::vector<double>& q) {
    update_kintree(q);
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
  std::shared_ptr<tinyfk::KinematicModel> kin_;

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
    update_kintree(q);
    return is_valid_dirty();
  }
  bool is_equality() const override { return false; }
  virtual bool is_valid_dirty() = 0;
};

class ConfigPointCst : public EqConstraintBase {
 public:
  using Ptr = std::shared_ptr<ConfigPointCst>;
  ConfigPointCst(std::shared_ptr<tinyfk::KinematicModel> kin,
                 const std::vector<std::string>& control_joint_names,
                 bool with_base,
                 const Eigen::VectorXd& q)
      : EqConstraintBase(kin, control_joint_names, with_base), q_(q) {
    size_t dof = control_joint_names.size() + (with_base ? 6 : 0);
    if (q.size() != dof) {
      throw std::runtime_error(
          "q must have the same size as the number of control joints");
    }
  }
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate_dirty() override {
    size_t dof = q_dim();
    std::vector<double> q_now_joint_std =
        kin_->get_joint_angles(control_joint_ids_);

    Eigen::VectorXd q_now(dof);
    for (size_t i = 0; i < control_joint_ids_.size(); i++) {
      q_now[i] = q_now_joint_std[i];
    }
    if (with_base_) {
      size_t head = control_joint_ids_.size();
      auto& base_pose = kin_->base_pose_;
      q_now(head) = base_pose.position.x;
      q_now(head + 1) = base_pose.position.y;
      q_now(head + 2) = base_pose.position.z;
      auto base_rpy = base_pose.rotation.getRPY();
      q_now(head + 3) = base_rpy.x;
      q_now(head + 4) = base_rpy.y;
      q_now(head + 5) = base_rpy.z;
    }
    return {q_now - q_, Eigen::MatrixXd::Identity(dof, dof)};
  }
  size_t cst_dim() const { return q_.size(); }
  std::string get_name() const override { return "ConfigPointCst"; }

 private:
  Eigen::VectorXd q_;
};

class LinkPoseCst : public EqConstraintBase {
 public:
  using Ptr = std::shared_ptr<LinkPoseCst>;
  LinkPoseCst(std::shared_ptr<tinyfk::KinematicModel> kin,
              const std::vector<std::string>& control_joint_names,
              bool with_base,
              const std::vector<std::string>& link_names,
              const std::vector<Eigen::VectorXd>& poses)
      : EqConstraintBase(kin, control_joint_names, with_base),
        link_ids_(kin_->get_link_ids(link_names)),
        poses_(poses) {
    for (auto& pose : poses_) {
      if (pose.size() != 3 && pose.size() != 6 && pose.size() != 7) {
        throw std::runtime_error("All poses must be 3 or 6 or 7 dimensional");
      }
    }
  }
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate_dirty() override;
  size_t cst_dim() const {
    size_t dim = 0;
    for (auto& pose : poses_) {
      dim += pose.size();
    }
    return dim;
  }
  std::string get_name() const override { return "LinkPoseCst"; }

 private:
  std::vector<size_t> link_ids_;
  std::vector<Eigen::VectorXd> poses_;
};

class RelativePoseCst : public EqConstraintBase {
 public:
  using Ptr = std::shared_ptr<RelativePoseCst>;
  RelativePoseCst(std::shared_ptr<tinyfk::KinematicModel> kin,
                  const std::vector<std::string>& control_joint_names,
                  bool with_base,
                  const std::string& link_name1,
                  const std::string& link_name2,
                  const Eigen::Vector3d& relative_pose)
      : EqConstraintBase(kin, control_joint_names, with_base),
        link_id2_(kin_->get_link_ids({link_name2})[0]),
        relative_pose_(relative_pose) {
    // TODO: because name is hard-coded, we cannot create two RelativePoseCst...
    auto dummy_link_name = link_name1 + "-relative-" + link_name2;
    tinyfk::Transform pose;
    pose.position.x = relative_pose[0];
    pose.position.y = relative_pose[1];
    pose.position.z = relative_pose[2];
    size_t link_id1_ = kin_->get_link_ids({link_name1})[0];
    auto new_link = kin_->add_new_link(dummy_link_name, link_id1_, pose, true);
    dummy_link_id_ = new_link->id;
  }

  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate_dirty() override;
  size_t cst_dim() const { return 7; }
  std::string get_name() const override { return "RelativePoseCst"; }

 private:
  size_t link_id2_;
  size_t dummy_link_id_;
  Eigen::Vector3d relative_pose_;
};

class FixedZAxisCst : public EqConstraintBase {
 public:
  using Ptr = std::shared_ptr<FixedZAxisCst>;
  FixedZAxisCst(std::shared_ptr<tinyfk::KinematicModel> kin,
                const std::vector<std::string>& control_joint_names,
                bool with_base,
                const std::string& link_name);

  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate_dirty() override;
  size_t cst_dim() const override { return 2; }
  std::string get_name() const override { return "FixedZAxisCst"; }

 private:
  size_t link_id_;
  std::vector<size_t> aux_link_ids_;
};

struct SphereAttachmentSpec {
  std::string postfix;
  std::string parent_link_name;
  Eigen::Matrix3Xd relative_positions;
  Eigen::VectorXd radii;
  bool ignore_collision;
};

struct SphereGroup {
  std::string parent_link_name;
  std::vector<size_t> sphere_ids;
  Eigen::VectorXd radii;
  size_t group_sphere_id;
  double group_radius;
  bool ignore_collision;
};

class SphereCollisionCst : public IneqConstraintBase {
 public:
  using Ptr = std::shared_ptr<SphereCollisionCst>;
  SphereCollisionCst(
      std::shared_ptr<tinyfk::KinematicModel> kin,
      const std::vector<std::string>& control_joint_names,
      bool with_base,
      const std::vector<SphereAttachmentSpec>& sphere_specs,
      const std::vector<std::pair<std::string, std::string>>& selcol_pairs,
      std::optional<SDFBase::Ptr> fixed_sdf);

  void set_sdf(const SDFBase::Ptr& sdf) {
    sdf_ = sdf;
    set_all_sdfs();
  }

  bool is_valid_dirty() override;
  bool check_ext_collision();
  bool check_self_collision();
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate_dirty() override;

  size_t cst_dim() const {
    if (selcol_group_id_pairs_.size() == 0) {
      return 1;
    } else {
      return 2;
    }
  }
  std::string get_name() const override { return "SphereCollisionCst"; }
  std::vector<std::pair<Eigen::Vector3d, double>> get_group_spheres() const;
  std::vector<std::pair<Eigen::Vector3d, double>> get_all_spheres() const;

 private:
  void set_all_sdfs();

  std::vector<SphereGroup> sphere_groups_;
  std::vector<std::pair<size_t, size_t>> selcol_group_id_pairs_;
  SDFBase::Ptr fixed_sdf_;
  SDFBase::Ptr sdf_;  // set later by user
  std::vector<SDFBase::Ptr> all_sdfs_cache_;
};

struct AppliedForceSpec {
  std::string link_name;
  double force;  // currently only z-axis force (minus direction) is supported
};

class ComInPolytopeCst : public IneqConstraintBase {
 public:
  using Ptr = std::shared_ptr<ComInPolytopeCst>;
  ComInPolytopeCst(std::shared_ptr<tinyfk::KinematicModel> kin,
                   const std::vector<std::string>& control_joint_names,
                   bool with_base,
                   BoxSDF::Ptr polytope_sdf,
                   const std::vector<AppliedForceSpec> applied_forces)
      : IneqConstraintBase(kin, control_joint_names, with_base),
        polytope_sdf_(polytope_sdf) {
    polytope_sdf_->width_[2] = 1000;  // adhoc to represent infinite height
    auto force_link_names = std::vector<std::string>();
    for (auto& force : applied_forces) {
      force_link_names.push_back(force.link_name);
      applied_force_values_.push_back(force.force);
    }
    force_link_ids_ = kin_->get_link_ids(force_link_names);
  }

  bool is_valid_dirty() override;
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate_dirty() override;

  size_t cst_dim() const { return 1; }
  std::string get_name() const override { return "ComInPolytopeCst"; }

 private:
  BoxSDF::Ptr polytope_sdf_;
  std::vector<size_t> force_link_ids_;
  std::vector<double> applied_force_values_;
};

};  // namespace cst
#endif
