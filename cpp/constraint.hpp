#ifndef CONSTRAINT_HPP
#define CONSTRAINT_HPP

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <memory>
#include <tinyfk.hpp>
#include <utility>
#include "primitive_sdf.hpp"

namespace cst {

namespace py = pybind11;
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

  virtual std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate_dirty()
      const = 0;
  virtual size_t cst_dim() const = 0;
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
  virtual bool is_valid_dirty() const = 0;
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
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate_dirty() const override;
  size_t cst_dim() const {
    size_t dim = 0;
    for (auto& pose : poses_) {
      dim += pose.size();
    }
    return dim;
  }

 private:
  std::vector<size_t> link_ids_;
  std::vector<Eigen::VectorXd> poses_;
};

struct SphereAttachentSpec {
  std::string parent_link_name;
  Eigen::Vector3d relative_position;
  double radius;
  bool ignore_collision;
};

class SphereCollisionCst : public IneqConstraintBase {
 public:
  using Ptr = std::shared_ptr<SphereCollisionCst>;
  SphereCollisionCst(
      std::shared_ptr<tinyfk::KinematicModel> kin,
      const std::vector<std::string>& control_joint_names,
      bool with_base,
      const std::vector<SphereAttachentSpec>& sphere_specs,
      const std::vector<std::pair<std::string, std::string>>& selcol_pairs,
      const std::vector<PrimitiveSDFBase::Ptr>& fixed_sdfs);

  void set_sdfs(const std::vector<PrimitiveSDFBase::Ptr>& sdfs) {
    sdfs_ = sdfs;
  }

  bool is_valid_dirty() const override;
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate_dirty() const override;

  size_t cst_dim() const {
    if (selcol_pairs_ids_.size() == 0) {
      return 1;
    } else {
      return 2;
    }
  }

 private:
  std::vector<PrimitiveSDFBase::Ptr> get_all_sdfs() const {
    // TODO: Consider using std::views::concat (but it's C++20)
    std::vector<PrimitiveSDFBase::Ptr> all_sdfs = fixed_sdfs_;
    all_sdfs.insert(all_sdfs.end(), sdfs_.begin(), sdfs_.end());
    if (all_sdfs.size() == 0) {
      throw std::runtime_error("(cpp) No SDFs are set");
    }
    return all_sdfs;
  }

  std::vector<size_t> sphere_ids_;
  std::vector<SphereAttachentSpec> sphere_specs_;
  std::vector<std::pair<size_t, size_t>> selcol_pairs_ids_;
  std::vector<PrimitiveSDFBase::Ptr> fixed_sdfs_;  // fixed after construction
  std::vector<PrimitiveSDFBase::Ptr> sdfs_;        // set later by user
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

  bool is_valid_dirty() const override;
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate_dirty() const override;

  size_t cst_dim() const { return 1; }

 private:
  BoxSDF::Ptr polytope_sdf_;
  std::vector<size_t> force_link_ids_;
  std::vector<double> applied_force_values_;
};

void bind_collision_constraints(py::module& m);

};  // namespace cst
#endif
