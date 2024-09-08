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

template <typename Scalar>
class ConstraintBase {
 public:
  using Ptr = std::shared_ptr<ConstraintBase>;
  using Transform = typename tinyfk::KinematicModel<Scalar>::Transform;
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using Values = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using MatrixDynamic = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  ConstraintBase(std::shared_ptr<tinyfk::KinematicModel<Scalar>> kin,
                 const std::vector<std::string>& control_joint_names,
                 bool with_base)
      : kin_(kin),
        control_joint_ids_(kin->get_joint_ids(control_joint_names)),
        with_base_(with_base) {}

  void update_kintree(const std::vector<Scalar>& q, bool high_accuracy = true) {
    if (with_base_) {
      std::vector<Scalar> q_head(control_joint_ids_.size());
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

  std::pair<Values, MatrixDynamic> evaluate(const std::vector<Scalar>& q) {
    update_kintree(q);
    post_update_kintree();
    return evaluate_dirty();
  }

  virtual std::pair<Values, MatrixDynamic> evaluate_dirty() = 0;
  virtual size_t cst_dim() const = 0;
  virtual std::string get_name() const = 0;
  virtual bool is_equality() const = 0;
  virtual ~ConstraintBase() = default;

 public:
  // want to make these protected, but will be used in CompositeConstraintBase
  // making this friend is also an option, but it's too complicated
  std::shared_ptr<tinyfk::KinematicModel<Scalar>> kin_;

 protected:
  std::vector<size_t> control_joint_ids_;
  bool with_base_;
};

template <typename Scalar>
class EqConstraintBase : public ConstraintBase<Scalar> {
 public:
  using Ptr = std::shared_ptr<EqConstraintBase>;
  using ConstraintBase<Scalar>::ConstraintBase;
  bool is_equality() const override { return true; }
};

template <typename Scalar>
class IneqConstraintBase : public ConstraintBase<Scalar> {
 public:
  using Ptr = std::shared_ptr<IneqConstraintBase>;
  using ConstraintBase<Scalar>::ConstraintBase;
  bool is_valid(const std::vector<Scalar>& q) {
    this->update_kintree(q, false);
    this->post_update_kintree();
    return is_valid_dirty();
  }
  bool is_equality() const override { return false; }
  virtual bool is_valid_dirty() = 0;
};

template <typename Scalar>
class ConfigPointCst : public EqConstraintBase<Scalar> {
 public:
  using Ptr = std::shared_ptr<ConfigPointCst>;
  using Transform = typename tinyfk::KinematicModel<Scalar>::Transform;
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using Values = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using MatrixDynamic = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  ConfigPointCst(std::shared_ptr<tinyfk::KinematicModel<Scalar>> kin,
                 const std::vector<std::string>& control_joint_names,
                 bool with_base,
                 const Values& q)
      : EqConstraintBase<Scalar>(kin, control_joint_names, with_base), q_(q) {
    size_t dof = this->control_joint_ids_.size() + (with_base ? 6 : 0);
    if (q.size() != dof) {
      throw std::runtime_error(
          "q must have the same size as the number of control joints");
    }
  }
  std::pair<Values, MatrixDynamic> evaluate_dirty() override {
    size_t dof = this->q_dim();
    std::vector<Scalar> q_now_joint_std =
        this->kin_->get_joint_angles(this->control_joint_ids_);

    Values q_now(dof);
    for (size_t i = 0; i < this->control_joint_ids_.size(); i++) {
      q_now[i] = q_now_joint_std[i];
    }
    if (this->with_base_) {
      size_t head = this->control_joint_ids_.size();
      auto base_pose = this->kin_->get_base_pose();
      q_now(head) = base_pose.trans().x();
      q_now(head + 1) = base_pose.trans().y();
      q_now(head + 2) = base_pose.trans().z();
      auto base_rpy = base_pose.getRPY();
      q_now(head + 3) = base_rpy.x();
      q_now(head + 4) = base_rpy.y();
      q_now(head + 5) = base_rpy.z();
    }
    return {q_now - q_, MatrixDynamic::Identity(dof, dof)};
  }
  size_t cst_dim() const { return q_.size(); }
  std::string get_name() const override { return "ConfigPointCst"; }

 private:
  Values q_;
};

template <typename Scalar>
class LinkPoseCst : public EqConstraintBase<Scalar> {
 public:
  using Ptr = std::shared_ptr<LinkPoseCst>;
  using Transform = typename tinyfk::KinematicModel<Scalar>::Transform;
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using Values = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using MatrixDynamic = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  LinkPoseCst(std::shared_ptr<tinyfk::KinematicModel<Scalar>> kin,
              const std::vector<std::string>& control_joint_names,
              bool with_base,
              const std::vector<std::string>& link_names,
              const std::vector<Values>& poses)
      : EqConstraintBase<Scalar>(kin, control_joint_names, with_base),
        link_ids_(this->kin_->get_link_ids(link_names)),
        poses_(poses) {
    for (auto& pose : poses_) {
      if (pose.size() != 3 && pose.size() != 6 && pose.size() != 7) {
        throw std::runtime_error("All poses must be 3 or 6 or 7 dimensional");
      }
    }
  }
  std::pair<Values, MatrixDynamic> evaluate_dirty() override;
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
  std::vector<Values> poses_;
};

template <typename Scalar>
class RelativePoseCst : public EqConstraintBase<Scalar> {
 public:
  using Ptr = std::shared_ptr<RelativePoseCst>;
  using Transform = typename tinyfk::KinematicModel<Scalar>::Transform;
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using Values = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using MatrixDynamic = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  RelativePoseCst(std::shared_ptr<tinyfk::KinematicModel<Scalar>> kin,
                  const std::vector<std::string>& control_joint_names,
                  bool with_base,
                  const std::string& link_name1,
                  const std::string& link_name2,
                  const Vector3& relative_pose)
      : EqConstraintBase<Scalar>(kin, control_joint_names, with_base),
        link_id2_(this->kin_->get_link_ids({link_name2})[0]),
        relative_pose_(relative_pose) {
    // TODO: because name is hard-coded, we cannot create two RelativePoseCst...
    auto pose = Transform::Identity();
    pose.trans() = relative_pose;
    size_t link_id1_ = this->kin_->get_link_ids({link_name1})[0];
    dummy_link_id_ = this->kin_->add_new_link(link_id1_, pose, true);
  }

  std::pair<Values, MatrixDynamic> evaluate_dirty() override;
  size_t cst_dim() const { return 7; }
  std::string get_name() const override { return "RelativePoseCst"; }

 private:
  size_t link_id2_;
  size_t dummy_link_id_;
  Vector3 relative_pose_;
};

template <typename Scalar>
class FixedZAxisCst : public EqConstraintBase<Scalar> {
 public:
  using Ptr = std::shared_ptr<FixedZAxisCst>;
  using Transform = typename tinyfk::KinematicModel<Scalar>::Transform;
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using Values = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using MatrixDynamic = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  FixedZAxisCst(std::shared_ptr<tinyfk::KinematicModel<Scalar>> kin,
                const std::vector<std::string>& control_joint_names,
                bool with_base,
                const std::string& link_name);

  std::pair<Values, MatrixDynamic> evaluate_dirty() override;
  size_t cst_dim() const override { return 2; }
  std::string get_name() const override { return "FixedZAxisCst"; }

 private:
  size_t link_id_;
  std::vector<size_t> aux_link_ids_;
};

template <typename Scalar>
struct SphereAttachmentSpec {
  using Points = Eigen::Matrix<Scalar, 3, Eigen::Dynamic>;
  using Values = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  std::string postfix;
  std::string parent_link_name;
  Points relative_positions;
  Values radii;
  bool ignore_collision;
};

template <typename Scalar>
struct SphereGroup {
  using Points = Eigen::Matrix<Scalar, 3, Eigen::Dynamic>;
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using Values = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  std::string parent_link_name;
  size_t parent_link_id;
  Values radii;
  Scalar group_radius;
  bool ignore_collision;
  Points sphere_relative_positions;
  Vector3 group_sphere_relative_position;
  // rot mat cache
  Points rot_mat_cache;
  bool is_rot_mat_dirty;

  // group sphere position
  Vector3 group_sphere_position_cache;
  bool is_group_sphere_position_dirty;

  // sphere positions cache
  Points sphere_positions_cache;
  bool is_sphere_positions_dirty;

  inline void clear_cache() {
    is_rot_mat_dirty = true;
    is_group_sphere_position_dirty = true;
    is_sphere_positions_dirty = true;
  }

  void create_group_sphere_position_cache(
      const std::shared_ptr<tinyfk::KinematicModel<Scalar>>& kin) {
    auto plink_pose = kin->get_link_pose(parent_link_id);
    // The code below is "safe" but not efficient so see the HACK below
    // if (is_rot_mat_dirty) {
    //   rot_mat_cache = plink_pose.quat().toRotationMatrix();
    //   this->is_rot_mat_dirty = false;
    // }
    // HACK: because is_group_sphere_position_dirty => is_sphere_positions_dirty
    rot_mat_cache = plink_pose.quat().toRotationMatrix();
    this->group_sphere_position_cache =
        rot_mat_cache * group_sphere_relative_position + plink_pose.trans();
    this->is_group_sphere_position_dirty = false;
  }

  void create_sphere_position_cache(
      const std::shared_ptr<tinyfk::KinematicModel<Scalar>>& kin) {
    // The code below is "safe" but not efficient so see the HACK below
    // auto plink_pose = kin->get_link_pose(parent_link_id);
    // if (is_rot_mat_dirty) {
    //   rot_mat_cache = plink_pose.quat().toRotationMatrix();
    //   this->is_rot_mat_dirty = false;
    // }

    // HACK: because the sub-sphere is evaluated after the group sphere
    // we know that there exiss matrix cache and transform cache. so...
    auto& plink_trans = kin->transform_cache_.data_[parent_link_id].trans();

    // NOTE: the above for-loop is faster than batch operation using Colwise
    for (int i = 0; i < sphere_positions_cache.cols(); i++) {
      sphere_positions_cache.col(i) =
          rot_mat_cache * sphere_relative_positions.col(i) + plink_trans;
    }
    this->is_sphere_positions_dirty = false;
  }
};

template <typename Scalar>
class SphereCollisionCst : public IneqConstraintBase<Scalar> {
 public:
  using Ptr = std::shared_ptr<SphereCollisionCst>;
  using Transform = typename tinyfk::KinematicModel<Scalar>::Transform;
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using Point = Eigen::Matrix<Scalar, 3, 1>;
  using Points = Eigen::Matrix<Scalar, 3, Eigen::Dynamic>;
  using Values = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using MatrixDynamic = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  SphereCollisionCst(
      std::shared_ptr<tinyfk::KinematicModel<Scalar>> kin,
      const std::vector<std::string>& control_joint_names,
      bool with_base,
      const std::vector<SphereAttachmentSpec<Scalar>>& sphere_specs,
      const std::vector<std::pair<std::string, std::string>>& selcol_pairs,
      std::optional<typename SDFBase<Scalar>::Ptr> fixed_sdf);

  void post_update_kintree() override {
    for (auto& group : sphere_groups_) {
      group.clear_cache();
    }
  }

  void set_sdf(const typename SDFBase<Scalar>::Ptr& sdf) {
    sdf_ = sdf;
    set_all_sdfs();
  }
  typename SDFBase<Scalar>::Ptr get_sdf() const { return sdf_; }

  bool is_valid_dirty() override;
  bool check_ext_collision();
  bool check_self_collision();
  std::pair<Values, MatrixDynamic> evaluate_dirty() override;

  size_t cst_dim() const override {
    if (selcol_group_id_pairs_.size() == 0) {
      return 1;
    } else {
      return 2;
    }
  }
  std::string get_name() const override { return "SphereCollisionCst"; }
  std::vector<std::pair<Vector3, Scalar>> get_group_spheres() const;
  std::vector<std::pair<Vector3, Scalar>> get_all_spheres() const;

 private:
  void set_all_sdfs();
  void set_all_sdfs_inner(typename SDFBase<Scalar>::Ptr sdf);

  std::vector<SphereGroup<Scalar>> sphere_groups_;
  std::vector<std::pair<size_t, size_t>> selcol_group_id_pairs_;
  typename SDFBase<Scalar>::Ptr fixed_sdf_;
  typename SDFBase<Scalar>::Ptr sdf_;  // set later by user
  std::vector<typename PrimitiveSDFBase<Scalar>::Ptr> all_sdfs_cache_;
  Scalar cutoff_dist_ = 0.1;
};

template <typename Scalar>
struct AppliedForceSpec {
  std::string link_name;
  Scalar force;  // currently only z-axis force (minus direction) is supported
};

template <typename Scalar>
class ComInPolytopeCst : public IneqConstraintBase<Scalar> {
 public:
  using Ptr = std::shared_ptr<ComInPolytopeCst>;
  using Transform = typename tinyfk::KinematicModel<Scalar>::Transform;
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using Values = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using MatrixDynamic = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  ComInPolytopeCst(std::shared_ptr<tinyfk::KinematicModel<Scalar>> kin,
                   const std::vector<std::string>& control_joint_names,
                   bool with_base,
                   typename BoxSDF<Scalar>::Ptr polytope_sdf,
                   const std::vector<AppliedForceSpec<Scalar>> applied_forces)
      : IneqConstraintBase<Scalar>(kin, control_joint_names, with_base),
        polytope_sdf_(polytope_sdf) {
    auto w = polytope_sdf_->get_width();
    w[2] = 1000;  // adhoc to represent infinite height
    polytope_sdf_->set_width(w);

    auto force_link_names = std::vector<std::string>();
    for (auto& force : applied_forces) {
      force_link_names.push_back(force.link_name);
      applied_force_values_.push_back(force.force);
    }
    force_link_ids_ = this->kin_->get_link_ids(force_link_names);
  }

  bool is_valid_dirty() override;
  std::pair<Values, MatrixDynamic> evaluate_dirty() override;

  size_t cst_dim() const { return 1; }
  std::string get_name() const override { return "ComInPolytopeCst"; }

 private:
  typename BoxSDF<Scalar>::Ptr polytope_sdf_;
  std::vector<size_t> force_link_ids_;
  std::vector<Scalar> applied_force_values_;
};

};  // namespace cst
#endif
