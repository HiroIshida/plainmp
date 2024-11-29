#ifndef tinyfk_hpp
#define tinyfk_hpp

#include <assert.h>
#include <Eigen/Core>  // slow compile...
#include <array>
#include <fstream>
#include <iostream>
#include <optional>
#include <stack>
#include <stdexcept>
#include <unordered_map>
#include "urdf_model/joint.h"
#include "urdf_model/pose.h"
#include "urdf_parser/urdf_parser.h"

#include "data_structure.hpp"

namespace kinematics {

struct RelevancePredicateTable {
  std::vector<std::vector<bool>> table_;
  RelevancePredicateTable() : RelevancePredicateTable(0, 0){};
  RelevancePredicateTable(int N_link, int N_joint) {
    // Jacobian computation typically iterates over all joint fixing a link,
    // and does not iterate over all links fixing a joint.
    // Therefore, we should put joint-related things inner for access
    // efficiency.
    for (int i = 0; i < N_link; i++) {
      table_.push_back(std::vector<bool>(N_joint));
    }
  }
  bool isRelevant(int link_id, int joint_id) const {
    return table_[link_id][joint_id];
  }
};

enum class RotationType { IGNORE, RPY, XYZW };

template <typename Scalar>
class KinematicModel {
 public:  // members
  using Transform = urdf::QuatTrans<Scalar>;
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using Quat = Eigen::Quaternion<Scalar>;
  using MatrixDynamic = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using Bound = std::pair<Scalar, Scalar>;

  // link stuff
  size_t root_link_id_;                                    // N_link
  std::unordered_map<std::string, int> link_name_id_map_;  // N_link
  std::vector<size_t> link_parent_link_ids_;               // N_link
  std::vector<std::vector<size_t>> link_child_link_idss_;  // N_link
  std::vector<bool> link_consider_rotation_;               // N_link
  std::vector<size_t> com_link_ids_;                       // N_COM_link
  std::vector<Scalar> link_masses_;                        // N_COM_link
  std::vector<Vector3> com_local_positions_;               // N_COM_link

  // joint stuff
  std::vector<int> joint_types_;
  std::vector<Vector3> joint_axes_;
  std::vector<Vector3> joint_positions_;
  std::vector<Quat> joint_orientations_;
  std::vector<bool> joint_orientation_identity_flags_;
  bool all_joint_orientation_identity_;
  std::vector<int> joint_child_link_ids_;
  std::vector<Bound> joint_position_limits_;

  std::unordered_map<std::string, int> joint_name_id_map_;
  std::vector<Scalar> joint_angles_;
  Transform base_pose_;

  RelevancePredicateTable rptable_;
  int num_dof_;
  Scalar total_mass_;

  // cache stuff
  mutable SizedCache<Transform> transform_cache_;
  mutable std::vector<Transform> tf_plink_to_hlink_cache_;

 public:  // functions
  KinematicModel(const std::string& xml_string);
  void init_link_info(const std::vector<urdf::LinkSharedPtr>& links);
  void init_joint_info(const std::vector<urdf::LinkSharedPtr>& links);
  void init_transform_cache(const std::vector<urdf::LinkSharedPtr>& links);

  virtual ~KinematicModel() {}

  void set_joint_angles(const std::vector<size_t>& joint_ids,
                        const std::vector<Scalar>& joint_angles,
                        bool high_accuracy = true);

  template <bool approx, bool all_quat_identity>
  void set_joint_angles_impl(const std::vector<size_t>& joint_ids,
                             const std::vector<Scalar>& joint_angles);

  inline Transform get_base_pose() const { return base_pose_; }

  inline void set_base_pose(const Transform& pose) {
    base_pose_ = pose;
    this->clear_cache();
  }

  inline void clear_cache() {
    transform_cache_.clear();
    transform_cache_.set_cache(root_link_id_, base_pose_);
  }

  void set_init_angles();

  std::vector<Scalar> get_joint_angles(
      const std::vector<size_t>& joint_ids) const;

  std::vector<size_t> get_joint_ids(std::vector<std::string> joint_names) const;

  std::vector<Bound> get_joint_position_limits(
      const std::vector<size_t>& joint_ids) const;

  std::vector<size_t> get_link_ids(std::vector<std::string> link_names) const;

  const Transform& get_link_pose(size_t link_id) const {
    if (!transform_cache_.is_cached(link_id)) {
      build_cache_until(link_id);
    }
    return transform_cache_.data_[link_id];
  }

  MatrixDynamic get_jacobian(size_t elink_id,
                             const std::vector<size_t>& joint_ids,
                             RotationType rot_type = RotationType::IGNORE,
                             bool with_base = false);

  MatrixDynamic get_attached_point_jacobian(
      size_t plink_id,
      Vector3 global_pos,  // TODO: provide relative pos is clearner though
      const std::vector<size_t>& joint_ids,
      bool with_base = false);

  Vector3 get_com();

  MatrixDynamic get_com_jacobian(const std::vector<size_t>& joint_ids,
                                 bool with_base);

  void set_joint_angle(size_t joint_id, Scalar angle) {
    joint_angles_[joint_id] = angle;
  }

  size_t add_new_link(size_t parent_id,
                      const std::array<Scalar, 3>& position,
                      const std::array<Scalar, 3>& rpy,
                      bool consider_rotation,
                      std::optional<std::string> link_name = std::nullopt);

  size_t add_new_link(size_t parent_id,
                      const Transform& pose,
                      bool consider_rotation,
                      std::optional<std::string> link_name = std::nullopt);

 private:
  void build_cache_until(size_t link_id) const;
  void build_cache_until_inner(size_t link_id) const;
  void update_rptable();
};

std::string load_urdf(const std::string& urdf_path);
};  // namespace tinyfk

#endif  // include guard
