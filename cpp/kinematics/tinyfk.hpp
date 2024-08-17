#ifndef tinyfk_hpp
#define tinyfk_hpp

#include "urdf_model/joint.h"
#include "urdf_model/pose.h"
#include "urdf_parser/urdf_parser.h"
#include <Eigen/Core> // slow compile...
#include <array>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <stack>
#include <stdexcept>
#include <unordered_map>

#include "data_structure.hpp"

namespace tinyfk {

using Bound = std::pair<double, double>;
using Transform = urdf::Pose;
using Vector3 = urdf::Vector3;
using Rotation = urdf::Rotation;

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

struct LinkIdAndTransform {
  size_t id;
  Transform pose;
};

enum class RotationType { IGNORE, RPY, XYZW };

class KinematicModel {
public: // members
  // change them all to private later
  urdf::ModelInterfaceSharedPtr robot_urdf_interface_;

  size_t root_link_id_;
  std::vector<urdf::LinkSharedPtr> links_;
  std::unordered_map<std::string, int> link_ids_;
  std::vector<urdf::LinkSharedPtr> com_dummy_links_;

  std::vector<urdf::JointSharedPtr> joints_;
  std::unordered_map<std::string, int> joint_ids_;
  std::vector<double> joint_angles_;
  Transform base_pose_;

  RelevancePredicateTable rptable_;
  int num_dof_;
  double total_mass_;

  mutable SizedStack<LinkIdAndTransform> transform_stack_;
  mutable SizedCache<Transform> transform_cache_;

public: // functions
  KinematicModel(const std::string &xml_string);

  virtual ~KinematicModel() {}

  void set_joint_angles( // this will clear all the cache stored
      const std::vector<size_t> &joint_ids,
      const std::vector<double> &joint_angles);
  void _set_joint_angles( // lower version of the set_joint_angle which does not
                          // clear cache
      const std::vector<size_t> &joint_ids,
      const std::vector<double> &joint_angles);

  void set_base_pose(Transform pose) {
    this->_set_base_pose(pose);
    this->clear_cache();
  }
  void _set_base_pose(Transform pose);

  void clear_cache();

  void set_init_angles();

  std::vector<double>
  get_joint_angles(const std::vector<size_t> &joint_ids) const;

  std::vector<size_t> get_joint_ids(std::vector<std::string> joint_names) const;

  std::vector<Bound>
  get_joint_position_limits(const std::vector<size_t> &joint_ids) const;

  std::vector<double>
  get_joint_velocity_limits(const std::vector<size_t> &joint_ids) const;

  std::vector<double>
  get_joint_effort_limits(const std::vector<size_t> &joint_ids) const;

  std::vector<std::string> get_joint_names() const {
    std::vector<std::string> joint_names;
    for (auto &joint : joints_) {
      joint_names.push_back(joint->name);
    }
    return joint_names;
  }

  std::vector<size_t> get_link_ids(std::vector<std::string> link_names) const;

  std::vector<std::string> get_link_names() const {
    std::vector<std::string> link_names;
    for (auto &link : links_) {
      link_names.push_back(link->name);
    }
    return link_names;
  }

  void get_link_pose(size_t link_id, Transform &out_tf_root_to_ef) const;

  Eigen::MatrixXd get_jacobian(size_t elink_id,
                               const std::vector<size_t> &joint_ids,
                               RotationType rot_type = RotationType::IGNORE,
                               bool with_base = false);

  Vector3 get_com();

  Eigen::MatrixXd get_com_jacobian(const std::vector<size_t> &joint_ids,
                                   bool with_base);

  Eigen::Matrix3d get_total_inertia_matrix();

  void set_joint_angle(size_t joint_id, double angle) {
    joint_angles_[joint_id] = angle;
  }

  urdf::LinkSharedPtr add_new_link(const std::string &link_name,
                                   size_t parent_id,
                                   const std::array<double, 3> &position,
                                   const std::array<double, 3> &rpy);

  urdf::LinkSharedPtr add_new_link(const std::string &link_name,
                                   size_t parent_id, const Transform &pose);

private:
  void get_link_pose_inner(size_t link_id, Transform &out_tf_root_to_ef) const;
  void update_rptable();
};

std::string load_urdf(const std::string &urdf_path);
}; // namespace tinyfk

#endif // include guard
