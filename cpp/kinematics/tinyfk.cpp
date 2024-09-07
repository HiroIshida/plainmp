#include "tinyfk.hpp"
#include "urdf_model/pose.h"
#include <Eigen/Geometry>
#include <functional>
#include <cmath>
#include <fstream>
#include <stdexcept>
#define VCL_NAMESPACE vcl
#include <vectorclass.h>
#include <vectormath_lib.h>

namespace tinyfk {

KinematicModel::KinematicModel(const std::string &xml_string) {
  if (xml_string.empty()) {
    throw std::runtime_error("xml string is empty");
  }
  urdf::ModelInterfaceSharedPtr robot_urdf_interface =
      urdf::parseURDF(xml_string);

  // numbering link id
  std::vector<urdf::LinkSharedPtr> links;
  std::unordered_map<std::string, int> link_ids;
  int lid = 0;
  std::stack<urdf::LinkSharedPtr> link_stack;
  link_stack.push(robot_urdf_interface->root_link_);
  while (!link_stack.empty()) {
    auto link = link_stack.top();
    link_stack.pop();
    link_ids[link->name] = lid;
    link->id = lid;
    links.push_back(link);
    lid++;
    for (auto &child_link : link->child_links) {
      link_stack.push(child_link);
    }
  }
  size_t N_link = lid; // starting from 0 and finally ++ increment, so it'S ok
  root_link_id_ = link_ids[robot_urdf_interface->root_link_->name];

  // compute link_parent_link_ids
  std::vector<size_t> link_parent_link_ids(N_link);
  for(const auto& link : links) {
    if(link->getParent() != nullptr) {
      link_parent_link_ids[link->id] = link->getParent()->id;
    }else{
      link_parent_link_ids[link->id] = 999999; // dummy value to cause segfault
    }
  }

  // compute total mass
  double total_mass = 0.0;
  for (const auto &link : links) {
    if (link->inertial != nullptr) {
      total_mass += link->inertial->mass;
    }
  }

  // set joint->_child_link.
  for(const auto& pair : robot_urdf_interface->joints_) {
    urdf::JointSharedPtr joint = pair.second;
    std::string clink_name = joint->child_link_name;
    int clink_id = link_ids[clink_name];
    urdf::LinkSharedPtr clink = links[clink_id];
    joint->setChildLink(clink);
  }

  // allign joint ids
  std::unordered_map<std::string, int> joint_ids;
  std::vector<urdf::JointSharedPtr> joints;
  std::vector<int> joint_types;
  std::vector<Eigen::Vector3d> joint_axes;
  std::vector<Eigen::Vector3d> joint_positions;
  std::vector<int> joint_child_link_ids;
  auto root_link = links[root_link_id_];
  std::stack<urdf::JointSharedPtr> joint_stack;
  for (auto &joint : root_link->child_joints) {
    joint_stack.push(joint);
  }
  size_t joint_counter = 0;
  while (!joint_stack.empty()) {
    // assign joint ids in the order of DFS
    auto joint = joint_stack.top();
    joint_stack.pop();
    auto jtype = joint->type;
    if (jtype == urdf::Joint::REVOLUTE || jtype == urdf::Joint::CONTINUOUS ||
        jtype == urdf::Joint::PRISMATIC) {
      joint->id = joint_counter;
      joints.push_back(joint);
      joint_types.push_back(jtype);
      joint_axes.push_back(joint->axis);
      joint_positions.push_back(joint->parent_to_joint_origin_transform.trans());
      joint_child_link_ids.push_back(joint->getChildLink()->id);
      joint_ids[joint->name] = joint_counter;
      joint_counter++;
    }
    if(joint->getChildLink() != nullptr) {
      for (auto &child_joint : joint->getChildLink()->child_joints) {
        joint_stack.push(child_joint);
      }
    }
  }

  int num_dof = joint_ids.size();
  std::vector<double> joint_angles(num_dof, 0.0);

  transform_cache_ = SizedCache<Transform>(N_link);
  tf_plink_to_hlink_cache_ = std::vector<Transform>(N_link);
  for(size_t hid = 0; hid < N_link; hid++) {
    auto pjoint = links[hid]->parent_joint;
    if(pjoint != nullptr) {
      // HACK: if joint is not fixed, the value (origin_transform, quat_identity)
      // will be overwritten in the set_joint_angles(q)
      tf_plink_to_hlink_cache_[hid] = pjoint->parent_to_joint_origin_transform;
      // HACK: assume that joint origin rotation is identity
      // actually this is checked in the urdf loading time, so this must be ok
      tf_plink_to_hlink_cache_[hid].is_quat_identity_ = true;
    }
  }

  links_ = links;
  link_ids_ = link_ids;
  link_parent_link_ids_ = link_parent_link_ids;
  joints_ = joints;
  joint_types_ = joint_types;
  joint_axes_ = joint_axes;
  joint_positions_ = joint_positions;
  joint_child_link_ids_ = joint_child_link_ids;
  joint_ids_ = joint_ids;
  num_dof_ = num_dof;
  total_mass_ = total_mass;
  joint_angles_ = joint_angles;

  // add COM of each link as new link
  {
    // NOTE: due to my bad design (add_new_link update internal state)
    // this procedure must come after initialization of member variables
    std::vector<urdf::LinkSharedPtr> com_dummy_links;
    for (const auto &link : links) {
      if (link->inertial == nullptr) {
        continue;
      }
      Transform new_link_pose;
      new_link_pose.trans() = link->inertial->origin.trans();
      const auto new_link = this->add_new_link(link->id, new_link_pose, false);
      // set new link's inertial as the same as the parent
      // except its origin is zero
      new_link->inertial = link->inertial;
      new_link->inertial->origin = Transform::Identity();
      com_dummy_links.push_back(new_link);
    }
    this->com_dummy_links_ = com_dummy_links;
  }

  this->set_base_pose(Transform::Identity());
}

void KinematicModel::set_joint_angles(const std::vector<size_t> &joint_ids,
                                      const std::vector<double> &joint_angles,
                                      bool high_accuracy) {
  for (size_t i = 0; i < joint_ids.size(); i++) {
    auto joint_id = joint_ids[i];
    joint_angles_[joint_id] = joint_angles[i];
    auto& tf_plink_to_hlink = tf_plink_to_hlink_cache_[joint_child_link_ids_[joint_id]];
    auto& tf_plink_to_pjoint_trans = joint_positions_[joint_id];
    if(joint_types_[joint_id] != urdf::Joint::PRISMATIC) {
      auto x = joint_angles[i] * 0.5;
      if(high_accuracy){
        tf_plink_to_hlink.quat().coeffs() << sin(x) * joint_axes_[joint_id], cos(x);
      }else{
        // DO NOT USE THIS in the gradient-based optimization
        // Approximate sin(x) = x - x^3/3! + x^5/5! - x^7/7! + x^9/9!
        // Approximate cos(x) = 1 - x^2/2! + x^4/4! - x^6/6! + x^8/8!
        if(x > 0.5 * M_PI || x < -0.5 * M_PI){
          if(x > M_PI || x < -M_PI){
            x = x - 2 * M_PI * std::floor(x * 0.1591549430918953 + 0.5);
          }
          if(x < -0.5 * M_PI){
            x = -x - M_PI;
          }else if(x > 0.5 * M_PI){
            x = -x + M_PI;
          }else{
          }
        }
        auto xx = x * x;
        auto xxx = x * xx;
        auto xxxx = xx * xx;
        auto xxxxx = xx * xxx;
        auto xxxxxx = xxx * xxx;
        auto xxxxxxx = xxx * xxxx;
        auto xxxxxxxx = xxxx * xxxx;
        auto xxxxxxxxx = xxxx * xxxxx;
        constexpr auto coeff2 = 1.0 / (1.0 * 2.0);
        constexpr auto coeff3 = 1.0 / (1.0 * 2.0 * 3.0);
        constexpr auto coeff4 = 1.0 / (1.0 * 2.0 * 3.0 * 4.0);
        constexpr auto coeff5 = 1.0 / (1.0 * 2.0 * 3.0 * 4.0 * 5.0);
        constexpr auto coeff6 = 1.0 / (1.0 * 2.0 * 3.0 * 4.0 * 5.0 * 6.0);
        constexpr auto coeff7 = 1.0 / (1.0 * 2.0 * 3.0 * 4.0 * 5.0 * 6.0 * 7.0);
        constexpr auto coeff8 = 1.0 / (1.0 * 2.0 * 3.0 * 4.0 * 5.0 * 6.0 * 7.0 * 8.0);
        constexpr auto coeff9 = 1.0 / (1.0 * 2.0 * 3.0 * 4.0 * 5.0 * 6.0 * 7.0 * 8.0 * 9.0);
        auto s = x - xxx * coeff3 + xxxxx * coeff5 - xxxxxxx * coeff7 + xxxxxxxxx * coeff9;
        auto c = 1 - xx * coeff2 + xxxx * coeff4 - xxxxxx * coeff6 + xxxxxxxx * coeff8;
        tf_plink_to_hlink.quat().coeffs() << s * joint_axes_[joint_id], c;
      }
      tf_plink_to_hlink.trans() = tf_plink_to_pjoint_trans;
      tf_plink_to_hlink.is_quat_identity_ = false;
    }else{
      Eigen::Vector3d&& trans = joint_axes_[joint_id] * joint_angles[i];
      tf_plink_to_hlink.trans() = tf_plink_to_pjoint_trans + trans;
      tf_plink_to_hlink.quat().setIdentity();
      tf_plink_to_hlink.is_quat_identity_ = true;
    }
  }
  clear_cache();
}


void KinematicModel::set_init_angles() {
  std::vector<double> joint_angles(num_dof_, 0.0);
  joint_angles_ = joint_angles;
  clear_cache();
}

std::vector<double>
KinematicModel::get_joint_angles(const std::vector<size_t> &joint_ids) const {
  std::vector<double> angles(joint_ids.size());
  for (size_t i = 0; i < joint_ids.size(); i++) {
    int idx = joint_ids[i];
    angles[i] = joint_angles_[idx];
  }
  return angles;
}

std::vector<size_t>
KinematicModel::get_joint_ids(std::vector<std::string> joint_names) const {
  int n_joint = joint_names.size();
  std::vector<size_t> joint_ids(n_joint);
  for (int i = 0; i < n_joint; i++) {
    auto iter = joint_ids_.find(joint_names[i]);
    if (iter == joint_ids_.end()) {
      throw std::invalid_argument("no joint named " + joint_names[i]);
    }
    joint_ids[i] = iter->second;
  }
  return joint_ids;
}

std::vector<Bound> KinematicModel::get_joint_position_limits(
    const std::vector<size_t> &joint_ids) const {
  const size_t n_joint = joint_ids.size();
  std::vector<Bound> limits(n_joint, Bound());
  for (size_t i = 0; i < n_joint; i++) {
    const auto &joint = joints_[joint_ids[i]];
    if (joint->type == urdf::Joint::CONTINUOUS) {
      limits[i].first = -std::numeric_limits<double>::infinity();
      limits[i].second = std::numeric_limits<double>::infinity();
    } else {
      limits[i].first = joint->limits->lower;
      limits[i].second = joint->limits->upper;
    }
  }
  return limits;
}

std::vector<double> KinematicModel::get_joint_velocity_limits(
    const std::vector<size_t> &joint_ids) const {
  const size_t n_joint = joint_ids.size();
  std::vector<double> limits(n_joint);
  for (size_t i = 0; i < n_joint; i++) {
    const auto &joint = joints_[joint_ids[i]];
    limits[i] = joint->limits->velocity;
  }
  return limits;
}

std::vector<double> KinematicModel::get_joint_effort_limits(
    const std::vector<size_t> &joint_ids) const {
  const size_t n_joint = joint_ids.size();
  std::vector<double> limits(n_joint);
  for (size_t i = 0; i < n_joint; i++) {
    const auto &joint = joints_[joint_ids[i]];
    limits[i] = joint->limits->effort;
  }
  return limits;
}

std::vector<size_t>
KinematicModel::get_link_ids(std::vector<std::string> link_names) const {
  int n_link = link_names.size();
  std::vector<size_t> link_ids(n_link);
  for (int i = 0; i < n_link; i++) {
    auto iter = link_ids_.find(link_names[i]);
    if (iter == link_ids_.end()) {
      throw std::invalid_argument("no link named " + link_names[i]);
    }
    link_ids[i] = iter->second;
  }
  return link_ids;
}

urdf::LinkSharedPtr KinematicModel::add_new_link(size_t parent_id,
                                 const std::array<double, 3> &position,
                                 const std::array<double, 3> &rpy,
                                 bool consider_rotation,
                                 std::optional<std::string> link_name){
  Transform pose;
  pose.trans()  = Eigen::Vector3d(position[0], position[1], position[2]);
  pose.setQuaternionFromRPY(rpy[0], rpy[1], rpy[2]);
  return this->add_new_link(parent_id, pose, consider_rotation, link_name);
}

urdf::LinkSharedPtr KinematicModel::add_new_link(size_t parent_id, const Transform &pose,
                                 bool consider_rotation,
                                 std::optional<std::string> link_name) {

  if(link_name == std::nullopt) {
    // if link_name is not given, generate a unique name
    std::hash<double> hasher;
    std::size_t hval = 0;
    hval ^= hasher(pose.trans()(0)) + 0x9e3779b9 + (hval << 6) + (hval >> 2);
    hval ^= hasher(pose.trans()(1)) + 0x9e3779b9 + (hval << 6) + (hval >> 2);
    hval ^= hasher(pose.trans()(2)) + 0x9e3779b9 + (hval << 6) + (hval >> 2);
    hval ^= hasher(pose.quat().x()) + 0x9e3779b9 + (hval << 6) + (hval >> 2);
    hval ^= hasher(pose.quat().y()) + 0x9e3779b9 + (hval << 6) + (hval >> 2);
    hval ^= hasher(pose.quat().z()) + 0x9e3779b9 + (hval << 6) + (hval >> 2);
    hval ^= hasher(pose.quat().w()) + 0x9e3779b9 + (hval << 6) + (hval >> 2);
    link_name = "hash_" + std::to_string(hval) + "_" + std::to_string(parent_id) + "_" + std::to_string(consider_rotation);
    bool link_name_exists = (link_ids_.find(link_name.value()) != link_ids_.end());
    if (link_name_exists) {
      return links_[link_ids_[link_name.value()]];
    }
  }else{
    bool link_name_exists = (link_ids_.find(link_name.value()) != link_ids_.end());
    if (link_name_exists) {
      std::string message = "link name " + link_name.value() + " already exists";
      throw std::runtime_error("link name : " + link_name.value() + " already exists");
    }
  }

  auto fixed_joint = std::make_shared<urdf::Joint>();

  fixed_joint->parent_to_joint_origin_transform = pose;
  fixed_joint->type = urdf::Joint::FIXED;

  int link_id = links_.size();
  auto new_link = std::make_shared<urdf::Link>();
  new_link->parent_joint = fixed_joint;
  new_link->setParent(links_[parent_id]);
  new_link->name = link_name.value();
  new_link->id = link_id;
  new_link->consider_rotation = consider_rotation;

  link_ids_[link_name.value()] = link_id;
  links_.push_back(new_link);
  links_[parent_id]->child_links.push_back(new_link);
  links_[parent_id]->child_joints.push_back(fixed_joint);

  transform_cache_.extend();
  tf_plink_to_hlink_cache_.push_back(pose);
  link_parent_link_ids_.push_back(parent_id);

  this->update_rptable(); // set _rptable

  return new_link;
}

void KinematicModel::update_rptable() {
  // this function usually must come in the end of a function

  // we must recreate from scratch
  int n_link = link_ids_.size();
  int n_dof = joint_ids_.size();
  auto rptable = RelevancePredicateTable(n_link, n_dof);

  for (urdf::JointSharedPtr joint : joints_) {
    int joint_id = joint_ids_.at(joint->name);
    urdf::LinkSharedPtr clink = joint->getChildLink();
    std::stack<urdf::LinkSharedPtr> link_stack;
    link_stack.push(clink);
    while (!link_stack.empty()) {
      auto here_link = link_stack.top();
      link_stack.pop();
      rptable.table_[here_link->id][joint_id] = true;
      for (auto &link : here_link->child_links) {
        link_stack.push(link);
      }
    }
  }
  rptable_ = rptable;
}

std::string load_urdf(const std::string &urdf_path) {
  std::string xml_string;
  std::fstream xml_file(urdf_path, std::fstream::in);
  while (xml_file.good()) {
    std::string line;
    std::getline(xml_file, line);
    xml_string += (line + "\n");
  }
  xml_file.close();
  return xml_string;
}

}; // end namespace tinyfk
