#include "tinyfk.hpp"
#include "urdf_model/pose.h"
#include <Eigen/Geometry>
#include <functional>
#include <cmath>
#include <fstream>
#include <stdexcept>

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
  for (const auto &map_pair : robot_urdf_interface->links_) {
    std::string name = map_pair.first;
    urdf::LinkSharedPtr link = map_pair.second;
    link_ids[name] = lid;
    link->id = lid;
    links.push_back(link);
    lid++;
  }
  size_t N_link = lid; // starting from 0 and finally ++ increment, so it'S ok

  // compute total mass
  double total_mass = 0.0;
  for (const auto &link : links) {
    if (link->inertial != nullptr) {
      total_mass += link->inertial->mass;
    }
  }

  // construct joints and joint_ids, and numbering joint id
  std::vector<urdf::JointSharedPtr> joints;
  std::unordered_map<std::string, int> joint_ids;
  int jid = 0;
  for (auto &map_pair : robot_urdf_interface->joints_) {
    std::string jname = map_pair.first;
    urdf::JointSharedPtr joint = map_pair.second;
    size_t jtype = joint->type;

    if (jtype == urdf::Joint::REVOLUTE || jtype == urdf::Joint::CONTINUOUS ||
        jtype == urdf::Joint::PRISMATIC) {
      joints.push_back(joint);
      joint_ids[jname] = jid;
      joint->id = jid;
      jid++;
    } else if (jtype == urdf::Joint::FIXED) {
    } else {
      throw std::invalid_argument("unsuported joint type is detected");
    }
  }

  // set joint->_child_link.
  for (urdf::JointSharedPtr joint : joints) {
    std::string clink_name = joint->child_link_name;
    int clink_id = link_ids[clink_name];
    urdf::LinkSharedPtr clink = links[clink_id];
    joint->setChildLink(clink);
  }

  int num_dof = joint_ids.size();
  std::vector<double> joint_angles(num_dof, 0.0);

  link_id_stack_ = SizedStack<size_t>(N_link);
  transform_stack2_ = SizedStack<std::pair<urdf::LinkSharedPtr, ExpTransform>>(
      N_link); // for batch update
  transform_cache_ = SizedCache<ExpTransform>(N_link);
  tf_plink_to_hlink_cache_ = std::vector<ExpTransform>(N_link);
  for(size_t hid = 0; hid < N_link; hid++) {
    auto pjoint = links[hid]->parent_joint;
    if(pjoint != nullptr) {
      tf_plink_to_hlink_cache_[hid] = pjoint->parent_to_joint_origin_transform;
    }
  }

  root_link_id_ = link_ids[robot_urdf_interface->root_link_->name];
  links_ = links;
  link_ids_ = link_ids;
  joints_ = joints;
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
      ExpTransform new_link_pose;
      new_link_pose.t = link->inertial->origin.t;
      const auto new_link = this->add_new_link(link->id, new_link_pose, false);
      // set new link's inertial as the same as the parent
      // except its origin is zero
      new_link->inertial = link->inertial;
      new_link->inertial->origin = ExpTransform::Identity();
      com_dummy_links.push_back(new_link);
    }
    this->com_dummy_links_ = com_dummy_links;
  }

  this->set_base_pose(Transform()); // initial base pose
}

void KinematicModel::set_joint_angles(const std::vector<size_t> &joint_ids,
                                      const std::vector<double> &joint_angles) {
  for (size_t i = 0; i < joint_ids.size(); i++) {
    auto joint_id = joint_ids[i];
    joint_angles_[joint_id] = joint_angles[i];
    auto joint = joints_[joint_id];
    auto& tf_plink_to_pjoint = joint->parent_to_joint_origin_transform;
    std::cout << "fix this!" << std::endl;
    std::cout << "fix this!" << std::endl;
    std::cout << "fix this!" << std::endl;
    auto&& tf_pjoint_to_hlink = joint->transform(joint_angles[i]).to_quattrans();
    std::cout << "fix this!" << std::endl;
    std::cout << "fix this!" << std::endl;
    std::cout << "fix this!" << std::endl;
    auto&& tf_plink_to_hlink = tf_plink_to_pjoint * tf_pjoint_to_hlink;
    tf_plink_to_hlink_cache_[joint->getChildLink()->id] = tf_plink_to_hlink;
  }
  transform_cache_.clear();
}

void KinematicModel::clear_cache() { transform_cache_.clear(); }

void KinematicModel::set_init_angles() {
  std::vector<double> joint_angles(num_dof_, 0.0);
  joint_angles_ = joint_angles;
  transform_cache_.clear();
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
  ExpTransform pose;
  pose.t  = Eigen::Vector3d(position[0], position[1], position[2]);
  // pose.rotation.setFromRPY(rpy[0], rpy[1], rpy[2]);
  pose.q = Eigen::AngleAxisd(rpy[0], Eigen::Vector3d::UnitX()) *
                           Eigen::AngleAxisd(rpy[1], Eigen::Vector3d::UnitY()) *
                           Eigen::AngleAxisd(rpy[2], Eigen::Vector3d::UnitZ());
  return this->add_new_link(parent_id, pose, consider_rotation, link_name);
}

urdf::LinkSharedPtr KinematicModel::add_new_link(size_t parent_id, const ExpTransform &pose,
                                 bool consider_rotation,
                                 std::optional<std::string> link_name) {

  if(link_name == std::nullopt) {
    // if link_name is not given, generate a unique name
    std::hash<double> hasher;
    std::size_t hval = 0;
    hval ^= hasher(pose.t(0)) + 0x9e3779b9 + (hval << 6) + (hval >> 2);
    hval ^= hasher(pose.t(1)) + 0x9e3779b9 + (hval << 6) + (hval >> 2);
    hval ^= hasher(pose.t(2)) + 0x9e3779b9 + (hval << 6) + (hval >> 2);
    hval ^= hasher(pose.q.x()) + 0x9e3779b9 + (hval << 6) + (hval >> 2);
    hval ^= hasher(pose.q.y()) + 0x9e3779b9 + (hval << 6) + (hval >> 2);
    hval ^= hasher(pose.q.z()) + 0x9e3779b9 + (hval << 6) + (hval >> 2);
    hval ^= hasher(pose.q.w()) + 0x9e3779b9 + (hval << 6) + (hval >> 2);
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
  link_id_stack_.extend();
  transform_stack2_.extend();
  tf_plink_to_hlink_cache_.push_back(pose);

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
