#include "tinyfk.hpp"
#include <Eigen/Geometry>
#include <cmath>
#include <fstream>
#include <functional>
#include <stdexcept>
#include "urdf_model/pose.h"

template <typename Scalar>
void compute_approx_sin_cos(Scalar x, Scalar& s, Scalar& c) {
  // Approximate sin(x) = x - x^3/3! + x^5/5! - x^7/7! + x^9/9!
  // Approximate cos(x) = 1 - x^2/2! + x^4/4! - x^6/6! + x^8/8!
  constexpr auto coeff2 = 1.0 / (1.0 * 2.0);
  constexpr auto coeff3 = 1.0 / (1.0 * 2.0 * 3.0);
  constexpr auto coeff4 = 1.0 / (1.0 * 2.0 * 3.0 * 4.0);
  constexpr auto coeff5 = 1.0 / (1.0 * 2.0 * 3.0 * 4.0 * 5.0);
  constexpr auto coeff6 = 1.0 / (1.0 * 2.0 * 3.0 * 4.0 * 5.0 * 6.0);
  constexpr auto coeff7 = 1.0 / (1.0 * 2.0 * 3.0 * 4.0 * 5.0 * 6.0 * 7.0);
  constexpr auto coeff8 = 1.0 / (1.0 * 2.0 * 3.0 * 4.0 * 5.0 * 6.0 * 7.0 * 8.0);
  constexpr auto coeff9 =
      1.0 / (1.0 * 2.0 * 3.0 * 4.0 * 5.0 * 6.0 * 7.0 * 8.0 * 9.0);
  constexpr auto half_pi = M_PI * 0.5;
  constexpr auto one_dev_2pi = 1.0 / (2 * M_PI);
  auto cos_sign = 1.0;
  if (x > half_pi || x < -half_pi) {
    if (x > M_PI || x < -M_PI) {
      x = x - 2 * M_PI * std::floor(x * one_dev_2pi + 0.5);
    }
    if (x < -half_pi) {
      x = -x - M_PI;
      cos_sign = -1.0;
    } else if (x > half_pi) {
      x = -x + M_PI;
      cos_sign = -1.0;
    } else {
    }
  }
  auto xx = x * x;
  auto xxxx = xx * xx;
  auto xxxxxx = xxxx * xx;
  auto xxxxxxxx = xxxx * xxxx;
  s = x *
      (1 - xx * coeff3 + xxxx * coeff5 - xxxxxx * coeff7 + xxxxxxxx * coeff9);
  c = cos_sign *
      (1 - xx * coeff2 + xxxx * coeff4 - xxxxxx * coeff6 + xxxxxxxx * coeff8);
}

namespace tinyfk {

template class KinematicModel<double>;
template class KinematicModel<float>;

template <typename Scalar>
void KinematicModel<Scalar>::init_link_info(
    const std::vector<urdf::LinkSharedPtr>& links) {
  // compute link struct of arrays
  size_t N_link = links.size();
  link_parent_link_ids_.resize(N_link);
  link_consider_rotation_.resize(N_link);
  link_child_link_idss_.resize(N_link);
  for (const auto& link : links) {
    if (link->getParent() != nullptr) {
      link_parent_link_ids_[link->id] = link->getParent()->id;
    } else {
      // root link does not have parent
      link_parent_link_ids_[link->id] = std::numeric_limits<size_t>::max();
    }
    link_consider_rotation_[link->id] = link->consider_rotation;
    for (const auto& child_link : link->child_links) {
      link_child_link_idss_[link->id].push_back(child_link->id);
    }
    if (link->inertial != nullptr) {
      com_link_ids_.push_back(link->id);
      link_masses_.push_back(link->inertial->mass);
      if constexpr (std::is_same<Scalar, double>::value) {
        com_local_positions_.push_back(link->inertial->origin.trans());
      } else if constexpr (std::is_same<Scalar, float>::value) {
        com_local_positions_.push_back(
            link->inertial->origin.trans().cast<float>());
      } else {
        static_assert(std::is_same<Scalar, double>::value ||
                          std::is_same<Scalar, float>::value,
                      "Scalar must be double or float");
      }
    }
  }
}

template <typename Scalar>
void KinematicModel<Scalar>::init_joint_info(
    const std::vector<urdf::LinkSharedPtr>& links) {
  // allign joint ids
  auto root_link = links[root_link_id_];
  std::stack<urdf::JointSharedPtr> joint_stack;
  for (auto& joint : root_link->child_joints) {
    joint_stack.push(joint);
  }
  size_t joint_counter = 0;
  while (!joint_stack.empty()) {
    // assign joint ids in the order of DFS
    auto joint = joint_stack.top();
    joint_stack.pop();

    // set child link of joint
    size_t clink_id = link_name_id_map_[joint->child_link_name];
    joint->setChildLink(links[clink_id]);

    auto jtype = joint->type;
    if (jtype == urdf::Joint::REVOLUTE || jtype == urdf::Joint::CONTINUOUS ||
        jtype == urdf::Joint::PRISMATIC) {
      joint->id = joint_counter;
      joint_types_.push_back(jtype);

      if constexpr (std::is_same<Scalar, double>::value) {
        joint_axes_.push_back(joint->axis);
        joint_positions_.push_back(
            joint->parent_to_joint_origin_transform.trans());
        joint_orientations_.push_back(
            joint->parent_to_joint_origin_transform.quat());
      } else if constexpr (std::is_same<Scalar, float>::value) {
        joint_axes_.push_back(joint->axis.cast<float>());
        joint_positions_.push_back(
            joint->parent_to_joint_origin_transform.trans().cast<float>());
        joint_orientations_.push_back(
            joint->parent_to_joint_origin_transform.quat().cast<float>());
      } else {
        static_assert(std::is_same<Scalar, double>::value ||
                          std::is_same<Scalar, float>::value,
                      "Scalar must be double or float");
      }

      auto& jo_quat = joint_orientations_.back();
      auto w = jo_quat.w();
      bool is_approx_identity = (std::abs(w - 1.0) < 1e-6);
      joint_orientation_identity_flags_.push_back(is_approx_identity);

      joint_child_link_ids_.push_back(joint->getChildLink()->id);
      joint_name_id_map_[joint->name] = joint_counter;

      Bound limit;
      if (jtype == urdf::Joint::CONTINUOUS) {
        limit.first = -std::numeric_limits<Scalar>::infinity();
        limit.second = std::numeric_limits<Scalar>::infinity();
      } else {
        limit.first = joint->limits->lower;
        limit.second = joint->limits->upper;
      }
      joint_position_limits_.push_back(limit);

      joint_counter++;
    }
    if (joint->getChildLink() != nullptr) {
      for (auto& child_joint : joint->getChildLink()->child_joints) {
        joint_stack.push(child_joint);
      }
    }
  }
  num_dof_ = joint_name_id_map_.size();
  joint_angles_.resize(num_dof_, 0.0);
}

template <typename Scalar>
void KinematicModel<Scalar>::init_transform_cache(
    const std::vector<urdf::LinkSharedPtr>& links) {
  size_t N_link = links.size();
  transform_cache_ = SizedCache<Transform>(N_link);
  tf_plink_to_hlink_cache_ = std::vector<Transform>(N_link);
  for (size_t hid = 0; hid < N_link; hid++) {
    auto pjoint = links[hid]->parent_joint;
    if (pjoint != nullptr) {
      // HACK: if joint is not fixed, the value (origin_transform,
      // quat_identity) will be overwritten in the set_joint_angles(q)
      if constexpr (std::is_same<Scalar, double>::value) {
        tf_plink_to_hlink_cache_[hid] =
            pjoint->parent_to_joint_origin_transform;
      } else if constexpr (std::is_same<Scalar, float>::value) {
        tf_plink_to_hlink_cache_[hid] =
            pjoint->parent_to_joint_origin_transform.cast<float>();
      } else {
        static_assert(std::is_same<Scalar, double>::value ||
                          std::is_same<Scalar, float>::value,
                      "Scalar must be double or float");
      }
      // HACK: assume that joint origin rotation is identity
      // actually this is checked in the urdf loading time, so this must be ok
      tf_plink_to_hlink_cache_[hid].is_quat_identity_ = true;
    }
  }
}

template <typename Scalar>
KinematicModel<Scalar>::KinematicModel(const std::string& xml_string) {
  if (xml_string.empty()) {
    throw std::runtime_error("xml string is empty");
  }
  urdf::ModelInterfaceSharedPtr robot_urdf_interface =
      urdf::parseURDF(xml_string);

  // numbering link id
  std::vector<urdf::LinkSharedPtr> links;
  int link_counter = 0;
  std::stack<urdf::LinkSharedPtr> link_stack;
  link_stack.push(robot_urdf_interface->root_link_);
  while (!link_stack.empty()) {
    auto link = link_stack.top();
    link_stack.pop();
    link_name_id_map_[link->name] = link_counter;
    link->id = link_counter;
    links.push_back(link);
    link_counter++;
    for (auto& child_link : link->child_links) {
      link_stack.push(child_link);
    }
  }
  size_t N_link = link_counter;
  root_link_id_ = link_name_id_map_[robot_urdf_interface->root_link_->name];

  // compute total mass
  total_mass_ = 0.0;
  for (const auto& link : links) {
    if (link->inertial != nullptr) {
      total_mass_ += link->inertial->mass;
    }
  }

  this->init_link_info(links);
  this->init_joint_info(links);
  this->init_transform_cache(links);
  this->set_base_pose(Transform::Identity());
  this->update_rptable();
}

template <typename Scalar>
void KinematicModel<Scalar>::set_joint_angles(
    const std::vector<size_t>& joint_ids,
    const std::vector<Scalar>& joint_angles,
    bool high_accuracy) {
  Quat tf_pjoint_to_hlink_quat;  // pre-allocate

  for (size_t i = 0; i < joint_ids.size(); i++) {
    auto joint_id = joint_ids[i];
    joint_angles_[joint_id] = joint_angles[i];
    auto& tf_plink_to_hlink =
        tf_plink_to_hlink_cache_[joint_child_link_ids_[joint_id]];
    auto& tf_plink_to_pjoint_trans = joint_positions_[joint_id];
    if (joint_types_[joint_id] != urdf::Joint::PRISMATIC) {
      auto x = joint_angles[i] * 0.5;

      // Here we will compute the multiplication of two transformation
      // tf_plink_to_hlink = tf_plink_to_pjoint * tf_pjoint_to_hlink
      // without instantiating the transformation object because
      // 1) dont want to instantiate the object
      // 2) the tf_pjoint_to_hlink does not have translation
      if (high_accuracy) {
        tf_pjoint_to_hlink_quat.coeffs() << sin(x) * joint_axes_[joint_id],
            cos(x);
      } else {
        Scalar s, c;
        compute_approx_sin_cos<Scalar>(x, s, c);
        tf_pjoint_to_hlink_quat.coeffs() << s * joint_axes_[joint_id], c;
      }
      const auto& tf_plink_to_pjoint_quat = joint_orientations_[joint_id];

      if (joint_orientation_identity_flags_[joint_id]) {
        tf_plink_to_hlink.quat() = tf_pjoint_to_hlink_quat;
      } else {
        tf_plink_to_hlink.quat() =
            tf_plink_to_pjoint_quat * tf_pjoint_to_hlink_quat;
      }

      tf_plink_to_hlink.trans() = tf_plink_to_pjoint_trans;
      tf_plink_to_hlink.is_quat_identity_ = false;
    } else {
      Vector3&& trans = joint_axes_[joint_id] * joint_angles[i];
      tf_plink_to_hlink.trans() = tf_plink_to_pjoint_trans + trans;
      tf_plink_to_hlink.quat().setIdentity();
      tf_plink_to_hlink.is_quat_identity_ = true;
    }
  }
  clear_cache();
}

template <typename Scalar>
void KinematicModel<Scalar>::set_init_angles() {
  std::vector<Scalar> joint_angles(num_dof_, 0.0);
  joint_angles_ = joint_angles;
  clear_cache();
}

template <typename Scalar>
std::vector<Scalar> KinematicModel<Scalar>::get_joint_angles(
    const std::vector<size_t>& joint_ids) const {
  std::vector<Scalar> angles(joint_ids.size());
  for (size_t i = 0; i < joint_ids.size(); i++) {
    int idx = joint_ids[i];
    angles[i] = joint_angles_[idx];
  }
  return angles;
}

template <typename Scalar>
std::vector<size_t> KinematicModel<Scalar>::get_joint_ids(
    std::vector<std::string> joint_names) const {
  int n_joint = joint_names.size();
  std::vector<size_t> joint_ids(n_joint);
  for (int i = 0; i < n_joint; i++) {
    auto iter = joint_name_id_map_.find(joint_names[i]);
    if (iter == joint_name_id_map_.end()) {
      throw std::invalid_argument("no joint named " + joint_names[i]);
    }
    joint_ids[i] = iter->second;
  }
  return joint_ids;
}

template <typename Scalar>
std::vector<typename KinematicModel<Scalar>::Bound>
KinematicModel<Scalar>::get_joint_position_limits(
    const std::vector<size_t>& joint_ids) const {
  const size_t n_joint = joint_ids.size();
  std::vector<Bound> limits(n_joint);
  for (size_t i = 0; i < n_joint; i++) {
    limits[i] = joint_position_limits_[joint_ids[i]];
  }
  return limits;
}

template <typename Scalar>
std::vector<size_t> KinematicModel<Scalar>::get_link_ids(
    std::vector<std::string> link_names) const {
  int n_link = link_names.size();
  std::vector<size_t> link_ids(n_link);
  for (int i = 0; i < n_link; i++) {
    auto iter = link_name_id_map_.find(link_names[i]);
    if (iter == link_name_id_map_.end()) {
      throw std::invalid_argument("no link named " + link_names[i]);
    }
    link_ids[i] = iter->second;
  }
  return link_ids;
}

template <typename Scalar>
size_t KinematicModel<Scalar>::add_new_link(
    size_t parent_id,
    const std::array<Scalar, 3>& position,
    const std::array<Scalar, 3>& rpy,
    bool consider_rotation,
    std::optional<std::string> link_name) {
  Transform pose;
  pose.trans() = Vector3(position[0], position[1], position[2]);
  pose.setQuaternionFromRPY(rpy[0], rpy[1], rpy[2]);
  return this->add_new_link(parent_id, pose, consider_rotation, link_name);
}

template <typename Scalar>
size_t KinematicModel<Scalar>::add_new_link(
    size_t parent_id,
    const Transform& pose,
    bool consider_rotation,
    std::optional<std::string> link_name) {
  if (link_name == std::nullopt) {
    // if link_name is not given, generate a unique name
    std::hash<Scalar> hasher;
    std::size_t hval = 0;
    hval ^= hasher(pose.trans()(0)) + 0x9e3779b9 + (hval << 6) + (hval >> 2);
    hval ^= hasher(pose.trans()(1)) + 0x9e3779b9 + (hval << 6) + (hval >> 2);
    hval ^= hasher(pose.trans()(2)) + 0x9e3779b9 + (hval << 6) + (hval >> 2);
    hval ^= hasher(pose.quat().x()) + 0x9e3779b9 + (hval << 6) + (hval >> 2);
    hval ^= hasher(pose.quat().y()) + 0x9e3779b9 + (hval << 6) + (hval >> 2);
    hval ^= hasher(pose.quat().z()) + 0x9e3779b9 + (hval << 6) + (hval >> 2);
    hval ^= hasher(pose.quat().w()) + 0x9e3779b9 + (hval << 6) + (hval >> 2);
    link_name = "hash_" + std::to_string(hval) + "_" +
                std::to_string(parent_id) + "_" +
                std::to_string(consider_rotation);
    bool link_name_exists =
        (link_name_id_map_.find(link_name.value()) != link_name_id_map_.end());
    if (link_name_exists) {
      return link_name_id_map_[link_name.value()];
    }
  } else {
    bool link_name_exists =
        (link_name_id_map_.find(link_name.value()) != link_name_id_map_.end());
    if (link_name_exists) {
      std::string message =
          "link name " + link_name.value() + " already exists";
      throw std::runtime_error("link name : " + link_name.value() +
                               " already exists");
    }
  }

  int link_id = link_name_id_map_.size();
  link_name_id_map_[link_name.value()] = link_id;

  transform_cache_.extend();
  tf_plink_to_hlink_cache_.push_back(pose);
  link_parent_link_ids_.push_back(parent_id);
  link_consider_rotation_.push_back(consider_rotation);
  link_child_link_idss_[parent_id].push_back(link_id);
  link_child_link_idss_.push_back(std::vector<size_t>());
  this->update_rptable();  // set _rptable
  return link_id;
}

template <typename Scalar>
void KinematicModel<Scalar>::update_rptable() {
  // this function usually must come in the end of a function

  // we must recreate from scratch
  int n_link = link_name_id_map_.size();
  int n_dof = joint_name_id_map_.size();
  auto rptable = RelevancePredicateTable(n_link, n_dof);

  for (size_t joint_id = 0; joint_id < n_dof; joint_id++) {
    auto clink_id = joint_child_link_ids_[joint_id];
    std::stack<size_t> link_stack;
    link_stack.push(clink_id);
    while (!link_stack.empty()) {
      auto hlink_id = link_stack.top();
      link_stack.pop();
      rptable.table_[hlink_id][joint_id] = true;
      for (auto clink_id : link_child_link_idss_[hlink_id]) {
        link_stack.push(clink_id);
      }
    }
  }
  rptable_ = rptable;
}

std::string load_urdf(const std::string& urdf_path) {
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

};  // end namespace tinyfk
