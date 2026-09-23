/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

// Standalone regression checks for the four-state collision kernel.
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

#include "plainmp/constraints/primitive_sphere_collision.hpp"

namespace pc = plainmp::constraint;
namespace pk = plainmp::kinematics;
namespace ps = plainmp::collision;

class OverrideBox : public ps::BoxSDF {
 public:
  using BoxSDF::BoxSDF;
  bool is_outside(const ps::Point&, double) const override { return true; }
};

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

int main() {
  try {
    std::ostringstream xml;
    xml << "<robot name='batch-check'><link name='l0'/>";
    for (int i = 1; i <= 9; ++i) {
      xml << "<link name='l" << i << "'/><joint name='j" << i << "' type='"
          << (i == 1 ? "prismatic" : "revolute") << "'><parent link='l" << i - 1
          << "'/><child link='l" << i << "'/><origin xyz='.11 .02 .09' rpy='"
          << (i % 2 ? "0 0 0" : "1.5707963267948966 .1 .2") << "'/><axis xyz='"
          << (i % 3 == 0   ? "1 0 0"
              : i % 3 == 1 ? "0 1 0"
                           : "0 0 1")
          << "'/><limit lower='-7' upper='7' effort='1' velocity='1'/></joint>";
    }
    xml << "<link name='side'/><joint name='branch' type='revolute'>"
           "<parent link='l2'/><child link='side'/>"
           "<origin xyz='.2 -.1 .4' rpy='.1 -.3 .2'/><axis xyz='.6 0 .8'/>"
           "<limit lower='-7' upper='7' effort='1' "
           "velocity='1'/></joint></robot>";
    auto kin = std::make_shared<pk::KinematicModel<double>>(xml.str());
    std::vector<std::string> names;
    for (int i = 1; i <= 7; ++i) names.push_back("j" + std::to_string(i));
    const auto ids = kin->get_joint_ids(names);
    const auto other_ids = kin->get_joint_ids({"j8", "j9", "branch"});
    std::vector<pc::SphereAttachmentSpec> specs;
    for (const auto &name : {"l0", "l9", "side", "l2", "l5", "l7"}) {
      Eigen::Matrix3Xd p(3, 5);
      p << 0., .05, .1, .15, .2, .02, -.02, 0., .01, -.01, 0., .03, .01, .02,
          .04;
      Eigen::VectorXd r(5);
      r << .03, .02, .05, 1e-7, 0.;
      specs.push_back(
          {name, p, r,
           std::string(name) == "l0" || std::string(name) == "side"});
    }
    pc::SphereCollisionCst cst(kin, names, pk::BaseType::FIXED, specs,
                               {{"l0", "l7"}, {"side", "l9"}, {"l2", "l9"}},
                               std::nullopt, true);
    const ps::Pose origin(Eigen::Vector3d(.4, .1, .6),
                          Eigen::Matrix3d::Identity());
    const ps::Pose tilted(
        Eigen::Vector3d(.3, -.1, .5),
        Eigen::AngleAxisd(.45, Eigen::Vector3d(1, 2, 3).normalized())
            .toRotationMatrix());
    std::vector<ps::SDFBase::Ptr> shapes{
        std::make_shared<ps::BoxSDF>(Eigen::Vector3d(.4, .6, .05), origin),
        std::make_shared<ps::BoxSDF>(Eigen::Vector3d(.4, .6, .05), tilted),
        std::make_shared<ps::SphereSDF>(.3, origin),
        std::make_shared<ps::CylinderSDF>(.2, .6, tilted),
        std::make_shared<ps::GroundSDF>(.1),
        std::make_shared<ps::UnionSDF>(std::vector<ps::SDFBase::Ptr>{}),
        std::make_shared<OverrideBox>(Eigen::Vector3d(10, 10, 10), origin)};
    std::mt19937 rng(19349663);
    std::uniform_real_distribution<double> angle(-18., 18.);
    size_t states_checked = 0, batches_checked = 0;
    for (const auto& shape : shapes) {
      cst.set_sdf(shape);
      if (shape == shapes.front()) {
#ifdef PLAINMP_HAS_AVX2_COLLISION
        require(cst.batch_supported() == bool(__builtin_cpu_supports("avx2")),
                "Runtime AVX2 dispatch mismatch");
#else
        require(!cst.batch_supported(), "Disabled build must use scalar checks");
#endif
        std::cout << "AVX2 batch dispatch: " << cst.batch_supported() << '\n';
      }
      if (dynamic_cast<OverrideBox*>(shape.get()))
        require(!cst.batch_supported(),
                "Subclass must use the scalar fallback");
      for (size_t trial = 0; trial < 1200; ++trial) {
        const size_t count = trial % 4 + 1;
        Eigen::VectorXd q[4];
        const double* ptr[4];
        for (size_t k = 0; k < count; ++k) {
          q[k].resize(7);
          for (int j = 0; j < 7; ++j) q[k][j] = angle(rng);
          q[k][0] *= .025;
          ptr[k] = q[k].data();
        }
        Eigen::Vector3d other(angle(rng), angle(rng), angle(rng));
        kin->set_joint_angles(other_ids, other, trial % 2 == 0);
        if (trial % 13 == 0)
          kin->set_base_pose(
              pk::QuatTrans<double>::fromXYZRPY(.1, -.2, .05, .2, -.1, .3));
        unsigned expected = 0;
        for (size_t k = 0; k < count; ++k)
          if (cst.is_valid(q[k])) expected |= 1u << k;
        const auto link_ids =
            kin->get_link_ids({"l0", "l2", "l5", "l7", "l9", "side"});
        std::vector<pk::QuatTrans<double>> poses;
        for (size_t id : link_ids)
          poses.push_back(kin->get_link_pose(id));
        require(cst.is_valid_batch(ptr, count) == expected,
                "Batch predicate mismatch");
        for (size_t j = 0; j < link_ids.size(); ++j) {
          const auto &pose = kin->get_link_pose(link_ids[j]);
          require((pose.trans() - poses[j].trans()).norm() < 1e-14 &&
                      (pose.quat().coeffs() - poses[j].quat().coeffs()).norm() <
                          1e-14,
                  "Batch restored transform mismatch");
        }
        auto actual = kin->get_joint_angles(ids);
        for (size_t j = 0; j < ids.size(); ++j)
          require(actual[j] == q[count - 1][j],
                  "Full batch final state mismatch");
        size_t first = 0;
        while (first < count && (expected & (1u << first))) ++first;
        require(cst.first_invalid_batch(ptr, count) == first,
                "Ordered prefix mismatch");
        actual = kin->get_joint_angles(ids);
        for (size_t j = 0; j < ids.size(); ++j)
          require(actual[j] == q[std::min(first, count - 1)][j],
                  "Prefix final state mismatch");
        states_checked += count;
        ++batches_checked;
      }
    }
    std::cout << batches_checked << " batches, " << states_checked
              << " states: predicates and visible kinematic state match\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
