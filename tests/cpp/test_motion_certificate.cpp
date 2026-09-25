/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2026 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include "plainmp/ompl/ompl_thin_wrap.hpp"

namespace {
namespace collision = plainmp::collision;
namespace constraint = plainmp::constraint;
namespace kin = plainmp::kinematics;
namespace planner = plainmp::ompl_wrapper;
namespace ob = ompl::base;
using Model = kin::KinematicModel<double>;
using SphereCst = constraint::SphereCollisionCst;
using Eigen::Vector2d;
using Eigen::Vector3d;

// No external model downloads: a revolute joint followed by a bounded slider.
constexpr char kRobot[] = R"(
<robot name="certificate_test">
  <link name="base"/>
  <link name="arm"/>
  <link name="tip"/>
  <joint name="turn" type="revolute">
    <parent link="base"/><child link="arm"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <joint name="slide" type="prismatic">
    <parent link="arm"/><child link="tip"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2" upper="2" effort="1" velocity="1"/>
  </joint>
</robot>)";

constraint::SphereAttachmentSpec sphere(const std::string& link,
                                        const Vector3d& center,
                                        double radius = 0.05) {
  Eigen::Matrix3Xd positions(3, 1);
  positions.col(0) = center;
  return {link, positions, Eigen::VectorXd::Constant(1, radius), false};
}

SphereCst::Ptr make_constraint(double radius = 0.05,
                               kin::BaseType base = kin::BaseType::FIXED) {
  return std::make_shared<SphereCst>(
      std::make_shared<Model>(kRobot),
      std::vector<std::string>{"turn", "slide"}, base,
      std::vector<constraint::SphereAttachmentSpec>{
          sphere("tip", Vector3d(1, 0, 0), radius)},
      std::vector<std::pair<std::string, std::string>>{}, std::nullopt);
}

collision::Pose pose(double x) {
  return {Vector3d(x, 0, 0), Eigen::Matrix3d::Identity()};
}

collision::SDFBase::Ptr obstacle(int kind) {
  switch (kind) {
    case 0:
      return std::make_shared<collision::SphereSDF>(0.1, pose(2));
    case 1:
      return std::make_shared<collision::BoxSDF>(Vector3d(0.2, 0.6, 0.6),
                                                 pose(2));
    case 2:
      return std::make_shared<collision::CylinderSDF>(0.1, 0.6, pose(2));
    default: {
      const Eigen::Matrix3d rotation =
          Eigen::AngleAxisd(0.4, Vector3d::UnitY()).toRotationMatrix();
      const collision::Pose tilted(Vector3d(2, 0, 0), rotation);
      if (kind == 3)
        return std::make_shared<collision::BoxSDF>(Vector3d(0.2, 0.6, 0.6),
                                                   tilted);
      return std::make_shared<collision::CylinderSDF>(0.1, 0.6, tilted);
    }
  }
}

// Probe both sides of each certificate using a separate point-checking model.
void expect_safe_interval(SphereCst& oracle,
                          const Vector2d& start,
                          const Vector2d& end,
                          double anchor,
                          double radius) {
  ASSERT_GT(radius, 0);
  ASSERT_TRUE(std::isfinite(radius));
  for (int i = -20; i <= 20; ++i) {
    const double t = std::clamp(anchor + radius * 0.999 * i / 20, 0.0, 1.0);
    SCOPED_TRACE(t);
    EXPECT_TRUE(oracle.is_valid(start + t * (end - start)));
  }
}

class PrimitiveCertificate : public testing::TestWithParam<int> {};

TEST_P(PrimitiveCertificate, ShrinksBeforeCollisionAndMatchesPointChecks) {
  auto cst = make_constraint();
  auto oracle = make_constraint();
  cst->set_sdf(obstacle(GetParam()));
  oracle->set_sdf(obstacle(GetParam()));
  const Vector2d start(0, 0), end(0, 1);
  ASSERT_TRUE(cst->prepare_motion_certificate(start, end, 1.0));
  double radius = 0;
  ASSERT_TRUE(cst->is_valid_with_motion_certificate(start, radius));
  ASSERT_GT(radius, 0.5);  // Ensure this test actually obtains a certificate.
  ASSERT_LT(radius, 0.9);  // At t=0.9 the sphere overlaps every primitive.
  expect_safe_interval(*oracle, start, end, 0, radius);
  EXPECT_FALSE(oracle->is_valid(start + 0.9 * (end - start)));

  int certificates = 0, collisions = 0;
  for (int i = 0; i <= 100; ++i) {
    const double t = i / 100.0;
    const Vector2d q = start + t * (end - start);
    SCOPED_TRACE(t);
    const bool valid = cst->is_valid_with_motion_certificate(q, radius);
    EXPECT_EQ(valid, oracle->is_valid(q));
    if (!valid) {
      ++collisions;
      EXPECT_DOUBLE_EQ(radius, 0);
    } else if (radius > 0) {
      ++certificates;
      expect_safe_interval(*oracle, start, end, t, radius);
    }
  }
  EXPECT_GT(certificates, 0);
  EXPECT_GT(collisions, 0);
}

TEST_P(PrimitiveCertificate, RotatingAndSlidingSegmentsHaveSafeCertificates) {
  auto cst = make_constraint();
  auto oracle = make_constraint();
  cst->set_sdf(obstacle(GetParam()));
  oracle->set_sdf(obstacle(GetParam()));
  std::mt19937 random(142857);
  std::uniform_real_distribution<double> angle(-1.5, 1.5), slide(-0.5, 1.2);
  int certificates = 0;
  for (int i = 0; i < 100; ++i) {
    const Vector2d start(angle(random), slide(random));
    const Vector2d end(angle(random), slide(random));
    ASSERT_TRUE(cst->prepare_motion_certificate(start, end, 0.3));
    for (double t : {0.0, 0.25, 0.5, 0.75, 1.0}) {
      SCOPED_TRACE(i);
      double radius = 0;
      const Vector2d q = start + t * (end - start);
      const bool valid = cst->is_valid_with_motion_certificate(q, radius);
      ASSERT_EQ(valid, oracle->is_valid(q));
      if (radius > 0) {
        ASSERT_TRUE(valid);
        EXPECT_LE(radius, 0.3);
        ++certificates;
        expect_safe_interval(*oracle, start, end, t, radius);
      }
    }
  }
  EXPECT_GT(certificates, 0);
}

INSTANTIATE_TEST_SUITE_P(Shapes,
                         PrimitiveCertificate,
                         testing::Values(0, 1, 2, 3, 4));

TEST(MotionCertificate, ContactAndNearbyPointsUseOrdinaryPredicate) {
  auto cst = make_constraint();
  cst->set_sdf(obstacle(0));
  ASSERT_TRUE(
      cst->prepare_motion_certificate(Vector2d(0, 0), Vector2d(0, 1), 1));
  // Analytic sphere contact at x=1.85; use both sides to avoid depending on
  // the legacy predicate's strictness exactly at contact.
  for (double delta : {-1e-6, 0.0, 1e-6}) {
    const Vector2d q(0, 0.85 + delta);
    const bool expected = cst->is_valid(q);
    if (delta != 0)
      EXPECT_EQ(expected, delta < 0);
    double radius = -1;
    EXPECT_EQ(cst->is_valid_with_motion_certificate(q, radius), expected);
    EXPECT_DOUBLE_EQ(radius, 0);
  }
}

TEST(MotionCertificate, FallbackContinuesToLaterSpheres) {
  auto model = std::make_shared<Model>(kRobot);
  Eigen::Matrix3Xd centers(3, 3);
  centers << -1, 1, 1.2, 0, 0, 0, 0, 0, 0;
  SphereCst cst(model, {"turn", "slide"}, kin::BaseType::FIXED,
                {{"tip", centers, Eigen::VectorXd::Constant(3, 0.05), false}},
                {}, std::nullopt, false);
  cst.set_sdf(std::make_shared<collision::SphereSDF>(0.1, pose(1.151)));
  const Vector2d start(0, 0), end(0, 0.5);
  ASSERT_TRUE(cst.prepare_motion_certificate(start, end, 1));
  // The first leaf is far away. The second has too little clearance for
  // certification, so fallback must still find the colliding third leaf.
  double radius = 1;
  EXPECT_FALSE(cst.is_valid(start));
  EXPECT_FALSE(cst.is_valid_with_motion_certificate(start, radius));
  EXPECT_DOUBLE_EQ(radius, 0);
}

TEST(MotionCertificate, GroundAndZeroMotion) {
  auto cst = make_constraint();
  auto oracle = make_constraint();
  auto base = Model::Transform::Identity();
  base.quat() = Eigen::AngleAxisd(-std::acos(-1.0) / 2, Vector3d::UnitY());
  for (auto c : {cst, oracle}) {
    c->kin_->set_base_pose(base);
    c->set_sdf(std::make_shared<collision::GroundSDF>(0));
  }
  const Vector2d start(0, 0), end(0, -1);
  ASSERT_TRUE(cst->prepare_motion_certificate(start, end, 1));
  double radius = 0;
  ASSERT_TRUE(cst->is_valid_with_motion_certificate(start, radius));
  EXPECT_LT(radius,
            0.95);  // Sphere radius and numerical guard reduce the range.
  expect_safe_interval(*oracle, start, end, 0, radius);
  EXPECT_FALSE(cst->is_valid_with_motion_certificate(end, radius));
  EXPECT_DOUBLE_EQ(radius, 0);
  for (const Vector2d& q : {start, end}) {
    ASSERT_TRUE(cst->prepare_motion_certificate(q, q, 0.3));
    EXPECT_EQ(cst->is_valid_with_motion_certificate(q, radius),
              oracle->is_valid(q));
    EXPECT_DOUBLE_EQ(radius, q == start ? 0.3 : 0.0);
  }
}

TEST(MotionCertificate, SelfCollisionUsesRelativeMotion) {
  auto create = []() {
    return std::make_shared<SphereCst>(
        std::make_shared<Model>(kRobot),
        std::vector<std::string>{"turn", "slide"}, kin::BaseType::FIXED,
        std::vector<constraint::SphereAttachmentSpec>{
            sphere("arm", Vector3d(1, 0, 0), 0.1),
            sphere("tip", Vector3d::Zero(), 0.1)},
        std::vector<std::pair<std::string, std::string>>{{"arm", "tip"}},
        std::nullopt);
  };
  auto cst = create(), oracle = create();
  const Vector2d start(-0.5, 0.3), end(0.5, 0.9);
  ASSERT_TRUE(cst->prepare_motion_certificate(start, end, 1));
  double radius = 0;
  ASSERT_TRUE(cst->is_valid_with_motion_certificate(start, radius));
  EXPECT_LT(radius, 5.0 / 6.0);
  expect_safe_interval(*oracle, start, end, 0, radius);
  EXPECT_FALSE(cst->is_valid_with_motion_certificate(end, radius));
  EXPECT_DOUBLE_EQ(radius, 0);
  // A common ancestor rotation cannot reduce the distance between the spheres.
  ASSERT_TRUE(
      cst->prepare_motion_certificate(Vector2d(-1, 0.5), Vector2d(1, 0.5), 1));
  ASSERT_TRUE(cst->is_valid_with_motion_certificate(Vector2d(0, 0.5), radius));
  EXPECT_DOUBLE_EQ(radius, 1);
}

TEST(MotionCertificate, ReplacingSdfInvalidatesPreparedBounds) {
  auto cst = make_constraint();
  cst->set_sdf(obstacle(0));
  const Vector2d start(0, 0), end(0, 0.5);
  ASSERT_TRUE(cst->prepare_motion_certificate(start, end, 0.3));
  double radius = 0;
  ASSERT_TRUE(cst->is_valid_with_motion_certificate(start, radius));
  ASSERT_GT(radius, 0);
  cst->set_sdf(std::make_shared<collision::SphereSDF>(0.1, pose(1)));
  // Even without re-prepare, an old certificate must not survive set_sdf().
  EXPECT_FALSE(cst->is_valid_with_motion_certificate(start, radius));
  EXPECT_DOUBLE_EQ(radius, 0);
  ASSERT_TRUE(cst->prepare_motion_certificate(start, end, 0.3));
  EXPECT_FALSE(cst->is_valid_with_motion_certificate(start, radius));
}

TEST(MotionCertificate, BaseAndUncontrolledJointsAreReflectedOnNextEdge) {
  auto cst = make_constraint();
  cst->set_sdf(obstacle(0));
  const Vector2d start(0, 0), end(0, 0.5);
  ASSERT_TRUE(cst->prepare_motion_certificate(start, end, 0.3));
  double radius = 0;
  ASSERT_TRUE(cst->is_valid_with_motion_certificate(start, radius));
  auto base = Model::Transform::Identity();
  base.trans().x() = 1;
  cst->kin_->set_base_pose(base);
  ASSERT_TRUE(cst->prepare_motion_certificate(start, end, 0.3));
  EXPECT_FALSE(cst->is_valid_with_motion_certificate(start, radius));
  EXPECT_DOUBLE_EQ(radius, 0);

  auto model = std::make_shared<Model>(kRobot);
  SphereCst partial(model, {"turn"}, kin::BaseType::FIXED,
                    {sphere("tip", Vector3d(1, 0, 0))}, {}, std::nullopt);
  partial.set_sdf(obstacle(0));
  const Eigen::VectorXd q = Eigen::VectorXd::Zero(1);
  ASSERT_TRUE(partial.prepare_motion_certificate(q, q, 0.3));
  ASSERT_TRUE(partial.is_valid_with_motion_certificate(q, radius));
  model->set_joint_angles(model->get_joint_ids({"slide"}),
                          Eigen::VectorXd::Ones(1));
  ASSERT_TRUE(partial.prepare_motion_certificate(q, q, 0.3));
  EXPECT_FALSE(partial.is_valid_with_motion_certificate(q, radius));
  EXPECT_DOUBLE_EQ(radius, 0);
}

TEST(MotionCertificate, AttachmentAndLinkAdditionCanBeReprepared) {
  auto model = std::make_shared<Model>(kRobot);
  model->add_new_link(model->get_link_ids({"tip"})[0], {1.0, 0.0, 0.0},
                      {0.0, 0.0, 0.0}, true, "tool");
  SphereCst cst(model, {"turn", "slide"}, kin::BaseType::FIXED,
                {sphere("tool", Vector3d::Zero())}, {}, std::nullopt);
  cst.set_sdf(obstacle(0));
  const Vector2d start(0, 0), end(0, 1);
  ASSERT_TRUE(cst.prepare_motion_certificate(start, end, 1));
  double before = 0, after = 0;
  ASSERT_TRUE(cst.is_valid_with_motion_certificate(start, before));
  ASSERT_GT(before, 0);
  model->add_new_link(model->root_link_id_, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
                      true, "extra");
  ASSERT_TRUE(cst.prepare_motion_certificate(start, end, 1));
  ASSERT_TRUE(cst.is_valid_with_motion_certificate(start, after));
  EXPECT_DOUBLE_EQ(after, before);
  EXPECT_FALSE(cst.is_valid_with_motion_certificate(end, after));
}

TEST(MotionCertificate, UnsupportedInputsFallBackAndClearEarlierPreparation) {
  auto cst = make_constraint();
  cst->set_sdf(obstacle(0));
  const Vector2d start(0, 0), end(0, 1);
  ASSERT_TRUE(cst->prepare_motion_certificate(start, end, 0.3));
  EXPECT_FALSE(cst->prepare_motion_certificate(start, Vector2d(0, 3), 0.3));
  double radius = 1;
  EXPECT_TRUE(cst->is_valid_with_motion_certificate(start, radius));
  EXPECT_DOUBLE_EQ(radius, 0);
  EXPECT_FALSE(cst->prepare_motion_certificate(
      start, Vector2d(std::numeric_limits<double>::quiet_NaN(), 0), 0.3));
  EXPECT_FALSE(cst->prepare_motion_certificate(start, end, 0));
  const std::vector<Vector3d> cloud{Vector3d(1, 0, 0)};
  cst->set_sdf(std::make_shared<collision::CloudSDF>(cloud, 0.1));
  EXPECT_FALSE(cst->prepare_motion_certificate(start, end, 0.3));
  EXPECT_FALSE(cst->is_valid_with_motion_certificate(start, radius));
  EXPECT_DOUBLE_EQ(radius, 0);
  auto tiny = make_constraint(1e-7);
  EXPECT_FALSE(tiny->prepare_motion_certificate(start, end, 0.3));
  for (auto type : {kin::BaseType::PLANAR, kin::BaseType::FLOATING}) {
    auto mobile = make_constraint(0.05, type);
    const Eigen::VectorXd q = Eigen::VectorXd::Zero(mobile->q_dim());
    EXPECT_FALSE(mobile->prepare_motion_certificate(q, q, 0.3));
    EXPECT_TRUE(mobile->is_valid_with_motion_certificate(q, radius));
    EXPECT_DOUBLE_EQ(radius, 0);
  }
}

struct EdgeResult {
  bool valid;
  size_t calls;
  size_t ordinary_checks;
  bool terminated;
  Eigen::VectorXd final_joints;
};

EdgeResult check_edge(bool enabled,
                      const Vector2d& start,
                      const Vector2d& end,
                      double resolution,
                      planner::ValidatorConfig::Type type) {
  auto cst = make_constraint();
  cst->set_sdf(obstacle(0));
  planner::ValidatorConfig config;
  config.type = type;
  config.resolution = resolution;
  config.box_width = {resolution, resolution};
  config.enable_interval_pruning = enabled;
  planner::CollisionAwareSpaceInformation csi({-3, -2}, {3, 2}, cst, 3, config);
  size_t ordinary = 0;
  csi.si_->setStateValidityChecker([&](const ob::State* state) {
    ++ordinary;
    return csi.is_valid(state);
  });
  csi.si_->setup();
  ob::ScopedState<> a(csi.si_), b(csi.si_);
  for (size_t j = 0; j < 2; ++j) {
    a[j] = start[j];
    b[j] = end[j];
  }
  // OMPL supplies an already-valid start; set the same initial state in each
  // run.
  EXPECT_TRUE(cst->is_valid(start));
  const bool valid = csi.si_->checkMotion(a.get(), b.get());
  return {valid, csi.is_valid_call_count_, ordinary, csi.is_terminatable(),
          cst->kin_->joint_angles_};
}

TEST(MotionValidator, RuntimeSwitchPreservesResultBudgetAndFinalState) {
  EXPECT_FALSE(planner::ValidatorConfig{}.enable_interval_pruning);
  for (auto type : {planner::ValidatorConfig::Type::BOX,
                    planner::ValidatorConfig::Type::EUCLIDEAN}) {
    for (double resolution : {0.005, 0.05, 1.0}) {
      for (const Vector2d& end :
           {Vector2d(0, 0.7), Vector2d(0, 1), Vector2d(1, 0.5)}) {
        SCOPED_TRACE(resolution);
        const auto off =
            check_edge(false, Vector2d::Zero(), end, resolution, type);
        const auto on =
            check_edge(true, Vector2d::Zero(), end, resolution, type);
        EXPECT_EQ(on.valid, off.valid);
        EXPECT_EQ(on.calls, off.calls);
        EXPECT_EQ(on.terminated, off.terminated);
        EXPECT_TRUE(on.final_joints.isApprox(off.final_joints, 1e-14));
        if (resolution == 0.05) {
          // Prove the runtime flag actually installs the certificate path.
          EXPECT_EQ(on.ordinary_checks, 0);
          EXPECT_GT(off.ordinary_checks, 0);
        } else {
          // Very short and very long edges take the ordinary path.
          EXPECT_EQ(on.ordinary_checks, off.ordinary_checks);
        }
      }
    }
  }
}

TEST(MotionValidator, CoveredSamplesAreSkippedButCountedInOriginalOrder) {
  auto space = std::make_shared<ob::RealVectorStateSpace>(1);
  ob::RealVectorBounds bounds(1);
  bounds.setLow(0);
  bounds.setHigh(1);
  space->setBounds(bounds);
  auto si = std::make_shared<ob::SpaceInformation>(space);
  auto validator = std::make_shared<planner::EuclideanMotionValidator>(si, 0.1);
  si->setMotionValidator(validator);
  std::vector<double> baseline;
  si->setStateValidityChecker([&](const ob::State* state) {
    baseline.push_back(
        state->as<ob::RealVectorStateSpace::StateType>()->values[0]);
    return true;
  });
  si->setup();
  ob::ScopedState<> start(si), end(si);
  start[0] = 0;
  end[0] = 1;
  ASSERT_TRUE(si->checkMotion(start.get(), end.get()));
  for (double requested_radius : {0.26, 1.0}) {
    size_t calls = 0, checks = 0, skips = 0, restores = 0;
    double final_rate = -1;
    planner::CustomValidatorBase::MotionCertificate certificate;
    certificate.prepare = [](const ob::State*, const ob::State*, double) {
      return true;
    };
    certificate.check = [&](const ob::State* state, double& radius) {
      const double rate =
          state->as<ob::RealVectorStateSpace::StateType>()->values[0];
      EXPECT_DOUBLE_EQ(rate, baseline.at(calls));
      ++calls;
      ++checks;
      radius = requested_radius;
      return true;
    };
    certificate.skip = [&]() {
      ++calls;
      ++skips;
    };
    certificate.restore = [&](const ob::State* state) {
      ++restores;
      final_rate = state->as<ob::RealVectorStateSpace::StateType>()->values[0];
    };
    validator->set_motion_certificate(std::move(certificate));
    ASSERT_TRUE(si->checkMotion(start.get(), end.get()));
    EXPECT_EQ(checks, requested_radius == 1.0 ? 1 : 3);
    EXPECT_GT(skips, 0);
    EXPECT_EQ(calls, baseline.size());
    EXPECT_EQ(checks + skips, calls);
    EXPECT_EQ(restores, 1);
    EXPECT_DOUBLE_EQ(final_rate, baseline.back());
  }
}
}  // namespace
