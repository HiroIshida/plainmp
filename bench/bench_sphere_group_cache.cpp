// Build against the same core objects and flags as the Python extension.
// Define PLAINMP_BENCH_PREVIOUS_AABB when building against the first AABB
// implementation, which does not have initialize_radius_cache().
#include <chrono>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include "plainmp/constraints/primitive_sphere_collision.hpp"

using namespace plainmp::constraint;

// Compare the fused path to the existing position-only path and to bounds
// reduced from those positions. This also exercises nontrivial rotations.
void verify_cache(SphereGroup& group,
                  const std::shared_ptr<kin::KinematicModel<double>>& model) {
  std::mt19937 random(42);
  std::uniform_real_distribution<double> sample(-3.0, 3.0);
  for (int uniform = 0; uniform < 2; ++uniform) {
    if (uniform) {
      group.radii.setConstant(0.05);
    }
#ifndef PLAINMP_BENCH_PREVIOUS_AABB
    group.initialize_radius_cache();
#endif
    for (int trial = 0; trial < 1000; ++trial) {
      kin::KinematicModel<double>::Transform pose;
      pose.trans() =
          Eigen::Vector3d(sample(random), sample(random), sample(random));
      pose.setQuaternionFromRPY(sample(random), sample(random), sample(random));
      model->set_base_pose(pose);
      group.clear_cache();
      group.create_group_sphere_position_cache(model);
      group.create_sphere_position_cache(model);
      const Eigen::Matrix3Xd reference_positions = group.sphere_positions_cache;
      group.create_aabb_cache(model);
      const Eigen::Vector3d reference_lower = group.aabb_lb_cache;
      const Eigen::Vector3d reference_upper = group.aabb_ub_cache;
      group.clear_cache();
      group.create_group_sphere_position_cache(model);
      group.create_aabb_cache(model);
      if (!group.is_aabb_valid ||
          !(group.sphere_positions_cache.array() == reference_positions.array())
               .all() ||
          !(group.aabb_lb_cache.array() == reference_lower.array()).all() ||
          !(group.aabb_ub_cache.array() == reference_upper.array()).all()) {
        throw std::runtime_error(
            "fused cache differs from position-only reference");
      }
    }
  }
  std::cout << "Verified 2000 poses with " << group.radii.size()
            << " spheres\n";
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: bench_sphere_group_cache N "
                 "aabb|aabb_uniform|positions|verify\n";
    return 1;
  }
  const int n = std::stoi(argv[1]);
  const std::string mode = argv[2];
  if (n < 1 || n > 20000000 ||
      (mode != "aabb" && mode != "aabb_uniform" && mode != "positions" &&
       mode != "verify")) {
    return 1;
  }
  const bool aabb = mode == "aabb" || mode == "aabb_uniform";
  const int repetitions = 20000000 / n;
  auto model = std::make_shared<kin::KinematicModel<double>>(
      "<robot name=\"bench\"><link name=\"base\"/></robot>");
  SphereGroup group{};
  group.parent_link_id = model->get_link_ids({"base"})[0];
  group.radii = Eigen::VectorXd::LinSpaced(n, 0.03, 0.09);
  if (mode == "aabb_uniform") {
    group.radii.setConstant(0.05);
  }
  group.sphere_relative_positions.resize(3, n);
  group.sphere_positions_cache.resize(3, n);
  group.group_sphere_relative_position.setZero();
  for (int i = 0; i < n; ++i) {
    group.sphere_relative_positions.col(i) =
        Eigen::Vector3d(0.01 * i, 0.005 * (i % 3), 0.007 * (i % 5));
  }
#ifndef PLAINMP_BENCH_PREVIOUS_AABB
  group.initialize_radius_cache();
#endif
  group.clear_cache();
  group.create_group_sphere_position_cache(model);
  if (mode == "verify") {
    verify_cache(group, model);
    return 0;
  }
  for (int pass = 0; pass < 6; ++pass) {
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repetitions; ++i) {
      group.clear_cache();
      if (aabb) {
        group.create_aabb_cache(model);
      } else {
        group.create_sphere_position_cache(model);
      }
      asm volatile("" : : "g"(&group) : "memory");
    }
    const auto end = std::chrono::steady_clock::now();
    if (pass > 0) {
      std::cout
          << n << ' ' << mode << ' '
          << std::chrono::duration<double, std::nano>(end - start).count() /
                 repetitions
          << '\n';
    }
  }
}
