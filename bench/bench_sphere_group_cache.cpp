// Build against the same core objects and flags as the Python extension.
// Define PLAINMP_BENCH_PREVIOUS_AABB when building against the first AABB
// implementation, which does not have initialize_radius_cache().
#include <chrono>
#include <iostream>
#include <string>
#include "plainmp/constraints/primitive_sphere_collision.hpp"

using namespace plainmp::constraint;

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: bench_sphere_group_cache N aabb|positions\n";
    return 1;
  }
  const int n = std::stoi(argv[1]);
  const std::string mode = argv[2];
  if (n < 1 || n > 20000000 || (mode != "aabb" && mode != "positions")) {
    return 1;
  }
  const bool aabb = mode == "aabb";
  const int repetitions = 20000000 / n;
  auto model = std::make_shared<kin::KinematicModel<double>>(
      "<robot name=\"bench\"><link name=\"base\"/></robot>");
  SphereGroup group{};
  group.parent_link_id = model->get_link_ids({"base"})[0];
  group.radii = Eigen::VectorXd::LinSpaced(n, 0.03, 0.09);
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
