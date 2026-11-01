#include <algorithm>
#include <chrono>
#include <cstdlib>
#include "plainmp/kinematics/kinematics.hpp"

using Kin = plainmp::kinematics::KinematicModel<double>;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: native MODEL.txt [calls]\n";
    return 1;
  }
  std::ifstream input(argv[1]);
  std::string path;
  std::getline(input, path);
  size_t n, samples;
  input >> n >> samples;
  std::vector<std::string> names(n);
  for (auto& name : names)
    input >> name;
  std::vector<Kin::Vector> qs(samples, Kin::Vector(n));
  for (auto& q : qs) {
    for (size_t j = 0; j < n; ++j)
      input >> q[j];
  }
  if (!input || samples == 0)
    return 1;
  Kin kin(plainmp::kinematics::load_urdf(path));
  auto ids = kin.get_joint_ids(names);
  volatile double checksum = 0;
  auto run = [&](size_t count) {
    for (size_t i = 0; i < count; ++i) {
      kin.set_joint_angles(ids, qs[i % samples]);
      checksum += kin.get_gravity_term(ids)[0];
    }
  };
  const size_t count = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 3000000;
  run(10000);
  const auto start = std::chrono::steady_clock::now();
  run(count);
  const auto end = std::chrono::steady_clock::now();
  std::cout << "calls " << count << " ns/call "
            << std::chrono::duration<double, std::nano>(end - start).count() /
                   count
            << " checksum " << checksum << '\n';
  std::cout << "cached_poses "
            << std::count(kin.transform_cache_.cache_predicate_vector_.begin(),
                          kin.transform_cache_.cache_predicate_vector_.end(), 1)
            << " total_links " << kin.link_parent_link_ids_.size() << '\n';
}
