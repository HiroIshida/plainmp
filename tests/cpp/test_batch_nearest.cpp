/* Copyright (C) 2026 Hirokazu Ishida
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. See https://mozilla.org/MPL/2.0/.
 */
// Run through CMake with -DPLAINMP_BUILD_TESTS=ON, then ctest --test-dir
// build/native.
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <new>
#include <random>
#include <stdexcept>
#include <vector>
#include "plainmp/ompl/batch_nearest.hpp"

static size_t allocations = 0;
void* operator new(size_t size) {
  ++allocations;
  if (void* p = std::malloc(size ? size : 1))
    return p;
  throw std::bad_alloc();
}
void* operator new[](size_t size) {
  return ::operator new(size);
}
void operator delete(void* p) noexcept {
  std::free(p);
}
void operator delete[](void* p) noexcept {
  std::free(p);
}
void operator delete(void* p, size_t) noexcept {
  std::free(p);
}
void operator delete[](void* p, size_t) noexcept {
  std::free(p);
}

void require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

void test_nearest() {
  using plainmp::ompl_wrapper::BatchNearest;
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> uniform(-1, 1);
  for (size_t dim : {1, 2, 7, 8, 14}) {
    constexpr size_t capacity = 4096;
    std::vector<double> points(capacity * dim), query(dim);
    for (double& x : points)
      x = uniform(rng);
    BatchNearest tree, linear;
    tree.allocate(points.data(), dim, capacity);
    linear.allocate(points.data(), dim, capacity, false);
    require(tree.nearest(query.data()).first == BatchNearest::none &&
                linear.nearest(query.data()).first == BatchNearest::none,
            "empty nearest");
    for (bool duplicates : {false, true}) {
      tree.clear();
      linear.clear();
      if (duplicates)
        std::fill(points.begin(), points.end(), 0.0);
      const size_t before = allocations;
      for (size_t i = 0; i < capacity; ++i) {
        tree.insert(i);
        linear.insert(i);
        if (i % 37 != 0 && i != capacity - 1)
          continue;
        for (double& x : query)
          x = uniform(rng);
        const auto actual = tree.nearest(query.data());
        require(actual == linear.nearest(query.data()),
                "KD-tree/linear mismatch");
        double best = std::numeric_limits<double>::infinity();
        size_t index = 0;
        for (size_t j = 0; j <= i; ++j) {
          double distance = 0;
          for (size_t d = 0; d < dim; ++d) {
            const double delta = points[j * dim + d] - query[d];
            distance += delta * delta;
          }
          if (distance < best) {
            best = distance;
            index = j;
          }
        }
        require(actual.first == index && actual.second == best,
                "nearest disagrees with brute force");
      }
      require(allocations == before, "allocation inside nearest insert/query");
    }
  }
}

int main() {
  test_nearest();
  std::cout << "BatchNearest native tests passed\n";
}
