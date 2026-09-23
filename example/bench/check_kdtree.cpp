// Exact-distance regression check for KD-tree pruning, independent of Python.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

#include "plainmp/collision/kdtree.hpp"

using Point = Eigen::Vector3d;

int main() {
  std::mt19937 generator(67867967);
  std::uniform_real_distribution<double> uniform(-1., 1.);
  auto random_point = [&]() {
    return Point(uniform(generator), uniform(generator), uniform(generator));
  };
  size_t checked = 0;
  auto check = [&](const std::vector<Point>& points,
                   const std::vector<Point>& queries) {
    plainmp::collision::KDTree tree(points);
    for (const Point& target : queries) {
      double expected = std::numeric_limits<double>::max();
      for (const Point& point : points) {
        const double distance = (point - target).squaredNorm();
        if (distance < expected) expected = distance;
      }
      if (tree.sqdist(target) != expected)
        throw std::runtime_error(
            "KD-tree squared distance differs from brute force");
      if (!points.empty() && expected < std::numeric_limits<double>::max() &&
          (tree.query(target) - target).squaredNorm() != expected)
        throw std::runtime_error(
            "KD-tree nearest point differs from brute force");
      ++checked;
    }
  };
  for (size_t count : {0, 1, 2, 3, 31, 1000, 100000}) {
    for (int geometry = 0; geometry < 5; ++geometry) {
      std::vector<Point> points;
      for (size_t i = 0; i < count; ++i) {
        Point p = random_point();
        if (geometry == 1) p.z() *= .025;  // Thin point-cloud table.
        if (geometry == 2) p.z() = 0.;     // Plane.
        if (geometry == 3) {
          p.y() = 0.;
          p.z() = 0.;
        }                                      // Line.
        if (geometry == 4) p = Point::Zero();  // Coincident points.
        points.push_back(p);
      }
      std::vector<Point> queries;
      for (size_t i = 0; i < 200; ++i) queries.push_back(3. * random_point());
      for (size_t i = 0; i < std::min(count, size_t(30)); ++i) {
        queries.push_back(points[i]);
        for (int axis = 0; axis < 3; ++axis) {
          Point p = points[i];
          p(axis) =
              std::nextafter(p(axis), std::numeric_limits<double>::infinity());
          queries.push_back(p);
          p(axis) = std::nextafter(points[i](axis),
                                   -std::numeric_limits<double>::infinity());
          queries.push_back(p);
        }
      }
      check(points, queries);
    }
  }
  // Underflow, overflow and nearly coincident split planes.
  for (double scale : {1e-160, 1e-100, 1., 1e100, 1e160}) {
    std::vector<Point> points, queries;
    for (int i = 0; i < 100; ++i) {
      points.push_back(scale * random_point());
      queries.push_back(scale * random_point());
    }
    check(points, queries);
  }
  // Compacted preorder storage must retain the old strict-less-than tie
  // rule, including a two-point subtree that has only a left child.
  for (int count : {2, 3}) {
    std::vector<Point> points{Point(-1., 0., 0.), Point(1., 0., 0.)};
    const Point expected = count == 2 ? Point(1., 0., 0.) : Point(0., 1., 0.);
    if (count == 3) points.push_back(expected);
    plainmp::collision::KDTree tree(points);
    if ((tree.query(Point::Zero()) - expected).squaredNorm() != 0. ||
        tree.sqdist(Point::Zero()) != 1.)
      throw std::runtime_error("KD-tree changed nearest-point tie selection");
    ++checked;
  }
  std::cout << checked << " exact KD-tree distances match brute force\n";
}
