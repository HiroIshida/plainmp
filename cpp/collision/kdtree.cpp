#include "kdtree.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <limits>
#include <random>
#include <vector>

KDTree::KDTree(const std::vector<Eigen::Vector3d>& points) {
  nodes.reserve(points.size());
  std::vector<Eigen::Vector3d> points_copy = points;
  root_index = build(points_copy.begin(), points_copy.end(), 0);
}

Eigen::Vector3d KDTree::query(const Eigen::Vector3d& target) const {
  Eigen::Vector3d best_point;
  double best_dist = std::numeric_limits<double>::max();
  nearest(root_index, target, best_dist, best_point);
  return best_point;
}

int KDTree::build(std::vector<Eigen::Vector3d>::iterator begin,
                  std::vector<Eigen::Vector3d>::iterator end,
                  int depth) {
  if (begin >= end)
    return -1;

  int axis = depth % 3;
  auto comparator = [axis](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return a(axis) < b(axis);
  };

  auto n = std::distance(begin, end);
  auto median_it = begin + n / 2;

  std::nth_element(begin, median_it, end, comparator);
  const Eigen::Vector3d& median_point = *median_it;

  nodes.emplace_back(median_point, axis);
  int node_index = static_cast<int>(nodes.size() - 1);

  // Recursively build left and right subtrees and store their indices
  nodes[node_index].left = build(begin, median_it, depth + 1);
  nodes[node_index].right = build(median_it + 1, end, depth + 1);

  return node_index;
}

void KDTree::nearest(int node_index,
                     const Eigen::Vector3d& target,
                     double& best_dist,
                     Eigen::Vector3d& best_point) const {
  if (node_index == -1)
    return;

  const KDNode& node = nodes[node_index];

  double dist = (node.point - target).squaredNorm();
  if (dist < best_dist) {
    best_dist = dist;
    best_point = node.point;
  }

  int axis = node.axis;
  double diff = target(axis) - node.point(axis);

  int first_index = (diff < 0) ? node.left : node.right;
  int second_index = (diff < 0) ? node.right : node.left;

  // Explore the side of the split where the target lies
  nearest(first_index, target, best_dist, best_point);

  // If there's a possibility that the other side could contain a closer
  // point, explore it
  if (diff * diff < best_dist) {
    nearest(second_index, target, best_dist, best_point);
  }
}
