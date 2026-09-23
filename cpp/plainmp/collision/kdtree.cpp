/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "plainmp/collision/kdtree.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <limits>
#include <random>
#include <vector>

namespace plainmp::collision {

KDTree::KDTree(const std::vector<Eigen::Vector3d>& points) {
  lower_.setConstant(std::numeric_limits<double>::infinity());
  upper_.setConstant(-std::numeric_limits<double>::infinity());
  for (const auto& point : points) {
    lower_ = lower_.cwiseMin(point);
    upper_ = upper_.cwiseMax(point);
  }
  nodes_.reserve(points.size());
  std::vector<Eigen::Vector3d> points_copy = points;
  root_index_ = build(points_copy.begin(), points_copy.end(), 0);
}

Eigen::Vector3d KDTree::query(const Eigen::Vector3d& target) const {
  Eigen::Vector3d best_point;
  double best_sqdist = std::numeric_limits<double>::max();
  nearest(root_index_, target, best_sqdist, best_point);
  return best_point;
}

double KDTree::sqdist(const Eigen::Vector3d& target) const {
  // Start with the cloud's finite bounds. This matters for queries outside
  // thin clouds: their distance along the thin axis is already unavoidable.
  const Eigen::Vector3d lower_delta =
      (lower_ - target).cwiseMax(target - upper_).cwiseMax(0.);
  // Inside the cloud bounds, the initial lower bound is zero and carrying
  // it through recursion costs more on dense volume clouds than it saves.
  if (lower_delta.isZero(0.))
    return nearest_sqdist(root_index_, target, std::numeric_limits<double>::max());
  return nearest_sqdist(root_index_, target, std::numeric_limits<double>::max(),
                        lower_delta);
}

double KDTree::nearest_sqdist(int node_index,
                             const Eigen::Vector3d& target,
                             double best_sqdist) const {
  // Distance-only traversal: no nearest-point copies or reference updates.
  while (node_index != -1) {
    const KDNode& node = nodes_[node_index];
    const double sqdist = (node.point - target).squaredNorm();
    if (sqdist < best_sqdist)
      best_sqdist = sqdist;
    const double diff = target(node.axis) - node.point(node.axis);
    const int first = diff < 0 ? node.left : node.right;
    const int second = diff < 0 ? node.right : node.left;
    if (first != -1)
      best_sqdist = nearest_sqdist(first, target, best_sqdist);
    if (!(diff * diff < best_sqdist))
      break;
    node_index = second;
  }
  return best_sqdist;
}

double KDTree::nearest_sqdist(int node_index,
                             const Eigen::Vector3d& target,
                             double best_sqdist,
                             Eigen::Vector3d lower_delta) const {
  while (node_index != -1) {
    const KDNode& node = nodes_[node_index];
    const double sqdist = (node.point - target).squaredNorm();
    if (sqdist < best_sqdist)
      best_sqdist = sqdist;
    const double diff = target(node.axis) - node.point(node.axis);
    const int first = diff < 0 ? node.left : node.right;
    const int second = diff < 0 ? node.right : node.left;
    if (first != -1)
      best_sqdist = nearest_sqdist(first, target, best_sqdist, lower_delta);
    // The far child's region inherits all ancestor bounds and tightens this
    // axis. Using the same squaredNorm operation order as the point distance
    // preserves a conservative lower bound, without a subtraction-based
    // update of an accumulated squared distance. No approximate pruning.
    lower_delta(node.axis) = diff;
    if (!(lower_delta.squaredNorm() < best_sqdist))
      break;
    node_index = second;
  }
  return best_sqdist;
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

  nodes_.emplace_back(median_point, axis);
  int node_index = static_cast<int>(nodes_.size() - 1);

  // Recursively build left and right subtrees and store their indices
  nodes_[node_index].left = build(begin, median_it, depth + 1);
  nodes_[node_index].right = build(median_it + 1, end, depth + 1);

  return node_index;
}

void KDTree::nearest(int node_index,
                     const Eigen::Vector3d& target,
                     double& best_sqdist,
                     Eigen::Vector3d& best_point) const {
  if (node_index == -1)
    return;

  const KDNode& node = nodes_[node_index];

  double sqdist = (node.point - target).squaredNorm();
  if (sqdist < best_sqdist) {
    best_sqdist = sqdist;
    best_point = node.point;
  }

  int axis = node.axis;
  double diff = target(axis) - node.point(axis);

  int first_index = (diff < 0) ? node.left : node.right;
  int second_index = (diff < 0) ? node.right : node.left;

  // Explore the side of the split where the target lies
  nearest(first_index, target, best_sqdist, best_point);

  // If there's a possibility that the other side could contain a closer
  // point, explore it
  if (diff * diff < best_sqdist) {
    nearest(second_index, target, best_sqdist, best_point);
  }
}

}  // namespace plainmp::collision
