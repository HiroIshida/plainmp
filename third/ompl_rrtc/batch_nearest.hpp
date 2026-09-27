/* Copyright (C) 2026 Hirokazu Ishida
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. See https://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace plainmp::ompl_wrapper {

// Exact Euclidean nearest neighbors: incremental batch KD-tree or linear
// scan. Both modes use preallocated storage with no per-insert or per-query
// allocation. Coordinates are owned by the caller.
class BatchNearest {
 public:
  using Index = uint32_t;
  static constexpr Index none = std::numeric_limits<Index>::max();
  static constexpr size_t batch = 128;

  void allocate(const double* points,
                size_t dimension,
                size_t capacity,
                bool use_kdtree = true) {
    points_ = points;
    dimension_ = dimension;
    capacity_ = capacity;
    // After splitting, each leaf has at least batch/2 entries. Internal
    // nodes number leaves-1. Include space for the initial empty root.
    nodes_.reset(use_kdtree ? new Node[2 * (capacity / (batch / 2) + 1) + 1]
                            : nullptr);
    linear_ids_.reset(use_kdtree ? nullptr : new Index[capacity]);
    clear();
  }

  void clear() {
    size_ = 0;
    used_ = 1;
    if (nodes_)
      init_leaf(0);
  }
  size_t size() const { return size_; }
  bool uses_kdtree() const { return nodes_ != nullptr; }

  void insert(Index point) {
    if (size_ >= capacity_)
      throw std::length_error("nearest-neighbor capacity exceeded");
    if (linear_ids_) {
      linear_ids_[size_++] = point;
      return;
    }
    Index n = 0;
    for (;;) {
      Node& node = nodes_[n];
      if (node.left != none) {
        n = coord(point, node.axis) < node.split ? node.left : node.right;
      } else if (node.count < batch) {
        node.ids[node.count++] = point;
        ++size_;
        return;
      } else {
        split(n);
      }
    }
  }

  // Returns squared distance; take a square root only for the selected node.
  std::pair<Index, double> nearest(const double* query) const {
    Index best = none;
    double distance = std::numeric_limits<double>::infinity();
    if (linear_ids_)
      scan(linear_ids_.get(), size_, query, best, distance);
    else
      search(0, query, best, distance);
    return {best, distance};
  }

 private:
  struct Node {
    Index left, right;
    size_t axis;
    double split;
    size_t count;
    Index ids[batch];
  };
  double coord(Index i, size_t axis) const {
    return points_[i * dimension_ + axis];
  }
  void init_leaf(Index n) {
    nodes_[n].left = none;
    nodes_[n].count = 0;
  }
  void split(Index n) {
    Node& node = nodes_[n];
    size_t axis = 0;
    double spread = -1;
    for (size_t d = 0; d < dimension_; ++d) {
      double lo = coord(node.ids[0], d), hi = lo;
      for (size_t j = 1; j < batch; ++j) {
        const double x = coord(node.ids[j], d);
        lo = std::min(lo, x);
        hi = std::max(hi, x);
      }
      if (hi - lo > spread) {
        spread = hi - lo;
        axis = d;
      }
    }
    auto middle = node.ids + batch / 2;
    std::nth_element(node.ids, middle, node.ids + batch, [&](Index a, Index b) {
      return coord(a, axis) < coord(b, axis);
    });
    node.axis = axis;
    node.split = coord(*middle, axis);
    node.left = used_++;
    node.right = used_++;
    init_leaf(node.left);
    init_leaf(node.right);
    auto& left = nodes_[node.left];
    auto& right = nodes_[node.right];
    left.count = right.count = batch / 2;
    std::copy(node.ids, middle, left.ids);
    std::copy(middle, node.ids + batch, right.ids);
  }
  void scan(const Index* ids,
            size_t count,
            const double* query,
            Index& best,
            double& distance) const {
    for (size_t j = 0; j < count; ++j) {
      const Index id = ids[j];
      const double* point = points_ + id * dimension_;
      double d2 = 0;
      for (size_t d = 0; d < dimension_; ++d) {
        const double delta = point[d] - query[d];
        d2 += delta * delta;
      }
      if (d2 < distance || (d2 == distance && id < best)) {
        best = id;
        distance = d2;
      }
    }
  }
  void search(Index n,
              const double* query,
              Index& best,
              double& distance) const {
    const Node& node = nodes_[n];
    if (node.left == none) {
      scan(node.ids, node.count, query, best, distance);
      return;
    }
    const double delta = query[node.axis] - node.split;
    const Index near = delta < 0 ? node.left : node.right;
    const Index far = delta < 0 ? node.right : node.left;
    search(near, query, best, distance);
    if (delta * delta <= distance)
      search(far, query, best, distance);
  }

  const double* points_ = nullptr;
  size_t dimension_ = 0, capacity_ = 0, size_ = 0;
  Index used_ = 0;
  std::unique_ptr<Node[]> nodes_;
  std::unique_ptr<Index[]> linear_ids_;
};

}  // namespace plainmp::ompl_wrapper
