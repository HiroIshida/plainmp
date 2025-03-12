/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <random>
#include <vector>
#include "plainmp/constraints/primitive.hpp"

namespace plainmp::experimental {

class MultiGoalRRT {
 public:
  MultiGoalRRT(const Eigen::VectorXd& start,
               const Eigen::VectorXd& lb,
               const Eigen::VectorXd& ub,
               const plainmp::constraint::IneqConstraintBase::Ptr& ineq,
               size_t n_max_node)
      : lb_(lb), ub_(ub), ineq_(ineq), n_max_node_(n_max_node) {
    states_ = Eigen::MatrixXd::Zero(start.size(), n_max_node);
    states_.col(0) = start;
    parents_ = std::vector<size_t>(n_max_node);
    parents_[0] = 0;
    n_node_ = 1;
    populate();
  }

  bool is_reachable(const Eigen::VectorXd& goal, double search_radius) {
    if (!ineq_->is_valid(goal)) {
      return false;
    }
    const auto [i_nearest, dist] = find_nearest(goal);
    if (dist > search_radius) {
      return false;
    }
    if (!is_valid_edge(states_.col(i_nearest), goal)) {
      return false;
    }
    return true;
  }

  std::vector<bool> is_reachable_batch(const Eigen::MatrixXd& goals,
                                       double search_radius) {
    if (!goals.rows() == lb_.size()) {
      throw std::runtime_error("Dimension mismatch");
    }
    std::vector<bool> ret(goals.cols());
    for (size_t i = 0; i < goals.cols(); ++i) {
      if (!is_reachable(goals.col(i), search_radius)) {
        ret[i] = false;
      } else {
        ret[i] = true;
      }
    }
    return ret;
  }

  Eigen::MatrixXd get_debug_states() const {
    return states_.leftCols(n_node_).transpose();  // python numpy in mind
  }

  std::vector<size_t> get_debug_parents() const {
    std::vector<size_t> ret(n_node_);
    std::copy(parents_.begin(), parents_.begin() + n_node_, ret.begin());
    return ret;
  }

 private:
  void populate() {
    const double max_dist = 2.0;
    for (size_t i = 0; i < n_max_node_ - 1; ++i) {
      Eigen::VectorXd q_rand = 0.5 * (Eigen::VectorXd::Random(lb_.size()) +
                                      Eigen::VectorXd::Ones(lb_.size()))
                                         .cwiseProduct(ub_ - lb_) +
                               lb_;
      if (!ineq_->is_valid(q_rand)) {
        continue;
      }
      const auto [i_nearest, dist] = find_nearest(q_rand);
      const Eigen::VectorXd& nearest = states_.col(i_nearest);

      if (dist > max_dist) {
        q_rand = nearest + (q_rand - nearest).normalized() * max_dist;
      }
      if (!is_valid_edge(nearest, q_rand)) {
        continue;
      }
      n_node_ += 1;
      states_.col(n_node_ - 1) = q_rand;
      parents_[n_node_ - 1] = i_nearest;
    }
  }

  std::pair<size_t, double> find_nearest(const Eigen::VectorXd& target) const {
    double min_dist = std::numeric_limits<double>::max();
    size_t i_nearest = 0;
    for (size_t i = 0; i < n_node_; ++i) {
      double dist = (states_.col(i) - target).norm();
      if (dist < min_dist) {
        min_dist = dist;
        i_nearest = i;
      }
    }
    return std::make_pair(i_nearest, min_dist);
  }

  bool is_valid_edge(const Eigen::VectorXd& from,
                     const Eigen::VectorXd& to) const {
    const double dist = (to - from).norm();
    const size_t n_sample = dist / resolution;
    const Eigen::VectorXd width = (to - from) * resolution / dist;
    Eigen::VectorXd q_tmp = from;
    for (size_t i = 1; i < n_sample; ++i) {
      q_tmp += width;
      if (!ineq_->is_valid(q_tmp)) {
        return false;
      }
    }
    return true;
  }

  Eigen::MatrixXd states_;
  std::vector<size_t> parents_;
  Eigen::VectorXd lb_;
  Eigen::VectorXd ub_;
  plainmp::constraint::IneqConstraintBase::Ptr ineq_;
  size_t n_max_node_;
  size_t n_node_;
  double resolution = 0.1;  // tmp
};

}  // namespace plainmp::experimental
