/* Copyright (C) 2026 Hirokazu Ishida
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. See https://mozilla.org/MPL/2.0/.
 * Algorithm adapted from KavrakiLab/vamp (Apache-2.0); see vamp_rrtc.hpp
 * and third/vamp/LICENSE for attribution and changes.
 */
#include "plainmp/ompl/vamp_rrtc.hpp"

#include <ompl/base/goals/GoalSampleableRegion.h>
#include <ompl/geometric/PathGeometric.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace plainmp::ompl_wrapper {
namespace ob = ompl::base;

VampRRTC::VampRRTC(const ob::SpaceInformationPtr& si)
    : ob::Planner(si, "vamp_rrtc") {
  specs_.recognizedGoal = ob::GOAL_SAMPLEABLE_REGION;
}

void VampRRTC::setRange(double range) {
  if (!std::isfinite(range) || range <= 0)
    throw std::invalid_argument("range must be positive and finite");
  range_ = range;
}

void VampRRTC::setSettings(const VampRRTCSettings& s) {
  if (!std::isfinite(s.radius) || !std::isfinite(s.min_radius) ||
      s.min_radius <= 0 || s.radius < s.min_radius || !std::isfinite(s.alpha) ||
      s.alpha < 0 || s.alpha >= 1 || !std::isfinite(s.tree_ratio) ||
      s.tree_ratio <= 0 || s.max_samples < 2 ||
      s.max_samples >= BatchNearest::none || s.halton_skip > 1000000000) {
    throw std::invalid_argument("invalid VampRRTCSettings");
  }
  settings_ = s;
  if (capacity_ &&
      (capacity_ != s.max_samples || start_tree_.uses_kdtree() != s.use_kdtree))
    setup_ = false;
}

void VampRRTC::setup() {
  ob::Planner::setup();
  if (dynamic_cast<ob::RealVectorStateSpace*>(si_->getStateSpace().get()) ==
      nullptr)
    throw std::invalid_argument("vamp_rrtc requires RealVectorStateSpace");
  dimension_ = si_->getStateDimension();
  if (!dimension_ ||
      settings_.max_samples >
          std::numeric_limits<size_t>::max() / dimension_ / sizeof(double))
    throw std::invalid_argument("invalid vamp_rrtc pool dimensions");
  const auto& bounds =
      si_->getStateSpace()->as<ob::RealVectorStateSpace>()->getBounds();
  for (size_t d = 0; d < dimension_; ++d) {
    if (!std::isfinite(bounds.low[d]) || !std::isfinite(bounds.high[d]) ||
        !std::isfinite(bounds.high[d] - bounds.low[d]))
      throw std::invalid_argument("vamp_rrtc requires finite bounds");
  }
  if (capacity_ != settings_.max_samples ||
      start_tree_.uses_kdtree() != settings_.use_kdtree) {
    capacity_ = settings_.max_samples;
    // Default initialization deliberately leaves unused pages untouched.
    points_.reset(new double[capacity_ * dimension_]);
    nodes_.reset(new Node[capacity_]);
    scratch_.reset(new double[3 * dimension_]);
    start_tree_.allocate(points_.get(), dimension_, capacity_,
                         settings_.use_kdtree);
    goal_tree_.allocate(points_.get(), dimension_, capacity_,
                        settings_.use_kdtree);
    bases_.resize(dimension_);
    numerators_.resize(dimension_);
    denominators_.resize(dimension_);
    uint64_t candidate = 3;
    for (size_t d = 0; d < dimension_; ++d) {
      for (;; ++candidate) {
        bool prime = true;
        for (uint64_t divisor = 2; divisor <= candidate / divisor; ++divisor)
          if (candidate % divisor == 0) {
            prime = false;
            break;
          }
        if (prime)
          break;
      }
      bases_[d] = candidate++;
    }
  }
}

void VampRRTC::clear() {
  ob::Planner::clear();
  size_ = 0;
  start_tree_.clear();
  goal_tree_.clear();
}

VampRRTC::Index VampRRTC::insert(BatchNearest& tree,
                                 const double* q,
                                 Index parent) {
  const Index i = size_++;
  std::copy(q, q + dimension_, point(i));
  nodes_[i] = {parent == BatchNearest::none ? i : parent,
               std::numeric_limits<double>::infinity()};
  tree.insert(i);
  return i;
}

bool VampRRTC::motion(Index from, const double* to) {
  State a, b;
  a.values = point(from);
  b.values = const_cast<double*>(to);
  return si_->checkMotion(&a, &b);
}

VampRRTC::Index VampRRTC::root(Index i) const {
  while (nodes_[i].parent != i)
    i = nodes_[i].parent;
  return i;
}

void VampRRTC::solution(Index a, Index b, bool a_is_start) {
  std::vector<Index> path;
  for (Index i = a;; i = nodes_[i].parent) {
    path.push_back(i);
    if (nodes_[i].parent == i)
      break;
  }
  std::reverse(path.begin(), path.end());
  // Both connection states are exactly equal, so omit one copy.
  if (nodes_[b].parent != b) {
    for (Index i = nodes_[b].parent;; i = nodes_[i].parent) {
      path.push_back(i);
      if (nodes_[i].parent == i)
        break;
    }
  }
  if (!a_is_start)
    std::reverse(path.begin(), path.end());
  auto result = std::make_shared<ompl::geometric::PathGeometric>(si_);
  result->getStates().reserve(path.size());
  State state;
  for (Index i : path) {
    state.values = point(i);
    result->append(&state);
  }
  pdef_->addSolutionPath(result, false, 0.0, getName());
}

ob::PlannerStatus VampRRTC::solve(const ob::PlannerTerminationCondition& ptc) {
  checkValidity();
  if (!settings_.use_halton && !rng_)
    rng_.emplace();
  for (size_t d = 0; d < dimension_; ++d) {
    uint64_t index = settings_.halton_skip;
    numerators_[d] = 0;
    denominators_[d] = 1;
    while (index) {
      numerators_[d] = numerators_[d] * bases_[d] + index % bases_[d];
      denominators_[d] *= bases_[d];
      index /= bases_[d];
    }
  }
  auto* goal = dynamic_cast<ob::GoalSampleableRegion*>(pdef_->getGoal().get());
  if (!goal)
    return ob::PlannerStatus::UNRECOGNIZED_GOAL_TYPE;
  // A solve starts a fresh search, reusing all allocated buffers.
  size_ = 0;
  start_tree_.clear();
  goal_tree_.clear();
  pis_.restart();
  while (size_ < capacity_ - 1 && !ptc && pis_.haveMoreStartStates()) {
    const auto* s = pis_.nextStart();
    if (s)
      insert(start_tree_, s->as<State>()->values, BatchNearest::none);
  }
  if (ptc)
    return ob::PlannerStatus::TIMEOUT;
  if (!start_tree_.size())
    return ob::PlannerStatus::INVALID_START;
  const Index start_count = size_;
  while (size_ < capacity_ && !ptc && pis_.haveMoreGoalStates()) {
    const auto* g = pis_.nextGoal();
    if (!g)
      continue;
    const Index gi =
        insert(goal_tree_, g->as<State>()->values, BatchNearest::none);
    for (Index s = 0; s < start_count && !ptc; ++s) {
      State start;
      start.values = point(s);
      if (goal->isStartGoalPairValid(&start, g) && motion(s, point(gi)) &&
          !ptc) {
        auto path = std::make_shared<ompl::geometric::PathGeometric>(si_);
        path->append(&start);
        path->append(g);
        pdef_->addSolutionPath(path, false, 0.0, getName());
        return ob::PlannerStatus::EXACT_SOLUTION;
      }
    }
  }
  if (ptc)
    return ob::PlannerStatus::TIMEOUT;
  if (!goal_tree_.size())
    return ob::PlannerStatus::INVALID_GOAL;

  BatchNearest* a = settings_.start_tree_first ? &goal_tree_ : &start_tree_;
  BatchNearest* b = settings_.start_tree_first ? &start_tree_ : &goal_tree_;
  bool a_is_start = !settings_.start_tree_first;
  double* sample = scratch_.get();
  double* next = sample + dimension_;
  double* anchor = next + dimension_;
  const auto& bounds =
      si_->getStateSpace()->as<ob::RealVectorStateSpace>()->getBounds();

  for (size_t iter = 0;
       iter < settings_.max_iterations && size_ < capacity_ && !ptc; ++iter) {
    const double asize = a->size(), bsize = b->size();
    // Preserve VAMP's asymmetric ratio rule (not simply "choose smaller").
    if (!settings_.balance ||
        std::abs(asize - bsize) / asize < settings_.tree_ratio) {
      std::swap(a, b);
      a_is_start = !a_is_start;
    }
    for (size_t d = 0; d < dimension_; ++d) {
      if (settings_.use_halton) {
        auto& n = numerators_[d];
        auto& denominator = denominators_[d];
        const auto base = bases_[d];
        const auto x = denominator - n;
        if (x == 1) {
          // Unlike VAMP's float recurrence, integer arithmetic does not
          // lose digits around 1.4M samples. Guard the eventual overflow.
          if (denominator > std::numeric_limits<uint64_t>::max() / base)
            denominator = 1;
          n = 1;
          denominator *= base;
        } else {
          auto y = denominator / base;
          while (x <= y)
            y /= base;
          n = (base + 1) * y - x;
        }
        const double u = static_cast<double>(n) / denominator;
        sample[d] = bounds.low[d] + u * (bounds.high[d] - bounds.low[d]);
      } else {
        sample[d] = rng_->uniformReal(bounds.low[d], bounds.high[d]);
      }
    }
    const auto nearest = a->nearest(sample);
    const Index near = nearest.first;
    const double distance = std::sqrt(nearest.second);
    const double radius = nodes_[near].radius;
    if (distance == 0 || (settings_.dynamic_domain && radius < distance))
      continue;
    const double fraction = std::min(1.0, range_ / distance);
    for (size_t d = 0; d < dimension_; ++d)
      next[d] = point(near)[d] + fraction * (sample[d] - point(near)[d]);
    if (!motion(near, next)) {
      if (settings_.dynamic_domain) {
        nodes_[near].radius = std::isinf(radius)
                                  ? settings_.radius
                                  : std::max(radius * (1.0 - settings_.alpha),
                                             settings_.min_radius);
      }
      continue;
    }
    if (ptc)
      break;
    Index current = insert(*a, next, near);
    if (settings_.dynamic_domain && std::isfinite(radius))
      nodes_[near].radius *= 1.0 + settings_.alpha;

    const auto other = b->nearest(point(current));
    const Index target = other.first;
    const double length = std::sqrt(other.second);
    std::copy(point(current), point(current) + dimension_, anchor);
    // Keep the active tree growing toward one fixed nearest node in the
    // opposite tree; no repeated nearest queries during CONNECT.
    const double steps = std::ceil(length / range_);
    if (!std::isfinite(steps) ||
        steps > static_cast<double>(std::numeric_limits<size_t>::max() / 2))
      return ob::PlannerStatus::TIMEOUT;
    const size_t count = static_cast<size_t>(steps);
    size_t step = 0;
    while (step < count && size_ < capacity_ && !ptc) {
      const double t = static_cast<double>(step + 1) / count;
      for (size_t d = 0; d < dimension_; ++d)
        next[d] = step + 1 == count
                      ? point(target)[d]
                      : anchor[d] + t * (point(target)[d] - anchor[d]);
      if (!motion(current, next) || ptc)
        break;
      current = insert(*a, next, current);
      ++step;
    }
    if (step == count && !ptc) {
      State first, second;
      first.values = point(root(a_is_start ? current : target));
      second.values = point(root(a_is_start ? target : current));
      if (goal->isStartGoalPairValid(&first, &second)) {
        solution(current, target, a_is_start);
        return ob::PlannerStatus::EXACT_SOLUTION;
      }
    }
  }
  return ob::PlannerStatus::TIMEOUT;
}
}  // namespace plainmp::ompl_wrapper
