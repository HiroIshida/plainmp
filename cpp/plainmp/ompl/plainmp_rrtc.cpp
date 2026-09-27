/* Copyright (C) 2026 Hirokazu Ishida
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. See https://mozilla.org/MPL/2.0/.
 * RRTConnect search follows OMPL 1.6; see third/ompl_rrtc/LICENSE.
 */
#include "plainmp/ompl/plainmp_rrtc.hpp"

#include <ompl/geometric/PathGeometric.h>
#include <typeinfo>

namespace plainmp::ompl_wrapper {
namespace ob = ompl::base;

PlainmpRRTC::PlainmpRRTC(const ob::SpaceInformationPtr& si,
                         const PlainmpRRTCSettings& settings)
    : ob::Planner(si, "plainmp_rrtc"),
      dimension_(si->getStateDimension()),
      capacity_(settings.max_samples) {
  // KD distances and the contiguous views require the standard Euclidean
  // metric/layout. Reject subclasses that could override their semantics.
  if (typeid(*si->getStateSpace()) != typeid(ob::RealVectorStateSpace))
    throw std::invalid_argument("plainmp_rrtc requires RealVectorStateSpace");
  if (!dimension_ || capacity_ < 2 || capacity_ >= BatchNearest::none ||
      capacity_ >
          std::numeric_limits<size_t>::max() / dimension_ / sizeof(double))
    throw std::invalid_argument("invalid PlainmpRRTCSettings pool dimensions");
  const auto& bounds =
      si->getStateSpace()->as<ob::RealVectorStateSpace>()->getBounds();
  for (size_t d = 0; d < dimension_; ++d)
    if (!std::isfinite(bounds.low[d]) || !std::isfinite(bounds.high[d]) ||
        bounds.low[d] >= bounds.high[d] ||
        !std::isfinite(bounds.high[d] - bounds.low[d]))
      throw std::invalid_argument(
          "plainmp_rrtc requires finite ordered bounds");
  points_.reset(new double[capacity_ * dimension_]);
  parents_.reset(new Index[capacity_]);
  roots_.reset(new Index[capacity_]);
  path_.reset(new Index[capacity_]);
  scratch_.reset(new double[2 * dimension_]);
  start_tree_.allocate(points_.get(), dimension_, capacity_);
  goal_tree_.allocate(points_.get(), dimension_, capacity_);
  sampler_ = si->allocStateSampler();
  specs_.recognizedGoal = ob::GOAL_SAMPLEABLE_REGION;
  specs_.directed = true;
  declareParam<double>("range", this, &PlainmpRRTC::setRange,
                       &PlainmpRRTC::getRange, "0.:1.:10000.");
}

void PlainmpRRTC::setRange(double range) {
  if (!std::isfinite(range) || range <= 0)
    throw std::invalid_argument("range must be positive and finite");
  range_ = range;
}

void PlainmpRRTC::reset() {
  size_ = 0;
  path_size_ = 0;
  start_first_ = true;
  start_tree_.clear();
  goal_tree_.clear();
}

void PlainmpRRTC::clear() {
  ob::Planner::clear();
  reset();
}

PlainmpRRTC::Index PlainmpRRTC::insert(BatchNearest& tree,
                                       const double* q,
                                       Index parent) {
  const Index id = size_++;
  std::copy_n(q, dimension_, point(id));
  parents_[id] = parent;
  roots_[id] = parent == BatchNearest::none ? id : roots_[parent];
  tree.insert(id);
  return id;
}

PlainmpRRTC::Grow PlainmpRRTC::grow(BatchNearest& tree,
                                    bool from_start,
                                    const double* target,
                                    Index& added) {
  if (size_ == capacity_)
    return Grow::TRAPPED;
  const auto nearest = tree.nearest(target);
  const Index parent = nearest.first;
  State from, to;
  from.values = point(parent);
  to.values = const_cast<double*>(target);
  // Use OMPL arithmetic for the chosen edge length/interpolation. Small
  // rounding differences can otherwise change discrete validation grids.
  const double distance = si_->distance(&from, &to);
  const bool reached = distance <= range_;
  if (!reached) {
    State interpolated;
    interpolated.values = scratch_.get() + dimension_;
    si_->getStateSpace()->interpolate(&from, &to, range_ / distance,
                                      &interpolated);
    if (si_->equalStates(&from, &interpolated))
      return Grow::TRAPPED;
    to.values = interpolated.values;
  }
  // Match RRTConnect's directed goal-tree validation, including the explicit
  // destination validity test before checking the motion toward its parent.
  if (!(from_start ? si_->checkMotion(&from, &to)
                   : si_->isValid(&to) && si_->checkMotion(&to, &from)))
    return Grow::TRAPPED;
  added = insert(tree, to.values, parent);
  return reached ? Grow::REACHED : Grow::ADVANCED;
}

void PlainmpRRTC::tracePath(Index start, Index goal) {
  // Both connection nodes have identical coordinates. Use OMPL's rule for
  // removing the duplicate, retaining the exact input endpoints.
  if (parents_[start] != BatchNearest::none)
    start = parents_[start];
  else
    goal = parents_[goal];
  path_size_ = 0;
  for (Index i = start; i != BatchNearest::none; i = parents_[i])
    path_[path_size_++] = i;
  std::reverse(path_.get(), path_.get() + path_size_);
  for (Index i = goal; i != BatchNearest::none; i = parents_[i])
    path_[path_size_++] = i;
}

ob::PlannerStatus PlainmpRRTC::search(
    const ob::PlannerTerminationCondition& ptc,
    ob::GoalSampleableRegion* goal) {
  State sample;
  sample.values = scratch_.get();
  while (size_ < capacity_ && !ptc) {
    const bool from_start = start_first_;
    start_first_ = !start_first_;
    auto& active = from_start ? start_tree_ : goal_tree_;
    auto& other = from_start ? goal_tree_ : start_tree_;
    if (goal && (goal_tree_.size() == 0 ||
                 pis_.getSampledGoalsCount() < goal_tree_.size() / 2)) {
      const auto* state =
          goal_tree_.size() == 0 ? pis_.nextGoal(ptc) : pis_.nextGoal();
      if (state)
        insert(goal_tree_, state->as<State>()->values, BatchNearest::none);
      if (goal_tree_.size() == 0)
        return ob::PlannerStatus::INVALID_GOAL;
    }
    sampler_->sampleUniform(&sample);
    Index a = BatchNearest::none, b = BatchNearest::none;
    if (grow(active, from_start, sample.values, a) == Grow::TRAPPED)
      continue;
    // CONNECT grows the OTHER tree toward A's new node, repeating nearest
    // neighbor queries and fixed-range steps, just as OMPL RRTConnect does.
    Grow state;
    do {
      if (ptc)
        return ob::PlannerStatus::TIMEOUT;
      state = grow(other, !from_start, point(a), b);
    } while (state == Grow::ADVANCED);
    if (state == Grow::REACHED) {
      const Index start = from_start ? a : b, end = from_start ? b : a;
      State first, last;
      first.values = point(roots_[start]);
      last.values = point(roots_[end]);
      if (!goal || goal->isStartGoalPairValid(&first, &last)) {
        tracePath(start, end);
        return ob::PlannerStatus::EXACT_SOLUTION;
      }
    }
  }
  return ob::PlannerStatus::TIMEOUT;
}

ob::PlannerStatus PlainmpRRTC::solveFixed(
    const double* start,
    const double* goal,
    const ob::PlannerTerminationCondition& ptc) {
  reset();
  if (ptc)
    return ob::PlannerStatus::TIMEOUT;
  State a, b;
  a.values = const_cast<double*>(start);
  b.values = const_cast<double*>(goal);
  for (size_t d = 0; d < dimension_; ++d) {
    if (!std::isfinite(start[d]))
      return ob::PlannerStatus::INVALID_START;
    if (!std::isfinite(goal[d]))
      return ob::PlannerStatus::INVALID_GOAL;
  }
  if (!si_->satisfiesBounds(&a) || !si_->isValid(&a))
    return ob::PlannerStatus::INVALID_START;
  if (!si_->satisfiesBounds(&b) || !si_->isValid(&b))
    return ob::PlannerStatus::INVALID_GOAL;
  insert(start_tree_, start, BatchNearest::none);
  insert(goal_tree_, goal, BatchNearest::none);
  return search(ptc, nullptr);
}

ob::PlannerStatus PlainmpRRTC::solve(
    const ob::PlannerTerminationCondition& ptc) {
  checkValidity();
  path_size_ = 0;
  auto* goal = dynamic_cast<ob::GoalSampleableRegion*>(pdef_->getGoal().get());
  if (!goal)
    return ob::PlannerStatus::UNRECOGNIZED_GOAL_TYPE;
  while (size_ < capacity_) {
    const auto* start = pis_.nextStart();
    if (!start)
      break;
    insert(start_tree_, start->as<State>()->values, BatchNearest::none);
  }
  if (!start_tree_.size())
    return ob::PlannerStatus::INVALID_START;
  if (!goal->couldSample())
    return ob::PlannerStatus::INVALID_GOAL;
  const auto status = search(ptc, goal);
  if (status == ob::PlannerStatus::EXACT_SOLUTION) {
    // Allocations here are solely for the standard OMPL result interface.
    auto path = std::make_shared<ompl::geometric::PathGeometric>(si_);
    path->getStates().reserve(path_size_);
    State state;
    for (size_t i = 0; i < path_size_; ++i) {
      state.values = const_cast<double*>(pathPoint(i));
      path->append(&state);
    }
    pdef_->addSolutionPath(path, false, 0.0, getName());
  }
  return status;
}
}  // namespace plainmp::ompl_wrapper
