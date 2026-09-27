/* Copyright (C) 2026 Hirokazu Ishida
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. See https://mozilla.org/MPL/2.0/.
 * RRTConnect search follows OMPL 1.6; see third/ompl_rrtc/LICENSE.
 */
#pragma once

#include <ompl/base/Planner.h>
#include <ompl/base/goals/GoalSampleableRegion.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include "plainmp/ompl/batch_nearest.hpp"

namespace plainmp::ompl_wrapper {

struct PlainmpRRTCSettings {
  // Combined capacity of both trees. Exhaustion returns TIMEOUT, never grows.
  size_t max_samples = 100000;
};

// OMPL RRTConnect (without intermediate states), for Euclidean real vectors.
// All search storage and the uniform sampler are allocated in the constructor.
// solveFixed avoids ProblemDefinition/PathGeometric allocations; its result is
// a view valid until clear(), the next solve(), or destruction. Not thread
// safe.
class PlainmpRRTC : public ompl::base::Planner {
 public:
  explicit PlainmpRRTC(const ompl::base::SpaceInformationPtr& si,
                       const PlainmpRRTCSettings& settings = {});
  void clear() override;
  void setRange(double range);
  double getRange() const { return range_; }
  ompl::base::PlannerStatus solve(
      const ompl::base::PlannerTerminationCondition& ptc) override;
  ompl::base::PlannerStatus solveFixed(
      const double* start,
      const double* goal,
      const ompl::base::PlannerTerminationCondition& ptc);
  size_t pathSize() const { return path_size_; }
  const double* pathPoint(size_t i) const { return point(path_[i]); }
  size_t nodeCount() const { return size_; }

 private:
  using Index = BatchNearest::Index;
  using State = ompl::base::RealVectorStateSpace::StateType;
  enum class Grow { TRAPPED, ADVANCED, REACHED };
  const double* point(Index i) const { return points_.get() + i * dimension_; }
  double* point(Index i) { return points_.get() + i * dimension_; }
  void reset();
  Index insert(BatchNearest& tree, const double* q, Index parent);
  Grow grow(BatchNearest& tree,
            bool from_start,
            const double* target,
            Index& added);
  void tracePath(Index start, Index goal);
  ompl::base::PlannerStatus search(
      const ompl::base::PlannerTerminationCondition& ptc,
      ompl::base::GoalSampleableRegion* goal);

  double range_ = 2.0;
  size_t dimension_, capacity_, path_size_ = 0;
  Index size_ = 0;
  bool start_first_ = true;
  // Coordinates are contiguous per configuration, with compact separate
  // parent/root arrays. NN scans never load the tree's ancestry metadata.
  std::unique_ptr<double[]> points_, scratch_;
  std::unique_ptr<Index[]> parents_, roots_, path_;
  BatchNearest start_tree_, goal_tree_;
  ompl::base::StateSamplerPtr sampler_;
};
}  // namespace plainmp::ompl_wrapper
