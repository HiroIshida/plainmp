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
#include <ompl/base/GoalTypes.h>
#include <ompl/geometric/planners/kpiece/KPIECE1.h>
#include <ompl/geometric/planners/rrt/RRT.h>

namespace ompl {

namespace custom {

class RRTModified : public geometric::RRT {
 public:
  using geometric::RRT::RRT;
  base::PlannerStatus solve(
      const base::PlannerTerminationCondition& ptc) override {
    base::Goal* goal = pdef_->getGoal().get();
    if (goal->getType() == base::GOAL_STATE) {
      return geometric::RRT::solve(ptc);  // as the original RRT
    } else if (goal->getType() == base::GOAL_SAMPLEABLE_REGION) {
      return solve_with_goal_const(ptc);
    } else {
      throw ompl::Exception("Unknown goal type");
    }
  }

 private:
  base::PlannerStatus solve_with_goal_const(
      const base::PlannerTerminationCondition& ptc);
};

class KPIECE1Modified : public geometric::KPIECE1 {
 public:
  using geometric::KPIECE1::KPIECE1;

  base::PlannerStatus solve(
      const base::PlannerTerminationCondition& ptc) override {
    base::Goal* goal = pdef_->getGoal().get();
    if (goal->getType() == base::GOAL_STATE) {
      return geometric::KPIECE1::solve(ptc);  // as the original KPIECE1
    } else if (goal->getType() == base::GOAL_SAMPLEABLE_REGION) {
      return solve_with_goal_const(ptc);
    } else {
      throw ompl::Exception("Unknown goal type");
    }
  }

 private:
  base::PlannerStatus solve_with_goal_const(
      const base::PlannerTerminationCondition& ptc);
};

}  // namespace custom

};  // namespace ompl
