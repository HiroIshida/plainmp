/* Copyright (C) 2026 Hirokazu Ishida
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. See https://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "plainmp/ompl/ompl_thin_wrap.hpp"
#include "plainmp/ompl/vamp_rrtc.hpp"

namespace plainmp::ompl_wrapper {

// Dedicated wrapper so allocation and termination improvements apply only to
// VampRRTC. The other planners keep their existing wrapper and checker.
class VampRRTCPlanner : public PlannerBase {
 public:
  VampRRTCPlanner(
      const std::vector<double>& lb,
      const std::vector<double>& ub,
      constraint::IneqConstraintBase::Ptr constraint,
      size_t max_calls,
      const ValidatorConfig& validator,
      std::optional<double> range = std::nullopt,
      const std::optional<VampRRTCSettings>& settings = std::nullopt)
      : PlannerBase(lb, ub, constraint, max_calls, validator),
        validity_buffer_(lb.size()) {
    ns_internal_measurement_ = 0;
    if (constraint->q_dim() != lb.size())
      throw std::invalid_argument("constraint and state dimensions differ");
    const auto positive = [](double x) { return std::isfinite(x) && x > 0; };
    if ((validator.type == ValidatorConfig::Type::BOX &&
         !std::all_of(validator.box_width.begin(), validator.box_width.end(),
                      positive)) ||
        (validator.type == ValidatorConfig::Type::EUCLIDEAN &&
         !positive(validator.resolution)))
      throw std::invalid_argument(
          "motion resolution must be positive and finite");
    auto planner = std::make_shared<VampRRTC>(csi_->si_);
    if (range)
      planner->setRange(*range);
    if (settings)
      planner->setSettings(*settings);
    setup_->setPlanner(planner);
    setup_->setStateValidityChecker([this](const ob::State* state) {
      if (csi_->is_valid_call_count_ >= csi_->max_is_valid_call_)
        return false;
      // Also stop within long edges, without reading the clock on every query.
      if (timeout_ && csi_->is_valid_call_count_ % 64 == 0 && timed_out())
        return false;
      // Constraint::is_valid takes an owning VectorXd. A Map would allocate
      // a temporary on each check, so this planner reuses its own vector.
      validity_buffer_ = Eigen::Map<const Eigen::VectorXd>(
          state->as<ob::RealVectorStateSpace::StateType>()->values,
          validity_buffer_.size());
      ++csi_->is_valid_call_count_;
      return csi_->ineq_cst_->is_valid(validity_buffer_);
    });
  }

  std::optional<Eigen::MatrixXd> solve(
      const std::vector<double>& start,
      const std::optional<std::vector<double>>& goal,
      const std::vector<RefineType>& refine_seq,
      std::optional<double> timeout,
      const std::optional<GoalSamplerFn>& goal_sampler,
      std::optional<size_t> /*max_goal_sample_count*/ = std::nullopt) {
    if (goal_sampler || !goal)
      throw std::invalid_argument(
          "VampRRTCPlanner requires a fixed goal; use OMPLSolver for IK goals");
    const size_t dimension = csi_->si_->getStateDimension();
    const auto valid = [dimension](const std::vector<double>& q) {
      return q.size() == dimension &&
             std::all_of(q.begin(), q.end(),
                         [](double x) { return std::isfinite(x); });
    };
    if (!valid(start) || !valid(*goal))
      throw std::invalid_argument(
          "start and goal must have the space dimension and finite values");
    if (timeout && (!std::isfinite(*timeout) || *timeout < 0))
      throw std::invalid_argument("timeout must be nonnegative and finite");
    setup_->clear();
    csi_->resetCount();
    timeout_ = timeout;
    ob::ScopedState<> a(csi_->si_->getStateSpace()),
        b(csi_->si_->getStateSpace());
    std::copy(start.begin(), start.end(),
              a->as<ob::RealVectorStateSpace::StateType>()->values);
    std::copy(goal->begin(), goal->end(),
              b->as<ob::RealVectorStateSpace::StateType>()->values);
    setup_->setStartAndGoalStates(a, b);
    solve_start_ = std::chrono::steady_clock::now();
    const ob::PlannerTerminationCondition stop([this] {
      return csi_->is_valid_call_count_ >= csi_->max_is_valid_call_ ||
             timed_out();
    });
    const auto status = setup_->solve(stop);
    record_time();
    if (status != ob::PlannerStatus::EXACT_SOLUTION)
      return std::nullopt;

    auto& path = setup_->getSolutionPath();
    std::optional<og::PathSimplifier> simplifier;
    if (!refine_seq.empty())
      simplifier.emplace(csi_->si_);
    for (auto refine : refine_seq) {
      if (stop)
        break;
      if (refine == RefineType::SHORTCUT) {
#ifdef OMPL_OLD_VERSION
        simplifier->shortcutPath(path);
#else
        simplifier->partialShortcutPath(path);
#endif
      } else if (refine == RefineType::BSPLINE) {
        simplifier->smoothBSpline(path);
      } else {
        throw std::invalid_argument("unknown refine type");
      }
    }
    Eigen::MatrixXd result(path.getStateCount(), dimension);
    for (size_t i = 0; i < path.getStateCount(); ++i)
      result.row(i) = Eigen::Map<const Eigen::VectorXd>(
                          path.getState(i)
                              ->as<ob::RealVectorStateSpace::StateType>()
                              ->values,
                          dimension)
                          .transpose();
    record_time();
    return result;
  }

 private:
  bool timed_out() const {
    return timeout_ && std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - solve_start_)
                               .count() >= *timeout_;
  }
  void record_time() {
    ns_internal_measurement_ =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - solve_start_)
            .count();
  }
  Eigen::VectorXd validity_buffer_;
  std::optional<double> timeout_;
  std::chrono::steady_clock::time_point solve_start_;
};
}  // namespace plainmp::ompl_wrapper
