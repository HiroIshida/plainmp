/* Copyright (C) 2026 Hirokazu Ishida
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. See https://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "plainmp/ompl/ompl_thin_wrap.hpp"
#include "plainmp/ompl/plainmp_rrtc.hpp"

namespace plainmp::ompl_wrapper {

// Fixed-goal Python interface. Input conversion, returned matrix and optional
// OMPL refinement are boundary allocations; the search reuses constructor
// storage, including the termination condition and constraint input vector.
class PlainmpRRTCPlanner {
 public:
  PlainmpRRTCPlanner(
      const std::vector<double>& lb,
      const std::vector<double>& ub,
      constraint::IneqConstraintBase::Ptr constraint,
      size_t max_calls,
      const ValidatorConfig& validator,
      std::optional<double> range = std::nullopt,
      const std::optional<PlainmpRRTCSettings>& settings = std::nullopt)
      : validity_buffer_(lb.size()), stop_([this] {
          return csi_->is_valid_call_count_ >= csi_->max_is_valid_call_ ||
                 timed_out();
        }) {
    if (!constraint || constraint->q_dim() != lb.size() || lb.empty() ||
        lb.size() != ub.size())
      throw std::invalid_argument("constraint and bounds dimensions differ");
    for (size_t i = 0; i < lb.size(); ++i)
      if (!std::isfinite(lb[i]) || !std::isfinite(ub[i]) || lb[i] >= ub[i])
        throw std::invalid_argument("bounds must be finite and ordered");
    const auto positive = [](double x) { return std::isfinite(x) && x > 0; };
    if ((validator.type == ValidatorConfig::Type::BOX &&
         (validator.box_width.size() != lb.size() ||
          !std::all_of(validator.box_width.begin(), validator.box_width.end(),
                       positive))) ||
        (validator.type == ValidatorConfig::Type::EUCLIDEAN &&
         !positive(validator.resolution)))
      throw std::invalid_argument(
          "motion resolution must be positive and finite");
    csi_ = std::make_unique<CollisionAwareSpaceInformation>(
        lb, ub, constraint, max_calls, validator);
    csi_->si_->setStateValidityChecker([this](const ob::State* state) {
      if (csi_->is_valid_call_count_ >= csi_->max_is_valid_call_ ||
          (timeout_ && csi_->is_valid_call_count_ % 64 == 0 && timed_out()))
        return false;
      validity_buffer_ = Eigen::Map<const Eigen::VectorXd>(
          state->as<ob::RealVectorStateSpace::StateType>()->values,
          validity_buffer_.size());
      ++csi_->is_valid_call_count_;
      return csi_->ineq_cst_->is_valid(validity_buffer_);
    });
    csi_->si_->setup();
    planner_ = std::make_unique<PlainmpRRTC>(
        csi_->si_, settings.value_or(PlainmpRRTCSettings{}));
    if (range)
      planner_->setRange(*range);
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
          "PlainmpRRTCPlanner requires a fixed goal; use OMPLSolver for IK "
          "goals");
    const size_t dimension = validity_buffer_.size();
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
    for (auto refine : refine_seq)
      if (refine != RefineType::SHORTCUT && refine != RefineType::BSPLINE)
        throw std::invalid_argument("unknown refine type");
    timeout_ = timeout;
    csi_->resetCount();
    solve_start_ = std::chrono::steady_clock::now();
    const auto status = planner_->solveFixed(start.data(), goal->data(), stop_);
    // As in the existing wrapper, exclude output matrix allocation/copy.
    record_time();
    if (status != ob::PlannerStatus::EXACT_SOLUTION)
      return std::nullopt;

    if (refine_seq.empty()) {
      Eigen::MatrixXd result(planner_->pathSize(), dimension);
      for (size_t i = 0; i < planner_->pathSize(); ++i)
        result.row(i) =
            Eigen::Map<const Eigen::VectorXd>(planner_->pathPoint(i), dimension)
                .transpose();
      return result;
    }
    og::PathGeometric path(csi_->si_);
    path.getStates().reserve(planner_->pathSize());
    ob::RealVectorStateSpace::StateType state;
    for (size_t i = 0; i < planner_->pathSize(); ++i) {
      state.values = const_cast<double*>(planner_->pathPoint(i));
      path.append(&state);
    }
    og::PathSimplifier simplifier(csi_->si_);
    for (auto refine : refine_seq) {
      if (stop_)
        break;
      if (refine == RefineType::SHORTCUT) {
#ifdef OMPL_OLD_VERSION
        simplifier.shortcutPath(path);
#else
        simplifier.partialShortcutPath(path);
#endif
      } else {
        simplifier.smoothBSpline(path);
      }
    }
    record_time();
    Eigen::MatrixXd result(path.getStateCount(), dimension);
    for (size_t i = 0; i < path.getStateCount(); ++i)
      result.row(i) = Eigen::Map<const Eigen::VectorXd>(
                          path.getState(i)
                              ->as<ob::RealVectorStateSpace::StateType>()
                              ->values,
                          dimension)
                          .transpose();
    return result;
  }

  size_t getCallCount() const { return csi_->is_valid_call_count_; }
  size_t get_ns_internal() const { return ns_internal_; }
  size_t getNodeCount() const { return planner_->nodeCount(); }

 private:
  bool timed_out() const {
    return timeout_ && std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - solve_start_)
                               .count() >= *timeout_;
  }
  void record_time() {
    ns_internal_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now() - solve_start_)
                       .count();
  }
  std::unique_ptr<CollisionAwareSpaceInformation> csi_;
  std::unique_ptr<PlainmpRRTC> planner_;
  Eigen::VectorXd validity_buffer_;
  std::optional<double> timeout_;
  std::chrono::steady_clock::time_point solve_start_;
  size_t ns_internal_ = 0;
  ob::PlannerTerminationCondition stop_;
};
}  // namespace plainmp::ompl_wrapper
