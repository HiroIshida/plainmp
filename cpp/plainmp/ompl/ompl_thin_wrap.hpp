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

#include <ertconnect/ERTConnect.h>
#include <ompl/base/MotionValidator.h>
#include <ompl/base/PlannerTerminationCondition.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/SimpleSetup.h>
#include <Eigen/Dense>
#include <chrono>
#include <optional>
#include "plainmp/constraints/primitive.hpp"
#include "plainmp/ompl/algorithm_selector.hpp"
#include "plainmp/ompl/custom_goal_samplable_region.hpp"
#include "plainmp/ompl/motion_validator.hpp"

namespace plainmp::ompl_wrapper {

namespace ob = ompl::base;
namespace og = ompl::geometric;

og::PathGeometric points_to_pathgeometric(
    const std::vector<std::vector<double>>& points,
    ob::SpaceInformationPtr si) {
  auto pg = og::PathGeometric(si);
  for (const auto& point : points) {
    ob::State* s = si->getStateSpace()->allocState();
    auto rs = s->as<ob::RealVectorStateSpace::StateType>();
    for (size_t i = 0; i < si->getStateDimension(); ++i) {
      rs->values[i] = point.at(i);
    }
    pg.append(rs);
  }
  return pg;
}

struct ValidatorConfig {
  enum class Type { BOX, EUCLIDEAN };
  Type type;
  // cannot use std::variant to work with pybind11
  // but either double or vector<double> is expected
  // and BOX => vector<double> is expected
  // and EUCLIDEAN => double is expected
  double resolution;
  std::vector<double> box_width;
};

struct CollisionAwareSpaceInformation {
  CollisionAwareSpaceInformation(const std::vector<double>& lb,
                                 const std::vector<double>& ub,
                                 constraint::IneqConstraintBase::Ptr ineq_cst,
                                 size_t max_is_valid_call,
                                 const ValidatorConfig& vconfig)
      : si_(nullptr),
        ineq_cst_(ineq_cst),
        is_valid_call_count_(0),
        max_is_valid_call_(max_is_valid_call) {
    if (!ineq_cst_) {
      throw std::runtime_error(
          "ineq_cst is nullptr. You should provide a constraint");
    }
    const auto space = bound2space(lb, ub);
    si_ = std::make_shared<ob::SpaceInformation>(space);

    size_t dim = space->getDimension();

    if (vconfig.type == ValidatorConfig::Type::EUCLIDEAN) {
      si_->setMotionValidator(
          std::make_shared<EuclideanMotionValidator>(si_, vconfig.resolution));
    } else if (vconfig.type == ValidatorConfig::Type::BOX) {
      if (vconfig.box_width.size() != dim) {
        throw std::runtime_error("box dimension and space dimension mismatch");
      }
      si_->setMotionValidator(
          std::make_shared<BoxMotionValidator>(si_, vconfig.box_width));
    } else {
      throw std::runtime_error("unknown validator type");
    }
    // si_->setup();
  }

  void resetCount() { this->is_valid_call_count_ = 0; }

  static std::shared_ptr<ob::StateSpace> bound2space(
      const std::vector<double>& lb,
      const std::vector<double>& ub) {
    const size_t dim = lb.size();
    auto bounds = ob::RealVectorBounds(dim);
    bounds.low = lb;
    bounds.high = ub;
    const auto space(std::make_shared<ob::RealVectorStateSpace>(dim));
    space->setBounds(bounds);
    // space->setup();
    return space;
  }

  bool is_terminatable() const {
    return is_valid_call_count_ > max_is_valid_call_;
  }

  bool is_valid(const ob::State* state) {
    auto rs = state->as<ob::RealVectorStateSpace::StateType>();
    const Eigen::Map<Eigen::VectorXd> tmp_vec(rs->values,
                                              si_->getStateDimension());
    this->is_valid_call_count_++;
    return ineq_cst_->is_valid(tmp_vec);
  }

  ob::SpaceInformationPtr si_;
  constraint::IneqConstraintBase::Ptr ineq_cst_;
  size_t is_valid_call_count_;
  const size_t max_is_valid_call_;
};

struct PlannerBase {
  enum class RefineType { SHORTCUT, BSPLINE };
  PlannerBase(const std::vector<double>& lb,
              const std::vector<double>& ub,
              constraint::IneqConstraintBase::Ptr ineq_cst,
              size_t max_is_valid_call,
              const ValidatorConfig& vconfig) {
    csi_ = std::make_unique<CollisionAwareSpaceInformation>(
        lb, ub, ineq_cst, max_is_valid_call, vconfig);
    setup_ = std::make_unique<og::SimpleSetup>(csi_->si_);
    setup_->setStateValidityChecker(
        [this](const ob::State* s) { return this->csi_->is_valid(s); });
  }
  std::optional<Eigen::MatrixXd> solve(
      const std::vector<double>& start,
      const std::optional<std::vector<double>>& goal,
      std::vector<RefineType> refine_seq,
      std::optional<double> timeout,
      const std::optional<GoalSamplerFn>& goal_sampler,
      std::optional<size_t> max_goal_sample_count = std::nullopt) {
    setup_->clear();
    csi_->resetCount();

    // args shold be eigen maybe?
    Eigen::VectorXd vec_start =
        Eigen::Map<const Eigen::VectorXd>(&start[0], start.size());
    ob::ScopedState<> sstart(csi_->si_->getStateSpace());
    auto rstart = sstart->as<ob::RealVectorStateSpace::StateType>();
    std::copy(start.begin(), start.end(), rstart->values);
    setup_->setStartState(sstart);

    if (goal.has_value() == goal_sampler.has_value()) {  // xor
      throw std::runtime_error("goal and goal_sampler should be exclusive");
    }
    if (goal_sampler) {
      auto goal_region = std::make_shared<CustomGoalSamplableRegion>(
          csi_->si_, *goal_sampler, max_goal_sample_count);
      setup_->setGoal(goal_region);
    } else {
      Eigen::VectorXd vec_goal =
          Eigen::Map<const Eigen::VectorXd>(&goal->at(0), goal->size());
      ob::ScopedState<> sgoal(csi_->si_->getStateSpace());
      auto rgoal = sgoal->as<ob::RealVectorStateSpace::StateType>();
      std::copy(goal->begin(), goal->end(), rgoal->values);
      setup_->setGoalState(sgoal);
    }

    std::function<bool()> fn = [this]() { return csi_->is_terminatable(); };
    ob::PlannerTerminationCondition ptc = ob::PlannerTerminationCondition(fn);
    if (timeout) {  // override
      ptc = ob::timedPlannerTerminationCondition(*timeout);
    }
    auto start_time = std::chrono::steady_clock::now();
    const auto result = setup_->solve(ptc);
    if (not result) {
      return {};
    }
    if (result == ob::PlannerStatus::APPROXIMATE_SOLUTION) {
      OMPL_INFORM(
          "reporeted to be solved. But reject it because it'S approx solution");
      return {};
    }
    auto p = setup_->getSolutionPath().as<og::PathGeometric>();

    og::PathSimplifier simplifier(csi_->si_);
    for (auto refine : refine_seq) {
      if (refine == RefineType::SHORTCUT) {
        simplifier.shortcutPath(*p);
      } else if (refine == RefineType::BSPLINE) {
        simplifier.smoothBSpline(*p);
      } else {
        throw std::runtime_error("unknown refine type");
      }
    }
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_time - start_time);
    ns_internal_measurement_ = elapsed.count();

    auto& states = p->getStates();
    const size_t dim = start.size();

    // use Eigen::MatrixXd to rerun numpy array in python
    Eigen::MatrixXd trajectory(dim, states.size());
    std::vector<double> tmp_vec(dim);
    for (size_t i = 0; i < states.size(); ++i) {
      auto rs = states[i]->as<ob::RealVectorStateSpace::StateType>();
      trajectory.col(i) = Eigen::Map<Eigen::VectorXd>(rs->values, dim);
    }
    return trajectory.transpose();
  }

  size_t getCallCount() const { return csi_->is_valid_call_count_; }
  size_t get_ns_internal() const {
    // for benchmarking with other libraries using internal measurement
    return ns_internal_measurement_;
  }
  std::unique_ptr<CollisionAwareSpaceInformation> csi_;
  std::unique_ptr<og::SimpleSetup> setup_;
  size_t ns_internal_measurement_;  // last measurement
};

struct OMPLPlanner : public PlannerBase {
  OMPLPlanner(const std::vector<double>& lb,
              const std::vector<double>& ub,
              constraint::IneqConstraintBase::Ptr ineq_cst,
              size_t max_is_valid_call,
              const ValidatorConfig& vconfig,
              const std::string& algo_name,
              std::optional<double> range)
      : PlannerBase(lb, ub, ineq_cst, max_is_valid_call, vconfig) {
    const auto algo = get_algorithm(algo_name, csi_->si_, range);
    setup_->setPlanner(algo);
  }
};

struct ERTConnectPlanner : public PlannerBase {
  ERTConnectPlanner(const std::vector<double>& lb,
                    const std::vector<double>& ub,
                    constraint::IneqConstraintBase::Ptr ineq_cst,
                    size_t max_is_valid_call,
                    const ValidatorConfig& vconfig)
      : PlannerBase(lb, ub, ineq_cst, max_is_valid_call, vconfig) {
    auto ert_connect = std::make_shared<og::ERTConnect>(csi_->si_);
    setup_->setPlanner(ert_connect);
  }

  void set_heuristic(const std::vector<std::vector<double>>& points) {
    auto geo_path = points_to_pathgeometric(points, this->csi_->si_);
    const auto heuristic = geo_path.getStates();
    const auto ert_connect = setup_->getPlanner()->as<og::ERTConnect>();
    ert_connect->setExperience(heuristic);
  }

  void set_parameters(std::optional<double> omega_min,
                      std::optional<double> omega_max,
                      std::optional<double> eps) {
    const auto planner = setup_->getPlanner();
    const auto ert_connect = planner->as<og::ERTConnect>();
    if (omega_min) {
      ert_connect->setExperienceFractionMin(*omega_min);
    }
    if (omega_max) {
      ert_connect->setExperienceFractionMax(*omega_max);
    }
    if (eps) {
      ert_connect->setExperienceTubularRadius(*eps);
    }
  }
};

void setGlobalSeed(size_t seed) {
  ompl::RNG::setSeed(seed);
}

void setLogLevelNone() {
  ompl::msg::setLogLevel(ompl::msg::LOG_NONE);
}

}  // namespace plainmp::ompl_wrapper
