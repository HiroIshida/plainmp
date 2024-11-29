#pragma once

#include <ompl/base/PlannerTerminationCondition.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/experience/ERTConnect.h>
#include <optional>
#include "algorithm_selector.hpp"
#include "constraints/primitive.hpp"
#include "custom_goal_samplable_region.hpp"
#include "motion_validator.hpp"
#include "ompl/base/MotionValidator.h"

namespace plainmp::ompl_wrapper {

namespace ob = ompl::base;
namespace og = ompl::geometric;

inline void state_to_vec(const ob::State* state, std::vector<double>& vec) {
  const ob::RealVectorStateSpace::StateType* rs;
  rs = state->as<ob::RealVectorStateSpace::StateType>();
  std::memcpy(vec.data(), rs->values, vec.size() * sizeof(double));
};

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
        max_is_valid_call_(max_is_valid_call),
        tmp_vec_(lb.size()) {
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
    si_->setup();
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
    space->setup();
    return space;
  }

  bool is_terminatable() const {
    return is_valid_call_count_ > max_is_valid_call_;
  }

  bool is_valid(const ob::State* state) {
    const size_t dim = si_->getStateDimension();
    state_to_vec(state, tmp_vec_);
    this->is_valid_call_count_++;
    return ineq_cst_->is_valid(tmp_vec_);
  }

  ob::SpaceInformationPtr si_;
  constraint::IneqConstraintBase::Ptr ineq_cst_;
  size_t is_valid_call_count_;
  const size_t max_is_valid_call_;
  std::vector<double>
      tmp_vec_;  // to avoid dynamic allocation (used in is_valid)
};

struct PlannerBase {
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
  std::optional<std::vector<std::vector<double>>> solve(
      const std::vector<double>& start,
      const std::optional<std::vector<double>>& goal,
      bool simplify,
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
    const auto result = setup_->solve(ptc);
    if (not result) {
      return {};
    }
    if (result == ob::PlannerStatus::APPROXIMATE_SOLUTION) {
      OMPL_INFORM(
          "reporeted to be solved. But reject it because it'S approx solution");
      return {};
    }
    if (simplify) {
      setup_->simplifySolution(ptc);
    }
    const auto p = setup_->getSolutionPath().as<og::PathGeometric>();
    auto& states = p->getStates();
    const size_t dim = start.size();

    // states
    auto trajectory = std::vector<std::vector<double>>();

    std::vector<double> tmp_vec(dim);
    for (const auto& state : states) {
      state_to_vec(state, tmp_vec);
      trajectory.push_back(tmp_vec);
    }
    return trajectory;
  }

  size_t getCallCount() const { return csi_->is_valid_call_count_; }
  std::unique_ptr<CollisionAwareSpaceInformation> csi_;
  std::unique_ptr<og::SimpleSetup> setup_;
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

    if (algo_name.compare("AITstarStop") == 0) {
      auto pdef = setup_->getProblemDefinition();
      auto objective =
          std::make_shared<ob::PathLengthOptimizationObjective>(csi_->si_);
      objective->setCostThreshold(
          ob::Cost(std::numeric_limits<double>::infinity()));
      pdef->setOptimizationObjective(objective);
    }
    if (algo_name.compare("AITstar") == 0 ||
        algo_name.compare("AITstarStop") == 0) {
      // probably ait star's bug: clear requires to pdef to be set already,
      // which is usually set in solve() function but we need to set it now
      auto pdef = setup_->getProblemDefinition();
      algo->setProblemDefinition(pdef);
      algo->setup();
    }
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
