#include <ompl/base/PlannerTerminationCondition.h>
#include <ompl/base/State.h>
#include <ompl/base/StateSampler.h>
#include <ompl/base/StateSpace.h>
#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/experience/ERTConnect.h>
#include <ompl/geometric/planners/kpiece/BKPIECE1.h>
#include <ompl/geometric/planners/kpiece/KPIECE1.h>
#include <ompl/geometric/planners/kpiece/LBKPIECE1.h>
#include <ompl/geometric/planners/rrt/RRT.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/tools/experience/ExperienceSetup.h>
#include <ompl/tools/lightning/Lightning.h>
#include <ompl/tools/lightning/LightningDB.h>
#include <ompl/util/Console.h>
#include <ompl/util/PPM.h>
#include <ompl/util/Time.h>

#include <boost/filesystem.hpp>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stack>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ompl/base/DiscreteMotionValidator.h"
#include "ompl/base/MotionValidator.h"
#include "ompl/base/Planner.h"
#include "ompl/base/PlannerData.h"
#include "ompl/base/PlannerStatus.h"
#include "ompl/base/SpaceInformation.h"
#include "ompl/base/StateValidityChecker.h"
#include "repair_planner.hpp"
#include "constraint.hpp"

#define STRING(str) #str
#define ALLOCATE_ALGO(ALGO)                                   \
  if (name.compare(#ALGO) == 0) {                             \
    const auto algo = std::make_shared<og::ALGO>(space_info); \
    return std::static_pointer_cast<ob::Planner>(algo);       \
  }

namespace ob = ompl::base;
namespace og = ompl::geometric;
namespace ot = ompl::tools;

// TODO: I wanted to pass and return eigen::matrix / vector, but
// pybind fail to convert numpy to eigen in callback case
using ConstFn = std::function<std::vector<double>(std::vector<double>)>;
using ConstJacFn = std::function<std::vector<std::vector<double>>(std::vector<double>)>;

template <typename T>
std::shared_ptr<T> create_algorithm(const ob::SpaceInformationPtr si, std::optional<double> range)
{
  auto algo = std::make_shared<T>(si);
  if (range) {
    algo->setRange(*range);
  }
  return algo;
}

inline void state_to_vec(const ob::State* state, std::vector<double>& vec)
{
  const ob::RealVectorStateSpace::StateType* rs;
  rs = state->as<ob::RealVectorStateSpace::StateType>();
  std::memcpy(vec.data(), rs->values, vec.size() * sizeof(double));
};

og::PathGeometric points_to_pathgeometric(const std::vector<std::vector<double>>& points,
                                          ob::SpaceInformationPtr si)
{
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

class BoxMotionValidator : public ob::MotionValidator
{
 public:
  BoxMotionValidator(const ob::SpaceInformationPtr& si, std::vector<double> width)
      : ob::MotionValidator(si), width_(width)
  {
      // NOTE: precompute inv width, because devide is more expensive than multiply
      for(size_t i = 0; i < width.size(); ++i) {
          inv_width_.push_back(1.0 / width[i]);
      }
  }

  bool checkMotion(const ob::State* s1, const ob::State* s2) const
  {
    const auto rs1 = s1->as<ob::RealVectorStateSpace::StateType>();
    const auto rs2 = s2->as<ob::RealVectorStateSpace::StateType>();

    // find longest (relative) axis index
    double diff_longest_axis;
    double max_diff = -std::numeric_limits<double>::infinity();
    size_t longest_idx = 0;
    for (size_t idx = 0; idx < si_->getStateDimension(); ++idx) {
      const double diff = rs2->values[idx] - rs1->values[idx];
      const double abs_scaled_diff = std::abs(diff) * inv_width_[idx];
      if (abs_scaled_diff > max_diff) {
        max_diff = abs_scaled_diff;
        longest_idx = idx;
        diff_longest_axis = diff;
      }
    }
    if (std::abs(diff_longest_axis) < 1e-6) {
      return true;
    }

    // main
    const auto s_test = si_->allocState()->as<ob::RealVectorStateSpace::StateType>();

    const auto space = si_->getStateSpace();
    const double step_ratio = width_[longest_idx] / std::abs(diff_longest_axis);

    double travel_rate = 0.0;
    while (travel_rate + step_ratio < 1.0) {
      travel_rate += step_ratio;
      space->interpolate(rs1, rs2, travel_rate, s_test);
      if (!si_->isValid(s_test)) {
        return false;
      }
    }
    return (si_->isValid(rs2));
  }

  bool checkMotion(const ob::State* s1,
                   const ob::State* s2,
                   std::pair<ob::State*, double>& lastValid) const
  {
    return checkMotion(s1, s2);
  }

 private:
  std::vector<double> width_;
  std::vector<double> inv_width_;
};

struct CollisionAwareSpaceInformation {
  void resetCount() { this->is_valid_call_count_ = 0; }

  static std::shared_ptr<ob::StateSpace> bound2space(const std::vector<double>& lb,
                                                     const std::vector<double>& ub)
  {
    const size_t dim = lb.size();
    auto bounds = ob::RealVectorBounds(dim);
    bounds.low = lb;
    bounds.high = ub;
    const auto space(std::make_shared<ob::RealVectorStateSpace>(dim));
    space->setBounds(bounds);
    space->setup();
    return space;
  }

  bool is_terminatable() const { return is_valid_call_count_ > max_is_valid_call_; }

  bool is_valid(const ob::State* state)
  {
    const size_t dim = si_->getStateDimension();
    state_to_vec(state, tmp_vec_);
    this->is_valid_call_count_++;
    return ineq_cst_->is_valid(tmp_vec_);
  }

  ob::SpaceInformationPtr si_;
  cst::IneqConstraintBase::Ptr ineq_cst_;
  size_t is_valid_call_count_;
  const size_t max_is_valid_call_;
  std::vector<double> box_width_;
  std::vector<double> tmp_vec_; // to avoid dynamic allocation (used in is_valid)
};

struct UnconstrianedCollisoinAwareSpaceInformation : public CollisionAwareSpaceInformation {
  UnconstrianedCollisoinAwareSpaceInformation(
      const std::vector<double>& lb,
      const std::vector<double>& ub,
      cst::IneqConstraintBase::Ptr ineq_cst,
      size_t max_is_valid_call,
      const std::vector<double>& box_width)
      : CollisionAwareSpaceInformation{nullptr, ineq_cst, 0, max_is_valid_call, box_width, std::vector<double>(box_width.size())}
  {
    const auto space = bound2space(lb, ub);
    si_ = std::make_shared<ob::SpaceInformation>(space);
    if (box_width.size() != space->getDimension()) {
      throw std::runtime_error("box dimension and space dimension mismatch");
    }
    si_->setMotionValidator(std::make_shared<BoxMotionValidator>(si_, box_width));
    si_->setup();
  }
};


struct PlannerBase {
  std::optional<std::vector<std::vector<double>>> solve(const std::vector<double>& start,
                                                        const std::vector<double>& goal,
                                                        bool simplify)
  {
    setup_->clear();
    csi_->resetCount();

    // args shold be eigen maybe?
    Eigen::VectorXd vec_start = Eigen::Map<const Eigen::VectorXd>(&start[0], start.size());
    Eigen::VectorXd vec_goal = Eigen::Map<const Eigen::VectorXd>(&goal[0], goal.size());

    ob::ScopedState<> sstart(csi_->si_->getStateSpace());
    ob::ScopedState<> sgoal(csi_->si_->getStateSpace());

    auto rstart = sstart->as<ob::RealVectorStateSpace::StateType>();
    auto rgoal = sgoal->as<ob::RealVectorStateSpace::StateType>();
    std::copy(start.begin(), start.end(), rstart->values);
    std::copy(goal.begin(), goal.end(), rgoal->values);
    setup_->setStartAndGoalStates(sstart, sgoal);

    std::function<bool()> fn = [this]() { return csi_->is_terminatable(); };
    const auto result = setup_->solve(fn);
    if (not result) {
      return {};
    }
    if (result == ob::PlannerStatus::APPROXIMATE_SOLUTION) {
      OMPL_INFORM("reporeted to be solved. But reject it because it'S approx solution");
      return {};
    }
    if (simplify) {
      setup_->simplifySolution(fn);
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

  std::shared_ptr<ob::Planner> get_algorithm(const std::string& name, std::optional<double> range)
  {
    const auto space_info = csi_->si_;
    if (name.compare("BKPIECE1") == 0) {
      return create_algorithm<og::BKPIECE1>(space_info, range);
    } else if (name.compare("KPIECE1") == 0) {
      return create_algorithm<og::KPIECE1>(space_info, range);
    } else if (name.compare("RRT") == 0) {
      return create_algorithm<og::RRT>(space_info, range);
    } else if (name.compare("RRTConnect") == 0) {
      return create_algorithm<og::RRTConnect>(space_info, range);
    } else if (name.compare("RRTstar") == 0) {
      return create_algorithm<og::RRTstar>(space_info, range);
    }
    throw std::runtime_error("algorithm " + name + " is not supported");
  }

  size_t getCallCount() const {
    return csi_->is_valid_call_count_;
  }
  std::unique_ptr<UnconstrianedCollisoinAwareSpaceInformation> csi_;
  std::unique_ptr<og::SimpleSetup> setup_;
};

struct UnconstrainedPlannerBase : public PlannerBase {
  UnconstrainedPlannerBase(const std::vector<double>& lb,
                           const std::vector<double>& ub,
                           cst::IneqConstraintBase::Ptr ineq_cst,
                           size_t max_is_valid_call,
                           const std::vector<double>& box_width)
      : PlannerBase{nullptr, nullptr}
  {
    csi_ = std::make_unique<UnconstrianedCollisoinAwareSpaceInformation>(
        lb, ub, ineq_cst, max_is_valid_call, box_width);
    setup_ = std::make_unique<og::SimpleSetup>(csi_->si_);
    setup_->setStateValidityChecker([this](const ob::State* s) { return this->csi_->is_valid(s); });
  }
};

struct OMPLPlanner : public UnconstrainedPlannerBase {
  OMPLPlanner(const std::vector<double>& lb,
              const std::vector<double>& ub,
              cst::IneqConstraintBase::Ptr ineq_cst,
              size_t max_is_valid_call,
              const std::vector<double>& box_width,
              const std::string& algo_name,
              std::optional<double> range)
      : UnconstrainedPlannerBase(lb, ub, ineq_cst, max_is_valid_call, box_width)
  {
    const auto algo = get_algorithm(algo_name, range);
    setup_->setPlanner(algo);
  }
};
 
struct ERTConnectPlanner : public UnconstrainedPlannerBase {
  ERTConnectPlanner(const std::vector<double>& lb,
                    const std::vector<double>& ub,
                    cst::IneqConstraintBase::Ptr ineq_cst,
                    size_t max_is_valid_call,
                    const std::vector<double>& box_width)
      : UnconstrainedPlannerBase(lb, ub, ineq_cst, max_is_valid_call, box_width)
  {
    auto ert_connect = std::make_shared<og::ERTConnect>(csi_->si_);
    setup_->setPlanner(ert_connect);
  }

  void set_heuristic(const std::vector<std::vector<double>>& points)
  {
    auto geo_path = points_to_pathgeometric(points, this->csi_->si_);
    const auto heuristic = geo_path.getStates();
    const auto ert_connect = setup_->getPlanner()->as<og::ERTConnect>();
    ert_connect->setExperience(heuristic);
  }

  void set_parameters(std::optional<double> omega_min,
                      std::optional<double> omega_max,
                      std::optional<double> eps)
  {
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

void setGlobalSeed(size_t seed) { ompl::RNG::setSeed(seed); }

void setLogLevelNone() { ompl::msg::setLogLevel(ompl::msg::LOG_NONE); }
