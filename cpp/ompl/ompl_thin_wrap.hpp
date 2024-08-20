#include <ompl/base/ConstrainedSpaceInformation.h>
#include <ompl/base/Constraint.h>
#include <ompl/base/PlannerTerminationCondition.h>
#include <ompl/base/State.h>
#include <ompl/base/StateSampler.h>
#include <ompl/base/StateSpace.h>
#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/constraint/AtlasStateSpace.h>
#include <ompl/base/spaces/constraint/ConstrainedStateSpace.h>
#include <ompl/base/spaces/constraint/ProjectedStateSpace.h>
#include <ompl/base/spaces/constraint/TangentBundleStateSpace.h>
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

template <bool Constrained>
std::vector<double> state_to_vec(const ob::State* state, size_t dim)
{
  const ob::RealVectorStateSpace::StateType* rs;
  if constexpr (Constrained) {
    rs = state->as<ompl::base::ConstrainedStateSpace::StateType>()
             ->getState()
             ->as<ob::RealVectorStateSpace::StateType>();
  } else {
    rs = state->as<ob::RealVectorStateSpace::StateType>();
  }
  std::vector<double> vec(dim);
  for (size_t i = 0; i < dim; ++i) {
    vec[i] = rs->values[i];
  }
  return vec;
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

class AllPassMotionValidator : public ob::MotionValidator
{
 public:
  using ob::MotionValidator::MotionValidator;
  bool checkMotion(const ob::State* s1, const ob::State* s2) const { return true; }

  bool checkMotion(const ob::State* s1,
                   const ob::State* s2,
                   std::pair<ob::State*, double>& lastValid) const
  {
    return true;
  }
};

class BoxMotionValidator : public ob::MotionValidator
{
 public:
  BoxMotionValidator(const ob::SpaceInformationPtr& si, std::vector<double> width)
      : ob::MotionValidator(si), width_(width)
  {
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
      const double check_width = width_[idx];
      const double diff = rs2->values[idx] - rs1->values[idx];
      const double abs_scaled_diff = std::abs(diff) / check_width;
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
};

std::optional<std::vector<double>> split_geodesic_with_box(const ob::State* s1,
                                                           const ob::State* s2,
                                                           const ob::SpaceInformation* si,
                                                           const std::vector<double>& width)
{
  // returns list of states if all states are valid, otherwise return std::nulopt

  // this function involves a lot of dynamic allocation (e.g. stack, hashmap, vector, state)
  // but compard to collision detection cost, it should be negligible.
  // TODO: do performance profiling later
  // unlike non-constrianed case, the constrained case traverse on manifold
  const auto space = si->getStateSpace();
  const size_t dim = space->getDimension();

  const auto is_state_pair_inside_box = [&](const ob::State* s1, const ob::State* s2) -> bool {
    const auto vec_left = state_to_vec<true>(s1, dim);
    const auto vec_right = state_to_vec<true>(s2, dim);

    for (size_t i = 0; i < dim; ++i) {
      const double&& abs_diff_i = std::abs(vec_left[i] - vec_right[i]);
      if (abs_diff_i > width[i]) {
        return false;
      }
    }
    return true;
  };

  std::stack<std::pair<double, double>> range_stack;
  range_stack.push({0.0, 1.0});

  std::unordered_map<double, ob::State*> fraction_to_state_table;
  std::set<double> fraction_set;

  // adaptively divide the path
  while (!range_stack.empty()) {
    const auto range_pair = range_stack.top();
    range_stack.pop();
    const double left = range_pair.first;
    const double right = range_pair.second;

    fraction_set.insert(left);
    fraction_set.insert(right);

    bool is_too_small_interval = (right - left) < 1e-5;
    if (is_too_small_interval) {
      // note: although this error can be avoided by manualy setting smaller delta size, it
      // usually deteriorate the performance a lot. So, I want to stick to the default ompl
      // pamater. (default is almost always right)
      std::string message =
          "seems that css->delta (step size to traverse geodestic) is too large for the box "
          "width";
      throw std::runtime_error(message);
    }

    // interpolate state given fraction if not interpolate yet
    // note that as well as interpolation we check if the interpolated state is valid or not
    // and if not valid, returning false.
    for (double fraction : std::array<double, 2>{left, right}) {
      const bool not_evaluated_yet =
          fraction_to_state_table.find(fraction) == fraction_to_state_table.end();
      if (not_evaluated_yet) {
        const auto s_new = si->allocState();
        space->interpolate(s1, s2, fraction, s_new);
        if (!si->isValid(s_new)) {
          return {};
        }
        fraction_to_state_table.insert({fraction, s_new});
      }
    }
    assert(fraction_to_state_table.find(left) != fraction_to_state_table.end());
    assert(fraction_to_state_table.find(right) != fraction_to_state_table.end());

    // check if two states are inside the constraint box
    const ob::State* s_left = fraction_to_state_table.find(left)->second;
    const ob::State* s_right = fraction_to_state_table.find(right)->second;

    const bool require_split = !is_state_pair_inside_box(s_left, s_right);
    if (require_split) {
      const double middle = 0.5 * (left + right);
      range_stack.push({left, middle});
      range_stack.push({middle, right});
    }
  }

  std::vector<double> fractions(fraction_set.size());
  std::copy(fraction_set.begin(), fraction_set.end(), fractions.begin());
  return fractions;
}

class GeodesicBoxMotionValidator : public ob::MotionValidator
{
 public:
  GeodesicBoxMotionValidator(const ob::SpaceInformationPtr& si, std::vector<double> width)
      : ob::MotionValidator(si), width_(width)
  {
  }

  bool checkMotion(const ob::State* s1, const ob::State* s2) const
  {
    const auto states = split_geodesic_with_box(s1, s2, si_, width_);
    return (states != std::nullopt);
  }

  bool checkMotion(const ob::State* s1,
                   const ob::State* s2,
                   std::pair<ob::State*, double>& lastValid) const
  {
    return checkMotion(s1, s2);
  }

 private:
  std::vector<double> width_;
};

class ConstraintWrap : public ob::Constraint
{
 public:
  ConstraintWrap(const ConstFn& f, const ConstJacFn& jac, size_t dim_ambient, size_t dim_constraint)
      : ob::Constraint(dim_ambient, dim_constraint), f_(f), jac_(jac)
  {
  }

  void function(const Eigen::Ref<const Eigen::VectorXd>& x,
                Eigen::Ref<Eigen::VectorXd> out) const override
  {
    std::vector<double> xvec(x.data(), x.data() + x.size());
    const auto yvec = f_(xvec);
    for (size_t i = 0; i < yvec.size(); ++i) {
      out[i] = yvec[i];
    }
  }

  void jacobian(const Eigen::Ref<const Eigen::VectorXd>& x,
                Eigen::Ref<Eigen::MatrixXd> out) const override
  {
    std::vector<double> xvec(x.data(), x.data() + x.size());
    const auto yvecvec = jac_(xvec);
    for (size_t i = 0; i < yvecvec.size(); ++i) {
      for (size_t j = 0; j < yvecvec.front().size(); ++j) {
        out(i, j) = yvecvec[i][j];
      }
    }
  }

  ConstFn f_;
  ConstJacFn jac_;
};

template <bool Constrained>
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
    const auto vec = state_to_vec<Constrained>(state, dim);
    this->is_valid_call_count_++;
    return ineq_cst_->is_valid(vec);
  }

  ob::SpaceInformationPtr si_;
  cst::IneqConstraintBase::Ptr ineq_cst_;
  size_t is_valid_call_count_;
  const size_t max_is_valid_call_;
  std::vector<double> box_width_;
};

struct UnconstrianedCollisoinAwareSpaceInformation : public CollisionAwareSpaceInformation<false> {
  UnconstrianedCollisoinAwareSpaceInformation(
      const std::vector<double>& lb,
      const std::vector<double>& ub,
      cst::IneqConstraintBase::Ptr ineq_cst,
      size_t max_is_valid_call,
      const std::vector<double>& box_width)
      : CollisionAwareSpaceInformation<false>{nullptr, ineq_cst, 0, max_is_valid_call, box_width}
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

enum ConstStateType { PROJECTION, ATLAS, TANGENT };

struct ConstrainedCollisoinAwareSpaceInformation : public CollisionAwareSpaceInformation<true> {
  ConstrainedCollisoinAwareSpaceInformation(
      const ConstFn& f_const,
      const ConstJacFn& jac_const,
      const std::vector<double>& lb,
      const std::vector<double>& ub,
      cst::IneqConstraintBase::Ptr ineq_cst,
      size_t max_is_valid_call,
      const std::vector<double>& box_width,
      ConstStateType cs_type = ConstStateType::PROJECTION)
      : CollisionAwareSpaceInformation<true>{nullptr, ineq_cst, 0, max_is_valid_call, box_width}
  {
    size_t dim_ambient = lb.size();
    size_t dim_constraint = f_const(lb).size();  // detect by dummy input
    std::shared_ptr<ob::Constraint> constraint =
        std::make_shared<ConstraintWrap>(f_const, jac_const, dim_ambient, dim_constraint);
    const auto space = bound2space(lb, ub);

    ob::ConstrainedStateSpacePtr css;
    if (cs_type == ConstStateType::PROJECTION) {
      css = std::make_shared<ob::ProjectedStateSpace>(space, constraint);
    } else if (cs_type == ConstStateType::ATLAS) {
      css = std::make_shared<ob::AtlasStateSpace>(space, constraint);
    } else if (cs_type == ConstStateType::TANGENT) {
      css = std::make_shared<ob::TangentBundleStateSpace>(space, constraint);
    }
    const auto csi = std::make_shared<ob::ConstrainedSpaceInformation>(css);
    si_ = std::static_pointer_cast<ob::SpaceInformation>(csi);
    si_->setMotionValidator(std::make_shared<GeodesicBoxMotionValidator>(si_, box_width));
    si_->setup();
  }
};

template <bool Constrained>
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

    if constexpr (Constrained) {
      sstart->as<ob::ConstrainedStateSpace::StateType>()->copy(vec_start);
      sgoal->as<ob::ConstrainedStateSpace::StateType>()->copy(vec_goal);
    } else {
      auto rstart = sstart->as<ob::RealVectorStateSpace::StateType>();
      auto rgoal = sgoal->as<ob::RealVectorStateSpace::StateType>();
      std::copy(start.begin(), start.end(), rstart->values);
      std::copy(goal.begin(), goal.end(), rgoal->values);
    }
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
      if constexpr (Constrained) {
        std::runtime_error("simplify does not seem to work well in constrinaed case");
      }
      setup_->simplifySolution(fn);
    }
    const auto p = setup_->getSolutionPath().as<og::PathGeometric>();
    auto& states = p->getStates();
    const size_t dim = start.size();

    // states
    auto trajectory = std::vector<std::vector<double>>();

    if constexpr (Constrained) {
      trajectory.push_back(state_to_vec<Constrained>(states[0], dim));

      for (size_t i = 0; i < states.size() - 1; ++i) {
        const ob::SpaceInformation* si = csi_->si_.get();
        const auto fractions =
            split_geodesic_with_box(states.at(i), states.at(i + 1), si, csi_->box_width_);

        const auto space = csi_->si_->getStateSpace();
        for (size_t j = 0; j < fractions->size(); ++j) {
          const auto s_new = si->allocState();
          space->interpolate(states.at(i), states.at(i + 1), fractions->at(j), s_new);
          trajectory.push_back(state_to_vec<Constrained>(s_new, dim));
        }
      }
      OMPL_INFORM("interpolate trajectory. original %d points => interped %d points",
                  states.size(),
                  trajectory.size());
    } else {
      for (const auto& state : states) {
        trajectory.push_back(state_to_vec<Constrained>(state, dim));
      }
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

  typedef typename std::conditional<Constrained,
                                    ConstrainedCollisoinAwareSpaceInformation,
                                    UnconstrianedCollisoinAwareSpaceInformation>::type CsiType;
  std::unique_ptr<CsiType> csi_;
  std::unique_ptr<og::SimpleSetup> setup_;
};

// struct ConstrainedPlanner : public PlannerBase<true> {
//   ConstrainedPlanner(const ConstFn& f_const,
//                      const ConstJacFn& jac_const,
//                      const std::vector<double>& lb,
//                      const std::vector<double>& ub,
//                      const std::function<bool(std::vector<double>)>& is_valid,
//                      size_t max_is_valid_call,
//                      const std::vector<double>& box_width,
//                      const std::string& algo_name,
//                      std::optional<double> range,
//                      ConstStateType cs_type = ConstStateType::PROJECTION)
//       : PlannerBase<true>{nullptr, nullptr}
//   {
//     csi_ = std::make_unique<ConstrainedCollisoinAwareSpaceInformation>(
//         f_const, jac_const, lb, ub, is_valid, max_is_valid_call, box_width, cs_type);
//     const auto algo = get_algorithm(algo_name, range);
// 
//     setup_ = std::make_unique<og::SimpleSetup>(csi_->si_);
//     setup_->setPlanner(algo);
//     setup_->setStateValidityChecker([this](const ob::State* s) { return this->csi_->is_valid(s); });
//   }
// };

struct UnconstrainedPlannerBase : public PlannerBase<false> {
  UnconstrainedPlannerBase(const std::vector<double>& lb,
                           const std::vector<double>& ub,
                           cst::IneqConstraintBase::Ptr ineq_cst,
                           size_t max_is_valid_call,
                           const std::vector<double>& box_width)
      : PlannerBase<false>{nullptr, nullptr}
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

// struct LightningDBWrap {
//   LightningDBWrap(size_t n_dim)
//   {
//     auto space = ob::StateSpacePtr(new ob::RealVectorStateSpace(n_dim));
//     si = std::make_shared<ob::SpaceInformation>(space);
//     db = std::make_shared<ot::LightningDB>(space);
//   }
// 
//   void addExperience(const std::vector<std::vector<double>>& points)
//   {
//     auto pg = points_to_pathgeometric(points, si);
//     double insertion_time;
//     db->addPath(pg, insertion_time);
//   }
// 
//   std::vector<std::vector<std::vector<double>>> getExperiencedPaths()
//   {
//     const auto dim = si->getStateSpace()->getDimension();
// 
//     std::vector<std::vector<std::vector<double>>> paths;
// 
//     std::vector<ob::PlannerDataPtr> datas;
//     db->getAllPlannerDatas(datas);
//     for (const auto& data : datas) {
//       std::vector<std::vector<double>> path;
//       for (std::size_t i = 0; i < data->numVertices(); ++i) {
//         const auto vert = data->getVertex(i);
//         path.push_back(state_to_vec<false>(vert.getState(), dim));
//       }
//       paths.push_back(path);
//     }
//     return paths;
//   }
// 
//   size_t getExperiencesCount() { return db->getExperiencesCount(); }
// 
//   void save(const std::string& fileName) { db->save(fileName); }
// 
//   void load(const std::string& fileName) { db->load(fileName); }
// 
//   ob::SpaceInformationPtr si;
//   ot::LightningDBPtr db;
// };
// 
// struct LightningPlanner : public UnconstrainedPlannerBase {
//   LightningPlanner(const LightningDBWrap& dbwrap,
//                    const std::vector<double>& lb,
//                    const std::vector<double>& ub,
//                    const std::function<bool(std::vector<double>)>& is_valid,
//                    size_t max_is_valid_call,
//                    const std::vector<double>& box_width,
//                    const std::string& algo_name,
//                    std::optional<double> range)
//       : UnconstrainedPlannerBase(lb, ub, is_valid, max_is_valid_call, box_width)
//   {
//     auto repair_planner = std::make_shared<og::LightningRetrieveRepair>(csi_->si_, dbwrap.db);
//     setup_->setPlanner(repair_planner);
//   }
// };
// 
// struct LightningRepairPlanner : public UnconstrainedPlannerBase {
//   // a variant of lightning that does not use dabase. Rather, user must
//   // give the trajectory before solve() explicitely as the heuristic.
// 
//   LightningRepairPlanner(const std::vector<double>& lb,
//                          const std::vector<double>& ub,
//                          const std::function<bool(std::vector<double>)>& is_valid,
//                          size_t max_is_valid_call,
//                          const std::vector<double>& box_width,
//                          const std::string& algo_name,
//                          std::optional<double> range)
//       : UnconstrainedPlannerBase(lb, ub, is_valid, max_is_valid_call, box_width)
//   {
//     ot::LightningDBPtr db = nullptr;  // we dont use database
//     auto repair_planner = std::make_shared<LightningRetrieveRepairWrap>(csi_->si_, db);
//     repair_planner_ = repair_planner;
//     setup_->setPlanner(repair_planner);
//   }
// 
//   void set_heuristic(const std::vector<std::vector<double>>& points)
//   {
//     repair_planner_->trajectory_heuristic_ = points_to_pathgeometric(points, this->csi_->si_);
//   }
// 
//  protected:
//   std::shared_ptr<LightningRetrieveRepairWrap> repair_planner_;
// };
 
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
