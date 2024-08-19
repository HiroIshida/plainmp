#include "repair_planner.hpp"

#include <optional>
#include <stdexcept>

#include "ompl/util/Time.h"

namespace og = ompl::geometric;
namespace ob = ompl::base;

ob::PlannerStatus LightningRetrieveRepairWrap::solve(const ob::PlannerTerminationCondition &ptc)
{
  if (this->trajectory_heuristic_ == std::nullopt) {
    throw std::runtime_error("trajectory heuristic must be set beforehand");
  }

  bool solved = false;

  // Restart the Planner Input States so that the first start and goal state can be fetched
  pis_.restart();

  const ob::State *startState = pis_.nextStart();
  const ob::State *goalState = pis_.nextGoal(ptc);

  // Error check start/goal states
  if ((startState == nullptr) || (goalState == nullptr)) {
    OMPL_ERROR("LightningRetrieveRepair: Start or goal states are null");
    return ob::PlannerStatus::UNRECOGNIZED_GOAL_TYPE;
  }

  auto primaryPath(std::make_shared<og::PathGeometric>(si_));
  primaryPath->append(startState);
  for (size_t i = 0; i < this->trajectory_heuristic_->getStateCount(); ++i) {
    const auto &state = this->trajectory_heuristic_->getState(i);
    primaryPath->append(state);
  }
  primaryPath->append(goalState);

  // Repair chosen path
  if (!repairPath(ptc, *primaryPath)) {
    OMPL_INFORM("LightningRetrieveRepair: repairPath failed or aborted");
    return ob::PlannerStatus::ABORT;
  }

  // Smooth the result
  if (smoothingEnabled_) {
    OMPL_INFORM("LightningRetrieveRepair solve: Simplifying solution (smoothing)...");
    ompl::time::point simplifyStart = ompl::time::now();
    std::size_t numStates = primaryPath->getStateCount();
    psk_->simplify(*primaryPath, ptc);
    double simplifyTime = ompl::time::seconds(ompl::time::now() - simplifyStart);
    OMPL_INFORM(
        "LightningRetrieveRepair: Path simplification took %f seconds and removed %d states",
        simplifyTime,
        numStates - primaryPath->getStateCount());
  }

  // Finished
  pdef_->addSolutionPath(primaryPath, false, 0., getName());
  solved = true;
  return {solved, false};
}
