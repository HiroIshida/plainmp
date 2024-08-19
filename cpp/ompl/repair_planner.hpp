#include <optional>

#include "ompl/geometric/PathGeometric.h"
#include "ompl/geometric/planners/experience/LightningRetrieveRepair.h"

namespace og = ompl::geometric;
namespace ob = ompl::base;

class LightningRetrieveRepairWrap : public og::LightningRetrieveRepair
{
 public:
  using og::LightningRetrieveRepair::LightningRetrieveRepair;
  ob::PlannerStatus solve(const ob::PlannerTerminationCondition &ptc);

  std::optional<og::PathGeometric> trajectory_heuristic_;
};
