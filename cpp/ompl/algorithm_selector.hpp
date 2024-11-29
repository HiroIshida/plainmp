#include <ompl/geometric/planners/PlannerIncludes.h>

namespace plainmp::ompl_wrapper {

std::shared_ptr<ompl::base::Planner> get_algorithm(
    const std::string& name,
    const ompl::base::SpaceInformationPtr& si,
    std::optional<double> range);

}  // namespace plainmp::ompl_wrapper
