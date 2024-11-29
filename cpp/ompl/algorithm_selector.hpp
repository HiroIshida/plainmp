#include <ompl/geometric/planners/PlannerIncludes.h>
#include <optional>

namespace plainmp::ompl_wrapper {

std::shared_ptr<ompl::base::Planner> get_algorithm(
    const std::string& name,
    const ompl::base::SpaceInformationPtr& si,
    std::optional<double> range = std::nullopt);

}  // namespace plainmp::ompl_wrapper
