#include "algorithm_selector.hpp"
#include <ompl/geometric/planners/est/BiEST.h>
#include <ompl/geometric/planners/est/EST.h>
#include <ompl/geometric/planners/kpiece/BKPIECE1.h>
#include <ompl/geometric/planners/kpiece/LBKPIECE1.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <optional>
#include "unidirectional_modified.hpp"

namespace plainmp::ompl_wrapper {

namespace ocustom = ompl::custom;
namespace ob = ompl::base;
namespace og = ompl::geometric;

template <typename T>
std::shared_ptr<T> create_with_range(const ob::SpaceInformationPtr& si,
                                     std::optional<double> range) {
  auto algo = std::make_shared<T>(si);
  if (range) {
    algo->setRange(*range);
  }
  return algo;
}

std::shared_ptr<ompl::base::Planner> get_algorithm(
    const std::string& name,
    const ompl::base::SpaceInformationPtr& si,
    std::optional<double> range) {
  // clang-format off
    if (name == "BKPIECE1") return create_with_range<og::BKPIECE1>(si, range);
    if (name == "LBKPIECE1") return create_with_range<og::LBKPIECE1>(si, range);
    if (name == "KPIECE1") return create_with_range<ocustom::KPIECE1Modified>(si, range);
    if (name == "RRT") return create_with_range<ocustom::RRTModified>(si, range);
    if (name == "RRTConnect") return create_with_range<og::RRTConnect>(si, range);
    if (name == "RRTstar") return create_with_range<og::RRTstar>(si, range);
    if (name == "EST") return create_with_range<og::EST>(si, range);
    if (name == "BiEST") return create_with_range<og::BiEST>(si, range);
  // clang-format on
  throw std::runtime_error(std::string("Algorithm '") + std::string(name) +
                           "' is not supported");
}

}  // namespace plainmp::ompl_wrapper
