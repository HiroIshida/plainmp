#include <ompl/geometric/planners/est/BiEST.h>
#include <ompl/geometric/planners/est/EST.h>
#include <ompl/geometric/planners/informedtrees/AITstar.h>
#include <ompl/geometric/planners/informedtrees/BITstar.h>
#include <ompl/geometric/planners/kpiece/BKPIECE1.h>
#include <ompl/geometric/planners/kpiece/LBKPIECE1.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <optional>
#include "unidirectional_modified.hpp"

namespace ocustom = ompl::custom;
namespace ob = ompl::base;
namespace og = ompl::geometric;

template <typename T>
std::shared_ptr<T> create_algorithm(const ob::SpaceInformationPtr si,
                                    std::optional<double> range) {
  auto algo = std::make_shared<T>(si);
  if (range) {
    algo->setRange(*range);
  }
  return algo;
}

std::shared_ptr<ob::Planner> get_algorithm(const std::string& name,
                                           const ob::SpaceInformationPtr& si,
                                           std::optional<double> range) {
  if (name.compare("BKPIECE1") == 0) {
    return create_algorithm<og::BKPIECE1>(si, range);
  } else if (name.compare("LBKPIECE1") == 0) {
    return create_algorithm<og::LBKPIECE1>(si, range);
  } else if (name.compare("KPIECE1") == 0) {
    return create_algorithm<ocustom::KPIECE1Modified>(si, range);
  } else if (name.compare("RRT") == 0) {
    return create_algorithm<ocustom::RRTModified>(si, range);
  } else if (name.compare("RRTConnect") == 0) {
    return create_algorithm<og::RRTConnect>(si, range);
  } else if (name.compare("RRTstar") == 0) {
    return create_algorithm<og::RRTstar>(si, range);
  } else if (name.compare("EST") == 0) {
    return create_algorithm<og::EST>(si, range);
  } else if (name.compare("BiEST") == 0) {
    return create_algorithm<og::BiEST>(si, range);
  } else if (name.compare("AITstar") == 0) {
    return std::make_shared<og::AITstar>(si);
  } else if (name.compare("AITstarStop") == 0) {
    return std::make_shared<og::AITstar>(si);
  } else if (name.compare("BITstar") == 0) {
    return std::make_shared<og::BITstar>(si);
  } else if (name.compare("BITstarStop") == 0) {
    auto bit = std::make_shared<og::BITstar>(si);
    bit->setStopOnSolnImprovement(true);
    return bit;
  }
  throw std::runtime_error("algorithm " + name + " is not supported");
}
