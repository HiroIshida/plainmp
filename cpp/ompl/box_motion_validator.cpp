#include <ompl/base/spaces/RealVectorStateSpace.h>
#include "box_motion_validator.hpp"

BoxMotionValidator::BoxMotionValidator(const ob::SpaceInformationPtr& si,
                                       std::vector<double> width)
    : ob::MotionValidator(si), width_(width) {
  // NOTE: precompute inv width, because devide is more expensive than
  // multiply
  for (size_t i = 0; i < width.size(); ++i) {
    inv_width_.push_back(1.0 / width[i]);
  }
}

bool BoxMotionValidator::checkMotion(const ob::State* s1,
                                     const ob::State* s2) const {
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
  const auto s_test =
      si_->allocState()->as<ob::RealVectorStateSpace::StateType>();

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

bool BoxMotionValidator::checkMotion(
    const ob::State* s1,
    const ob::State* s2,
    std::pair<ob::State*, double>& lastValid) const {
  return checkMotion(s1, s2);
}
