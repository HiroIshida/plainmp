#include "motion_validator.hpp"
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include "sequence_table.hpp"

namespace plainmp::ompl_wrapper {

BoxMotionValidator::BoxMotionValidator(const ob::SpaceInformationPtr& si,
                                       std::vector<double> width)
    : ob::MotionValidator(si), width_(width) {
  // NOTE: precompute inv width, because devide is more expensive than
  // multiply
  for (size_t i = 0; i < width.size(); ++i) {
    inv_width_.push_back(1.0 / width[i]);
  }
  s_test_ = si_->allocState()->as<ob::RealVectorStateSpace::StateType>();
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
  const auto space = si_->getStateSpace();
  const double step_ratio = width_[longest_idx] / std::abs(diff_longest_axis);
  size_t n_test = std::floor(1 / step_ratio) + 2;  // including start and end
  if (n_test < SEQUENCE_TABLE.size() + 1) {
    // TABLE[i] for i+1 steps
    auto& sequence = SEQUENCE_TABLE[n_test - 1];
    // sequence[0] is already checked
    // sequence[1] is the end, thus
    if (!si_->isValid(rs2)) {
      return false;
    }
    // start from 2
    for (size_t i = 2; i < n_test; i++) {
      double travel_rate = sequence[i] * step_ratio;
      space->interpolate(rs1, rs2, travel_rate, s_test_);
      if (!si_->isValid(s_test_)) {
        return false;
      }
    }
    return true;
  } else {
    if (!si_->isValid(rs2)) {
      return false;
    }
    for (size_t i = 1; i < n_test - 1; i++) {
      double travel_rate = i * step_ratio;
      space->interpolate(rs1, rs2, travel_rate, s_test_);
      if (!si_->isValid(s_test_)) {
        return false;
      }
    }
    return true;
  }
}

}  // namespace plainmp::ompl_wrapper
