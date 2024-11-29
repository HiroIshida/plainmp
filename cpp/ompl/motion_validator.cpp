#include "motion_validator.hpp"
#include <ompl/base/spaces/RealVectorStateSpace.h>

namespace plainmp::ompl_wrapper {

std::vector<int64_t> find_farthest_sequence(size_t N) {
  std::vector<int64_t> result(N);
  std::vector<bool> used(N, false);
  used[0] = true;
  int current_size = 1;

  for (int i = 1; i < N; i++) {
    int64_t max_min_distance = -1;
    int64_t best_num = -1;

    for (int num = 0; num < N; num++) {
      if (used[num])
        continue;

      int64_t min_distance = std::numeric_limits<int64_t>::max();
      for (int j = 0; j < current_size; j++) {
        int64_t distance = std::abs(num - result[j]);
        if (distance < min_distance) {
          min_distance = distance;
        }
      }

      if (min_distance > max_min_distance ||
          (min_distance == max_min_distance && num > best_num)) {
        max_min_distance = min_distance;
        best_num = num;
      }
    }

    result[i] = best_num;
    used[best_num] = true;
    current_size++;
  }
  return result;
}

std::vector<std::vector<int64_t>> compute_sequence_table(size_t n_element) {
  std::vector<std::vector<int64_t>> result(n_element + 1);
  for (size_t i = 0; i < n_element; i++) {
    std::vector<int64_t> sequence = find_farthest_sequence(i + 1);
    result[i] = sequence;
  }
  return result;
}

BoxMotionValidator::BoxMotionValidator(const ob::SpaceInformationPtr& si,
                                       std::vector<double> width)
    : ob::MotionValidator(si), width_(width) {
  // NOTE: precompute inv width, because devide is more expensive than
  // multiply
  for (size_t i = 0; i < width.size(); ++i) {
    inv_width_.push_back(1.0 / width[i]);
  }
  s_test_ = si_->allocState()->as<ob::RealVectorStateSpace::StateType>();
  sequence_table_ = compute_sequence_table(200);
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
  auto& sequence = sequence_table_[n_test - 1];
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
}

}  // namespace plainmp::ompl_wrapper
