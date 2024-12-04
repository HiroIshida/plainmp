/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "motion_validator.hpp"
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include "sequence_table.hpp"

namespace plainmp::ompl_wrapper {

bool CustomValidatorBase::checkMotion(const ob::State* s1,
                                      const ob::State* s2) const {
  const double step_ratio = determine_step_ratio(s1, s2);
  if (step_ratio == std::numeric_limits<double>::infinity()) {
    return true;
  }
  /* double check the logic of the following code
   * If step_ratio = 0.15, then
   * n_test = floor(1/0.15) + 2 = 8
   * TABLE_SEQUENCE[7] = [0, 7, 4, 2, 6, 5, 3, 1, 0, 0, 0, ...]
   * This first index is already checked in the algrithm side (travel_rate = 0)
   * second index is checked directly by isValid(s2) (travel_rate = 1)
   * In the for loop, the travel_rate is computed in the following order
   * i=2: travel_rate = 0.15 * 4 = 0.6
   * i=3: travel_rate = 0.15 * 2 = 0.3
   * i=4: travel_rate = 0.15 * 6 = 0.9
   * i=5: travel_rate = 0.15 * 5 = 0.75
   * i=6: travel_rate = 0.15 * 3 = 0.45
   * i=7: travel_rate = 0.15 * 1 = 0.15
   * so, travel rates of [0, 1.0, 0.6, 0.3, 0.9, 0.75, 0.45, 0.15] are checked
   *
   * TABLE_SEQUENCE is precomputed for n_test <= 128
   * and SEQUENCE_TABLE size is 128
   * if n_test = 128, SEQUENCE_TABLE.size() + 1 = 129, then proceed
   * if n_test = 129, SEQUENCE_TABLE.size() + 1 = 130, then use naive method
   */

  const auto space = si_->getStateSpace();
  size_t n_test = std::floor(1 / step_ratio) + 2;  // including start and end
  if (n_test < SEQUENCE_TABLE.size() + 1) {
    // TABLE[i] for i+1 steps because n_test = 0 never happens
    auto& sequence = SEQUENCE_TABLE[n_test - 1];
    // NOTE: OMPL's algorithm assumes that the first state is valid
    // e.g., see comment in ompl::base::DiscreteMotionValidator::checkMotion of
    // https://github.com/ompl/ompl/blob/main/src/ompl/base/src/DiscreteMotionValidator.cpp
    if (!si_->isValid(s2)) {
      // This check corresponds to sequence[1] (which must be the end of the
      // path) omit this from for loop to avoid unnecessary clamp operation
      return false;
    }
    // start from 2
    for (size_t i = 2; i < n_test; i++) {
      double travel_rate = sequence[i] * step_ratio;
      space->interpolate(s1, s2, travel_rate, s_test_);
      if (!si_->isValid(s_test_)) {
        return false;
      }
    }
    return true;
  } else {
    if (!si_->isValid(s2)) {
      return false;
    }
    for (size_t i = 1; i < n_test - 1; i++) {
      double travel_rate = i * step_ratio;
      space->interpolate(s1, s2, travel_rate, s_test_);
      if (!si_->isValid(s_test_)) {
        return false;
      }
    }
    return true;
  }
}

BoxMotionValidator::BoxMotionValidator(const ob::SpaceInformationPtr& si,
                                       std::vector<double> width)
    : CustomValidatorBase(si), width_(width) {
  for (size_t i = 0; i < width.size(); ++i) {
    inv_width_.push_back(1.0 / width[i]);
  }
}

double BoxMotionValidator::determine_step_ratio(const ob::State* s1,
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
    return std::numeric_limits<double>::infinity();
  }
  return width_[longest_idx] / std::abs(diff_longest_axis);
}

EuclideanMotionValidator::EuclideanMotionValidator(
    const ob::SpaceInformationPtr& si,
    double resolution)
    : CustomValidatorBase(si), resolution_(resolution) {}

double EuclideanMotionValidator::determine_step_ratio(
    const ob::State* s1,
    const ob::State* s2) const {
  const auto rs1 = s1->as<ob::RealVectorStateSpace::StateType>();
  const auto rs2 = s2->as<ob::RealVectorStateSpace::StateType>();
  double dist = si_->distance(s1, s2);
  double step_ratio = resolution_ / dist;
  return step_ratio;
}

}  // namespace plainmp::ompl_wrapper
