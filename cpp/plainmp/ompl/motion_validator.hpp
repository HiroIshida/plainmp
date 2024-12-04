/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include <ompl/base/MotionValidator.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <vector>

namespace plainmp::ompl_wrapper {

namespace ob = ompl::base;

std::vector<std::vector<int64_t>> compute_sequence_table(size_t n_element);

class CustomValidatorBase : public ob::MotionValidator {
 public:
  CustomValidatorBase(const ob::SpaceInformationPtr& si)
      : ob::MotionValidator(si) {
    s_test_ = si_->allocState()->as<ob::RealVectorStateSpace::StateType>();
  }
  ~CustomValidatorBase() override { si_->freeState(s_test_); }
  bool checkMotion(const ob::State* s1, const ob::State* s2) const;
  bool checkMotion(const ob::State* s1,
                   const ob::State* s2,
                   std::pair<ob::State*, double>& lastValid) const {
    return checkMotion(s1, s2);
  }
  virtual double determine_step_ratio(const ob::State* s1,
                                      const ob::State* s2) const = 0;

 private:
  ob::RealVectorStateSpace::StateType* s_test_;  // pre-allocated memory
};

class BoxMotionValidator : public CustomValidatorBase {
 public:
  BoxMotionValidator(const ob::SpaceInformationPtr& si,
                     std::vector<double> width);
  double determine_step_ratio(const ob::State* s1,
                              const ob::State* s2) const override;

 private:
  std::vector<double> width_;
  std::vector<double> inv_width_;
};

class EuclideanMotionValidator : public CustomValidatorBase {
 public:
  EuclideanMotionValidator(const ob::SpaceInformationPtr& si,
                           double resolution);
  double determine_step_ratio(const ob::State* s1,
                              const ob::State* s2) const override;

 private:
  double resolution_;
};

}  // namespace plainmp::ompl_wrapper
