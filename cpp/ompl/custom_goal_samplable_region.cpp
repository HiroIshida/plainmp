#include "custom_goal_samplable_region.hpp"
#include <ompl/base/spaces/RealVectorStateSpace.h>

namespace plainmp::ompl_wrapper {

CustomGoalSamplableRegion::CustomGoalSamplableRegion(
    const ob::SpaceInformationPtr& si,
    const GoalSamplerFn& sampler,
    std::optional<size_t> max_sample_count)
    : ob::GoalSampleableRegion(si),
      sampler_(sampler),
      max_sample_count_(max_sample_count),
      sample_count_(0),
      round_robin_idx_(0) {
  if (max_sample_count_) {
    size_t n_dof = si->getStateSpace()->getDimension();
    past_samples_ = std::vector<double>(n_dof * *max_sample_count_);
  }
}

void CustomGoalSamplableRegion::sampleGoal(ob::State* st) const {
  size_t n_dof = si_->getStateSpace()->getDimension();
  if (max_sample_count_ && sample_count_ >= *max_sample_count_) {
    const double* begin = past_samples_.data() + n_dof * round_robin_idx_;
    std::copy(begin, begin + n_dof,
              st->as<ob::RealVectorStateSpace::StateType>()->values);
    round_robin_idx_ = (round_robin_idx_ + 1) % *max_sample_count_;
  } else {
    std::vector<double> vec = sampler_();
    auto rs = st->as<ob::RealVectorStateSpace::StateType>();
    std::copy(vec.begin(), vec.end(), rs->values);
    if (max_sample_count_) {
      std::copy(vec.begin(), vec.end(),
                past_samples_.data() + n_dof * sample_count_);
    }
    sample_count_++;
  }
}

}  // namespace plainmp::ompl_wrapper
