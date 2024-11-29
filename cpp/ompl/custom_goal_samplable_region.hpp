#include <ompl/base/SpaceInformation.h>
#include <ompl/base/goals/GoalSampleableRegion.h>
#include <memory>
#include <optional>

namespace plainmp::ompl_wrapper {

namespace ob = ompl::base;

using GoalSamplerFn = std::function<std::vector<double>()>;

class CustomGoalSamplableRegion : public ob::GoalSampleableRegion {
 public:
  using Ptr = std::shared_ptr<CustomGoalSamplableRegion>;
  CustomGoalSamplableRegion(
      const ob::SpaceInformationPtr& si,
      const GoalSamplerFn& sampler,
      std::optional<size_t> max_sample_count = std::nullopt);
  void sampleGoal(ob::State* st) const override;
  unsigned int maxSampleCount() const override {
    return std::numeric_limits<unsigned int>::max();
  }
  double distanceGoal(const ob::State* st) const override {
    // NOTE: distnace goal is 0.0 because we assume that
    // the sample solve the collision free IK with joint limit
    // in python side and the goal is the solution
    return 0.0;
  }

 private:
  GoalSamplerFn sampler_;
  std::optional<size_t> max_sample_count_;
  mutable size_t sample_count_;
  mutable std::vector<double> past_samples_;  // note: contiguous memory
  mutable size_t round_robin_idx_;
};
}  // namespace plainmp::ompl_wrapper
