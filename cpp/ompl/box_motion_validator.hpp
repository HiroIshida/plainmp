#include <ompl/base/MotionValidator.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <vector>

namespace plainmp::ompl_wrapper {

namespace ob = ompl::base;

class BoxMotionValidator : public ob::MotionValidator {
 public:
  BoxMotionValidator(const ob::SpaceInformationPtr& si,
                     std::vector<double> width);
  ~BoxMotionValidator() override { si_->freeState(s_test_); }

  bool checkMotion(const ob::State* s1, const ob::State* s2) const;
  bool checkMotion(const ob::State* s1,
                   const ob::State* s2,
                   std::pair<ob::State*, double>& lastValid) const;

 private:
  std::vector<double> width_;
  std::vector<double> inv_width_;
  ob::RealVectorStateSpace::StateType* s_test_;  // pre-allocated memory
};

}  // namespace plainmp::ompl_wrapper
