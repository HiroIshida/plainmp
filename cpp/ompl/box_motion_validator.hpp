#include <ompl/base/MotionValidator.h>
#include <ompl/base/SpaceInformation.h>
#include <vector>

namespace plainmp::ompl_wrapper {

namespace ob = ompl::base;

class BoxMotionValidator : public ob::MotionValidator {
 public:
  BoxMotionValidator(const ob::SpaceInformationPtr& si,
                     std::vector<double> width);

  bool checkMotion(const ob::State* s1, const ob::State* s2) const;
  bool checkMotion(const ob::State* s1,
                   const ob::State* s2,
                   std::pair<ob::State*, double>& lastValid) const;

 private:
  std::vector<double> width_;
  std::vector<double> inv_width_;
};

}  // namespace plainmp::ompl_wrapper
