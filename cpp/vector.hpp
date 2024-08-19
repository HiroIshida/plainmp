#include <vectorclass.h>
#include <vector>

class SimdVector {
 public:
  SimdVector(size_t size) : vecs_4d_((size + 3) / 4) {}

 private:
  std::vector<Vec4d> vecs_4d_;
}
