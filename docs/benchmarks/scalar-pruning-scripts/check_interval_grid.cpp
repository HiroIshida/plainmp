#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

int main() {
  std::mt19937_64 rng(982451653);
  size_t tested = 0;
  for (int k=0; k<500000; ++k) {
    const int grid = 2 + rng()%125;
    const double inverse_step = grid + std::generate_canonical<double,53>(rng);
    const double step = 1/inverse_step;
    const int n = std::floor(1/step)+2;
    if (n>128) continue;
    const double rate = (rng()%4 == 0) ? 1.0 : (1+rng()%(n-2))*step;
    double inner = std::generate_canonical<double,53>(rng);
    if (k%3 == 0) {
      inner = std::abs((1+rng()%(n-2))*step-rate);
      inner = std::nextafter(inner, k%2 ? 0.0 : INFINITY);
    }
    std::array<bool,129> expected{}, actual{};
    for (int j=1; j+1<n; ++j)
      if (std::abs(j*step-rate)<=inner) expected[j]=true;
    const int first=std::max(1,static_cast<int>((rate-inner)*(1/step))-1);
    const int last=std::min(n-2,static_cast<int>((rate+inner)*(1/step))+1);
    for (int j=first; j<=last; ++j)
      if (std::abs(j*step-rate)<=inner) actual[j]=true;
    assert(actual==expected);
    ++tested;
  }
  std::cout << tested << " interval grid sets match exactly\n";
}
