// Run through CMake with -DPLAINMP_BUILD_TESTS=ON, then ctest --test-dir
// build/native.
#include <ompl/base/ScopedState.h>
#include <ompl/base/goals/GoalStates.h>
#include <ompl/geometric/PathGeometric.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <new>
#include <random>
#include <stdexcept>
#include <vector>
#include "plainmp/ompl/batch_nearest.hpp"
#include "plainmp/ompl/vamp_rrtc.hpp"

static size_t allocations = 0;
void* operator new(size_t size) {
  ++allocations;
  if (void* p = std::malloc(size ? size : 1))
    return p;
  throw std::bad_alloc();
}
void* operator new[](size_t size) {
  return ::operator new(size);
}
void operator delete(void* p) noexcept {
  std::free(p);
}
void operator delete[](void* p) noexcept {
  std::free(p);
}
void operator delete(void* p, size_t) noexcept {
  std::free(p);
}
void operator delete[](void* p, size_t) noexcept {
  std::free(p);
}

void require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

void test_nearest() {
  using plainmp::ompl_wrapper::BatchNearest;
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> uniform(-1, 1);
  for (size_t dim : {1, 2, 7, 8, 14}) {
    constexpr size_t capacity = 4096;
    std::vector<double> points(capacity * dim), query(dim);
    for (double& x : points)
      x = uniform(rng);
    BatchNearest tree, linear;
    tree.allocate(points.data(), dim, capacity);
    linear.allocate(points.data(), dim, capacity, false);
    require(tree.nearest(query.data()).first == BatchNearest::none &&
                linear.nearest(query.data()).first == BatchNearest::none,
            "empty nearest");
    for (bool duplicates : {false, true}) {
      tree.clear();
      linear.clear();
      if (duplicates)
        std::fill(points.begin(), points.end(), 0.0);
      const size_t before = allocations;
      for (size_t i = 0; i < capacity; ++i) {
        tree.insert(i);
        linear.insert(i);
        if (i % 37 != 0 && i != capacity - 1)
          continue;
        for (double& x : query)
          x = uniform(rng);
        const auto actual = tree.nearest(query.data());
        require(actual == linear.nearest(query.data()),
                "KD-tree/linear mismatch");
        double best = std::numeric_limits<double>::infinity();
        size_t index = 0;
        for (size_t j = 0; j <= i; ++j) {
          double distance = 0;
          for (size_t d = 0; d < dim; ++d) {
            const double delta = points[j * dim + d] - query[d];
            distance += delta * delta;
          }
          if (distance < best) {
            best = distance;
            index = j;
          }
        }
        require(actual.first == index && actual.second == best,
                "nearest disagrees with brute force");
      }
      require(allocations == before, "allocation inside nearest insert/query");
    }
  }
}

void test_planner() {
  namespace ob = ompl::base;
  using State = ob::RealVectorStateSpace::StateType;
  using namespace plainmp::ompl_wrapper;
  auto space = std::make_shared<ob::RealVectorStateSpace>(2);
  ob::RealVectorBounds bounds(2);
  bounds.setLow(-1);
  bounds.setHigh(1);
  space->setBounds(bounds);
  auto si = std::make_shared<ob::SpaceInformation>(space);
  // A wall with a gap around either end; direct start-goal motion is blocked.
  si->setStateValidityChecker([](const ob::State* s) {
    auto q = s->as<State>()->values;
    return std::abs(q[0]) > 0.1 || std::abs(q[1]) > 0.7;
  });
  si->setStateValidityCheckingResolution(0.001);
  si->setup();
  for (bool start_first : {true, false}) {
    VampRRTCSettings settings;
    settings.start_tree_first = start_first;
    settings.max_samples = 4096;
    settings.max_iterations = 10000;
    auto planner = std::make_shared<VampRRTC>(si);
    planner->setRange(0.2);
    planner->setSettings(settings);
    for (int trial = 0; trial < 4; ++trial) {
      planner->clear();
      // Also exercise changing the backend on an already allocated planner.
      settings.use_kdtree = trial % 2 == 0;
      planner->setSettings(settings);
      auto problem = std::make_shared<ob::ProblemDefinition>(si);
      ob::ScopedState<> start(space), goal(space);
      start[0] = -0.8;
      start[1] = 0;
      goal[0] = 0.8;
      goal[1] = 0;
      problem->addStartState(start);
      auto goals = std::make_shared<ob::GoalStates>(si);
      goals->addState(goal);
      goal[1] = 0.1;
      goals->addState(goal);
      problem->setGoal(goals);
      planner->setProblemDefinition(problem);
      planner->setup();
      require(planner->solve(ob::timedPlannerTerminationCondition(2.0)) ==
                  ob::PlannerStatus::EXACT_SOLUTION,
              "wall planning failed");
      auto path = std::dynamic_pointer_cast<ompl::geometric::PathGeometric>(
          problem->getSolutionPath());
      require(path->check(), "invalid wall path");
      require(space->equalStates(path->getState(0), start.get()),
              "wrong start");
      const auto* last = path->getState(path->getStateCount() - 1)->as<State>();
      require(last->values[0] == 0.8 &&
                  (last->values[1] == 0 || last->values[1] == 0.1),
              "wrong goal");
    }
  }
}

int main() {
  test_nearest();
  test_planner();
  std::cout << "VampRRTC native tests passed\n";
}
