/* Copyright (C) 2026 Hirokazu Ishida
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. See https://mozilla.org/MPL/2.0/.
 */
#include <ompl/base/ScopedState.h>
#include <ompl/base/goals/GoalStates.h>
#include <ompl/datastructures/NearestNeighborsLinear.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <array>
#include <cstdlib>
#include <iostream>
#include <new>
#include <random>
#include <vector>
#include "ompl_rrtc/plainmp_rrtc.hpp"
#include "plainmp/ompl/motion_validator.hpp"

static size_t allocations = 0, releases = 0;
void* operator new(size_t n) {
  ++allocations;
  if (void* p = std::malloc(n ? n : 1))
    return p;
  throw std::bad_alloc();
}
void* operator new[](size_t n) {
  return ::operator new(n);
}
void operator delete(void* p) noexcept {
  if (p)
    ++releases;
  std::free(p);
}
void operator delete[](void* p) noexcept {
  ::operator delete(p);
}
void operator delete(void* p, size_t) noexcept {
  ::operator delete(p);
}
void operator delete[](void* p, size_t) noexcept {
  ::operator delete(p);
}

namespace ob = ompl::base;
namespace og = ompl::geometric;
using namespace plainmp::ompl_wrapper;
using State = ob::RealVectorStateSpace::StateType;
using Trace = std::vector<std::array<double, 5>>;
void require(bool b, const char* message) {
  if (!b)
    throw std::runtime_error(message);
}

class Sampler : public ob::StateSampler {
 public:
  Sampler(const ob::StateSpace* space, unsigned seed)
      : ob::StateSampler(space), rng_(seed) {}
  void sampleUniform(ob::State* state) override {
    auto* q = state->as<State>()->values;
    q[0] = uniform_(rng_);
    q[1] = uniform_(rng_);
  }
  void sampleUniformNear(ob::State*, const ob::State*, double) override {
    throw std::runtime_error("unexpected near sampling");
  }
  void sampleGaussian(ob::State*, const ob::State*, double) override {
    throw std::runtime_error("unexpected Gaussian sampling");
  }

 private:
  std::mt19937 rng_;
  std::uniform_real_distribution<double> uniform_{-1, 1};
};

class MotionValidator : public ob::MotionValidator {
 public:
  MotionValidator(const ob::SpaceInformationPtr& si, Trace& trace)
      : ob::MotionValidator(si), trace_(trace) {}
  bool checkMotion(const ob::State* a, const ob::State* b) const override {
    auto x = a->as<State>()->values, y = b->as<State>()->values;
    trace_.push_back({1, x[0], x[1], y[0], y[1]});
    const size_t steps = std::ceil(si_->distance(a, b) / 0.01);
    double data[2];
    State state;
    state.values = data;
    for (size_t i = 1; i <= std::max(size_t{1}, steps); ++i) {
      si_->getStateSpace()->interpolate(
          a, b, double(i) / std::max(size_t{1}, steps), &state);
      if (!si_->isValid(&state))
        return false;
    }
    return true;
  }
  bool checkMotion(const ob::State*,
                   const ob::State*,
                   std::pair<ob::State*, double>&) const override {
    throw std::runtime_error("unexpected last valid overload");
  }

 private:
  Trace& trace_;
};

ob::SpaceInformationPtr makeSpace(unsigned seed,
                                  Trace& trace,
                                  bool sealed = false) {
  auto space = std::make_shared<ob::RealVectorStateSpace>(2);
  ob::RealVectorBounds bounds(2);
  bounds.setLow(-1);
  bounds.setHigh(1);
  space->setBounds(bounds);
  space->setStateSamplerAllocator([seed](const ob::StateSpace* s) {
    return std::make_shared<Sampler>(s, seed);
  });
  auto si = std::make_shared<ob::SpaceInformation>(space);
  si->setStateValidityChecker([&trace, sealed](const ob::State* state) {
    auto q = state->as<State>()->values;
    trace.push_back({0, q[0], q[1], 0, 0});
    return std::abs(q[0]) > .1 || (!sealed && std::abs(q[1]) > .7);
  });
  si->setMotionValidator(std::make_shared<MotionValidator>(si, trace));
  si->setup();
  return si;
}

void compareReference() {
  // Feed exactly the same stream to both planners. Linear NN gives OMPL the
  // same first-insertion tie rule as our exact KD-tree. Compare every validity
  // call, directed motion check and returned waypoint, not merely success.
  for (double range : {.2, 2.0})
    for (unsigned seed = 0; seed < 20; ++seed) {
      Trace expected, actual;
      expected.reserve(1000000);
      actual.reserve(1000000);
      auto si = makeSpace(seed, expected);
      auto problem = std::make_shared<ob::ProblemDefinition>(si);
      ob::ScopedState<> a(si), b(si);
      a[0] = -.8;
      a[1] = 0;
      b[0] = .8;
      b[1] = 0;
      problem->setStartAndGoalStates(a, b);
      og::RRTConnect reference(si);
      reference.setRange(range);
      reference.setNearestNeighbors<ompl::NearestNeighborsLinear>();
      reference.setProblemDefinition(problem);
      reference.setup();
      const auto stop = ob::plannerNonTerminatingCondition();
      require(reference.solve(stop) == ob::PlannerStatus::EXACT_SOLUTION,
              "reference failed");
      auto fast_si = makeSpace(seed, actual);
      PlainmpRRTCSettings settings;
      settings.max_samples = 10000;
      PlainmpRRTC fast(fast_si, settings);
      fast.setRange(range);
      const size_t before = allocations, freed_before = releases;
      const auto status =
          fast.solveFixed(a->as<State>()->values, b->as<State>()->values, stop);
      require(allocations == before && releases == freed_before,
              "allocation/release in first search or path reconstruction");
      require(status == ob::PlannerStatus::EXACT_SOLUTION,
              "new planner failed");
      require(actual.size() == expected.size(),
              "validation trace lengths differ");
      for (size_t i = 0; i < actual.size(); ++i)
        for (size_t j = 0; j < 5; ++j)
          if (std::abs(actual[i][j] - expected[i][j]) >= 1e-12) {
            std::cerr << "seed=" << seed << " range=" << range << " entry=" << i
                      << " field=" << j << " expected=" << expected[i][j]
                      << " actual=" << actual[i][j] << "\n";
            throw std::runtime_error("validation trace differs from OMPL");
          }
      auto path = std::dynamic_pointer_cast<og::PathGeometric>(
          problem->getSolutionPath());
      require(fast.pathSize() == path->getStateCount(), "path sizes differ");
      for (size_t i = 0; i < fast.pathSize(); ++i)
        for (size_t d = 0; d < 2; ++d)
          require(std::abs(fast.pathPoint(i)[d] -
                           path->getState(i)->as<State>()->values[d]) < 1e-12,
                  "path differs from OMPL");
      actual.clear();
      const size_t before_reuse = allocations, freed_reuse = releases;
      require(fast.solveFixed(a->as<State>()->values, b->as<State>()->values,
                              stop) == ob::PlannerStatus::EXACT_SOLUTION,
              "reuse failed");
      require(allocations == before_reuse && releases == freed_reuse,
              "allocation/release on reuse");
    }
}

void limitsAndAdapter() {
  Trace trace;
  trace.reserve(1000000);
  auto si = makeSpace(42, trace);
  double a[2] = {-.8, 0}, b[2] = {.8, 0};
  const auto stop = ob::plannerNonTerminatingCondition();
  for (size_t capacity : {2, 3, 4, 129, 1024}) {
    PlainmpRRTCSettings settings;
    settings.max_samples = capacity;
    auto sealed = makeSpace(1, trace, true);
    PlainmpRRTC fast(sealed, settings);
    fast.setRange(.2);
    trace.clear();
    auto n = allocations, f = releases;
    require(fast.solveFixed(a, b, stop) == ob::PlannerStatus::TIMEOUT,
            "capacity exhaustion should fail");
    require(fast.nodeCount() == capacity && fast.pathSize() == 0,
            "incorrect capacity handling");
    require(allocations == n && releases == f,
            "allocation at capacity exhaustion");
  }
  PlainmpRRTC fast(si);
  auto problem = std::make_shared<ob::ProblemDefinition>(si);
  ob::ScopedState<> start(si), end(si);
  start[0] = a[0];
  start[1] = a[1];
  end[0] = b[0];
  end[1] = b[1];
  problem->addStartState(start);
  auto goals = std::make_shared<ob::GoalStates>(si);
  goals->addState(end);
  end[1] = .8;
  goals->addState(end);
  problem->setGoal(goals);
  fast.setProblemDefinition(problem);
  fast.setup();
  for (int i = 0; i < 3; ++i) {
    fast.clear();
    problem->clearSolutionPaths();
    trace.clear();
    require(fast.solve(stop) == ob::PlannerStatus::EXACT_SOLUTION,
            "OMPL adapter failed");
    require(
        std::dynamic_pointer_cast<og::PathGeometric>(problem->getSolutionPath())
            ->check(),
        "invalid adapter path");
  }
  const auto immediate = ob::plannerAlwaysTerminatingCondition();
  require(fast.solveFixed(a, b, immediate) == ob::PlannerStatus::TIMEOUT &&
              fast.pathSize() == 0,
          "stale solution after timeout");
  a[0] = 2;
  require(fast.solveFixed(a, b, stop) == ob::PlannerStatus::INVALID_START,
          "out of bounds start accepted");
}

void defaultSamplerAndValidators() {
  for (bool box : {false, true}) {
    auto space = std::make_shared<ob::RealVectorStateSpace>(2);
    ob::RealVectorBounds bounds(2);
    bounds.setLow(-1);
    bounds.setHigh(1);
    space->setBounds(bounds);
    auto si = std::make_shared<ob::SpaceInformation>(space);
    si->setStateValidityChecker([](const ob::State* s) {
      const auto* q = s->as<State>()->values;
      return std::abs(q[0]) > .1 || std::abs(q[1]) > .7;
    });
    if (box)
      si->setMotionValidator(std::make_shared<BoxMotionValidator>(
          si, std::vector<double>{.01, .01}));
    else
      si->setMotionValidator(
          std::make_shared<EuclideanMotionValidator>(si, .01));
    si->setup();
    PlainmpRRTC fast(si);
    fast.setRange(.2);
    double a[2] = {-.8, 0}, b[2] = {.8, 0};
    const auto stop = ob::timedPlannerTerminationCondition(2.0);
    auto n = allocations, f = releases;
    const auto status = fast.solveFixed(a, b, stop);
    require(allocations == n && releases == f,
            "default sampler/validator allocated during search");
    require(status == ob::PlannerStatus::EXACT_SOLUTION,
            "default sampler/validator failed");
  }
}

int main() {
  ompl::msg::setLogLevel(ompl::msg::LOG_NONE);
  compareReference();
  defaultSamplerAndValidators();
  limitsAndAdapter();
  std::cout << "PlainmpRRTC: OMPL traces match; search/reconstruction "
               "new/delete counts are zero\n";
}
