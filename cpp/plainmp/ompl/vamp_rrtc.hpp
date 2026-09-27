/* Copyright (C) 2026 Hirokazu Ishida
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. See https://mozilla.org/MPL/2.0/.
 *
 * VAMP-inspired RRTC, adapted for plainmp's double precision state space.
 * Algorithm source: KavrakiLab/vamp, planning/rrtc.hh, revision
 * f6de6a72e0725ba08e0829f0a34261a3f465aed7 (Apache-2.0).
 * Modifications (2026): OMPL integration, pooled storage and an independent
 * batch KD-tree, bounded termination, exact endpoint handling.
 * See third/vamp/LICENSE for the upstream license.
 */
#pragma once

#include <ompl/base/Planner.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/util/RandomNumbers.h>
#include <optional>
#include <vector>
#include "plainmp/ompl/batch_nearest.hpp"

namespace plainmp::ompl_wrapper {

struct VampRRTCSettings {
  bool use_kdtree = true;
  // VAMP's examples/benchmarks use Halton with bases 3, 5, 7, ... .
  bool use_halton = true;
  size_t halton_skip = 0;
  bool dynamic_domain = true;
  double radius = 4.0;
  double alpha = 0.0001;
  double min_radius = 1.0;
  bool balance = true;
  double tree_ratio = 1.0;
  size_t max_iterations = 100000;
  size_t max_samples = 100000;
  bool start_tree_first = true;
};

// RealVectorStateSpace and reversible geometric motion validity only.
// OMPL states are stack views into our pool. OMPL owns only the final path.
class VampRRTC : public ompl::base::Planner {
 public:
  explicit VampRRTC(const ompl::base::SpaceInformationPtr& si);
  void setup() override;
  void clear() override;
  ompl::base::PlannerStatus solve(
      const ompl::base::PlannerTerminationCondition& ptc) override;
  void setRange(double range);
  void setSettings(const VampRRTCSettings& settings);

 private:
  using Index = BatchNearest::Index;
  using State = ompl::base::RealVectorStateSpace::StateType;
  struct Node {
    Index parent;
    double radius;
  };
  double* point(Index i) { return points_.get() + i * dimension_; }
  Index insert(BatchNearest& tree, const double* q, Index parent);
  bool motion(Index from, const double* to);
  Index root(Index i) const;
  void solution(Index a, Index b, bool a_is_start);

  VampRRTCSettings settings_;
  double range_ = 2.0;
  size_t dimension_ = 0, capacity_ = 0;
  Index size_ = 0;
  std::unique_ptr<double[]> points_, scratch_;
  std::unique_ptr<Node[]> nodes_;
  BatchNearest start_tree_, goal_tree_;
  std::optional<ompl::RNG> rng_;
  std::vector<uint64_t> bases_, numerators_, denominators_;
};

}  // namespace plainmp::ompl_wrapper
