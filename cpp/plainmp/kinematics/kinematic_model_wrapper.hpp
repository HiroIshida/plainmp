/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <Eigen/Dense>
#include <array>
#include <string>
#include "plainmp/kinematics/kinematics.hpp"

namespace plainmp::kinematics {

using Transform = KinematicModel<double>::Transform;
using Vector7d = Eigen::Matrix<double, 7, 1>;

Vector7d pose_to_vector(const Transform& pose);
Transform vector_to_pose(const Vector7d& pose_vec);

class _KinematicModel : public KinematicModel<double> {
  // a utility class for easy binding
 public:
  using KinematicModel::KinematicModel;
  size_t add_new_link_py(const std::string& link_name,
                         const std::string& parent_name,
                         const std::array<double, 3>& position,
                         const std::array<double, 3>& rpy,
                         bool consider_rotation);

  Vector7d get_base_pose();
  void set_base_pose(const Vector7d& pose_vec);
  Vector7d debug_get_link_pose(const std::string& link_name);
};

}  // namespace plainmp::kinematics