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
#include <Eigen/Geometry>

namespace plainmp::kinematics {

enum class RotAxis { NoRotation, PureX, PureY, PureZ, General };

template <typename Scalar>
void multiply_quaternions(const Eigen::Quaternion<Scalar>& q1,
                          const Eigen::Quaternion<Scalar>& q2,
                          Eigen::Quaternion<Scalar>& result,
                          RotAxis type1,
                          RotAxis type2);

extern template void multiply_quaternions<float>(
    const Eigen::Quaternion<float>& q1,
    const Eigen::Quaternion<float>& q2,
    Eigen::Quaternion<float>& result,
    RotAxis type1,
    RotAxis type2);

extern template void multiply_quaternions<double>(
    const Eigen::Quaternion<double>& q1,
    const Eigen::Quaternion<double>& q2,
    Eigen::Quaternion<double>& result,
    RotAxis type1,
    RotAxis type2);

}  // namespace plainmp::kinematics
