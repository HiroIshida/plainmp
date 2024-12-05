/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "plainmp/kinematics/quaternion_mult.hpp"
#include <array>
#include "plainmp/kinematics/_quaternion_mult.hpp"

template <typename Scalar>
using QuatMultFn = void (*)(const Eigen::Quaternion<Scalar>&,
                            const Eigen::Quaternion<Scalar>&,
                            Eigen::Quaternion<Scalar>&);

template <typename Scalar>
using MultTable = std::array<std::array<QuatMultFn<Scalar>, 5>, 5>;

namespace plainmp::kinematics {
template <typename Scalar>
constexpr MultTable<Scalar> make_mult_table() {
  // note: we will filter out the no rotation case before using this table, so
  // we don't need to fill elments for no rotation
  MultTable<Scalar> table = {};

  // self is pure x
  table[static_cast<int>(RotAxis::PureX)][static_cast<int>(RotAxis::PureX)] =
      mult_quat_xaxis_xaxis;
  table[static_cast<int>(RotAxis::PureX)][static_cast<int>(RotAxis::PureY)] =
      mult_quat_xaxis_yaxis;
  table[static_cast<int>(RotAxis::PureX)][static_cast<int>(RotAxis::PureZ)] =
      mult_quat_xaxis_zaxis;
  table[static_cast<int>(RotAxis::PureX)][static_cast<int>(RotAxis::General)] =
      mult_quat_xaxis_general;

  // self is pure y
  table[static_cast<int>(RotAxis::PureY)][static_cast<int>(RotAxis::PureX)] =
      mult_quat_yaxis_xaxis;
  table[static_cast<int>(RotAxis::PureY)][static_cast<int>(RotAxis::PureY)] =
      mult_quat_yaxis_yaxis;
  table[static_cast<int>(RotAxis::PureY)][static_cast<int>(RotAxis::PureZ)] =
      mult_quat_yaxis_zaxis;
  table[static_cast<int>(RotAxis::PureY)][static_cast<int>(RotAxis::General)] =
      mult_quat_yaxis_general;

  // self is pure z
  table[static_cast<int>(RotAxis::PureZ)][static_cast<int>(RotAxis::PureX)] =
      mult_quat_zaxis_xaxis;
  table[static_cast<int>(RotAxis::PureZ)][static_cast<int>(RotAxis::PureY)] =
      mult_quat_zaxis_yaxis;
  table[static_cast<int>(RotAxis::PureZ)][static_cast<int>(RotAxis::PureZ)] =
      mult_quat_zaxis_zaxis;
  table[static_cast<int>(RotAxis::PureZ)][static_cast<int>(RotAxis::General)] =
      mult_quat_zaxis_general;

  // self is general
  table[static_cast<int>(RotAxis::General)][static_cast<int>(RotAxis::PureX)] =
      mult_quat_general_xaxis;
  table[static_cast<int>(RotAxis::General)][static_cast<int>(RotAxis::PureY)] =
      mult_quat_general_yaxis;
  table[static_cast<int>(RotAxis::General)][static_cast<int>(RotAxis::PureZ)] =
      mult_quat_general_zaxis;
  table[static_cast<int>(RotAxis::General)]
       [static_cast<int>(RotAxis::General)] = mult_quat_general_general;

  return table;
}

template <typename Scalar>
static constexpr auto _QUAT_MULT_TABLE = make_mult_table<Scalar>();

template <typename Scalar>
void multiply_quaternions(const Eigen::Quaternion<Scalar>& q1,
                          const Eigen::Quaternion<Scalar>& q2,
                          Eigen::Quaternion<Scalar>& result,
                          RotAxis type1,
                          RotAxis type2) {
  if (type1 == RotAxis::NoRotation) {
    result = q2;
    return;
  }
  if (type2 == RotAxis::NoRotation) {
    result = q1;
    return;
  }
  _QUAT_MULT_TABLE<Scalar>[static_cast<int>(type1)][static_cast<int>(type2)](
      q1, q2, result);
}

template void multiply_quaternions<double>(const Eigen::Quaternion<double>& q1,
                                           const Eigen::Quaternion<double>& q2,
                                           Eigen::Quaternion<double>& result,
                                           RotAxis type1,
                                           RotAxis type2);

template void multiply_quaternions<float>(const Eigen::Quaternion<float>& q1,
                                          const Eigen::Quaternion<float>& q2,
                                          Eigen::Quaternion<float>& result,
                                          RotAxis type1,
                                          RotAxis type2);

}  // namespace plainmp::kinematics
