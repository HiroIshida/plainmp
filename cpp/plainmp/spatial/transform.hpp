/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace plainmp::spatial {

template <typename Scalar>
struct QuatTrans {
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  // NOTE: considering memory layout, (quat, trans) is much better than (trans,
  // quat)
  Eigen::Quaternion<Scalar> quat_;
  Eigen::Matrix<Scalar, 3, 1> trans_;
  bool is_quat_identity_ = false;

  inline QuatTrans<Scalar>& operator=(const QuatTrans<Scalar>& other) {
    quat_ = other.quat_;
    // somehow eigen's assginment operator is quite slow. so
    std::memcpy(trans_.data(), other.trans().data(), sizeof(Scalar) * 3);
    return *this;
  }

  template <typename ScalarTo>
  QuatTrans<ScalarTo> cast() const {
    auto&& quat_new = quat_.template cast<ScalarTo>();
    auto&& trans_new = trans_.template cast<ScalarTo>();
    return {quat_new, trans_new, is_quat_identity_};
  }

  // acceessor
  inline Eigen::Quaternion<Scalar>& quat() { return quat_; }
  inline Eigen::Matrix<Scalar, 3, 1>& trans() { return trans_; }
  // const accessor
  inline const Eigen::Quaternion<Scalar>& quat() const { return quat_; }
  inline const Eigen::Matrix<Scalar, 3, 1>& trans() const { return trans_; }

  static QuatTrans<Scalar> Identity() {
    QuatTrans<Scalar> qt;
    qt.quat_ = Eigen::Quaternion<Scalar>::Identity();
    qt.trans_ = Eigen::Matrix<Scalar, 3, 1>::Zero();
    return qt;
  }
  void clear() {
    quat_ = Eigen::Quaternion<Scalar>::Identity();
    trans_ = Eigen::Matrix<Scalar, 3, 1>::Zero();
  }
  inline QuatTrans<Scalar> operator*(const QuatTrans<Scalar>& other) const {
    if (other.is_quat_identity_) {
      return {quat_, trans_ + quat_ * other.trans_};
    } else {
      return {quat_ * other.quat_, trans_ + quat_ * other.trans_};
    }
  }

  inline void quat_identity_sensitive_mult_and_assign(
      const QuatTrans<Scalar>& other,
      QuatTrans<Scalar>& result) const {
    if (other.is_quat_identity_) {
      result.quat_ = quat_;
      result.trans_ = trans_ + quat_ * other.trans_;
    } else {
      result.quat_ = quat_ * other.quat_;
      result.trans_ = trans_ + quat_ * other.trans_;
    }
  }

  inline QuatTrans<Scalar> quat_identity_sensitive_mul(
      const QuatTrans<Scalar>& other) const {
    // NOTE: in kin tree update, left side is from root => current transform
    // which is usually not quat-identity. but other is from pair-link-wise
    // transform thus more likely to be quat-identity. Thus...
    if (other.is_quat_identity_) {
      return {quat_, trans_ + quat_ * other.trans_};
    } else {
      return {quat_ * other.quat_, trans_ + quat_ * other.trans_};
    }
  }

  QuatTrans<Scalar> getInverse() const {
    Eigen::Quaternion<Scalar> q_inv = quat_.inverse();
    return {q_inv, q_inv * (-trans_)};
  }

  Vector3 getRPY() const {
    auto sqx = quat_.x() * quat_.x();
    auto sqy = quat_.y() * quat_.y();
    auto sqz = quat_.z() * quat_.z();
    auto sqw = quat_.w() * quat_.w();

    // Cases derived from https://orbitalstation.wordpress.com/tag/quat_ernion/
    auto sarg = -2 * (quat_.x() * quat_.z() - quat_.w() * quat_.y());
    const Scalar pi_2 = 1.57079632679489661923;

    Scalar roll, pitch, yaw;
    if (sarg <= -0.99999) {
      pitch = -pi_2;
      roll = 0;
      yaw = -2 * atan2(quat_.x(), quat_.y());
    } else if (sarg >= 0.99999) {
      pitch = pi_2;
      roll = 0;
      yaw = 2 * atan2(quat_.x(), quat_.y());
    } else {
      pitch = asin(sarg);
      roll = atan2(2 * (quat_.y() * quat_.z() + quat_.w() * quat_.x()),
                   sqw - sqx - sqy + sqz);
      yaw = atan2(2 * (quat_.x() * quat_.y() + quat_.w() * quat_.z()),
                  sqw + sqx - sqy - sqz);
    }
    return {roll, pitch, yaw};
  }

  void setQuaternionFromRPY(const Vector3& rpy) {
    auto phi = rpy[0] / 2.0;
    auto the = rpy[1] / 2.0;
    auto psi = rpy[2] / 2.0;
    quat_.x() = sin(phi) * cos(the) * cos(psi) - cos(phi) * sin(the) * sin(psi);
    quat_.y() = cos(phi) * sin(the) * cos(psi) + sin(phi) * cos(the) * sin(psi);
    quat_.z() = cos(phi) * cos(the) * sin(psi) - sin(phi) * sin(the) * cos(psi);
    quat_.w() = cos(phi) * cos(the) * cos(psi) + sin(phi) * sin(the) * sin(psi);
  }

  void setQuaternionFromRPY(Scalar roll, Scalar pitch, Scalar yaw) {
    setQuaternionFromRPY(Eigen::Matrix<Scalar, 3, 1>(roll, pitch, yaw));
  }

  static QuatTrans<Scalar> fromXYZRPY(const Vector3& xyz, const Vector3& rpy) {
    Eigen::Quaternion<Scalar> q;
    auto phi = rpy[0] / 2.0;
    auto the = rpy[1] / 2.0;
    auto psi = rpy[2] / 2.0;
    auto x = sin(phi) * cos(the) * cos(psi) - cos(phi) * sin(the) * sin(psi);
    auto y = cos(phi) * sin(the) * cos(psi) + sin(phi) * cos(the) * sin(psi);
    auto z = cos(phi) * cos(the) * sin(psi) - sin(phi) * sin(the) * cos(psi);
    auto w = cos(phi) * cos(the) * cos(psi) + sin(phi) * sin(the) * sin(psi);
    return {Eigen::Quaternion<Scalar>(w, x, y, z), xyz};
  }
  static QuatTrans<Scalar> fromXYZRPY(Scalar x,
                                      Scalar y,
                                      Scalar z,
                                      Scalar roll,
                                      Scalar pitch,
                                      Scalar yaw) {
    return fromXYZRPY(Vector3(x, y, z), Vector3(roll, pitch, yaw));
  }

  static QuatTrans<Scalar> fromXYZ(const Vector3& xyz) {
    return {Eigen::Quaternion<Scalar>::Identity(), xyz};
  }
  static QuatTrans<Scalar> fromXYZ(Scalar x, Scalar y, Scalar z) {
    return fromXYZ(Vector3(x, y, z));
  }
};

}  // namespace plainmp::spatial
