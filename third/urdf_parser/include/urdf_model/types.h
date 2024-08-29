/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2008, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/* Author: Steve Peters */

#ifndef URDF_MODEL_TYPES_H
#define URDF_MODEL_TYPES_H

#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#define URDF_TYPEDEF_CLASS_POINTER(Class) \
class Class; \
typedef std::shared_ptr<Class> Class##SharedPtr; \
typedef std::shared_ptr<const Class> Class##ConstSharedPtr; \
typedef std::weak_ptr<Class> Class##WeakPtr

namespace urdf{

// shared pointer used in joint.h
typedef std::shared_ptr<double> DoubleSharedPtr;

URDF_TYPEDEF_CLASS_POINTER(Box);
URDF_TYPEDEF_CLASS_POINTER(Collision);
URDF_TYPEDEF_CLASS_POINTER(Cylinder);
URDF_TYPEDEF_CLASS_POINTER(Geometry);
URDF_TYPEDEF_CLASS_POINTER(Inertial);
URDF_TYPEDEF_CLASS_POINTER(Joint);
URDF_TYPEDEF_CLASS_POINTER(JointCalibration);
URDF_TYPEDEF_CLASS_POINTER(JointDynamics);
URDF_TYPEDEF_CLASS_POINTER(JointLimits);
URDF_TYPEDEF_CLASS_POINTER(JointMimic);
URDF_TYPEDEF_CLASS_POINTER(JointSafety);
URDF_TYPEDEF_CLASS_POINTER(Link);
URDF_TYPEDEF_CLASS_POINTER(Material);
URDF_TYPEDEF_CLASS_POINTER(Mesh);
URDF_TYPEDEF_CLASS_POINTER(Sphere);
URDF_TYPEDEF_CLASS_POINTER(Visual);

// create *_pointer_cast functions in urdf namespace
template<class T, class U>
std::shared_ptr<T> const_pointer_cast(std::shared_ptr<U> const & r)
{
  return std::const_pointer_cast<T>(r);
}

template<class T, class U>
std::shared_ptr<T> dynamic_pointer_cast(std::shared_ptr<U> const & r)
{
  return std::dynamic_pointer_cast<T>(r);
}

template<class T, class U>
std::shared_ptr<T> static_pointer_cast(std::shared_ptr<U> const & r)
{
  return std::static_pointer_cast<T>(r);
}

template<typename Scalar>
struct QuatTrans {
    Eigen::Quaternion<Scalar> quat_;
    Eigen::Matrix<Scalar, 3, 1> trans_;

    inline Eigen::Quaternion<Scalar> get_quat() const { return quat_; }
    inline Eigen::Matrix<Scalar, 3, 1> get_trans() const { return trans_; }
    inline Eigen::Matrix<Scalar, 3, 3> get_rotmat() const { return quat_.toRotationMatrix(); }

    inline Scalar get_trans_x() const { return trans_.x(); }
    inline Scalar get_trans_y() const { return trans_.y(); }
    inline Scalar get_trans_z() const { return trans_.z(); }

    // setter
    inline void set_trans(const Eigen::Matrix<Scalar, 3, 1>& trans){ 
        trans_ = trans;
    }
    inline void set_trans(Scalar x, Scalar y, Scalar z){
        trans_ = Eigen::Matrix<Scalar, 3, 1>(x, y, z);
    }
    inline void set_trans_x(Scalar x){ trans_.x() = x; }
    inline void set_trans_y(Scalar y){ trans_.y() = y; }
    inline void set_trans_z(Scalar z){ trans_.z() = z; }
    inline void set_trans_identity(){ trans_.setZero(); }
    inline void set_rot(const Scalar x, const Scalar y, const Scalar z, const Scalar w){ 
        quat_ = Eigen::Quaternion<Scalar>(w, x, y, z);
    }
    inline void set_rot(const Eigen::Vector3d axis, Scalar angle){ 
        auto tmp = axis * sin(angle * 0.5);
        quat_.x() = tmp.x();
        quat_.y() = tmp.y();
        quat_.z() = tmp.z();
        quat_.w() = cos(angle * 0.5);
    }
    inline void set_rot_identity(){ quat_ = Eigen::Quaternion<Scalar>::Identity(); }

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
        return {quat_ * other.quat_, trans_ + quat_ * other.trans_};
    }

    QuatTrans<Scalar> getInverse() const {
        Eigen::Quaternion<Scalar> q_inv = quat_.inverse();
        return {q_inv, q_inv * (-trans_)};
    }

    Eigen::Vector3d getRPY() const {
      auto sqx = quat_.x() * quat_.x();
      auto sqy = quat_.y() * quat_.y();
      auto sqz = quat_.z() * quat_.z();
      auto sqw = quat_.w() * quat_.w();

      // Cases derived from https://orbitalstation.wordpress.com/tag/quat_ernion/
      auto sarg = -2 * (quat_.x() * quat_.z() - quat_.w() * quat_.y());
      const double pi_2 = 1.57079632679489661923;

      Scalar roll, pitch, yaw;
      if (sarg <= -0.99999) {
        pitch = -pi_2;
        roll  = 0;
        yaw   = -2 * atan2(quat_.x(), quat_.y());
      } else if (sarg >= 0.99999) {
        pitch = pi_2;
        roll  = 0;
        yaw   = 2 * atan2(quat_.x(), quat_.y());
      } else {
        pitch = asin(sarg);
        roll = atan2(2 * (quat_.y() * quat_.z() + quat_.w() * quat_.x()), sqw - sqx - sqy + sqz);
        yaw = atan2(2 * (quat_.x() * quat_.y() + quat_.w() * quat_.z()), sqw + sqx - sqy - sqz);
      }
      return {roll, pitch, yaw};
    }

    void set_rot_from_rpy(const Eigen::Vector3d& rpy) {
        auto phi = rpy[0] / 2.0;
        auto the = rpy[1] / 2.0;
        auto psi = rpy[2] / 2.0;
        quat_.x() = sin(phi) * cos(the) * cos(psi) - cos(phi) * sin(the) * sin(psi);
        quat_.y() = cos(phi) * sin(the) * cos(psi) + sin(phi) * cos(the) * sin(psi);
        quat_.z() = cos(phi) * cos(the) * sin(psi) - sin(phi) * sin(the) * cos(psi);
        quat_.w() = cos(phi) * cos(the) * cos(psi) + sin(phi) * sin(the) * sin(psi);
    }

    void set_rot_from_rpy(Scalar roll, Scalar pitch, Scalar yaw) {
        set_rot_from_rpy(Eigen::Matrix<Scalar, 3, 1>(roll, pitch, yaw));
    }

    static QuatTrans<Scalar> fromXYZRPY(const Eigen::Vector3f& xyz, const Eigen::Vector3f& rpy) {
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
    static QuatTrans<Scalar> fromXYZRPY(Scalar x, Scalar y, Scalar z, Scalar roll, Scalar pitch, Scalar yaw) {
        return fromXYZRPY(Eigen::Vector3d(x, y, z), Eigen::Vector3d(roll, pitch, yaw));
    }

    static QuatTrans<Scalar> fromXYZ(const Eigen::Vector3d& xyz) {
        return {Eigen::Quaternion<Scalar>::Identity(), xyz};
    }
    static QuatTrans<Scalar> fromXYZ(Scalar x, Scalar y, Scalar z) {
        return fromXYZ(Eigen::Vector3d(x, y, z));
    }
};

}

#endif
