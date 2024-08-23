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
    Eigen::Quaternion<Scalar> q;
    Eigen::Matrix<Scalar, 3, 1> t;

    static QuatTrans<Scalar> Identity() {
        QuatTrans<Scalar> qt;
        qt.q = Eigen::Quaternion<Scalar>::Identity();
        qt.t = Eigen::Matrix<Scalar, 3, 1>::Zero();
        return qt;
    }
    void clear() {
        q = Eigen::Quaternion<Scalar>::Identity();
        t = Eigen::Matrix<Scalar, 3, 1>::Zero();
    }
    inline QuatTrans<Scalar> operator*(const QuatTrans<Scalar>& other) const {
        return {q * other.q, t + q * other.t};
    }

    Eigen::Vector3d getRPY() const {
      auto sqx = q.x() * q.x();
      auto sqy = q.y() * q.y();
      auto sqz = q.z() * q.z();
      auto sqw = q.w() * q.w();

      // Cases derived from https://orbitalstation.wordpress.com/tag/quaternion/
      auto sarg = -2 * (q.x() * q.z() - q.w() * q.y());
      const double pi_2 = 1.57079632679489661923;

      Scalar roll, pitch, yaw;
      if (sarg <= -0.99999) {
        pitch = -pi_2;
        roll  = 0;
        yaw   = -2 * atan2(q.x(), q.y());
      } else if (sarg >= 0.99999) {
        pitch = pi_2;
        roll  = 0;
        yaw   = 2 * atan2(q.x(), q.y());
      } else {
        pitch = asin(sarg);
        // roll  = atan2(2 * (this->y*this->z + this->w*this->x), sqw - sqx - sqy + sqz);
        roll = atan2(2 * (q.y() * q.z() + q.w() * q.x()), sqw - sqx - sqy + sqz);
        // yaw   = atan2(2 * (this->x*this->y + this->w*this->z), sqw + sqx - sqy - sqz);
        yaw = atan2(2 * (q.x() * q.y() + q.w() * q.z()), sqw + sqx - sqy - sqz);
      }
      return {roll, pitch, yaw};
    }

    static QuatTrans<Scalar> fromXYZRPY(const Eigen::Vector3d& xyz, const Eigen::Vector3d& rpy) {
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
