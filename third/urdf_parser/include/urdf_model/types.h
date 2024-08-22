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

    static QuatTrans<Scalar> fromXYZRPY(const Eigen::Vector3d& xyz, const Eigen::Vector3d& rpy) {
        Eigen::Quaternion<Scalar> q;
        q = Eigen::AngleAxis<Scalar>(rpy[0], Eigen::Matrix<Scalar, 3, 1>::UnitX())
            * Eigen::AngleAxis<Scalar>(rpy[1], Eigen::Matrix<Scalar, 3, 1>::UnitY())
            * Eigen::AngleAxis<Scalar>(rpy[2], Eigen::Matrix<Scalar, 3, 1>::UnitZ());
        return {q, xyz};
    }
};

}

#endif
