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

/* Author: Wim Meeussen */

#ifndef URDF_INTERFACE_POSE_H
#define URDF_INTERFACE_POSE_H

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <urdf_exception/exception.h>
#include <urdf_model/utils.h>
#include <urdf_model/types.h>

namespace urdf{

class Vector3
{
public:
  Vector3(double _x,double _y, double _z) {this->x=_x;this->y=_y;this->z=_z;};
  Vector3() {this->clear();};
  double x;
  double y;
  double z;

  double& operator[](size_t index) {
    switch (index) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
      default: throw std::out_of_range("Index out of range");
    }
  }

  void clear() {this->x=this->y=this->z=0.0;};
  void init(const std::string &vector_str)
  {
    this->clear();
    std::vector<std::string> pieces;
    std::vector<double> xyz;
    urdf::split_string( pieces, vector_str, " ");
    for (unsigned int i = 0; i < pieces.size(); ++i){
      if (pieces[i] != ""){
        try {
          xyz.push_back(strToDouble(pieces[i].c_str()));
        } catch(std::runtime_error &) {
          throw ParseError("Unable to parse component [" + pieces[i] + "] to a double (while parsing a vector value)");
        }
      }
    }

    if (xyz.size() != 3)
      throw ParseError("Parser found " + std::to_string(xyz.size())  + " elements but 3 expected while parsing vector [" + vector_str + "]");

    this->x = xyz[0];
    this->y = xyz[1];
    this->z = xyz[2];
  }

  void inverse_inplace(){
    this->x *= -1;
    this->y *= -1;
    this->z *= -1;
  }

  Vector3 operator+(const Vector3& vec) const
  {
    return Vector3(this->x+vec.x,this->y+vec.y,this->z+vec.z);
  };

  Vector3 operator-(const Vector3& vec) const
  {
    return Vector3(this->x-vec.x,this->y-vec.y,this->z-vec.z);
  };

  Vector3 operator/(double deno) const
  {
    return Vector3(this->x/deno,this->y/deno,this->z/deno);
  };
};

void cross_product(const Vector3& a, const Vector3& b, Vector3& out);

class Rotation
{
public:
  Rotation(double _x,double _y, double _z, double _w) {this->x=_x;this->y=_y;this->z=_z;this->w=_w;};
  Rotation() {this->clear();};
  void getQuaternion(double &quat_x,double &quat_y,double &quat_z, double &quat_w) const
  {
    quat_x = this->x;
    quat_y = this->y;
    quat_z = this->z;
    quat_w = this->w;
  };
  void getRPY(double &roll,double &pitch,double &yaw) const
  {
    double sqw;
    double sqx;
    double sqy;
    double sqz;

    sqx = this->x * this->x;
    sqy = this->y * this->y;
    sqz = this->z * this->z;
    sqw = this->w * this->w;

    // Cases derived from https://orbitalstation.wordpress.com/tag/quaternion/
    double sarg = -2 * (this->x*this->z - this->w*this->y);
    const double pi_2 = 1.57079632679489661923;
    if (sarg <= -0.99999) {
      pitch = -pi_2;
      roll  = 0;
      yaw   = 2 * atan2(this->x, -this->y);
    } else if (sarg >= 0.99999) {
      pitch = pi_2;
      roll  = 0;
      yaw   = 2 * atan2(-this->x, this->y);
    } else {
      pitch = asin(sarg);
      roll  = atan2(2 * (this->y*this->z + this->w*this->x), sqw - sqx - sqy + sqz);
      yaw   = atan2(2 * (this->x*this->y + this->w*this->z), sqw + sqx - sqy - sqz);
    }

  };

  Vector3 getRPY() const
  {
    double sqw;
    double sqx;
    double sqy;
    double sqz;

    sqx = this->x * this->x;
    sqy = this->y * this->y;
    sqz = this->z * this->z;
    sqw = this->w * this->w;

    // Cases derived from https://orbitalstation.wordpress.com/tag/quaternion/
    double sarg = -2 * (this->x*this->z - this->w*this->y);
    const double pi_2 = 1.57079632679489661923;

    double roll, pitch, yaw;
    if (sarg <= -0.99999) {
      pitch = -pi_2;
      roll  = 0;
      yaw   = 2 * atan2(this->x, -this->y);
    } else if (sarg >= 0.99999) {
      pitch = pi_2;
      roll  = 0;
      yaw   = 2 * atan2(-this->x, this->y);
    } else {
      pitch = asin(sarg);
      roll  = atan2(2 * (this->y*this->z + this->w*this->x), sqw - sqx - sqy + sqz);
      yaw   = atan2(2 * (this->x*this->y + this->w*this->z), sqw + sqx - sqy - sqz);
    }
    return Vector3(roll, pitch, yaw);
  };

  void setFromQuaternion(double quat_x,double quat_y,double quat_z,double quat_w)
  {
    this->x = quat_x;
    this->y = quat_y;
    this->z = quat_z;
    this->w = quat_w;
    this->normalize();
    this->rot_matrix_dirty = true;
  };
  void setFromRPY(double roll, double pitch, double yaw)
  {
    double phi, the, psi;

    phi = roll / 2.0;
    the = pitch / 2.0;
    psi = yaw / 2.0;

    this->x = sin(phi) * cos(the) * cos(psi) - cos(phi) * sin(the) * sin(psi);
    this->y = cos(phi) * sin(the) * cos(psi) + sin(phi) * cos(the) * sin(psi);
    this->z = cos(phi) * cos(the) * sin(psi) - sin(phi) * sin(the) * cos(psi);
    this->w = cos(phi) * cos(the) * cos(psi) + sin(phi) * sin(the) * sin(psi);

    this->normalize();
    this->rot_matrix_dirty = true;
  };

  double x,y,z,w;

  // these are used to cache the rotation matrix
  mutable bool rot_matrix_dirty;
  mutable double rot_matrix[3][3];

  void init(const std::string &rotation_str)
  {
    this->clear();
    Vector3 rpy;
    rpy.init(rotation_str);
    setFromRPY(rpy.x, rpy.y, rpy.z);
  }

  void clear() { this->x=this->y=this->z=0.0;this->w=1.0; rot_matrix_dirty = true;}

  void normalize()
  {
    double s = sqrt(this->x * this->x +
                    this->y * this->y +
                    this->z * this->z +
                    this->w * this->w);
    if (s == 0.0)
    {
      this->x = 0.0;
      this->y = 0.0;
      this->z = 0.0;
      this->w = 1.0;
    }
    else
    {
      this->x /= s;
      this->y /= s;
      this->z /= s;
      this->w /= s;
    }
  };

  // Multiplication operator (copied from gazebo)
  Rotation operator*( const Rotation &qt ) const
  {
    Rotation c;

    c.x = this->w * qt.x + this->x * qt.w + this->y * qt.z - this->z * qt.y;
    c.y = this->w * qt.y - this->x * qt.z + this->y * qt.w + this->z * qt.x;
    c.z = this->w * qt.z + this->x * qt.y - this->y * qt.x + this->z * qt.w;
    c.w = this->w * qt.w - this->x * qt.x - this->y * qt.y - this->z * qt.z;

    return c;
  };
  /// Rotate a vector using the quaternion
  Vector3 operator*(Vector3 vec) const
  {
    double q_xx = this->x * this->x;
    double q_yy = this->y * this->y;
    double q_zz = this->z * this->z;
    double q_xy = this->x * this->y;
    double q_xz = this->x * this->z;
    double q_xw = this->x * this->w;
    double q_yz = this->y * this->z;
    double q_yw = this->y * this->w;
    double q_zw = this->z * this->w;

    // fill rotation matrix
    if (this->rot_matrix_dirty)
    {
      this->rot_matrix[0][0] = 1 - 2*q_yy - 2*q_zz;
      this->rot_matrix[0][1] = 2*q_xy - 2*q_zw;
      this->rot_matrix[0][2] = 2*q_xz + 2*q_yw;
      this->rot_matrix[1][0] = 2*q_xy + 2*q_zw;
      this->rot_matrix[1][1] = 1 - 2*q_xx - 2*q_zz;
      this->rot_matrix[1][2] = 2*q_yz - 2*q_xw;
      this->rot_matrix[2][0] = 2*q_xz - 2*q_yw;
      this->rot_matrix[2][1] = 2*q_yz + 2*q_xw;
      this->rot_matrix[2][2] = 1 - 2*q_xx - 2*q_yy;
      this->rot_matrix_dirty = false;
    }
    return {this->rot_matrix[0][0]*vec.x + this->rot_matrix[0][1]*vec.y + this->rot_matrix[0][2]*vec.z,
            this->rot_matrix[1][0]*vec.x + this->rot_matrix[1][1]*vec.y + this->rot_matrix[1][2]*vec.z,
            this->rot_matrix[2][0]*vec.x + this->rot_matrix[2][1]*vec.y + this->rot_matrix[2][2]*vec.z};
  };
  // Get the inverse of this quaternion
  Rotation GetInverse() const
  {
    Rotation q;

    double norm = this->w*this->w+this->x*this->x+this->y*this->y+this->z*this->z;

    if (norm > 0.0)
    {
      q.w = this->w / norm;
      q.x = -this->x / norm;
      q.y = -this->y / norm;
      q.z = -this->z / norm;
    }

    return q;
  };

  Rotation inverse() const{
    double norm = this->w*this->w+this->x*this->x+this->y*this->y+this->z*this->z;
    return Rotation(-x/norm, -y/norm, -z/norm, w/norm);
  }

  void inverse_inplace(){
    double norm = this->w*this->w+this->x*this->x+this->y*this->y+this->z*this->z;
    this->w = this->w / norm;
    this->x = -this->x / norm;
    this->y = -this->y / norm;
    this->z = -this->z / norm;
  }
};

class Pose
{
public:
  Pose() { this->clear(); };

  Vector3  position;
  Rotation rotation;

  void clear()
  {
    this->position.clear();
    this->rotation.clear();
  };

  void inverse_inplace(){
    this->rotation.inverse_inplace();
    this->position.inverse_inplace();
    this->position = this->rotation * this->position;
  }

  Pose inverse() const{
    Pose pose_out = *this;
    pose_out.inverse_inplace();
    return pose_out;
  }

  QuatTrans<double> to_quattrans() const{
    QuatTrans<double> qt;
    qt.q.x() = rotation.x;
    qt.q.y() = rotation.y;
    qt.q.z() = rotation.z;
    qt.q.w() = rotation.w;
    qt.t = {position.x, position.y, position.z};
    return qt;
  }
};

// avoid defining this by operator overload. because this transformation is 
// more like function composition rather than multiplication or addition.
Pose pose_transform(const Pose& pose12, const Pose& pose23);

}

#endif