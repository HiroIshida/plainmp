#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace primitive_sdf {

using Point = Eigen::Vector3d;
using Points = Eigen::Matrix3Xd;
using Values = Eigen::VectorXd;

struct Pose {
  Pose(const Eigen::Vector3d& position, const Eigen::Matrix3d& rotation)
      : position_(position), rot_(rotation), rot_inv_(rotation.inverse()) {}

  Points transform_points(const Points& p) const {
    return rot_inv_ * (p.colwise() - position_);
  }

  Point transform_point(const Point& p) const {
    return rot_inv_ * (p - position_);
  }

  void set_position(const Eigen::Vector3d& position) { position_ = position; }

  Pose inverse() const { return Pose(-rot_ * position_, rot_inv_); }

  Eigen::Vector3d position_;
  Eigen::Matrix3d rot_;
  Eigen::Matrix3d rot_inv_;
};

class SDFBase {
 public:
  using Ptr = std::shared_ptr<SDFBase>;
  virtual Values evaluate_batch(const Points& p) const 
  {
    // naive implementation. please override this function if you have a better implementation
    Values vals(p.cols());
    for (int i = 0; i < p.cols(); i++) {
      vals(i) = evaluate(p.col(i));
    }
    return vals;
  }
  virtual double evaluate(const Point& p) const = 0;
  virtual bool is_outside(const Point& p, double radius) const = 0;
};

struct UnionSDF : public SDFBase {
  using Ptr = std::shared_ptr<UnionSDF>;
  UnionSDF(std::vector<SDFBase::Ptr> sdfs, bool create_bvh) : sdfs_(sdfs) {
      if(create_bvh){
          throw std::runtime_error("Not implemented yet");
      }
  }

  Values evaluate_batch(const Points& p) const override {
    Values vals = sdfs_[0]->evaluate_batch(p);
    for (size_t i = 1; i < sdfs_.size(); i++) {
      vals = vals.cwiseMin(sdfs_[i]->evaluate_batch(p));
    }
    return vals;
  }

  double evaluate(const Point& p) const override {
    double val = std::numeric_limits<double>::max();
    for (const auto& sdf : sdfs_) {
      val = std::min(val, sdf->evaluate(p));
    }
    return val;
  }

  bool is_outside(const Point& p, double radius) const override {
    for (const auto& sdf : sdfs_) {
      if (!sdf->is_outside(p, radius)) {
        return false;
      }
    }
    return true;
  }
 private:
  std::vector<std::shared_ptr<SDFBase>> sdfs_;
};

struct PrimitiveSDFBase : public SDFBase {
 public:
  using Ptr = std::shared_ptr<PrimitiveSDFBase>;
};

struct GroundSDF : public PrimitiveSDFBase {
  using Ptr = std::shared_ptr<GroundSDF>;
  GroundSDF(double height) : height_(height) {}
  Values evaluate_batch(const Points& p) const override {
    return p.row(2).array() + height_;
  }
  double evaluate(const Point& p) const override { return p(2) + height_; }
  bool is_outside(const Point& p, double radius) const override {
    return p(2) + height_ > radius;
  }
  double height_;
};

struct BoxSDF : public PrimitiveSDFBase {
  using Ptr = std::shared_ptr<BoxSDF>;
  BoxSDF(const Eigen::Vector3d& width, const Pose& pose)
      : width_(width), half_width_(0.5 * width), pose_(pose) {}
  double evaluate(const Point& p) const override {
      throw std::runtime_error("Not implemented yet");
      return 0;
  }
  bool is_outside(const Point& p, double radius) const override {
      // axis aligned case (TODO: create new type to do this)
      // auto abs_relative_point = (p - pose_.position_).cwiseAbs();
      // if(abs_relative_point.x() > half_width_(0) + radius){
      //     return true;
      // }
      // if(abs_relative_point.y() > half_width_(1) + radius){
      //     return true;
      // }
      // if(abs_relative_point.z() > half_width_(2) + radius){
      //     return true;
      // }
      // return false;
      auto p_from_center = p - pose_.position_;
      double xdot_abs = abs(p_from_center.dot(pose_.rot_.col(0)));
      if(xdot_abs > half_width_(0) + radius){
          return true;
      }
      double ydot_abs = abs(p_from_center.dot(pose_.rot_.col(1)));
      if(ydot_abs > half_width_(1) + radius){
          return true;
      }
      double zdot_abs = abs(p_from_center.dot(pose_.rot_.col(2)));
      return zdot_abs > half_width_(2) + radius;
  }
  Eigen::Vector3d width_;
  Eigen::Vector3d half_width_;
  Pose pose_;
};

struct CylinderSDF : public PrimitiveSDFBase {
  using Ptr = std::shared_ptr<CylinderSDF>;
  CylinderSDF(double radius, double height, const Pose& pose)
      : r_cylinder_(radius), rsq_cylinder_(radius * radius), height_(height), half_height_(0.5 * height), pose_(pose) {}
  double evaluate(const Point& p) const override {
      throw std::runtime_error("Not implemented yet");
      return 0;
  }
  bool is_outside(const Point& p, double radius) const override {
      auto p_from_center = p - pose_.position_;

      // collision with top and bottom
      double zdot_abs = abs(p_from_center.dot(pose_.rot_.col(2)));
      if(zdot_abs > half_height_ + radius){
          return true;
      }

      // collision with side
      double xdot_abs = abs(p_from_center.dot(pose_.rot_.col(0)));
      double ydot_abs = abs(p_from_center.dot(pose_.rot_.col(1)));
      double dist_sq = xdot_abs * xdot_abs + ydot_abs * ydot_abs;

      // Note: this for solerly for avoiding sqrt operation
      // dist_sq > (r_cylinder_ + radius)^2
      //         = (r_cylinder_^2 + 2 * r_cylinder_ * radius + radius^2)
      // now we compute 2 * r_cylinder_ * radius + radius^2 as ...
      double remain = radius * (2 * r_cylinder_ + radius);
      return dist_sq > rsq_cylinder_ + remain;
  }
  double r_cylinder_;
  double rsq_cylinder_;
  double height_;
  double half_height_;
  Pose pose_;
};


}  // namespace primitive_sdf
