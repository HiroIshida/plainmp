#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

using Point = Eigen::Vector3d;
using Points = Eigen::Matrix3Xd;
using Values = Eigen::VectorXd;

struct Pose {
  Pose(const Eigen::Vector3d& position, const Eigen::Matrix3d& rotation)
      : position_(position), rot_(rotation), rot_inv_(rotation.inverse()) {
    axis_aligned_ = rot_.isApprox(Eigen::Matrix3d::Identity());
    if (axis_aligned_) {
      z_axis_aligned_ = true;
    } else {
      double tol = 1e-6;
      z_axis_aligned_ =
          std::abs(rot_(2, 2) - 1.0) < tol && std::abs(rot_(0, 2)) < tol &&
          std::abs(rot_(1, 2)) < tol && std::abs(rot_(2, 0)) < tol &&
          std::abs(rot_(2, 1)) < tol;
    }
  }

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
  bool axis_aligned_;
  bool z_axis_aligned_;
};

enum SDFType { UNION, BOX, CYLINDER, SPHERE, GROUND };

class SDFBase {
 public:
  using Ptr = std::shared_ptr<SDFBase>;
  virtual SDFType get_type() const = 0;
  virtual Values evaluate_batch(const Points& p) const {
    // naive implementation. please override this function if you have a better
    // implementation
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
  SDFType get_type() const override { return SDFType::UNION; }
  UnionSDF(std::vector<SDFBase::Ptr> sdfs, bool create_bvh) : sdfs_(sdfs) {
    if (create_bvh) {
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

  std::vector<std::shared_ptr<SDFBase>> sdfs_;
};

struct PrimitiveSDFBase : public SDFBase {
 public:
  using Ptr = std::shared_ptr<PrimitiveSDFBase>;

  // this filtering is quite fast as it is not virtual function
  inline bool is_outside_aabb(const Point& p, double radius) const {
    return p(0) < lb(0) - radius || p(0) > ub(0) + radius ||
           p(1) < lb(1) - radius || p(1) > ub(1) + radius ||
           p(2) < lb(2) - radius || p(2) > ub(2) + radius;
  }

  inline bool is_outside_aabb_batch(const Points& ps,
                                    const Eigen::VectorXd& radii) const {
    // this is much faster than loop-based implementation
    double ps_x_min_minus_radius = (ps.row(0).transpose() - radii).minCoeff();
    if (ps_x_min_minus_radius > ub(0)) {
      return true;
    }
    double ps_x_max_plus_radius = (ps.row(0).transpose() + radii).maxCoeff();
    if (ps_x_max_plus_radius < lb(0)) {
      return true;
    }

    double ps_y_min_minus_radius = (ps.row(1).transpose() - radii).minCoeff();
    if (ps_y_min_minus_radius > ub(1)) {
      return true;
    }
    double ps_y_max_plus_radius = (ps.row(1).transpose() + radii).maxCoeff();
    if (ps_y_max_plus_radius < lb(1)) {
      return true;
    }

    double ps_z_min_minus_radius = (ps.row(2).transpose() - radii).minCoeff();
    if (ps_z_min_minus_radius > ub(2)) {
      return true;
    }
    double ps_z_max_plus_radius = (ps.row(2).transpose() + radii).maxCoeff();
    if (ps_z_max_plus_radius < lb(2)) {
      return true;
    }
    return false;
  }

  Eigen::Vector3d lb;
  Eigen::Vector3d ub;
};

struct GroundSDF : public PrimitiveSDFBase {
  using Ptr = std::shared_ptr<GroundSDF>;
  SDFType get_type() const override { return SDFType::GROUND; }
  GroundSDF(double height) : height_(height) {
    lb = Eigen::Vector3d(-std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity(), 0.0);
    ub = Eigen::Vector3d(std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::infinity(), 0.0);
  }
  Values evaluate_batch(const Points& p) const override {
    return p.row(2).array() + height_;
  }
  double evaluate(const Point& p) const override { return p(2) + height_; }
  bool is_outside(const Point& p, double radius) const override {
    return p(2) + height_ > radius;
  }

 private:
  double height_;
};

struct BoxSDF : public PrimitiveSDFBase {
  // should implement
  using Ptr = std::shared_ptr<BoxSDF>;
  SDFType get_type() const override { return SDFType::BOX; }
  BoxSDF(const Eigen::Vector3d& width, const Pose& pose)
      : width_(width), half_width_(0.5 * width), pose_(pose) {
    Eigen::Matrix3Xd local_vertices(3, 8);
    local_vertices.col(0) =
        Eigen::Vector3d(-width_(0) * 0.5, -width_(1) * 0.5, -width_(2) * 0.5);
    local_vertices.col(1) =
        Eigen::Vector3d(width_(0) * 0.5, -width_(1) * 0.5, -width_(2) * 0.5);
    local_vertices.col(2) =
        Eigen::Vector3d(-width_(0) * 0.5, width_(1) * 0.5, -width_(2) * 0.5);
    local_vertices.col(3) =
        Eigen::Vector3d(width_(0) * 0.5, width_(1) * 0.5, -width_(2) * 0.5);
    local_vertices.col(4) =
        Eigen::Vector3d(-width_(0) * 0.5, -width_(1) * 0.5, width_(2) * 0.5);
    local_vertices.col(5) =
        Eigen::Vector3d(width_(0) * 0.5, -width_(1) * 0.5, width_(2) * 0.5);
    local_vertices.col(6) =
        Eigen::Vector3d(-width_(0) * 0.5, width_(1) * 0.5, width_(2) * 0.5);
    local_vertices.col(7) =
        Eigen::Vector3d(width_(0) * 0.5, width_(1) * 0.5, width_(2) * 0.5);
    auto world_vertices = pose.inverse().transform_points(local_vertices);
    lb = world_vertices.rowwise().minCoeff();
    ub = world_vertices.rowwise().maxCoeff();
  }

  void set_width(const Eigen::Vector3d& width) {
    width_ = width;
    half_width_ = 0.5 * width;
  }

  const Eigen::Vector3d& get_width() const { return width_; }

  double evaluate(const Point& p) const override {
    Eigen::Vector3d sdists;
    if (pose_.axis_aligned_) {
      sdists = (p - pose_.position_).array().abs() - half_width_.array();
    } else {
      sdists = (pose_.rot_inv_ * (p - pose_.position_)).array().abs() -
               half_width_.array();
    }
    Eigen::Vector3d m = sdists.array().max(0.0);
    double outside_distance = m.norm();
    double inside_distance = (sdists.cwiseMin(0.0)).maxCoeff();
    return outside_distance + inside_distance;
  }

  bool is_outside(const Point& p, double radius) const override {
    // NOTE: you may think that the following code is more efficient than the
    // current implementation. However, the current implementation is way
    // faster than this code.
    /* >>>>>>>
    auto p_local = pose_.rot_.transpose() * (p - pose_.position_);
    Eigen::Vector3d q = p_local.cwiseAbs() - half_width_;
    if (q.maxCoeff() < -radius) {
      return false;  // Completely inside
    }
    double outside_distance = q.cwiseMax(0.0).norm();
    return outside_distance > radius;
    <<<<<<< */

    double x_signed_dist, y_signed_dist, z_signed_dist;
    if (pose_.axis_aligned_) {
      x_signed_dist = abs(p(0) - pose_.position_(0)) - half_width_(0);
      if (x_signed_dist > radius) {
        return true;
      }
      y_signed_dist = abs(p(1) - pose_.position_(1)) - half_width_(1);
      if (y_signed_dist > radius) {
        return true;
      }
      z_signed_dist = abs(p(2) - pose_.position_(2)) - half_width_(2);
      if (z_signed_dist > radius) {
        return true;
      }
    } else if (pose_.z_axis_aligned_) {
      z_signed_dist = abs(p(2) - pose_.position_(2)) - half_width_(2);
      if (z_signed_dist > radius) {
        return true;
      }
      auto p_from_center = p - pose_.position_;
      x_signed_dist =
          abs(p_from_center.dot(pose_.rot_.col(0))) - half_width_(0);
      if (x_signed_dist > radius) {
        return true;
      }
      y_signed_dist =
          abs(p_from_center.dot(pose_.rot_.col(1))) - half_width_(1);
      if (y_signed_dist > radius) {
        return true;
      }
    } else {
      auto p_from_center = p - pose_.position_;
      x_signed_dist =
          abs(p_from_center.dot(pose_.rot_.col(0))) - half_width_(0);
      if (x_signed_dist > radius) {
        return true;
      }
      y_signed_dist =
          abs(p_from_center.dot(pose_.rot_.col(1))) - half_width_(1);
      if (y_signed_dist > radius) {
        return true;
      }
      z_signed_dist =
          abs(p_from_center.dot(pose_.rot_.col(2))) - half_width_(2);
      if (z_signed_dist > radius) {
        return true;
      }
    }

    if (radius < 1e-6) {
      return false;
    }

    // (literally) edge case, which araises only when radius is considered
    bool is_x_out = x_signed_dist > 0;
    bool is_y_out = y_signed_dist > 0;
    bool is_z_out = z_signed_dist > 0;
    std::uint8_t out_count = is_x_out + is_y_out + is_z_out;
    if (out_count < 2) {
      return false;
    }
    if (out_count == 3) {
      return x_signed_dist * x_signed_dist + y_signed_dist * y_signed_dist +
                 z_signed_dist * z_signed_dist >
             radius * radius;
    }
    if (!is_x_out) {
      return y_signed_dist * y_signed_dist + z_signed_dist * z_signed_dist >
             radius * radius;
    }
    if (!is_y_out) {
      return x_signed_dist * x_signed_dist + z_signed_dist * z_signed_dist >
             radius * radius;
    }
    return x_signed_dist * x_signed_dist + y_signed_dist * y_signed_dist >
           radius * radius;
  }

 private:
  Eigen::Vector3d width_;
  Eigen::Vector3d half_width_;
  Pose pose_;
};

struct CylinderSDF : public PrimitiveSDFBase {
  using Ptr = std::shared_ptr<CylinderSDF>;
  SDFType get_type() const override { return SDFType::CYLINDER; }
  CylinderSDF(double radius, double height, const Pose& pose)
      : r_cylinder_(radius),
        rsq_cylinder_(radius * radius),
        height_(height),
        half_height_(0.5 * height),
        pose_(pose) {
    Eigen::Matrix3Xd local_vertices(3, 8);
    local_vertices.col(0) = Eigen::Vector3d(-radius, -radius, -height * 0.5);
    local_vertices.col(1) = Eigen::Vector3d(radius, -radius, -height * 0.5);
    local_vertices.col(2) = Eigen::Vector3d(-radius, radius, -height * 0.5);
    local_vertices.col(3) = Eigen::Vector3d(radius, radius, -height * 0.5);
    local_vertices.col(4) = Eigen::Vector3d(-radius, -radius, height * 0.5);
    local_vertices.col(5) = Eigen::Vector3d(radius, -radius, height * 0.5);
    local_vertices.col(6) = Eigen::Vector3d(-radius, radius, height * 0.5);
    local_vertices.col(7) = Eigen::Vector3d(radius, radius, height * 0.5);
    auto world_vertices = pose.inverse().transform_points(local_vertices);
    lb = world_vertices.rowwise().minCoeff();
    ub = world_vertices.rowwise().maxCoeff();
  }

  double evaluate(const Point& p) const override {
    double z_signed_dist, xdot_abs, ydot_abs;
    if (pose_.z_axis_aligned_) {
      z_signed_dist = abs(p(2) - pose_.position_(2)) - half_height_;
      xdot_abs = abs(p(0) - pose_.position_(0));
      ydot_abs = abs(p(1) - pose_.position_(1));
    } else {
      auto p_from_center = p - pose_.position_;
      z_signed_dist = abs(p_from_center.dot(pose_.rot_.col(2))) - half_height_;
      xdot_abs = abs(p_from_center.dot(pose_.rot_.col(0)));
      ydot_abs = abs(p_from_center.dot(pose_.rot_.col(1)));
    }
    double r_signed_dist =
        sqrt(xdot_abs * xdot_abs + ydot_abs * ydot_abs) - r_cylinder_;
    Eigen::Vector2d d_2d(r_signed_dist, z_signed_dist);
    auto outside_distance = (d_2d.cwiseMax(0.0)).norm();
    auto inside_distance = d_2d.cwiseMin(0.0).maxCoeff();
    return outside_distance + inside_distance;
  }

  bool is_outside(const Point& p, double radius) const override {
    double z_signed_dist, xdot_abs, ydot_abs;
    if (pose_.z_axis_aligned_) {
      z_signed_dist = abs(p(2) - pose_.position_(2)) - half_height_;
      if (z_signed_dist > radius) {
        return true;
      }
      xdot_abs = abs(p(0) - pose_.position_(0));
      ydot_abs = abs(p(1) - pose_.position_(1));
    } else {
      auto p_from_center = p - pose_.position_;
      z_signed_dist = abs(p_from_center.dot(pose_.rot_.col(2))) - half_height_;
      if (z_signed_dist > radius) {
        return true;
      }
      xdot_abs = abs(p_from_center.dot(pose_.rot_.col(0)));
      ydot_abs = abs(p_from_center.dot(pose_.rot_.col(1)));
    }
    double dist_sq = xdot_abs * xdot_abs + ydot_abs * ydot_abs;
    if (radius < 1e-6) {
      return dist_sq > rsq_cylinder_;
    }

    if (dist_sq > (r_cylinder_ + radius) * (r_cylinder_ + radius)) {
      return true;
    }
    bool h_out = z_signed_dist > 0;
    bool r_out = dist_sq > rsq_cylinder_;
    if (h_out && r_out) {
      double r_signed_dist = sqrt(dist_sq) - r_cylinder_;
      return z_signed_dist * z_signed_dist + r_signed_dist * r_signed_dist >
             radius * radius;
    }
    return false;
  }

 private:
  double r_cylinder_;
  double rsq_cylinder_;
  double height_;
  double half_height_;
  Pose pose_;
};

struct SphereSDF : public PrimitiveSDFBase {
  using Ptr = std::shared_ptr<SphereSDF>;
  SDFType get_type() const override { return SDFType::SPHERE; }
  SphereSDF(double radius, const Pose& pose)
      : r_sphere_(radius), rsq_sphere_(radius * radius), pose_(pose) {
    lb = pose.position_ - Eigen::Vector3d(radius, radius, radius);
    ub = pose.position_ + Eigen::Vector3d(radius, radius, radius);
  }

  double evaluate(const Point& p) const override {
    auto p_from_center = p - pose_.position_;
    double dist = p_from_center.norm() - r_sphere_;
    return dist;
  }

  bool is_outside(const Point& p, double radius) const override {
    if (radius < 1e-6) {
      return (p - pose_.position_).squaredNorm() > rsq_sphere_;
    }
    return (p - pose_.position_).squaredNorm() >
           (r_sphere_ + radius) * (r_sphere_ + radius);
  }

 private:
  double r_sphere_;
  double rsq_sphere_;
  Pose pose_;
};
