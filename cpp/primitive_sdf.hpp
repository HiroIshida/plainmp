#ifndef PRIMITIVE_SDF_HPP
#define PRIMITIVE_SDF_HPP

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

struct AABB {
  bool is_outside(const Point& p, double radius) const {
    return (p.array() + radius < lb.array()).any() ||
           (p.array() - radius > ub.array()).any();
  }
  Point lb;
  Point ub;
};

class Pose {
 public:
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

 private:
  Eigen::Vector3d position_;
  Eigen::Matrix3d rot_;
  Eigen::Matrix3d rot_inv_;
};

class SDFBase {
 public:
  using Ptr = std::shared_ptr<SDFBase>;
  // for ease of binding to python, we name different functions
  virtual Values evaluate_batch(const Points& p) const = 0;
  virtual double evaluate(const Point& p) const = 0;
  virtual bool is_outside(const Point& p, double radius) const = 0;
  virtual AABB get_aabb() const = 0;
};

class UnionSDF : public SDFBase {
 public:
  using Ptr = std::shared_ptr<UnionSDF>;
  UnionSDF(std::vector<SDFBase::Ptr> sdfs, bool create_bvh) : sdfs_(sdfs) {
    if (create_bvh) {
      throw std::runtime_error(
          "UnionSDF with BVH is buggy. Please disable it.");
    }
    if (create_bvh) {
      aabb_ = get_aabb();
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
    if (aabb_.has_value() && aabb_->is_outside(p, radius)) {
      return true;
    }
    for (const auto& sdf : sdfs_) {
      if (!sdf->is_outside(p, radius)) {
        return false;
      }
    }
    return true;
  }

  AABB get_aabb() const override {
    Point lb = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
    Point ub = Eigen::Vector3d::Constant(-std::numeric_limits<double>::max());
    for (const auto& sdf : sdfs_) {
      auto aabb = sdf->get_aabb();
      lb = lb.cwiseMin(aabb.lb);
      ub = ub.cwiseMax(aabb.ub);
    }
    return {lb, ub};
  }

 private:
  std::vector<std::shared_ptr<SDFBase>> sdfs_;
  std::optional<AABB> aabb_;
};

class PrimitiveSDFBase : public SDFBase {
 public:
  using Ptr = std::shared_ptr<PrimitiveSDFBase>;
  PrimitiveSDFBase(const Pose& tf) : tf_(tf), tf_inv_(tf.inverse()) {}

  Values evaluate_batch(const Points& p) const override {
    auto p_local = tf_.transform_points(p);
    return evaluate_in_local_frame(p_local);
  }

  double evaluate(const Point& p) const override {
    auto p_local = tf_.transform_point(p);
    return evaluate_in_local_frame(p_local);
  }

  bool is_outside(const Point& p, double radius) const override {
    auto p_local = tf_.transform_point(p);
    return is_outside_in_local_frame(p_local, radius);
  }

  AABB get_aabb() const override {
    auto local_vertices = get_local_aabb_vertices();
    auto world_vertices = tf_inv_.transform_points(local_vertices);
    auto lb = world_vertices.rowwise().minCoeff();
    auto ub = world_vertices.rowwise().maxCoeff();
    return {lb, ub};
  }

  Pose tf_;
  Pose tf_inv_;

 protected:
  virtual Values evaluate_in_local_frame(const Points& p) const = 0;
  virtual double evaluate_in_local_frame(const Point& p) const = 0;
  virtual bool is_outside_in_local_frame(const Point& p, double radius) const {
    return evaluate_in_local_frame(p) > radius;
  }  // maybe override this for performance
     //
  virtual Eigen::Matrix3Xd get_local_aabb_vertices() const = 0;
};

class BoxSDF : public PrimitiveSDFBase {
 public:
  using Ptr = std::shared_ptr<BoxSDF>;
  Eigen::Vector3d width_;

  BoxSDF(const Eigen::Vector3d& width, const Pose& tf)
      : PrimitiveSDFBase(tf), width_(width) {}

 private:
  Values evaluate_in_local_frame(const Points& p) const override {
    auto&& half_width = width_ * 0.5;
    auto d = p.cwiseAbs().colwise() - half_width;
    auto outside_distance = (d.cwiseMax(0.0)).colwise().norm();
    auto inside_distance = d.cwiseMin(0.0).colwise().maxCoeff();
    Values vals = outside_distance + inside_distance;
    return vals;
  }

  double evaluate_in_local_frame(const Point& p) const override {
    auto&& half_width = width_ * 0.5;
    auto d = p.cwiseAbs() - half_width;
    auto outside_distance = (d.cwiseMax(0.0)).norm();
    auto inside_distance = d.cwiseMin(0.0).maxCoeff();
    return outside_distance + inside_distance;
  }

  Eigen::Matrix3Xd get_local_aabb_vertices() const {
    Eigen::Matrix3Xd vertices(3, 8);
    vertices.col(0) =
        Eigen::Vector3d(-width_(0) * 0.5, -width_(1) * 0.5, -width_(2) * 0.5);
    vertices.col(1) =
        Eigen::Vector3d(width_(0) * 0.5, -width_(1) * 0.5, -width_(2) * 0.5);
    vertices.col(2) =
        Eigen::Vector3d(-width_(0) * 0.5, width_(1) * 0.5, -width_(2) * 0.5);
    vertices.col(3) =
        Eigen::Vector3d(width_(0) * 0.5, width_(1) * 0.5, -width_(2) * 0.5);
    vertices.col(4) =
        Eigen::Vector3d(-width_(0) * 0.5, -width_(1) * 0.5, width_(2) * 0.5);
    vertices.col(5) =
        Eigen::Vector3d(width_(0) * 0.5, -width_(1) * 0.5, width_(2) * 0.5);
    vertices.col(6) =
        Eigen::Vector3d(-width_(0) * 0.5, width_(1) * 0.5, width_(2) * 0.5);
    vertices.col(7) =
        Eigen::Vector3d(width_(0) * 0.5, width_(1) * 0.5, width_(2) * 0.5);
    return vertices;
  };
};

class CylinderSDF : public PrimitiveSDFBase {
 public:
  using Ptr = std::shared_ptr<CylinderSDF>;
  double radius_;
  double height_;
  CylinderSDF(double radius, double height, const Pose& tf)
      : PrimitiveSDFBase(tf), radius_(radius), height_(height) {}

 private:
  Values evaluate_in_local_frame(const Points& p) const override {
    Eigen::VectorXd&& d = p.topRows(2).colwise().norm();
    Eigen::Matrix2Xd p_projected(2, d.size());
    p_projected.row(0) = d;
    p_projected.row(1) = p.row(2);

    auto&& half_width = Eigen::Vector2d(radius_, height_ * 0.5);
    auto d_2d = p_projected.cwiseAbs().colwise() - half_width;
    auto outside_distance = (d_2d.cwiseMax(0.0)).colwise().norm();
    auto inside_distance = d_2d.cwiseMin(0.0).colwise().maxCoeff();
    Values vals = outside_distance + inside_distance;
    return vals;
  }

  double evaluate_in_local_frame(const Point& p) const override {
    double d = p.topRows(2).norm();
    Eigen::Vector2d p_projected(d, p(2));

    auto&& half_width = Eigen::Vector2d(radius_, height_ * 0.5);
    auto d_2d = p_projected.cwiseAbs() - half_width;
    auto outside_distance = (d_2d.cwiseMax(0.0)).norm();
    auto inside_distance = d_2d.cwiseMin(0.0).maxCoeff();
    return outside_distance + inside_distance;
  }

  Eigen::Matrix3Xd get_local_aabb_vertices() const {
    Eigen::Matrix3Xd vertices(3, 8);
    vertices.col(0) = Eigen::Vector3d(-radius_, -radius_, -height_ * 0.5);
    vertices.col(1) = Eigen::Vector3d(radius_, -radius_, -height_ * 0.5);
    vertices.col(2) = Eigen::Vector3d(-radius_, radius_, -height_ * 0.5);
    vertices.col(3) = Eigen::Vector3d(radius_, radius_, -height_ * 0.5);
    vertices.col(4) = Eigen::Vector3d(-radius_, -radius_, height_ * 0.5);
    vertices.col(5) = Eigen::Vector3d(radius_, -radius_, height_ * 0.5);
    vertices.col(6) = Eigen::Vector3d(-radius_, radius_, height_ * 0.5);
    vertices.col(7) = Eigen::Vector3d(radius_, radius_, height_ * 0.5);
    return vertices;
  }
};

class SphereSDF : public PrimitiveSDFBase {
 public:
  using Ptr = std::shared_ptr<SphereSDF>;
  double radius_;

  SphereSDF(double radius, const Pose& tf)
      : PrimitiveSDFBase(tf), radius_(radius) {}

 private:
  Values evaluate_in_local_frame(const Eigen::Matrix3Xd& p) const override {
    return (p.colwise().norm().array() - radius_);
  }

  double evaluate_in_local_frame(const Point& p) const override {
    return (p.norm() - radius_);
  }

  Eigen::Matrix3Xd get_local_aabb_vertices() const {
    Eigen::Matrix3Xd vertices(3, 8);
    vertices.col(0) = Eigen::Vector3d(-radius_, -radius_, -radius_);
    vertices.col(1) = Eigen::Vector3d(radius_, -radius_, -radius_);
    vertices.col(2) = Eigen::Vector3d(-radius_, radius_, -radius_);
    vertices.col(3) = Eigen::Vector3d(radius_, radius_, -radius_);
    vertices.col(4) = Eigen::Vector3d(-radius_, -radius_, radius_);
    vertices.col(5) = Eigen::Vector3d(radius_, -radius_, radius_);
    vertices.col(6) = Eigen::Vector3d(-radius_, radius_, radius_);
    vertices.col(7) = Eigen::Vector3d(radius_, radius_, radius_);
    return vertices;
  }
};

}  // namespace primitive_sdf
#endif
