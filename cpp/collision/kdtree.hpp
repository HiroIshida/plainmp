#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <vector>

struct KDNode {
  Eigen::Vector3d point;
  int axis;
  int left;
  int right;
  KDNode(const Eigen::Vector3d& pt, int ax)
      : point(pt), axis(ax), left(-1), right(-1) {}
};

class KDTree {
 public:
  using Ptr = std::shared_ptr<KDTree>;
  KDTree(const std::vector<Eigen::Vector3d>& points);
  Eigen::Vector3d query(const Eigen::Vector3d& target) const;
  double sqdist(const Eigen::Vector3d& target) const;

 private:
  std::vector<KDNode> nodes_;
  int root_index_;

  int build(std::vector<Eigen::Vector3d>::iterator begin,
            std::vector<Eigen::Vector3d>::iterator end,
            int depth);

  void nearest(int node_index,
               const Eigen::Vector3d& target,
               double& best_dist,
               Eigen::Vector3d& best_point) const;
};
