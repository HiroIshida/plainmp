/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "plainmp/bindings/bindings.hpp"
#include "plainmp/constraints/primitive_sphere_collision.hpp"
#include <chrono>
namespace pc = plainmp::constraint;
namespace py = pybind11;
using Clock = std::chrono::steady_clock;
class OverrideBox : public plainmp::collision::BoxSDF {
public:
  using BoxSDF::BoxSDF;
  bool is_outside(const plainmp::collision::Point &, double) const override {
    return true;
  }
};
void bind_batch_queries_check(py::module &m) {
  auto e = m.def_submodule("experiment");
  e.def(
      "replay_batch",
      [](pc::SphereCollisionCst::Ptr cst, const Eigen::MatrixXd &q,
         const std::vector<bool> &expected, size_t rounds, bool use_batch) {
        if (q.rows() != expected.size() || q.cols() != cst->q_dim())
          throw std::invalid_argument("shape mismatch");
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            flat = q;
        auto run = [&](bool verify) {
          uint64_t sum = 0, mismatches = 0;
          for (size_t i = 0; i < q.rows(); i += 4) {
            const size_t count = std::min(size_t(4), size_t(q.rows()) - i);
            const double *values[4];
            for (size_t k = 0; k < count; ++k)
              values[k] = flat.data() + (i + k) * q.cols();
            unsigned mask = 0;
            if (use_batch)
              mask = cst->is_valid_batch(values, count);
            else
              for (size_t k = 0; k < count; ++k)
                if (cst->is_valid(
                        Eigen::Map<const Eigen::VectorXd>(values[k], q.cols())))
                  mask |= 1u << k;
            sum += __builtin_popcount(mask);
            if (verify)
              for (size_t k = 0; k < count; ++k)
                mismatches += bool(mask & (1u << k)) != expected[i + k];
          }
          return std::make_pair(sum, mismatches);
        };
        auto validation = run(true);
        std::vector<double> ns;
        uint64_t sum = 0;
        for (size_t r = 0; r < rounds; ++r) {
          auto start = Clock::now();
          sum += run(false).first;
          ns.push_back(
              std::chrono::duration<double, std::nano>(Clock::now() - start)
                  .count());
        }
        return py::make_tuple(ns, validation.second, sum);
      },
      py::arg("cst"), py::arg("q"), py::arg("expected"), py::arg("rounds") = 1,
      py::arg("use_batch") = true);
  e.def("batch_mask",
        [](pc::SphereCollisionCst::Ptr cst, const Eigen::MatrixXd &q) {
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
              flat = q;
          if (q.rows() < 1 || q.rows() > 4 || q.cols() != cst->q_dim())
            throw std::invalid_argument("shape mismatch");
          const double *ptr[4];
          for (size_t i = 0; i < q.rows(); ++i)
            ptr[i] = flat.data() + i * q.cols();
          return cst->is_valid_batch(ptr, q.rows());
        });

  e.def("batch_supported", &pc::SphereCollisionCst::batch_supported);
  e.def("first_invalid",
        [](pc::SphereCollisionCst::Ptr cst, const Eigen::MatrixXd &q) {
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
              flat = q;
          if (q.rows() < 1 || q.rows() > 4 || q.cols() != cst->q_dim())
            throw std::invalid_argument("shape mismatch");
          const double *ptr[4];
          for (size_t i = 0; i < q.rows(); ++i)
            ptr[i] = flat.data() + i * q.cols();
          return cst->first_invalid_batch(ptr, q.rows());
        });
  e.def("override_box", []() -> plainmp::collision::SDFBase::Ptr {
    return std::make_shared<OverrideBox>(
        Eigen::Vector3d(10, 10, 10),
        plainmp::collision::Pose(Eigen::Vector3d::Zero(),
                                 Eigen::Matrix3d::Identity()));
  });
}
