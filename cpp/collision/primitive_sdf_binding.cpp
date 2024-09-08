#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "primitive_sdf.hpp"

namespace primitive_sdf {

namespace py = pybind11;

template <typename Scalar>
void bind_primitive_sdf(py::module& m) {
  auto m_psdf = m.def_submodule("primitive_sdf");
  py::class_<Pose<Scalar>>(m_psdf, "Pose", py::module_local())
      .def(py::init<const Eigen::Matrix<Scalar, 3, 1>&,
                    const Eigen::Matrix<Scalar, 3, 3>&>())
      .def_readonly("axis_aligned", &Pose<Scalar>::axis_aligned_)
      .def_readonly("z_axis_aligned", &Pose<Scalar>::z_axis_aligned_);
  py::class_<SDFBase<Scalar>, typename SDFBase<Scalar>::Ptr>(
      m_psdf, "SDFBase",
      py::module_local());  // user is not supposed to instantiate this class.
                            // This to tell pybind that this is a base class
  py::class_<PrimitiveSDFBase<Scalar>, typename PrimitiveSDFBase<Scalar>::Ptr,
             SDFBase<Scalar>>(
      m_psdf, "PrimitiveSDFBase",
      py::module_local());  // user is not supposed to instantiate this class.
                            // This to tell pybind that this is a base class
  py::class_<UnionSDF<Scalar>, typename UnionSDF<Scalar>::Ptr, SDFBase<Scalar>>(
      m_psdf, "UnionSDF", py::module_local())
      .def(py::init<std::vector<typename SDFBase<Scalar>::Ptr>, bool>())
      .def("evaluate_batch", &UnionSDF<Scalar>::evaluate_batch)
      .def("evaluate", &UnionSDF<Scalar>::evaluate)
      .def("is_outside", &UnionSDF<Scalar>::is_outside);
  py::class_<GroundSDF<Scalar>, typename GroundSDF<Scalar>::Ptr,
             PrimitiveSDFBase<Scalar>>(m_psdf, "GroundSDF", py::module_local())
      .def(py::init<double>())
      .def("evaluate_batch", &GroundSDF<Scalar>::evaluate_batch)
      .def("evaluate", &GroundSDF<Scalar>::evaluate)
      .def("is_outside", &GroundSDF<Scalar>::is_outside);
  py::class_<BoxSDF<Scalar>, typename BoxSDF<Scalar>::Ptr,
             PrimitiveSDFBase<Scalar>>(m_psdf, "BoxSDF", py::module_local())
      .def(py::init<const Eigen::Matrix<Scalar, 3, 1>&, const Pose<Scalar>&>())
      .def("evaluate_batch", &BoxSDF<Scalar>::evaluate_batch)
      .def("evaluate", &BoxSDF<Scalar>::evaluate)
      .def("is_outside", &BoxSDF<Scalar>::is_outside);
  py::class_<CylinderSDF<Scalar>, typename CylinderSDF<Scalar>::Ptr,
             PrimitiveSDFBase<Scalar>>(m_psdf, "CylinderSDF",
                                       py::module_local())
      .def(py::init<Scalar, Scalar, const Pose<Scalar>&>())
      .def("evaluate_batch", &CylinderSDF<Scalar>::evaluate_batch)
      .def("evaluate", &CylinderSDF<Scalar>::evaluate)
      .def("is_outside", &CylinderSDF<Scalar>::is_outside);
  py::class_<SphereSDF<Scalar>, typename SphereSDF<Scalar>::Ptr,
             PrimitiveSDFBase<Scalar>>(m_psdf, "SphereSDF", py::module_local())
      .def(py::init<Scalar, const Pose<Scalar>&>())
      .def("evaluate_batch", &SphereSDF<Scalar>::evaluate_batch)
      .def("evaluate", &SphereSDF<Scalar>::evaluate)
      .def("is_outside", &SphereSDF<Scalar>::is_outside);
}

template void bind_primitive_sdf<float>(py::module&);
template void bind_primitive_sdf<double>(py::module&);

}  // namespace primitive_sdf
