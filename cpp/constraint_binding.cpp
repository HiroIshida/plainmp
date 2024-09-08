#include "constraint_binding.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "composite_constraint.hpp"
#include "constraint.hpp"
#include "sequential_constraint.hpp"

namespace py = pybind11;

namespace cst {

template <typename Scalar>
void bind_collision_constraints(py::module& m) {
  auto cst_m = m.def_submodule("constraint");
  py::class_<ConstraintBase<Scalar>, typename ConstraintBase<Scalar>::Ptr>(
      cst_m, "ConstraintBase");
  py::class_<EqConstraintBase<Scalar>, typename EqConstraintBase<Scalar>::Ptr,
             ConstraintBase<Scalar>>(cst_m, "EqConstraintBase");
  py::class_<IneqConstraintBase<Scalar>,
             typename IneqConstraintBase<Scalar>::Ptr, ConstraintBase<Scalar>>(
      cst_m, "IneqConstraintBase");
  py::class_<ConfigPointCst<Scalar>, typename ConfigPointCst<Scalar>::Ptr,
             EqConstraintBase<Scalar>>(cst_m, "ConfigPointCst")
      .def(py::init<std::shared_ptr<tinyfk::KinematicModel<Scalar>>,
                    const std::vector<std::string>&, bool,
                    const typename ConfigPointCst<Scalar>::Values&>())
      .def("update_kintree", &ConfigPointCst<Scalar>::update_kintree)
      .def("evaluate", &ConfigPointCst<Scalar>::evaluate)
      .def("cst_dim", &ConfigPointCst<Scalar>::cst_dim);
  py::class_<LinkPoseCst<Scalar>, typename LinkPoseCst<Scalar>::Ptr,
             EqConstraintBase<Scalar>>(cst_m, "LinkPoseCst")
      .def(py::init<std::shared_ptr<tinyfk::KinematicModel<Scalar>>,
                    const std::vector<std::string>&, bool,
                    const std::vector<std::string>&,
                    const std::vector<typename LinkPoseCst<Scalar>::Values>&>())
      .def("update_kintree", &LinkPoseCst<Scalar>::update_kintree)
      .def("evaluate", &LinkPoseCst<Scalar>::evaluate)
      .def("cst_dim", &LinkPoseCst<Scalar>::cst_dim);
  py::class_<RelativePoseCst<Scalar>, typename RelativePoseCst<Scalar>::Ptr,
             EqConstraintBase<Scalar>>(cst_m, "RelativePoseCst")
      .def(py::init<std::shared_ptr<tinyfk::KinematicModel<Scalar>>,
                    const std::vector<std::string>&, bool, const std::string&,
                    const std::string&, const Eigen::Matrix<Scalar, 3, 1>&>())
      .def("update_kintree", &RelativePoseCst<Scalar>::update_kintree)
      .def("evaluate", &RelativePoseCst<Scalar>::evaluate);
  py::class_<FixedZAxisCst<Scalar>, typename FixedZAxisCst<Scalar>::Ptr,
             EqConstraintBase<Scalar>>(cst_m, "FixedZAxisCst")
      .def(
          py::init<std::shared_ptr<tinyfk::KinematicModel<Scalar>>,
                   const std::vector<std::string>&, bool, const std::string&>())
      .def("pdate_kintree", &FixedZAxisCst<Scalar>::update_kintree)
      .def("evaluate", &FixedZAxisCst<Scalar>::evaluate);
  py::class_<SphereAttachmentSpec<Scalar>>(cst_m, "SphereAttachmentSpec")
      .def(py::init<const std::string&, const std::string&,
                    const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>&, Eigen::Matrix<Scalar, Eigen::Dynamic, 1>, bool>())
      .def_readonly("parent_link_name",
                    &SphereAttachmentSpec<Scalar>::parent_link_name);

  py::class_<SphereCollisionCst<Scalar>,
             typename SphereCollisionCst<Scalar>::Ptr,
             IneqConstraintBase<Scalar>>(cst_m, "SphereCollisionCst")
      .def(py::init<std::shared_ptr<tinyfk::KinematicModel<Scalar>>,
                    const std::vector<std::string>&, bool,
                    const std::vector<SphereAttachmentSpec<Scalar>>&,
                    const std::vector<std::pair<std::string, std::string>>&,
                    std::optional<typename SDFBase<Scalar>::Ptr>>())
      .def("set_sdf", &SphereCollisionCst<Scalar>::set_sdf)
      .def("get_sdf", &SphereCollisionCst<Scalar>::get_sdf)
      .def("update_kintree", &SphereCollisionCst<Scalar>::update_kintree)
      .def("is_valid", &SphereCollisionCst<Scalar>::is_valid)
      .def("evaluate", &SphereCollisionCst<Scalar>::evaluate)
      .def("get_group_spheres", &SphereCollisionCst<Scalar>::get_group_spheres)
      .def("get_all_spheres", &SphereCollisionCst<Scalar>::get_all_spheres);

  py::class_<AppliedForceSpec<Scalar>>(cst_m, "AppliedForceSpec")
      .def(py::init<const std::string&, Scalar>())
      .def_readonly("link_name", &AppliedForceSpec<Scalar>::link_name)
      .def_readonly("force", &AppliedForceSpec<Scalar>::force);

  py::class_<ComInPolytopeCst<Scalar>, typename ComInPolytopeCst<Scalar>::Ptr,
             IneqConstraintBase<Scalar>>(cst_m, "ComInPolytopeCst")
      .def(py::init<std::shared_ptr<tinyfk::KinematicModel<Scalar>>,
                    const std::vector<std::string>&, bool,
                    typename primitive_sdf::BoxSDF<Scalar>::Ptr,
                    const std::vector<AppliedForceSpec<Scalar>>&>())
      .def("update_kintree", &ComInPolytopeCst<Scalar>::update_kintree)
      .def("is_valid", &ComInPolytopeCst<Scalar>::is_valid)
      .def("evaluate", &ComInPolytopeCst<Scalar>::evaluate);
  py::class_<EqCompositeCst<Scalar>, typename EqCompositeCst<Scalar>::Ptr>(
      cst_m, "EqCompositeCst")
      .def(py::init<std::vector<typename EqConstraintBase<Scalar>::Ptr>>())
      .def("update_kintree", &EqCompositeCst<Scalar>::update_kintree)
      .def("evaluate", &EqCompositeCst<Scalar>::evaluate)
      .def_readonly("constraints", &EqCompositeCst<Scalar>::constraints_);
  py::class_<IneqCompositeCst<Scalar>, typename IneqCompositeCst<Scalar>::Ptr>(
      cst_m, "IneqCompositeCst")
      .def(py::init<std::vector<typename IneqConstraintBase<Scalar>::Ptr>>())
      .def("update_kintree", &IneqCompositeCst<Scalar>::update_kintree)
      .def("evaluate", &IneqCompositeCst<Scalar>::evaluate)
      .def("is_valid", &IneqCompositeCst<Scalar>::is_valid)
      .def("__str__", &IneqCompositeCst<Scalar>::to_string)
      .def_readonly("constraints", &IneqCompositeCst<Scalar>::constraints_);
  py::class_<SequentialCst<Scalar>, typename SequentialCst<Scalar>::Ptr>(
      cst_m, "SequentialCst")
      .def(py::init<size_t, size_t>())
      .def("add_globally", &SequentialCst<Scalar>::add_globally)
      .def("add_at", &SequentialCst<Scalar>::add_at)
      .def("add_motion_step_box_constraint",
           &SequentialCst<Scalar>::add_motion_step_box_constraint)
      .def("add_fixed_point_at", &SequentialCst<Scalar>::add_fixed_point_at)
      .def("finalize", &SequentialCst<Scalar>::finalize)
      .def("evaluate", &SequentialCst<Scalar>::evaluate)
      .def("__str__", &SequentialCst<Scalar>::to_string)
      .def("x_dim", &SequentialCst<Scalar>::x_dim)
      .def("cst_dim", &SequentialCst<Scalar>::cst_dim);
}

template void bind_collision_constraints<float>(py::module& m);
template void bind_collision_constraints<double>(py::module& m);

}  // namespace cst
