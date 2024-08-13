#include "constraint_binding.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "composite_constraint.hpp"
#include "constraint.hpp"
#include "sequential_constraint.hpp"

namespace py = pybind11;

namespace cst {

void bind_collision_constraints(py::module& m) {
  auto cst_m = m.def_submodule("constraint");
  py::class_<ConstraintBase, ConstraintBase::Ptr>(cst_m, "ConstraintBase");
  py::class_<EqConstraintBase, EqConstraintBase::Ptr, ConstraintBase>(
      cst_m, "EqConstraintBase");
  py::class_<IneqConstraintBase, IneqConstraintBase::Ptr, ConstraintBase>(
      cst_m, "IneqConstraintBase");
  py::class_<ConfigPointCst, ConfigPointCst::Ptr, EqConstraintBase>(
      cst_m, "ConfigPointCst")
      .def(py::init<std::shared_ptr<tinyfk::KinematicModel>,
                    const std::vector<std::string>&, bool,
                    const Eigen::VectorXd&>())
      .def("update_kintree", &ConfigPointCst::update_kintree)
      .def("evaluate", &ConfigPointCst::evaluate)
      .def("cst_dim", &ConfigPointCst::cst_dim);
  py::class_<LinkPoseCst, LinkPoseCst::Ptr, EqConstraintBase>(cst_m,
                                                              "LinkPoseCst")
      .def(py::init<std::shared_ptr<tinyfk::KinematicModel>,
                    const std::vector<std::string>&, bool,
                    const std::vector<std::string>&,
                    const std::vector<Eigen::VectorXd>&>())
      .def("update_kintree", &LinkPoseCst::update_kintree)
      .def("evaluate", &LinkPoseCst::evaluate)
      .def("cst_dim", &LinkPoseCst::cst_dim);
  py::class_<RelativePoseCst, RelativePoseCst::Ptr, EqConstraintBase>(
      cst_m, "RelativePoseCst")
      .def(py::init<std::shared_ptr<tinyfk::KinematicModel>,
                    const std::vector<std::string>&, bool, const std::string&,
                    const std::string&, const Eigen::Vector3d&>())
      .def("update_kintree", &RelativePoseCst::update_kintree)
      .def("evaluate", &RelativePoseCst::evaluate);
  py::class_<SphereAttachmentSpec>(cst_m, "SphereAttachmentSpec")
      .def(py::init<const std::string&, const std::string&,
                    const Eigen::Vector3d&, double, bool>())
      .def_readonly("parent_link_name",
                    &SphereAttachmentSpec::parent_link_name);

  py::class_<SphereCollisionCst, SphereCollisionCst::Ptr, IneqConstraintBase>(
      cst_m, "SphereCollisionCst")
      .def(py::init<std::shared_ptr<tinyfk::KinematicModel>,
                    const std::vector<std::string>&, bool,
                    const std::vector<SphereAttachmentSpec>&,
                    const std::vector<std::pair<std::string, std::string>>&,
                    std::optional<SDFBase::Ptr>>())
      .def("set_sdf", &SphereCollisionCst::set_sdf)
      .def("update_kintree", &SphereCollisionCst::update_kintree)
      .def("is_valid", &SphereCollisionCst::is_valid)
      .def("evaluate", &SphereCollisionCst::evaluate);

  py::class_<AppliedForceSpec>(cst_m, "AppliedForceSpec")
      .def(py::init<const std::string&, double>())
      .def_readonly("link_name", &AppliedForceSpec::link_name)
      .def_readonly("force", &AppliedForceSpec::force);

  py::class_<ComInPolytopeCst, ComInPolytopeCst::Ptr, IneqConstraintBase>(
      cst_m, "ComInPolytopeCst")
      .def(py::init<std::shared_ptr<tinyfk::KinematicModel>,
                    const std::vector<std::string>&, bool,
                    primitive_sdf::BoxSDF::Ptr,
                    const std::vector<AppliedForceSpec>&>())
      .def("update_kintree", &ComInPolytopeCst::update_kintree)
      .def("is_valid", &ComInPolytopeCst::is_valid)
      .def("evaluate", &ComInPolytopeCst::evaluate);
  py::class_<EqCompositeCst, EqCompositeCst::Ptr>(cst_m, "EqCompositeCst")
      .def(py::init<std::vector<EqConstraintBase::Ptr>>())
      .def("update_kintree", &EqCompositeCst::update_kintree)
      .def("evaluate", &EqCompositeCst::evaluate);
  py::class_<IneqCompositeCst, IneqCompositeCst::Ptr>(cst_m, "IneqCompositeCst")
      .def(py::init<std::vector<IneqConstraintBase::Ptr>>())
      .def("update_kintree", &IneqCompositeCst::update_kintree)
      .def("evaluate", &IneqCompositeCst::evaluate)
      .def("is_valid", &IneqCompositeCst::is_valid);

  py::class_<SequentialCst, SequentialCst::Ptr>(cst_m, "SequentialCst")
      .def(py::init<size_t, size_t>())
      .def("add_globally", &SequentialCst::add_globally)
      .def("add_at", &SequentialCst::add_at)
      .def("add_motion_step_box_constraint",
           &SequentialCst::add_motion_step_box_constraint)
      .def("add_fixed_point_at", &SequentialCst::add_fixed_point_at)
      .def("finalize", &SequentialCst::finalize)
      .def("evaluate", &SequentialCst::evaluate)
      .def("x_dim", &SequentialCst::x_dim)
      .def("cst_dim", &SequentialCst::cst_dim);
}

}  // namespace cst
