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
#include "plainmp/constraints/composite_constraint.hpp"
#include "plainmp/constraints/primitive.hpp"
#include "plainmp/constraints/primitive_com_in_polytope.hpp"
#include "plainmp/constraints/primitive_config_point.hpp"
#include "plainmp/constraints/primitive_fixed_zaxis.hpp"
#include "plainmp/constraints/primitive_link_pose.hpp"
#include "plainmp/constraints/primitive_relative_pose.hpp"
#include "plainmp/constraints/primitive_sphere_collision.hpp"
#include "plainmp/constraints/sequential_constraint.hpp"

using namespace plainmp::constraint;
using namespace plainmp::collision;
using namespace plainmp::kinematics;

namespace plainmp::bindings {

void bind_constraint_submodule(py::module& m) {
  auto cst_m = m.def_submodule("constraint");
  py::class_<ConstraintBase, ConstraintBase::Ptr>(cst_m, "ConstraintBase");
  py::class_<EqConstraintBase, EqConstraintBase::Ptr, ConstraintBase>(
      cst_m, "EqConstraintBase");
  py::class_<IneqConstraintBase, IneqConstraintBase::Ptr, ConstraintBase>(
      cst_m, "IneqConstraintBase");
  py::class_<ConfigPointCst, ConfigPointCst::Ptr, EqConstraintBase>(
      cst_m, "ConfigPointCst")
      .def(py::init<std::shared_ptr<kin::KinematicModel<double>>,
                    const std::vector<std::string>&, BaseType,
                    const Eigen::VectorXd&>())
      .def("update_kintree", &ConfigPointCst::update_kintree)
      .def("evaluate", &ConfigPointCst::evaluate)
      .def("cst_dim", &ConfigPointCst::cst_dim);
  py::class_<LinkPoseCst, LinkPoseCst::Ptr, EqConstraintBase>(cst_m,
                                                              "LinkPoseCst")
      .def(py::init<std::shared_ptr<kin::KinematicModel<double>>,
                    const std::vector<std::string>&, BaseType,
                    const std::vector<std::string>&,
                    const std::vector<Eigen::VectorXd>&>())
      .def("update_kintree", &LinkPoseCst::update_kintree)
      .def("evaluate", &LinkPoseCst::evaluate)
      .def("cst_dim", &LinkPoseCst::cst_dim);
  py::class_<RelativePoseCst, RelativePoseCst::Ptr, EqConstraintBase>(
      cst_m, "RelativePoseCst")
      .def(py::init<std::shared_ptr<kin::KinematicModel<double>>,
                    const std::vector<std::string>&, BaseType,
                    const std::string&, const std::string&,
                    const Eigen::Vector3d&>())
      .def("update_kintree", &RelativePoseCst::update_kintree)
      .def("evaluate", &RelativePoseCst::evaluate);
  py::class_<FixedZAxisCst, FixedZAxisCst::Ptr, EqConstraintBase>(
      cst_m, "FixedZAxisCst")
      .def(py::init<std::shared_ptr<kin::KinematicModel<double>>,
                    const std::vector<std::string>&, BaseType,
                    const std::string&>())
      .def("pdate_kintree", &FixedZAxisCst::update_kintree)
      .def("evaluate", &FixedZAxisCst::evaluate);
  py::class_<SphereAttachmentSpec>(cst_m, "SphereAttachmentSpec")
      .def(py::init<const std::string&, const Eigen::Matrix3Xd&,
                    Eigen::VectorXd, bool>())
      .def_readonly("parent_link_name", &SphereAttachmentSpec::parent_link_name)
      .def_readwrite("relative_positions",
                     &SphereAttachmentSpec::relative_positions)
      .def_readwrite("radii", &SphereAttachmentSpec::radii);

  py::class_<SphereCollisionCst, SphereCollisionCst::Ptr, IneqConstraintBase>(
      cst_m, "SphereCollisionCst")
      .def(py::init<std::shared_ptr<kin::KinematicModel<double>>,
                    const std::vector<std::string>&, BaseType,
                    const std::vector<SphereAttachmentSpec>&,
                    const std::vector<std::pair<std::string, std::string>>&,
                    std::optional<SDFBase::Ptr>>())
      .def("set_sdf", &SphereCollisionCst::set_sdf)
      .def("get_sdf", &SphereCollisionCst::get_sdf)
      .def("update_kintree", &SphereCollisionCst::update_kintree)
      .def("is_valid", &SphereCollisionCst::is_valid)
      .def("evaluate", &SphereCollisionCst::evaluate)
      .def("get_group_spheres", &SphereCollisionCst::get_group_spheres)
      .def("get_all_spheres", &SphereCollisionCst::get_all_spheres);

  py::class_<AppliedForceSpec>(cst_m, "AppliedForceSpec")
      .def(py::init<const std::string&, double>())
      .def_readonly("link_name", &AppliedForceSpec::link_name)
      .def_readonly("force", &AppliedForceSpec::force);

  py::class_<ComInPolytopeCst, ComInPolytopeCst::Ptr, IneqConstraintBase>(
      cst_m, "ComInPolytopeCst")
      .def(py::init<std::shared_ptr<kin::KinematicModel<double>>,
                    const std::vector<std::string>&, BaseType, BoxSDF::Ptr,
                    const std::vector<AppliedForceSpec>&>())
      .def("update_kintree", &ComInPolytopeCst::update_kintree)
      .def("is_valid", &ComInPolytopeCst::is_valid)
      .def("evaluate", &ComInPolytopeCst::evaluate);

  py::class_<NullEqConstraint, NullEqConstraint::Ptr, EqConstraintBase>(
      cst_m, "NullEqConstraint")
      .def(py::init<std::shared_ptr<kin::KinematicModel<double>>,
                    const std::vector<std::string>&, BaseType>())
      .def("update_kintree", &NullEqConstraint::update_kintree)
      .def("evaluate", &NullEqConstraint::evaluate);

  py::class_<NullIneqConstraint, NullIneqConstraint::Ptr, IneqConstraintBase>(
      cst_m, "NullIneqConstraint")
      .def(py::init<std::shared_ptr<kin::KinematicModel<double>>,
                    const std::vector<std::string>&, BaseType>())
      .def("update_kintree", &NullIneqConstraint::update_kintree)
      .def("evaluate", &NullEqConstraint::evaluate)
      .def("is_valid", &NullIneqConstraint::is_valid);

  py::class_<EqCompositeCst, EqCompositeCst::Ptr>(cst_m, "EqCompositeCst")
      .def(py::init<std::vector<EqConstraintBase::Ptr>>())
      .def("update_kintree", &EqCompositeCst::update_kintree)
      .def("evaluate", &EqCompositeCst::evaluate)
      .def_readonly("constraints", &EqCompositeCst::constraints_);
  py::class_<IneqCompositeCst, IneqCompositeCst::Ptr>(cst_m, "IneqCompositeCst")
      .def(py::init<std::vector<IneqConstraintBase::Ptr>>())
      .def("update_kintree", &IneqCompositeCst::update_kintree)
      .def("evaluate", &IneqCompositeCst::evaluate)
      .def("is_valid", &IneqCompositeCst::is_valid)
      .def("__str__", &IneqCompositeCst::to_string)
      .def_readonly("constraints", &IneqCompositeCst::constraints_);
  py::class_<SequentialCst, SequentialCst::Ptr>(cst_m, "SequentialCst")
      .def(py::init<size_t, size_t>())
      .def("add_globally", &SequentialCst::add_globally)
      .def("add_at", &SequentialCst::add_at)
      .def("add_motion_step_box_constraint",
           &SequentialCst::add_motion_step_box_constraint)
      .def("add_fixed_point_at", &SequentialCst::add_fixed_point_at)
      .def("finalize", &SequentialCst::finalize)
      .def("evaluate", &SequentialCst::evaluate)
      .def("__str__", &SequentialCst::to_string)
      .def("x_dim", &SequentialCst::x_dim)
      .def("cst_dim", &SequentialCst::cst_dim);
}

}  // namespace plainmp::bindings
