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
#include "plainmp/constraints/primitive_link_position_bound.hpp"
#include "plainmp/constraints/primitive_relative_pose.hpp"
#include "plainmp/constraints/primitive_sphere_collision.hpp"
#include "plainmp/constraints/sequential_constraint.hpp"

using namespace plainmp::constraint;
using namespace plainmp::collision;
using namespace plainmp::kinematics;

namespace plainmp::bindings {

void bind_constraint_submodule(nb::module_& m) {
  auto cst_m = m.def_submodule("constraint");
  nb::class_<ConstraintBase>(cst_m, "ConstraintBase")
      .def("update_kintree", &ConstraintBase::update_kintree)
      .def("evaluate", &ConstraintBase::evaluate)
      .def("get_kin", &ConstraintBase::get_kin)
      .def("get_control_joint_names", &ConstraintBase::get_control_joint_names);
  nb::class_<EqConstraintBase, ConstraintBase>(cst_m, "EqConstraintBase");
  nb::class_<IneqConstraintBase, ConstraintBase>(cst_m, "IneqConstraintBase");
  nb::class_<ConfigPointCst, EqConstraintBase>(cst_m, "ConfigPointCst")
      .def(nb::init<std::shared_ptr<kin::KinematicModel<double>>,
                    const std::vector<std::string>&, BaseType,
                    const Eigen::VectorXd&>())
      .def("cst_dim", &ConfigPointCst::cst_dim);
  nb::class_<LinkPoseCst, EqConstraintBase>(cst_m, "LinkPoseCst")
      .def("__init__",
           [](LinkPoseCst* self,
              std::shared_ptr<kin::KinematicModel<double>> kin,
              const std::vector<std::string>& control_joint_names,
              BaseType base_type, const std::vector<std::string>& link_names,
              nb::sequence poses) {
             std::vector<Eigen::VectorXd> converted;
             converted.reserve(nb::len(poses));
             for (nb::handle pose : poses) {
               nb::sequence values = nb::cast<nb::sequence>(pose);
               converted.push_back(vector_from_sequence(values));
             }
             new (self) LinkPoseCst(std::move(kin), control_joint_names,
                                    base_type, link_names, converted);
           })
      .def("cst_dim", &LinkPoseCst::cst_dim)
      .def("get_desired_poses", &LinkPoseCst::get_desired_poses);
  nb::class_<RelativePoseCst, EqConstraintBase>(cst_m, "RelativePoseCst")
      .def(nb::init<std::shared_ptr<kin::KinematicModel<double>>,
                    const std::vector<std::string>&, BaseType,
                    const std::string&, const std::string&,
                    const Eigen::Vector3d&>());
  nb::class_<FixedZAxisCst, EqConstraintBase>(cst_m, "FixedZAxisCst")
      .def(nb::init<std::shared_ptr<kin::KinematicModel<double>>,
                    const std::vector<std::string>&, BaseType,
                    const std::string&>());
  nb::class_<SphereAttachmentSpec>(cst_m, "SphereAttachmentSpec")
      .def(nb::init<const std::string&, const Eigen::Matrix3Xd&,
                    Eigen::VectorXd, bool>())
      .def_ro("parent_link_name", &SphereAttachmentSpec::parent_link_name)
      .def_rw("relative_positions", &SphereAttachmentSpec::relative_positions)
      .def_rw("radii", &SphereAttachmentSpec::radii);

  nb::class_<SphereCollisionCst, IneqConstraintBase>(cst_m,
                                                     "SphereCollisionCst")
      .def(nb::init<std::shared_ptr<kin::KinematicModel<double>>,
                    const std::vector<std::string>&, BaseType,
                    const std::vector<SphereAttachmentSpec>&,
                    const std::vector<std::pair<std::string, std::string>>&,
                    std::optional<SDFBase::Ptr>, bool>(),
           nb::arg("kin"), nb::arg("control_joint_names"), nb::arg("base_type"),
           nb::arg("attachments"), nb::arg("self_collision_pairs"),
           nb::arg("sdf").none(), nb::arg("use_mesh"))
      .def("set_sdf", &SphereCollisionCst::set_sdf)
      .def("get_sdf", &SphereCollisionCst::get_sdf)
      .def("is_valid", &SphereCollisionCst::is_valid)
      .def("get_group_spheres", &SphereCollisionCst::get_group_spheres)
      .def("get_all_spheres", &SphereCollisionCst::get_all_spheres);

  nb::class_<AppliedForceSpec>(cst_m, "AppliedForceSpec")
      .def(nb::init<const std::string&, double>())
      .def_ro("link_name", &AppliedForceSpec::link_name)
      .def_ro("force", &AppliedForceSpec::force);

  nb::class_<ComInPolytopeCst, IneqConstraintBase>(cst_m, "ComInPolytopeCst")
      .def(nb::init<std::shared_ptr<kin::KinematicModel<double>>,
                    const std::vector<std::string>&, BaseType, BoxSDF::Ptr,
                    const std::vector<AppliedForceSpec>&>())
      .def("is_valid", &ComInPolytopeCst::is_valid);
  nb::class_<LinkPositionBoundCst, IneqConstraintBase>(cst_m,
                                                       "LinkPositionBoundCst")
      .def(nb::init<std::shared_ptr<kin::KinematicModel<double>>,
                    const std::vector<std::string>&, BaseType,
                    const std::string&, size_t, const std::optional<double>&,
                    const std::optional<double>&>(),
           nb::arg("kin"), nb::arg("control_joint_names"), nb::arg("base_type"),
           nb::arg("link_name"), nb::arg("axis"), nb::arg("lower_bound").none(),
           nb::arg("upper_bound").none())
      .def("is_valid", &LinkPositionBoundCst::is_valid);
  nb::class_<EqCompositeCst>(cst_m, "EqCompositeCst")
      .def(nb::init<std::vector<EqConstraintBase::Ptr>>())
      .def("evaluate", &EqCompositeCst::evaluate)
      .def_ro("constraints", &EqCompositeCst::constraints_);
  nb::class_<IneqCompositeCst>(cst_m, "IneqCompositeCst")
      .def(nb::init<std::vector<IneqConstraintBase::Ptr>>())
      .def("evaluate", &IneqCompositeCst::evaluate)
      .def("is_valid", &IneqCompositeCst::is_valid)
      .def("__str__", &IneqCompositeCst::to_string)
      .def_ro("constraints", &IneqCompositeCst::constraints_);
  nb::class_<SequentialCst>(cst_m, "SequentialCst")
      .def(nb::init<size_t, size_t>())
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
