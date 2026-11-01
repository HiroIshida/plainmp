/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "plainmp/kinematics/kinematics.hpp"
#include <cstring>
#include <memory>
#include "plainmp/bindings/bindings.hpp"
#include "plainmp/kinematics/kinematic_model_wrapper.hpp"

using namespace plainmp::kinematics;

namespace plainmp::bindings {

namespace {
// Copy native contiguous float64 directly; use nanobind/NumPy conversion for
// other layouts and dtypes. memcpy also supports unaligned and readonly input.
void gravity_positions(nb::handle value, Eigen::VectorXd& result) {
  Py_buffer buffer;
  if (PyObject_GetBuffer(value.ptr(), &buffer,
                         PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) == 0) {
    const std::unique_ptr<Py_buffer, decltype(&PyBuffer_Release)> release(
        &buffer, PyBuffer_Release);
    if (buffer.ndim == 1 && buffer.itemsize == sizeof(double) &&
        buffer.format && std::strcmp(buffer.format, "d") == 0) {
      result.resize(buffer.shape[0]);
      if (result.size() != 0) {
        std::memcpy(result.data(), buffer.buf, buffer.len);
      }
      return;
    }
  } else {
    PyErr_Clear();
  }
  if (!nb::try_cast(value, result)) {
    result = nb::cast<Eigen::VectorXd>(nb::module_::import_("numpy").attr(
        "asarray")(value, nb::arg("dtype") = "float64"));
  }
}

// NumPy owns the result; avoid the cost of exporting an Eigen allocation.
nb::object gravity_array(const Eigen::VectorXd& value, nb::handle empty) {
  nb::object result = empty(value.size());
  Py_buffer buffer;
  if (PyObject_GetBuffer(result.ptr(), &buffer, PyBUF_WRITABLE) != 0) {
    throw nb::python_error();
  }
  if (value.size() != 0) {
    std::memcpy(buffer.buf, value.data(), value.size() * sizeof(double));
  }
  PyBuffer_Release(&buffer);
  return result;
}
}  // namespace

void bind_kinematics_submodule(nb::module_& m) {
  const nb::object numpy_empty = nb::module_::import_("numpy").attr("empty");
  auto m_kin = m.def_submodule("kinematics");
  nb::class_<urdf::Link>(m_kin, "Link")
      .def_ro("name", &urdf::Link::name)
      .def_ro("id", &urdf::Link::id);

  nb::enum_<BaseType>(m_kin, "BaseType")
      .value("FIXED", BaseType::FIXED)
      .value("FLOATING", BaseType::FLOATING)
      .value("PLANAR", BaseType::PLANAR);

  // parent class
  nb::class_<KinematicModel<double>>(m_kin, "KinematicModel_cpp");

  // child "binding" class
  nb::class_<utils::_KinematicModel, KinematicModel<double>>(m_kin,
                                                             "KinematicModel")
      .def(nb::init<std::string&>())
      .def("add_new_link",
           [](utils::_KinematicModel& self, const std::string& link_name,
              const std::string& parent_name, nb::sequence position,
              nb::sequence rpy, bool consider_rotation) {
             const Eigen::Vector3d position_vec =
                 vector3_from_sequence(position);
             const Eigen::Vector3d rpy_vec = vector3_from_sequence(rpy);
             return self.add_new_link_py(
                 link_name, parent_name,
                 {position_vec.x(), position_vec.y(), position_vec.z()},
                 {rpy_vec.x(), rpy_vec.y(), rpy_vec.z()}, consider_rotation);
           })
      .def("debug_get_link_pose", &utils::_KinematicModel::debug_get_link_pose)
      .def("set_joint_positions", &utils::_KinematicModel::set_joint_angles,
           nb::arg("joint_ids"), nb::arg("positions"),
           nb::arg("accurate") = true)
      .def(
          "set_joint_positions",
          [](utils::_KinematicModel& self, const std::vector<size_t>& joint_ids,
             nb::sequence positions, bool accurate) {
            Eigen::VectorXd vector = vector_from_sequence(positions);
            self.set_joint_angles(joint_ids, vector, accurate);
          },
          nb::arg("joint_ids"), nb::arg("positions"),
          nb::arg("accurate") = true)
      .def("get_joint_positions", &utils::_KinematicModel::get_joint_angles)
      .def("set_base_pose", &utils::_KinematicModel::set_base_pose)
      .def("set_base_pose",
           [](utils::_KinematicModel& self, nb::sequence values) {
             if (nb::len(values) != 7) {
               throw std::invalid_argument("expected a 7-element pose");
             }
             const Eigen::VectorXd vector = vector_from_sequence(values);
             self.set_base_pose(vector);
           })
      .def("get_base_pose", &utils::_KinematicModel::get_base_pose)
      .def("get_joint_position_limits",
           &utils::_KinematicModel::get_joint_position_limits)
      .def(
          "get_gravity_term",
          [numpy_empty](utils::_KinematicModel& self,
                        const std::vector<size_t>& joint_ids,
                        BaseType base_type) {
            return gravity_array(self.get_gravity_term(joint_ids, base_type),
                                 numpy_empty);
          },
          nb::arg("joint_ids"), nb::arg("base_type") = BaseType::FIXED)
      .def(
          "get_gravity_term2",
          [numpy_empty](utils::_KinematicModel& self,
                        const std::vector<size_t>& joint_ids,
                        nb::handle positions, BaseType base_type,
                        bool accurate) {
            auto& q = self.gravity_input_buffer_;
            gravity_positions(positions, q);
            return gravity_array(
                self.get_gravity_term2(joint_ids, q, base_type, accurate),
                numpy_empty);
          },
          nb::arg("joint_ids"), nb::arg("positions"),
          nb::arg("base_type") = BaseType::FIXED, nb::arg("accurate") = true)
      .def("get_link_ids", &utils::_KinematicModel::get_link_ids)
      .def("get_link_mass", &utils::_KinematicModel::get_link_mass,
           nb::arg("link_id"))
      .def("set_link_mass", &utils::_KinematicModel::set_link_mass,
           nb::arg("link_id"), nb::arg("mass"))
      .def("get_joint_ids", &utils::_KinematicModel::get_joint_ids);
}

}  // namespace plainmp::bindings
