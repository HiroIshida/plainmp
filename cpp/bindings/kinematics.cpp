#include "bindings.hpp"
#include "kinematics/tinyfk.hpp"

using namespace tinyfk;

namespace plainmp::bindings {

class _KinematicModel : public KinematicModel<double> {
  // a utility class for easy binding
 public:
  using KinematicModel::KinematicModel;
  size_t add_new_link_py(const std::string& link_name,
                         const std::string& parent_name,
                         const std::array<double, 3>& position,
                         const std::array<double, 3>& rpy,
                         bool consider_rotation) {
    size_t parent_id = get_link_ids({parent_name})[0];
    return KinematicModel::add_new_link(parent_id, position, rpy,
                                        consider_rotation, link_name);
  }
};

void bind_kinematics_submodule(py::module& m) {
  auto m_tinyfk = m.def_submodule("tinyfk");
  py::class_<urdf::Link, urdf::LinkSharedPtr>(m_tinyfk, "Link")
      .def_readonly("name", &urdf::Link::name)
      .def_readonly("id", &urdf::Link::id);

  py::class_<KinematicModel<double>, std::shared_ptr<KinematicModel<double>>>(
      m_tinyfk, "KinematicModel_cpp", py::module_local());

  py::class_<_KinematicModel, std::shared_ptr<_KinematicModel>,
             KinematicModel<double>>(m_tinyfk, "KinematicModel",
                                     py::module_local())
      .def(py::init<std::string&>())
      .def("add_new_link", &_KinematicModel::add_new_link_py)
      .def("get_joint_position_limits",
           &_KinematicModel::get_joint_position_limits)
      .def("get_link_ids", &_KinematicModel::get_link_ids)
      .def("get_joint_ids", &_KinematicModel::get_joint_ids);
}

}  // namespace plainmp::bindings
