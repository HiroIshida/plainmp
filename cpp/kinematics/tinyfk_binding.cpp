#include "tinyfk.hpp"
#include "tinyfk_binding.hpp"
#include <pybind11/stl.h>
#include <memory>

namespace tinyfk {

namespace py = pybind11;


class _KinematicModel : public KinematicModel {
  // a utility class for easy binding
  public:
  using KinematicModel::KinematicModel;
  size_t add_new_link_py(const std::string &link_name,
                                   const std::string&  parent_name,
                                   const std::array<double, 3> &position,
                                   const std::array<double, 3> &rpy,
                                   bool consider_rotation) {
    size_t parent_id = get_link_ids({parent_name})[0];
    auto link = KinematicModel::add_new_link(parent_id, position, rpy, consider_rotation, link_name);
    return link->id;
  }
};

void bind_tinyfk(py::module& m) {

  auto m_tinyfk = m.def_submodule("tinyfk");
  py::class_<urdf::Link, urdf::LinkSharedPtr>(m_tinyfk, "Link")
    .def_readonly("name", &urdf::Link::name)
    .def_readonly("id", &urdf::Link::id);

  py::class_<KinematicModel, std::shared_ptr<KinematicModel>>(m_tinyfk, "KinematicModel_cpp", py::module_local());

  py::class_<_KinematicModel, std::shared_ptr<_KinematicModel>, KinematicModel>(m_tinyfk, "KinematicModel", py::module_local())
      .def(py::init<std::string &>())
      .def("add_new_link", &_KinematicModel::add_new_link_py)
      .def("get_joint_position_limits", &_KinematicModel::get_joint_position_limits)
      .def("get_link_ids", &_KinematicModel::get_link_ids)
      .def("get_joint_ids", &_KinematicModel::get_joint_ids);
}

}  // namespace tinyfk
