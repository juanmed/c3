#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "core/lcs.h"
#include "multibody/geom_geom_collider.h"
#include "multibody/lcs_factory.h"
#include "multibody/lcs_factory_options.h"
#include "multibody/multibody_utils.h"

#include "drake/bindings/pydrake/common/sorted_pair_pybind.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/input_port.h"

namespace py = pybind11;
namespace c3 {
namespace multibody {
namespace pyc3 {
PYBIND11_MODULE(multibody, m) {
  m.doc() = "C3 Multibody Utilities";

  py::module multibody_module = py::module::import("pydrake.multibody.plant");
  py::object multibody_class = multibody_module.attr("MultibodyPlant");
  multibody_class.attr("DeclareSurfaceVelocityInputPort") = py::cpp_function(
      [](drake::multibody::MultibodyPlant<double>* plant,
         const drake::geometry::GeometryId& geometry_id,
         const Eigen::Vector3d& default_velocity_normal, double default_speed) {
        plant->DeclareSurfaceVelocityInputPort(geometry_id,
                                               default_velocity_normal,
                                               default_speed);
      },
      py::is_method(multibody_class), py::arg("geometry_id"),
      py::arg("default_velocity_normal"), py::arg("default_speed"));
  multibody_class.attr("get_surface_speed_input_port") = py::cpp_function(
      [](drake::multibody::MultibodyPlant<double>* plant,
         const drake::geometry::GeometryId& geometry_id) -> py::object {
        auto maybe_port = plant->get_surface_speed_input_port(geometry_id);
        if (!maybe_port.has_value()) {
          return py::none();
        }
        auto& port = maybe_port.value().get();
        return py::cast(&port, py::return_value_policy::reference_internal,
                        py::cast(plant));
      },
      py::is_method(multibody_class), py::arg("geometry_id"));
  multibody_class.attr("get_surface_velocity_normal_input_port") =
      py::cpp_function(
          [](drake::multibody::MultibodyPlant<double>* plant,
             const drake::geometry::GeometryId& geometry_id) -> py::object {
            auto maybe_port =
                plant->get_surface_velocity_normal_input_port(geometry_id);
            if (!maybe_port.has_value()) {
              return py::none();
            }
            auto& port = maybe_port.value().get();
            return py::cast(&port,
                            py::return_value_policy::reference_internal,
                            py::cast(plant));
          },
          py::is_method(multibody_class), py::arg("geometry_id"));

  // LCSFactory Class and ContactModel enum
  py::enum_<c3::multibody::ContactModel>(m, "ContactModel")
      .value("Unknown", c3::multibody::ContactModel::kUnknown)
      .value("StewartAndTrinkle",
             c3::multibody::ContactModel::kStewartAndTrinkle)
      .value("Anitescu", c3::multibody::ContactModel::kAnitescu)
      .value("FrictionlessSpring",
             c3::multibody::ContactModel::kFrictionlessSpring)
      .export_values();

  py::class_<c3::multibody::LCSContactDescription>(m, "LCSContactDescription")
      .def(py::init<>())
      .def_readwrite("witness_point_A",
                     &c3::multibody::LCSContactDescription::witness_point_A)
      .def_readwrite("witness_point_B",
                     &c3::multibody::LCSContactDescription::witness_point_B)
      .def_readwrite("force_basis",
                     &c3::multibody::LCSContactDescription::force_basis)
      .def_readwrite("is_slack",
                     &c3::multibody::LCSContactDescription::is_slack)
      .def_static("CreateSlackVariableDescription",
                  &c3::multibody::LCSContactDescription::
                      CreateSlackVariableDescription);

  py::class_<c3::multibody::LCSFactory>(m, "LCSFactory")
      .def(py::init<const drake::multibody::MultibodyPlant<double>&,
                    drake::systems::Context<double>&,
                    const drake::multibody::MultibodyPlant<drake::AutoDiffXd>&,
                    drake::systems::Context<drake::AutoDiffXd>&,
                    const std::vector<
                        drake::SortedPair<drake::geometry::GeometryId>>&,
                    const c3::LCSFactoryOptions&>(),
           py::arg("plant"), py::arg("context"), py::arg("plant_ad"),
           py::arg("context_ad"), py::arg("contact_geoms"), py::arg("options"))
      .def("GenerateLCS", &c3::multibody::LCSFactory::GenerateLCS)
      .def("GetContactDescriptions",
           &c3::multibody::LCSFactory::GetContactDescriptions)
      .def("UpdateStateAndInput",
           &c3::multibody::LCSFactory::UpdateStateAndInput, py::arg("state"),
           py::arg("input"))
      .def_static("LinearizePlantToLCS",
                  &c3::multibody::LCSFactory::LinearizePlantToLCS,
                  py::arg("plant"), py::arg("context"), py::arg("plant_ad"),
                  py::arg("context_ad"), py::arg("contact_geoms"),
                  py::arg("options"), py::arg("state"), py::arg("input"))
      .def_static("FixSomeModes", &c3::multibody::LCSFactory::FixSomeModes,
                  py::arg("other"), py::arg("active_lambda_inds"),
                  py::arg("inactive_lambda_inds"))
      // Overload the function GetNumContactVariables
      .def_static("GetNumContactVariables",
                  py::overload_cast<c3::multibody::ContactModel, int, int>(
                      &c3::multibody::LCSFactory::GetNumContactVariables),
                  py::arg("contact_model"), py::arg("num_contacts"),
                  py::arg("num_friction_directions"))
      .def_static("GetNumContactVariables",
                  py::overload_cast<const c3::LCSFactoryOptions>(
                      &c3::multibody::LCSFactory::GetNumContactVariables),
                  py::arg("options"));

  py::class_<LCSFactoryOptions>(m, "LCSFactoryOptions")
      .def(py::init<>())
      .def_readwrite("dt", &LCSFactoryOptions::dt)
      .def_readwrite("N", &LCSFactoryOptions::N)
      .def_readwrite("contact_model", &LCSFactoryOptions::contact_model)
      .def_readwrite("num_friction_directions",
                     &LCSFactoryOptions::num_friction_directions)
      .def_readwrite("num_contacts", &LCSFactoryOptions::num_contacts)
      .def_readwrite("spring_stiffness", &LCSFactoryOptions::spring_stiffness)
      .def_readwrite("mu", &LCSFactoryOptions::mu);

  m.def("LoadLCSFactoryOptions", &LoadLCSFactoryOptions);
}
}  // namespace pyc3
}  // namespace multibody
}  // namespace c3
