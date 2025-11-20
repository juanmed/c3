#include <iostream>

#include <gflags/gflags.h>

#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"

int conveyor_belt_tool() {
  drake::multibody::MultibodyPlantConfig config;
  config.time_step = 0.0;  // continuous plant
  config.penetration_allowance = 0.001;
  config.contact_model = "point";
  config.contact_surface_representation = "polygon";

  drake::geometry::SceneGraphConfig scene_graph_config;
  scene_graph_config.default_proximity_properties.margin = 1e-3;

  drake::systems::DiagramBuilder<double> lcs_builder;
  auto [plant_lcs, scene_graph_lcs] = drake::multibody::AddMultibodyPlant(
      config, scene_graph_config, &lcs_builder);
  std::string conveyor_belt_tool_url =
      "examples/resources/conveyor_belt/conveyor_belt_tool.sdf";
  std::string box_url = "examples/resources/conveyor_belt/box.sdf";
  drake::multibody::Parser parser(&lcs_builder);
  parser.AddModels(conveyor_belt_tool_url);
  parser.AddModels(box_url);

  // Overrides the surface speed and surface velocity normal defined through
  // the sdf file, and also create their input ports to dynamic modify them.
  const drake::multibody::RigidBody<double>& conveyor_belt_body =
      plant_lcs.GetBodyByName("conveyor_belt_tool");
  const drake::geometry::GeometryId geom_id =
      plant_lcs.GetCollisionGeometriesForBody(conveyor_belt_body).at(0);
  plant_lcs.DeclareSurfaceVelocityInputPort(
      geom_id, Eigen::Vector3d(0.0, 1.0, 0.0), 0.5);
  plant_lcs.set_name("plant_lcs");
  plant_lcs.Finalize();

  return 0;
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  conveyor_belt_tool();
}
