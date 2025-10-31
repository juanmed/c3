#include <fstream>
#include <gflags/gflags.h>


#include "drake/geometry/meshcat.h"
#include "drake/geometry/meshcat_visualizer.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/meshcat/contact_visualizer.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/sine.h"

int surface_velocity_example() {
  drake::multibody::MultibodyPlantConfig config;
  // We allow only discrete systems.
  config.time_step = 0.005;
  config.penetration_allowance = 0.001;
  config.contact_model = "hydroelastic"; // "hydroelastic" or "point" or "hydroelastic_with_fallback"
  config.contact_surface_representation = "polygon"; // "polygon" or "triangle"
 
  drake::geometry::SceneGraphConfig scene_graph_config;
  scene_graph_config.default_proximity_properties.margin = 1e-3;

  drake::systems::DiagramBuilder<double> builder;
  auto [plant, scene_graph] =
      drake::multibody::AddMultibodyPlant(config, scene_graph_config, &builder);
  std::string conveyor_belt_url =
      "examples/resources/conveyor_belt/conveyor_belt.sdf";
  drake::multibody::Parser parser(&builder);
  parser.AddModels(conveyor_belt_url);

  // Overrides the surface speed and surface velocity normal defined through
  // the sdf file, and also create their input ports to dynamic modify them.
  const drake::multibody::RigidBody<double>& body =
      plant.GetBodyByName("conveyor_belt");
  const drake::geometry::GeometryId geom_id =
      plant.GetCollisionGeometriesForBody(body).at(0);
  plant.DeclareSurfaceVelocityInputPort(geom_id, Eigen::Vector3d(0.0, 1.0, 0.0),
                                        1.0);
  plant.Finalize();

  // Set up visualization
  auto meshcat = std::make_shared<drake::geometry::Meshcat>();
  drake::geometry::MeshcatVisualizer<double>::AddToBuilder(&builder, scene_graph,
                                                    meshcat);
  drake::geometry::MeshcatVisualizerParams meshcat_params;
  meshcat_params.delete_on_initialization_event = false;
  auto& visualizer = drake::geometry::MeshcatVisualizerd::AddToBuilder(
      &builder, scene_graph, meshcat, std::move(meshcat_params));
  drake::multibody::meshcat::ContactVisualizerParams cparams;
  cparams.newtons_per_meter = 60.0;
  drake::multibody::meshcat::ContactVisualizerd::AddToBuilder(&builder, plant, meshcat,
                                                       std::move(cparams));

  // Set up context
  std::unique_ptr<drake::systems::Diagram<double>> diagram = builder.Build();
  std::unique_ptr<drake::systems::Context<double>> context =
      diagram->CreateDefaultContext();
  diagram->SetDefaultContext(context.get());

  // Force visualization
  diagram->ForcedPublish(*context);

  // Draw diagram of plant
  // std::ofstream graphviz(FLAGS_graphviz);
  // std::map<std::string, std::string> options{{"plant/split", "I/O"}};
  // graphviz << diagram->GetGraphvizString({}, options);

  // Set up simulator
  drake::systems::Simulator<double> simulator(*diagram);
  simulator.set_target_realtime_rate(1.0);
  simulator.Initialize();
  visualizer.StartRecording();
  simulator.AdvanceTo(20.0);
  visualizer.PublishRecording();

  return 0;
}

int main(int argc, char* argv[]) {
  // Initialize gflags.
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  surface_velocity_example();
  return 0;
}