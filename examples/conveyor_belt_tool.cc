#include <iostream>
#include <memory>
#include <string>

#include <gflags/gflags.h>

#include "core/c3.h"
#include "examples/common_systems.hpp"
#include "systems/c3_controller.h"
#include "systems/c3_controller_options.h"
#include "systems/lcs_factory_system.h"
#include "systems/lcs_simulator.h"

#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/framework/diagram_builder.h"

struct ConveyorSystem {
  std::unique_ptr<drake::systems::DiagramBuilder<double>> builder;
  std::unique_ptr<drake::systems::Diagram<double>> diagram;
  drake::multibody::MultibodyPlant<double>* plant{};
  drake::geometry::SceneGraph<double>* scene_graph{};
};

ConveyorSystem setupLCSPlant(const std::string& name, bool build = true) {
  drake::multibody::MultibodyPlantConfig config;
  config.time_step = 0.0;  // continuous plant
  config.penetration_allowance = 0.001;
  config.contact_model = "point";
  config.contact_surface_representation = "polygon";

  drake::geometry::SceneGraphConfig scene_graph_config;
  scene_graph_config.default_proximity_properties.margin = 1e-3;

  auto lcs_builder = std::make_unique<drake::systems::DiagramBuilder<double>>();
  auto [plant_lcs, scene_graph_lcs] = drake::multibody::AddMultibodyPlant(
      config, scene_graph_config, lcs_builder.get());
  std::string conveyor_belt_tool_url =
      "examples/resources/conveyor_belt/conveyor_belt_tool.sdf";
  std::string box_url = "examples/resources/conveyor_belt/box.sdf";
  drake::multibody::Parser parser(lcs_builder.get());
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
  plant_lcs.set_name(name);
  plant_lcs.Finalize();

  std::unique_ptr<drake::systems::Diagram<double>> plant_diagram;
  if (build) plant_diagram = lcs_builder->Build();

  return {.builder = std::move(lcs_builder),
          .diagram = build ? std::move(plant_diagram) : nullptr,
          .plant = &plant_lcs,
          .scene_graph = &scene_graph_lcs};
}

std::vector<drake::SortedPair<drake::geometry::GeometryId>> extractContactPairs(
    const drake::multibody::MultibodyPlant<double>* plant) {
  std::vector<drake::SortedPair<drake::geometry::GeometryId>> contact_pairs;
  const drake::geometry::GeometryId geom_a =
      plant
          ->GetCollisionGeometriesForBody(
              plant->GetBodyByName("conveyor_belt_tool"))
          .at(0);
  const drake::geometry::GeometryId geom_b =
      plant->GetCollisionGeometriesForBody(plant->GetBodyByName("box")).at(0);
  contact_pairs.push_back({geom_a, geom_b});
  return contact_pairs;
}

int conveyor_belt_tool() {
  ConveyorSystem conveyor_lcs = setupLCSPlant("plant_for_lcs");
  ConveyorSystem conveyor_sim = setupLCSPlant("plant_for_sim", false);

  const auto prnt = [](const auto& e) { std::cout << e << std::endl; };
  auto u_ns = conveyor_lcs.plant->GetActuatorNames();
  std::cout << "inputs" << std::endl;
  std::for_each(u_ns.begin(), u_ns.end(), prnt);
  std::cout << "states" << std::endl;
  auto s_ns = conveyor_lcs.plant->GetStateNames();
  std::for_each(s_ns.begin(), s_ns.end(), prnt);

  // Get contact geometry pairs
  auto contact_pairs = extractContactPairs(conveyor_lcs.plant);

  // Create contexts for the plant and LCS factory system.
  std::unique_ptr<drake::systems::Context<double>> plant_diagram_context =
      conveyor_lcs.diagram->CreateDefaultContext();
  auto plant_autodiff =
      drake::systems::System<double>::ToAutoDiffXd(*conveyor_lcs.plant);
  auto& plant_for_lcs_context =
      conveyor_lcs.diagram->GetMutableSubsystemContext(
          *conveyor_lcs.plant, plant_diagram_context.get());
  auto plant_context_autodiff = plant_autodiff->CreateDefaultContext();

  // Add the LCS factory system.
  c3::systems::C3ControllerOptions options = drake::yaml::LoadYamlFile<
      c3::systems::C3ControllerOptions>(
      "examples/resources/conveyor_belt/conveyor_belt_tool_c3_options.yaml");
  auto lcs_factory_system =
      conveyor_sim.builder->AddSystem<c3::systems::LCSFactorySystem>(
          *conveyor_lcs.plant, plant_for_lcs_context, *plant_autodiff,
          *plant_context_autodiff, contact_pairs, options.lcs_factory_options);

  return 0;
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  conveyor_belt_tool();
}
