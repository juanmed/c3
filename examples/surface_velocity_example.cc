#include <fstream>

#include <drake/systems/primitives/constant_value_source.h>
#include <drake/systems/primitives/constant_vector_source.h>
#include <drake/systems/primitives/zero_order_hold.h>
#include <gflags/gflags.h>

#include "core/c3.h"
#include "examples/common_systems.hpp"
#include "systems/c3_controller.h"
#include "systems/c3_controller_options.h"
#include "systems/lcs_factory_system.h"
#include "systems/lcs_simulator.h"

#include "drake/common/proto/call_python.h"
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
#include "drake/systems/primitives/vector_log_sink.h"

using c3::C3;
using c3::systems::C3Controller;
using c3::systems::C3ControllerOptions;

using c3::systems::LCSFactorySystem;
using c3::systems::LCSSimulator;
using drake::common::CallPython;
using drake::common::ToPythonTuple;

class SineVectorGenerator : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SineVectorGenerator);
  SineVectorGenerator(int dims) : dims_(dims) {
    this->DeclareVectorOutputPort("sine_cosine",
                                  drake::systems::BasicVector<double>(dims_),
                                  &SineVectorGenerator::calc_output);
  }

  void calc_output(const drake::systems::Context<double>& context,
                   drake::systems::BasicVector<double>* output_vector) const {
    Eigen::VectorBlock<Eigen::VectorX<double>> output_value =
        output_vector->get_mutable_value();
    Eigen::VectorX<double> out = Eigen::VectorX<double>::Zero(dims_);
    if (context.get_time() < 0.5) {
      out(0) = 100;
    } else {
      out(0) = 0;
    }
    // out(1) = 10 * std::cos(3 * context.get_time()) + 2;
    // out(2) = 2 * std::sin(2.5 * context.get_time()) + 3;
    // out(3) = 3 * std::sin(2 * context.get_time()) + 4;
    // out(4) = 1 * std::cos(1.5 * context.get_time()) + 5;
    // out(5) = 4 * std::cos(1 * context.get_time()) + 0;
    output_value = out;
  }

 private:
  const int dims_;
};

int conveyor_belt_example() {
  drake::multibody::MultibodyPlantConfig config;
  config.time_step = 0.005;
  config.penetration_allowance = 0.001;
  config.contact_model =
      "hydroelastic";  // "hydroelastic" or "point" or "hydroelastic_with_fallback"
  config.contact_surface_representation = "polygon";  // "polygon" or "triangle"

  drake::geometry::SceneGraphConfig scene_graph_config;
  scene_graph_config.default_proximity_properties.margin = 1e-3;

  // Plant for LCS system
  drake::systems::DiagramBuilder<double> plant_for_lcs_builder;
  auto [plant_for_lcs, scene_graph_for_lcs] =
      drake::multibody::AddMultibodyPlant(config, scene_graph_config,
                                          &plant_for_lcs_builder);
  std::string conveyor_belt_url =
      "examples/resources/conveyor_belt/conveyor_belt.sdf";
  drake::multibody::Parser parser(&plant_for_lcs_builder);
  parser.AddModels(conveyor_belt_url);

  // Overrides the surface speed and surface velocity normal defined through
  // the sdf file, and also create their input ports to dynamically modify them.
  const drake::multibody::RigidBody<double>& conveyor_belt_body =
      plant_for_lcs.GetBodyByName("conveyor_belt");
  const drake::geometry::GeometryId geom_id =
      plant_for_lcs.GetCollisionGeometriesForBody(conveyor_belt_body).at(0);
  plant_for_lcs.DeclareSurfaceVelocityInputPort(
      geom_id, Eigen::Vector3d(0.0, 1.0, 0.0), 5.0);
  plant_for_lcs.set_name("plant_for_lcs");
  plant_for_lcs.Finalize();

  // auto plant_diagram = plant_for_lcs_builder.Build();

  // Set up visualization
  auto meshcat = std::make_shared<drake::geometry::Meshcat>();
  drake::geometry::MeshcatVisualizer<double>::AddToBuilder(
      &plant_for_lcs_builder, scene_graph_for_lcs, meshcat);
  drake::geometry::MeshcatVisualizerParams meshcat_params;
  meshcat_params.delete_on_initialization_event = false;
  auto& visualizer = drake::geometry::MeshcatVisualizerd::AddToBuilder(
      &plant_for_lcs_builder, scene_graph_for_lcs, meshcat,
      std::move(meshcat_params));
  drake::multibody::meshcat::ContactVisualizerParams cparams;
  cparams.newtons_per_meter = 60.0;
  drake::multibody::meshcat::ContactVisualizerd::AddToBuilder(
      &plant_for_lcs_builder, plant_for_lcs, meshcat, std::move(cparams));

  // Set up context
  std::unique_ptr<drake::systems::Diagram<double>> diagram =
      plant_for_lcs_builder.Build();
  std::unique_ptr<drake::systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram->SetDefaultContext(diagram_context.get());

  // auto& plant_context =
  //     diagram->GetMutableSubsystemContext(plant_for_lcs,
  //     diagram_context.get());
  // const auto q0 = plant_for_s.GetPositions(plant_context);
  // const auto v0 = plant_for_sim.GetVelocities(plant_context);
  // drake::VectorX<double> state(q0.size() + v0.size());
  // state << q0, v0;
  // plant_for_sim.SetPositionsAndVelocities(&plant_context, state);

  // // Force visualization
  // diagram->ForcedPublish(*diagram_context);

  // const std::string path =
  //     "/home/juanmedrano_eng/repos/c3/examples/conveyor_belt_diagram.dot";
  // std::ofstream graphviz(path);
  // std::map<std::string, std::string> options_gv{{"plant/split", "I/O"}};
  // graphviz << diagram->GetGraphvizString({}, options_gv);

  // Set up simulator
  drake::systems::Simulator<double> simulator(*diagram,
                                              std::move(diagram_context));
  simulator.set_target_realtime_rate(1.0);
  simulator.Initialize();
  visualizer.StartRecording();
  simulator.AdvanceTo(40.0);
  visualizer.PublishRecording();

  return 0;
}

int surface_velocity_example() {
  drake::multibody::MultibodyPlantConfig config;
  config.time_step = 0.0;
  config.penetration_allowance = 0.001;
  config.contact_model =
      "point";  // "hydroelastic" or "point" or "hydroelastic_with_fallback"
  config.contact_surface_representation = "polygon";  // "polygon" or "triangle"

  drake::geometry::SceneGraphConfig scene_graph_config;
  scene_graph_config.default_proximity_properties.margin = 1e-3;

  // Plant for LCS system
  drake::systems::DiagramBuilder<double> plant_for_lcs_builder;
  auto [plant_for_lcs, scene_graph_for_lcs] =
      drake::multibody::AddMultibodyPlant(config, scene_graph_config,
                                          &plant_for_lcs_builder);
  std::string conveyor_belt_url =
      "examples/resources/conveyor_belt/conveyor_belt.sdf";
  drake::multibody::Parser parser(&plant_for_lcs_builder);
  parser.AddModels(conveyor_belt_url);

  // Overrides the surface speed and surface velocity normal defined through
  // the sdf file, and also create their input ports to dynamic modify them.
  const drake::multibody::RigidBody<double>& conveyor_belt_body =
      plant_for_lcs.GetBodyByName("conveyor_belt");
  const drake::geometry::GeometryId geom_id =
      plant_for_lcs.GetCollisionGeometriesForBody(conveyor_belt_body).at(0);
  plant_for_lcs.DeclareSurfaceVelocityInputPort(
      geom_id, Eigen::Vector3d(0.0, 1.0, 0.0), 0.0);
  plant_for_lcs.set_name("plant_for_lcs");
  plant_for_lcs.Finalize();

  auto plant_diagram = plant_for_lcs_builder.Build();

  // Create contexts for the plant and LCS factory system.
  std::unique_ptr<drake::systems::Context<double>> plant_diagram_context =
      plant_diagram->CreateDefaultContext();
  auto plant_autodiff =
      drake::systems::System<double>::ToAutoDiffXd(plant_for_lcs);
  auto& plant_for_lcs_context = plant_diagram->GetMutableSubsystemContext(
      plant_for_lcs, plant_diagram_context.get());
  auto plant_context_autodiff = plant_autodiff->CreateDefaultContext();

  // Get contact geometry pairs
  std::vector<drake::SortedPair<drake::geometry::GeometryId>> contact_pairs;
  const drake::geometry::GeometryId geom_a =
      plant_for_lcs
          .GetCollisionGeometriesForBody(
              plant_for_lcs.GetBodyByName("conveyor_belt"))
          .at(0);
  const drake::geometry::GeometryId geom_b =
      plant_for_lcs
          .GetCollisionGeometriesForBody(plant_for_lcs.GetBodyByName("sphere"))
          .at(0);
  contact_pairs.push_back({geom_a, geom_b});

  // Add the LCS factory system.
  drake::systems::DiagramBuilder<double> plant_for_sim_builder;
  C3ControllerOptions options = drake::yaml::LoadYamlFile<C3ControllerOptions>(
      "examples/resources/conveyor_belt/conveyor_belt_c3_options.yaml");
  auto lcs_factory_system = plant_for_sim_builder.AddSystem<LCSFactorySystem>(
      plant_for_lcs, plant_for_lcs_context, *plant_autodiff,
      *plant_context_autodiff, contact_pairs, options.lcs_factory_options);

  // Create plant for simulation
  auto [plant_for_sim, scene_graph_for_sim] =
      drake::multibody::AddMultibodyPlant(config, scene_graph_config,
                                          &plant_for_sim_builder);
  drake::multibody::Parser parser_for_sim_plant(&plant_for_sim_builder);
  parser_for_sim_plant.AddModels(conveyor_belt_url);
  plant_for_sim.set_name("plant_for_sim");

  const drake::multibody::RigidBody<double>& sim_conveyor_belt_body =
      plant_for_sim.GetBodyByName("conveyor_belt");
  const drake::geometry::GeometryId sim_geom_id =
      plant_for_sim.GetCollisionGeometriesForBody(sim_conveyor_belt_body).at(0);
  plant_for_sim.DeclareSurfaceVelocityInputPort(
      sim_geom_id, Eigen::Vector3d(0.0, 1.0, 0.0), 0.0);
  plant_for_sim.Finalize();

  std::vector<std::string> q_names = plant_for_sim.GetPositionNames();
  std::vector<std::string> v_names = plant_for_sim.GetVelocityNames();
  std::vector<std::string> x_names = plant_for_sim.GetStateNames();

  std::cout << "Qs" << std::endl;
  for (const auto& q : q_names) {
    std::cout << q << std::endl;
  }

  std::cout << "Vs" << std::endl;
  for (const auto& q : v_names) {
    std::cout << q << std::endl;
  }

  std::cout << "Xs" << std::endl;
  for (const auto& q : x_names) {
    std::cout << q << std::endl;
  }

  // Add the C3 controller.
  C3::CostMatrices cost = C3::CreateCostMatricesFromC3Options(
      options.c3_options, options.lcs_factory_options.N);
  auto c3_controller = plant_for_sim_builder.AddSystem<C3Controller>(
      plant_for_lcs, cost, options,
      lcs_factory_system->GetNumContactVelocityBiases());
  c3_controller->set_name("c3_controller");
  c3_controller->AddLinearConstraint(Eigen::RowVectorXd::Ones(1), -12.0, 12.0,
                                     c3::ConstraintVariable::INPUT);

  // Add linear constratins to the controller
  // Eigen::MatrixXd A = Eigen::MatrixXd::Zero(14, 14);
  // A(3, 3) = 1;
  // A(4, 4) = 1;
  // A(5, 5) = 1;
  // A(6, 6) = 1;
  // Eigen::VectorXd lower_bound(14);
  // Eigen::VectorXd upper_bound(14);
  // lower_bound << 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  // upper_bound << 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0;
  // c3_controller->AddLinearConstraint(A, lower_bound, upper_bound,
  //                                    ConstraintVariable::STATE);

  // Add a constant vector source for the desired state.
  auto xdes = plant_for_sim_builder
                  .AddSystem<drake::systems::ConstantVectorSource<double>>(options.goal);

  // Add a vector-to-timestamped-vector converter.
  auto vector_to_timestamped_vector =
      plant_for_sim_builder.AddSystem<Vector2TimestampedVector>(6);

  // sim plant -> timestamped vector -> c3 controller
  plant_for_sim_builder.Connect(
      plant_for_sim.get_state_output_port(),
      vector_to_timestamped_vector->get_input_port_state());
  plant_for_sim_builder.Connect(
      vector_to_timestamped_vector->get_output_port_timestamped_state(),
      c3_controller->get_input_port_lcs_state());

  auto sim_state_logger = drake::systems::LogVectorOutput(
      plant_for_sim.get_state_output_port(), &plant_for_sim_builder);
  sim_state_logger->set_name("sim_state_logger");

  // lcs factory system -> c3 controller's LCS input
  // x_des -> c3 controller's x_des input
  plant_for_sim_builder.Connect(lcs_factory_system->get_output_port_lcs(),
                                c3_controller->get_input_port_lcs());
  plant_for_sim_builder.Connect(xdes->get_output_port(),
                                c3_controller->get_input_port_target());

  auto des_state_logger = drake::systems::LogVectorOutput(
      xdes->get_output_port(), &plant_for_sim_builder);
  des_state_logger->set_name("des_state_logger");


  const int lcs_num_inputs = plant_for_lcs.num_actuators();
  const int lcs_num_biases = lcs_factory_system->GetNumContactVelocityBiases();
  auto sine_vector_gen = plant_for_sim_builder.AddSystem<SineVectorGenerator>(
      lcs_num_inputs + lcs_num_biases);

  // c3 controller's output -> C3Solution2Input -> plant's conveyor speed input
  auto c3_input = plant_for_sim_builder.AddSystem<C3Solution2Input>(1);
  plant_for_sim_builder.Connect(c3_controller->get_output_port_c3_solution(),
                                c3_input->get_input_port_c3_solution());
  plant_for_sim_builder.Connect(
      //c3_input->get_output_port_c3_input(),
      sine_vector_gen->get_output_port(0),
      plant_for_sim.get_surface_speed_input_port(sim_geom_id).value().get());

  auto x_logger = drake::systems::LogVectorOutput(
      sine_vector_gen->get_output_port(0), 
      // c3_input->get_output_port_c3_input(), 
      &plant_for_sim_builder);
  x_logger->set_name("x_logger");

  // Add a ZeroOrderHold system for state updates.
  auto input_zero_order_hold =
      plant_for_sim_builder.AddSystem<drake::systems::ZeroOrderHold<double>>(
          1 / options.publish_frequency, 1);
  plant_for_sim_builder.Connect(c3_input->get_output_port_c3_input(),
                                input_zero_order_hold->get_input_port());
  plant_for_sim_builder.Connect(
      vector_to_timestamped_vector->get_output_port_timestamped_state(),
      lcs_factory_system->get_input_port_lcs_state());
  plant_for_sim_builder.Connect(input_zero_order_hold->get_output_port(),
                                lcs_factory_system->get_input_port_lcs_input());

  // Set up visualization
  auto meshcat = std::make_shared<drake::geometry::Meshcat>();
  drake::geometry::MeshcatVisualizer<double>::AddToBuilder(
      &plant_for_sim_builder, scene_graph_for_sim, meshcat);
  drake::geometry::MeshcatVisualizerParams meshcat_params;
  meshcat_params.delete_on_initialization_event = false;
  auto& visualizer = drake::geometry::MeshcatVisualizerd::AddToBuilder(
      &plant_for_sim_builder, scene_graph_for_sim, meshcat,
      std::move(meshcat_params));
  drake::multibody::meshcat::ContactVisualizerParams cparams;
  cparams.newtons_per_meter = 60.0;
  drake::multibody::meshcat::ContactVisualizerd::AddToBuilder(
      &plant_for_sim_builder, plant_for_sim, meshcat, std::move(cparams));

  // Set up context
  std::unique_ptr<drake::systems::Diagram<double>> diagram =
      plant_for_sim_builder.Build();
  std::unique_ptr<drake::systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram->SetDefaultContext(diagram_context.get());

  auto& plant_context =
      diagram->GetMutableSubsystemContext(plant_for_sim, diagram_context.get());
  // const auto q0 = plant_for_sim.GetPositions(plant_context);
  // const auto v0 = plant_for_sim.GetVelocities(plant_context);
  // drake::VectorX<double> state(q0.size() + v0.size());
  // state << q0, v0;
  // plant_for_sim.SetPositionsAndVelocities(&plant_context, state);

  // Force visualization
  diagram->ForcedPublish(*diagram_context);

  const std::string path =
      "/home/juanmedrano_eng/repos/c3/examples/conveyor_belt_diagram.dot";
  std::ofstream graphviz(path);
  std::map<std::string, std::string> options_gv{{"plant/split", "I/O"}};
  graphviz << diagram->GetGraphvizString({}, options_gv);

  // Set up simulator
  drake::systems::Simulator<double> simulator(*diagram,
                                              std::move(diagram_context));
  simulator.set_target_realtime_rate(1.0);
  simulator.Initialize();
  visualizer.StartRecording();
  simulator.AdvanceTo(2.0);
  visualizer.PublishRecording();

  // Plot data
  const auto& x_log = x_logger->FindLog(simulator.get_context());
  CallPython("figure", 1);
  CallPython("clf");
  CallPython("plot", x_log.sample_times(), x_log.data().transpose());
  CallPython("title", "Control input");

  const auto& state_log = sim_state_logger->FindLog(simulator.get_context());
  CallPython("figure", 2);
  CallPython("clf");
  CallPython("plot", state_log.sample_times(), state_log.data().transpose());
  CallPython("legend", ToPythonTuple("x", "z", "pitch", "vx", "vz", "wy"));
  CallPython("title", "Sim Plant State");
  CallPython("grid", true);

  const auto& des_state_log =
      des_state_logger->FindLog(simulator.get_context());
  CallPython("figure", 3);
  CallPython("clf");
  CallPython("plot", des_state_log.sample_times(),
             des_state_log.data().transpose());
  CallPython("legend",
             ToPythonTuple("x_d", "z_d", "pitch_d", "vx_d", "vz_d", "wy_d"));
  CallPython("title", "Sim Plant Desired State");
  CallPython("grid", true);

  return 0;
}

int main(int argc, char* argv[]) {
  // Initialize gflags.
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  surface_velocity_example();
  // conveyor_belt_example();
  return 0;
}
