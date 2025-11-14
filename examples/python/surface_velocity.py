"""Surface velocity example matching examples/surface_velocity_example.cc."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    Meshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    Simulator,
    System,
    ConstantVectorSource,
    ZeroOrderHold,
)
from pydrake.multibody.meshcat import ContactVisualizer, ContactVisualizerParams
from pydrake.systems.primitives import LogVectorOutput

from common_systems import C3Solution2Input, Vector2TimestampedVector
from pyc3 import (
    C3,
    C3Controller,
    LCSFactorySystem,
    LoadC3ControllerOptions,
)


def _declare_surface_velocity_port(plant):
    belt_body = plant.GetBodyByName("conveyor_belt")
    geom_id = plant.GetCollisionGeometriesForBody(belt_body)[0]
    plant.DeclareSurfaceVelocityInputPort(geom_id, np.array([0.0, 1.0, 0.0]), 0.0)
    return geom_id


def _make_conveyor_belt_plant(builder, sdf_path):
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
    parser = Parser(plant)
    parser.AddModels(sdf_path)
    geom_id = _declare_surface_velocity_port(plant)
    plant.set_name("conveyor_belt_plant")
    plant.Finalize()
    return plant, scene_graph, geom_id


def run_surface_velocity_example():
    conveyor_belt_sdf = "examples/resources/conveyor_belt/conveyor_belt.sdf"
    options_path = (
        "examples/resources/conveyor_belt/conveyor_belt_c3_options.yaml"
    )
    options = LoadC3ControllerOptions(options_path)
    costs = C3.CreateCostMatricesFromC3Options(
        options.c3_options, options.lcs_factory_options.N
    )

    # Build plant used for LCS linearization.
    plant_for_lcs_builder = DiagramBuilder()
    (
        plant_for_lcs,
        _scene_graph_for_lcs,
        lcs_geom_id,
    ) = _make_conveyor_belt_plant(plant_for_lcs_builder, conveyor_belt_sdf)
    plant_diagram = plant_for_lcs_builder.Build()
    plant_diagram_context = plant_diagram.CreateDefaultContext()
    plant_for_lcs_context = plant_diagram.GetMutableSubsystemContext(
        plant_for_lcs, plant_diagram_context
    )
    plant_autodiff = System.ToAutoDiffXd(plant_for_lcs)
    plant_autodiff_context = plant_autodiff.CreateDefaultContext()

    belt_geom = lcs_geom_id
    sphere_geom = plant_for_lcs.GetCollisionGeometriesForBody(
        plant_for_lcs.GetBodyByName("sphere")
    )[0]
    contact_pairs = [tuple(sorted((belt_geom, sphere_geom)))]

    # Build simulation diagram.
    builder = DiagramBuilder()
    plant_for_sim, scene_graph_for_sim, sim_geom_id = _make_conveyor_belt_plant(
        builder, conveyor_belt_sdf
    )
    plant_for_sim.set_name("plant_for_sim")

    q_names = plant_for_sim.GetPositionNames()
    v_names = plant_for_sim.GetVelocityNames()
    x_names = list(q_names) + list(v_names)

    print("Qs")
    for name in q_names:
        print(name)

    print("Vs")
    for name in v_names:
        print(name)

    print("Xs")
    for name in x_names:
        print(name)

    lcs_factory_system = builder.AddSystem(
        LCSFactorySystem(
            plant_for_lcs,
            plant_for_lcs_context,
            plant_autodiff,
            plant_autodiff_context,
            contact_pairs,
            options.lcs_factory_options,
        )
    )

    num_biases = lcs_factory_system.GetNumContactVelocityBiases()
    c3_controller = builder.AddSystem(
        C3Controller(plant_for_lcs, costs, options, num_biases)
    )
    c3_controller.set_name("c3_controller")

    xd = np.array([0.0, 0.0, 1.25, 0.0, 0.0, 0.0])
    xdes = builder.AddSystem(ConstantVectorSource(xd))

    n_x = plant_for_sim.num_positions() + plant_for_sim.num_velocities()
    vector_to_timestamped_vector = builder.AddSystem(Vector2TimestampedVector(n_x))
    builder.Connect(
        plant_for_sim.get_state_output_port(),
        vector_to_timestamped_vector.get_input_port_state(),
    )

    builder.Connect(
        vector_to_timestamped_vector.get_output_port_timestamped_state(),
        c3_controller.get_input_port_lcs_state(),
    )
    builder.Connect(
        lcs_factory_system.get_output_port_lcs(),
        c3_controller.get_input_port_lcs(),
    )
    builder.Connect(
        xdes.get_output_port(), c3_controller.get_input_port_target()
    )

    c3_input = builder.AddSystem(C3Solution2Input(1))
    builder.Connect(
        c3_controller.get_output_port_c3_solution(),
        c3_input.get_input_port_c3_solution(),
    )

    surface_speed_port = plant_for_sim.get_surface_speed_input_port(sim_geom_id)
    if surface_speed_port is None:
        raise RuntimeError("Surface speed input port was not declared.")
    builder.Connect(
        c3_input.get_output_port_c3_input(),
        surface_speed_port,
    )

    x_logger = LogVectorOutput(
        c3_input.get_output_port_c3_input(), builder
    )
    x_logger.set_name("surface_velocity_input")

    input_zero_order_hold = builder.AddSystem(
        ZeroOrderHold(1.0 / options.publish_frequency, 1)
    )
    builder.Connect(
        c3_input.get_output_port_c3_input(),
        input_zero_order_hold.get_input_port(),
    )
    builder.Connect(
        vector_to_timestamped_vector.get_output_port_timestamped_state(),
        lcs_factory_system.get_input_port_lcs_state(),
    )
    builder.Connect(
        input_zero_order_hold.get_output_port(),
        lcs_factory_system.get_input_port_lcs_input(),
    )

    meshcat = Meshcat()
    meshcat_params = MeshcatVisualizerParams()
    meshcat_params.delete_on_initialization_event = False
    meshcat_visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph_for_sim, meshcat, meshcat_params
    )
    cparams = ContactVisualizerParams()
    cparams.newtons_per_meter = 60.0
    ContactVisualizer.AddToBuilder(builder, plant_for_sim, meshcat, cparams)

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(
        plant_for_sim, diagram_context
    )
    q0 = plant_for_sim.GetPositions(plant_context)
    v0 = plant_for_sim.GetVelocities(plant_context)
    plant_for_sim.SetPositionsAndVelocities(plant_context, np.hstack([q0, v0]))

    diagram.ForcedPublish(diagram_context)

    dot_path = Path(__file__).resolve().parent / "conveyor_belt_diagram.dot"
    dot_path.write_text(diagram.GetGraphvizString())

    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()
    meshcat_visualizer.StartRecording()
    simulator.AdvanceTo(40.0)
    meshcat_visualizer.PublishRecording()

    log = x_logger.FindLog(simulator.get_context())
    plt.figure(1)
    plt.clf()
    plt.plot(log.sample_times(), log.data().T)
    plt.title("Control input")
    plt.xlabel("Time [s]")
    plt.ylabel("Surface speed [m/s]")
    plt.show()


if __name__ == "__main__":
    run_surface_velocity_example()
