"""Surface velocity example matching examples/surface_velocity_example.cc."""

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Dict, Optional, Sequence, Union

import yaml

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

EXAMPLES_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SDF_PATH = (
    EXAMPLES_DIR / "resources" / "conveyor_belt" / "conveyor_belt.sdf"
)
DEFAULT_OPTIONS_PATH = (
    EXAMPLES_DIR
    / "resources"
    / "conveyor_belt"
    / "conveyor_belt_c3_options.yaml"
)
DEFAULT_TARGET_STATE = np.array([0.0, 0.0, 1.25, 0.0, 0.0, 0.0])

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


@dataclass
class SurfaceVelocityResult:
    steady_state_error: float
    settling_time: float
    state_times: np.ndarray
    state_trajectory: np.ndarray
    target_state: np.ndarray
    error_norm: np.ndarray
    control_times: np.ndarray
    control_trajectory: np.ndarray
    c3_overrides: Optional[Dict[str, object]] = None


class SurfaceVelocityExperiment:
    """Builds and simulates the conveyor belt surface velocity example."""

    def __init__(
        self,
        *,
        conveyor_belt_sdf: Union[str, Path] = DEFAULT_SDF_PATH,
        options_path: Union[str, Path] = DEFAULT_OPTIONS_PATH,
        target_state: Optional[Sequence[float]] = None,
        settling_tolerance: float = 1e-2,
        enable_visualization: bool = False,
        verbose: bool = False,
    ) -> None:
        self.conveyor_belt_sdf = Path(conveyor_belt_sdf).resolve()
        self.options_path = Path(options_path).resolve()
        self.target_state = (
            np.array(target_state, dtype=float)
            if target_state is not None
            else DEFAULT_TARGET_STATE.copy()
        )
        self.settling_tolerance = settling_tolerance
        self.enable_visualization = enable_visualization
        self.verbose = verbose

    def run(
        self,
        *,
        sim_time: float = 40.0,
        c3_overrides: Optional[Dict[str, object]] = None,
    ) -> SurfaceVelocityResult:
        options = self._load_options(c3_overrides)
        costs = C3.CreateCostMatricesFromC3Options(
            options.c3_options, options.lcs_factory_options.N
        )

        plant_for_lcs_builder = DiagramBuilder()
        (
            plant_for_lcs,
            _scene_graph_for_lcs,
            lcs_geom_id,
        ) = _make_conveyor_belt_plant(
            plant_for_lcs_builder, str(self.conveyor_belt_sdf)
        )
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

        builder = DiagramBuilder()
        plant_for_sim, scene_graph_for_sim, sim_geom_id = _make_conveyor_belt_plant(
            builder, str(self.conveyor_belt_sdf)
        )
        plant_for_sim.set_name("plant_for_sim")

        if self.verbose:
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

        xdes = builder.AddSystem(ConstantVectorSource(self.target_state))

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

        control_logger = LogVectorOutput(
            c3_input.get_output_port_c3_input(), builder
        )
        control_logger.set_name("surface_velocity_input")

        state_logger = LogVectorOutput(
            plant_for_sim.get_state_output_port(), builder
        )
        state_logger.set_name("plant_state")

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

        meshcat_visualizer = None
        if self.enable_visualization:
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

        simulator = Simulator(diagram, diagram_context)
        # simulator.set_target_realtime_rate(1.0)
        simulator.Initialize()
        if meshcat_visualizer is not None:
            meshcat_visualizer.StartRecording()
        simulator.AdvanceTo(sim_time)
        if meshcat_visualizer is not None:
            meshcat_visualizer.PublishRecording()

        state_log = state_logger.FindLog(simulator.get_context())
        control_log = control_logger.FindLog(simulator.get_context())

        state_times = np.asarray(state_log.sample_times())
        state_trajectory = np.asarray(state_log.data())
        (
            steady_state_error,
            settling_time,
            error_norm,
        ) = self._compute_metrics(state_times, state_trajectory)

        control_times = np.asarray(control_log.sample_times())
        control_trajectory = np.asarray(control_log.data())

        overrides_copy = None
        if c3_overrides:
            overrides_copy = {
                key: self._normalize_override_value(value)
                for key, value in c3_overrides.items()
            }

        return SurfaceVelocityResult(
            steady_state_error=steady_state_error,
            settling_time=settling_time,
            state_times=state_times,
            state_trajectory=state_trajectory,
            target_state=self.target_state.copy(),
            error_norm=error_norm,
            control_times=control_times,
            control_trajectory=control_trajectory,
            c3_overrides=overrides_copy,
        )

    def _normalize_override_value(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return list(value)
        return value

    def _load_options(self, c3_overrides):
        if not c3_overrides:
            return LoadC3ControllerOptions(str(self.options_path))

        with self.options_path.open("r", encoding="utf-8") as handler:
            options_data = yaml.safe_load(handler)

        c3_data = options_data.get("c3_options", {})
        for key, value in c3_overrides.items():
            c3_data[key] = self._normalize_override_value(value)
        options_data["c3_options"] = c3_data

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".yaml", encoding="utf-8"
        ) as temp_file:
            yaml.safe_dump(options_data, temp_file)
            temp_file.flush()
            return LoadC3ControllerOptions(temp_file.name)

    def _compute_metrics(self, times, state_trajectory):
        if state_trajectory.size == 0:
            return 0.0, 0.0, np.array([])
        error = state_trajectory - self.target_state[:, None]
        error_norm = np.linalg.norm(error, axis=0)
        steady_state_error = float(error_norm[-1])
        settling_time = self._compute_settling_time(times, error_norm)
        return steady_state_error, settling_time, error_norm

    def _compute_settling_time(self, times, error_norm):
        if len(times) == 0:
            return 0.0
        satisfied = error_norm <= self.settling_tolerance
        for idx, is_satisfied in enumerate(satisfied):
            if is_satisfied and np.all(satisfied[idx:]):
                return float(times[idx])
        violating = np.where(~satisfied)[0]
        if violating.size == 0:
            return float(times[0])
        last_idx = int(violating[-1])
        last_idx = min(last_idx + 1, len(times) - 1)
        return float(times[last_idx])


def run_surface_velocity_example():
    experiment = SurfaceVelocityExperiment(
        enable_visualization=True,
        verbose=True,
    )
    result = experiment.run(sim_time=40.0)

    print(f"Steady-state error: {result.steady_state_error:.5f}")
    print(f"Settling time: {result.settling_time:.3f} s")

    plt.figure(1)
    plt.clf()
    plt.plot(result.control_times, result.control_trajectory.T)
    plt.title("Control input")
    plt.xlabel("Time [s]")
    plt.ylabel("Surface speed [m/s]")
    plt.show()
    return result



if __name__ == "__main__":
    run_surface_velocity_example()
