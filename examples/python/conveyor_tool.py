#!/usr/bin/env python3
"""Python replication of examples/conveyor_belt_tool.cc."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml
from pydrake.all import (
    ConstantVectorSource,
    DiagramBuilder,
    Demultiplexer,
    Meshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    Simulator,
    System,
    ZeroOrderHold,
)
from pydrake.geometry import SceneGraphConfig
from pydrake.multibody.meshcat import ContactVisualizer, ContactVisualizerParams
from pydrake.multibody.plant import AddMultibodyPlant, MultibodyPlantConfig
from pydrake.systems.primitives import LogVectorOutput

from common_systems import C3Solution2Input, Vector2TimestampedVector
from pyc3 import (
    C3,
    C3Controller,
    LCSFactorySystem,
    LoadC3ControllerOptions,
)

EXAMPLES_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = EXAMPLES_DIR / "resources" / "conveyor_belt"
DEFAULT_CONVEYOR_SDF = RESOURCES_DIR / "conveyor_belt_tool_1d.sdf"
DEFAULT_BOX_SDF = RESOURCES_DIR / "box.sdf"
DEFAULT_OPTIONS = RESOURCES_DIR / "conveyor_belt_tool_1d_c3_options.yaml"
DEFAULT_DIAGRAM_PATH = EXAMPLES_DIR / "conveyor_belt_tool_1d_diagram.dot"


@dataclass
class ConveyorToolSystem:
    builder: DiagramBuilder
    plant: object
    scene_graph: object
    diagram: Optional[object] = None


@dataclass
class ConveyorToolResult:
    steady_state_error: float
    settling_time: float
    state_times: np.ndarray
    state_trajectory: np.ndarray
    desired_state_times: np.ndarray
    desired_state_trajectory: np.ndarray
    control_times: np.ndarray
    control_trajectory: np.ndarray
    target_state: np.ndarray
    error_norm: np.ndarray
    c3_overrides: Optional[Dict[str, object]] = None


def _declare_surface_velocity_port(plant) -> None:
    belt_body = plant.GetBodyByName("conveyor_belt_tool")
    geom_id = plant.GetCollisionGeometriesForBody(belt_body)[0]
    plant.DeclareSurfaceVelocityInputPort(
        geom_id, np.array([0.0, 1.0, 0.0]), 0.5)


def _build_conveyor_system(
    *,
    name: str,
    conveyor_sdf: Union[str, Path],
    box_sdf: Union[str, Path],
    build_diagram: bool,
) -> ConveyorToolSystem:
    builder = DiagramBuilder()
    plant_config = MultibodyPlantConfig()
    plant_config.time_step = 0.005
    plant_config.penetration_allowance = 0.005
    plant_config.contact_model = "hydroelastic"
    plant_config.contact_surface_representation = "polygon"
    scene_graph_config = SceneGraphConfig()
    scene_graph_config.default_proximity_properties.margin = 1e-3
    plant, scene_graph = AddMultibodyPlant(
        plant_config=plant_config,
        scene_graph_config=scene_graph_config,
        builder=builder,
    )
    parser = Parser(plant)
    parser.AddModels(str(conveyor_sdf))
    parser.AddModels(str(box_sdf))
    _declare_surface_velocity_port(plant)
    plant.set_name(name)
    plant.Finalize()
    diagram = builder.Build() if build_diagram else None
    return ConveyorToolSystem(
        builder=builder, plant=plant, scene_graph=scene_graph, diagram=diagram
    )


def _extract_contact_pairs(plant) -> List[Tuple[object, object]]:
    belt_geom = plant.GetCollisionGeometriesForBody(
        plant.GetBodyByName("conveyor_belt_tool"))[0]
    box_geom = plant.GetCollisionGeometriesForBody(
        plant.GetBodyByName("box"))[0]
    floor_geom = plant.GetCollisionGeometriesForBody(
        plant.GetBodyByName("floor"))[0]
    return [
        tuple(sorted((belt_geom, box_geom))),
        tuple(sorted((box_geom, floor_geom))),
    ]


class ConveyorToolExperiment:
    """Builds and simulates the conveyor belt tool example."""

    def __init__(
        self,
        *,
        conveyor_sdf: Union[str, Path] = DEFAULT_CONVEYOR_SDF,
        box_sdf: Union[str, Path] = DEFAULT_BOX_SDF,
        options_path: Union[str, Path] = DEFAULT_OPTIONS,
        tracked_state_indices: Optional[Sequence[int]] = None,
        settling_tolerance: float = 5e-3,
        settling_epsilon: float = 2e-2,
        settling_window: float = 0.25,
        enable_visualization: bool = True,
        verbose: bool = False,
    ) -> None:
        self.conveyor_sdf = Path(conveyor_sdf).resolve()
        self.box_sdf = Path(box_sdf).resolve()
        self.options_path = Path(options_path).resolve()
        self.tracked_state_indices = (
            np.atleast_1d(tracked_state_indices).astype(int)
            if tracked_state_indices is not None
            else None
        )
        self.settling_tolerance = settling_tolerance
        self.settling_epsilon = settling_epsilon
        self.settling_window = settling_window
        self.enable_visualization = enable_visualization
        self.verbose = verbose

    def run(
        self,
        *,
        sim_time: float = 7.0,
        c3_overrides: Optional[Dict[str, object]] = None,
        diagram_path: Optional[Union[str, Path]] = DEFAULT_DIAGRAM_PATH,
    ) -> ConveyorToolResult:
        options = self._load_options(c3_overrides)

        costs = C3.CreateCostMatricesFromC3Options(
            options.c3_options, options.lcs_factory_options.N
        )

        plant_for_lcs_system = _build_conveyor_system(
            name="plant_for_lcs",
            conveyor_sdf=self.conveyor_sdf,
            box_sdf=self.box_sdf,
            build_diagram=True,
        )
        plant_for_sim_system = _build_conveyor_system(
            name="plant_for_sim",
            conveyor_sdf=self.conveyor_sdf,
            box_sdf=self.box_sdf,
            build_diagram=False,
        )

        contact_pairs = _extract_contact_pairs(plant_for_lcs_system.plant)
        plant_diagram_context = (
            plant_for_lcs_system.diagram.CreateDefaultContext())
        plant_for_lcs_context = (
            plant_for_lcs_system.diagram.GetMutableSubsystemContext(
                plant_for_lcs_system.plant, plant_diagram_context
            )
        )
        plant_autodiff = System.ToAutoDiffXd(plant_for_lcs_system.plant)
        plant_autodiff_context = plant_autodiff.CreateDefaultContext()

        builder = plant_for_sim_system.builder
        plant_for_sim = plant_for_sim_system.plant
        scene_graph_for_sim = plant_for_sim_system.scene_graph

        lcs_factory_system = builder.AddSystem(
            LCSFactorySystem(
                plant_for_lcs_system.plant,
                plant_for_lcs_context,
                plant_autodiff,
                plant_autodiff_context,
                contact_pairs,
                options.lcs_factory_options,
            )
        )

        num_biases = lcs_factory_system.GetNumContactVelocityBiases()
        c3_controller = builder.AddSystem(
            C3Controller(
                plant_for_lcs_system.plant,
                costs,
                options,
                num_biases,
            )
        )
        c3_controller.set_name("c3_controller")

        target_state = self._resolve_target_state(options, plant_for_sim)
        xdes = builder.AddSystem(ConstantVectorSource(target_state))

        n_x = plant_for_sim.num_positions() + plant_for_sim.num_velocities()
        vector_to_timestamped_vector = builder.AddSystem(
            Vector2TimestampedVector(n_x))
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
            xdes.get_output_port(),
            c3_controller.get_input_port_target(),
        )

        num_inputs = plant_for_sim.num_actuators()
        c3_input = builder.AddSystem(
            C3Solution2Input(num_inputs + num_biases))
        builder.Connect(
            c3_controller.get_output_port_c3_solution(),
            c3_input.get_input_port_c3_solution(),
        )

        state_demux_sizes: List[int]
        if num_biases:
            state_demux_sizes = [num_inputs, num_biases]
        else:
            state_demux_sizes = [num_inputs]
        input_demux = builder.AddSystem(Demultiplexer(state_demux_sizes))
        builder.Connect(
            c3_input.get_output_port_c3_input(),
            input_demux.get_input_port(),
        )
        builder.Connect(
            input_demux.get_output_port(0),
            plant_for_sim.get_actuation_input_port(),
        )

        if num_biases:
            geom_id = plant_for_sim.GetCollisionGeometriesForBody(
                plant_for_sim.GetBodyByName("conveyor_belt_tool"))[0]
            surface_port = plant_for_sim.get_surface_speed_input_port(geom_id)
            if surface_port is None:
                raise RuntimeError(
                    "Surface speed input port was not declared.")
            builder.Connect(
                input_demux.get_output_port(1),
                surface_port,
            )

        control_logger = LogVectorOutput(
            c3_input.get_output_port_c3_input(), builder)
        control_logger.set_name("c3_input_logger")
        state_logger = LogVectorOutput(
            plant_for_sim.get_state_output_port(), builder)
        state_logger.set_name("plant_state_logger")
        desired_state_logger = LogVectorOutput(
            xdes.get_output_port(), builder)
        desired_state_logger.set_name("desired_state_logger")

        input_zoh = builder.AddSystem(
            ZeroOrderHold(
                1.0 / options.publish_frequency,
                num_inputs + num_biases,
            )
        )
        builder.Connect(
            c3_input.get_output_port_c3_input(),
            input_zoh.get_input_port(),
        )
        builder.Connect(
            vector_to_timestamped_vector.get_output_port_timestamped_state(),
            lcs_factory_system.get_input_port_lcs_state(),
        )
        builder.Connect(
            input_zoh.get_output_port(),
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
            contact_params = ContactVisualizerParams()
            contact_params.newtons_per_meter = 60.0
            ContactVisualizer.AddToBuilder(
                builder, plant_for_sim, meshcat, contact_params)

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        plant_context = diagram.GetMutableSubsystemContext(
            plant_for_sim, context)

        if self.verbose:
            self._print_state_names(plant_for_sim)

        q0 = plant_for_sim.GetPositions(plant_context)
        v0 = plant_for_sim.GetVelocities(plant_context)
        plant_for_sim.SetPositionsAndVelocities(
            plant_context, np.hstack([q0, v0]))

        diagram.ForcedPublish(context)

        if diagram_path is not None:
            self._write_graphviz(diagram, diagram_path)

        simulator = Simulator(diagram, context)
        simulator.set_target_realtime_rate(1.0 if self.enable_visualization else 0.0)
        simulator.Initialize()
        if meshcat_visualizer is not None:
            meshcat_visualizer.StartRecording()
        simulator.AdvanceTo(sim_time)
        if meshcat_visualizer is not None:
            meshcat_visualizer.PublishRecording()

        state_log = state_logger.FindLog(simulator.get_context())
        control_log = control_logger.FindLog(simulator.get_context())
        desired_state_log = desired_state_logger.FindLog(
            simulator.get_context())

        state_times = np.asarray(state_log.sample_times())
        state_trajectory = np.asarray(state_log.data())
        control_times = np.asarray(control_log.sample_times())
        control_trajectory = np.asarray(control_log.data())
        desired_state_times = np.asarray(desired_state_log.sample_times())
        desired_state_traj = np.asarray(desired_state_log.data())

        steady_state_error, settling_time, error_norm = self._compute_metrics(
            state_times, state_trajectory, target_state
        )

        override_copy = None
        if c3_overrides:
            override_copy = {
                key: self._normalize_override_value(value)
                for key, value in c3_overrides.items()
            }

        return ConveyorToolResult(
            steady_state_error=steady_state_error,
            settling_time=settling_time,
            state_times=state_times,
            state_trajectory=state_trajectory,
            desired_state_times=desired_state_times,
            desired_state_trajectory=desired_state_traj,
            control_times=control_times,
            control_trajectory=control_trajectory,
            target_state=target_state,
            error_norm=error_norm,
            c3_overrides=override_copy,
        )

    def _print_state_names(self, plant) -> None:
        print("States:")
        for name in plant.GetStateNames():
            print(f"  {name}")

    def _write_graphviz(
        self, diagram, diagram_path: Union[str, Path]
    ) -> None:
        path = Path(diagram_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(diagram.GetGraphvizString())

    def _resolve_target_state(self, options, plant) -> np.ndarray:
        goal = getattr(options, "goal", None)
        if goal is None:
            goal = getattr(options, "goal_state", None)
        if goal is None:
            raise RuntimeError("C3 controller options do not specify a goal.")
        target_state = np.asarray(goal, dtype=float).flatten()
        n_states = plant.num_positions() + plant.num_velocities()
        if target_state.size != n_states:
            raise ValueError(
                f"Goal state size {target_state.size} does not match "
                f"plant state dimension {n_states}."
            )
        return target_state

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
        ) as tmp:
            yaml.safe_dump(options_data, tmp)
            tmp.flush()
            return LoadC3ControllerOptions(tmp.name)

    def _normalize_override_value(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return list(value)
        return value

    def _compute_metrics(
        self,
        times: np.ndarray,
        state_trajectory: np.ndarray,
        target_state: np.ndarray,
    ) -> Tuple[float, float, np.ndarray]:
        if state_trajectory.size == 0:
            return 0.0, 0.0, np.array([])
        tracked_indices = (
            self.tracked_state_indices
            if self.tracked_state_indices is not None
            else np.arange(target_state.size)
        )
        error = state_trajectory[tracked_indices, :] - target_state[tracked_indices, None]
        error_norm = np.linalg.norm(error, axis=0)
        steady_state_error = float(error_norm[-1])
        settling_time = self._compute_settling_time(times, error_norm)
        return steady_state_error, settling_time, error_norm

    def _compute_settling_time(self, times, error_norm):
        if len(times) == 0:
            return 0.0
        moving_average, ma_times = self._moving_average(
            error_norm,
            times,
            window=self.settling_window,
        )
        threshold = self.settling_tolerance + self.settling_epsilon
        for idx, value in enumerate(moving_average):
            if value <= threshold and np.all(moving_average[idx:] <= threshold):
                return float(ma_times[idx])
        return float(times[-1])

    def _moving_average(self, signal, timestamps, *, window):
        signal = np.asarray(signal, dtype=float)
        timestamps = np.asarray(timestamps, dtype=float)
        if signal.size == 0 or signal.size != timestamps.size:
            raise ValueError(
                "signal and timestamps must be non-empty and aligned.")
        dt = np.mean(np.diff(timestamps))
        if dt <= 0:
            raise ValueError("timestamps must be strictly increasing.")
        window_samples = max(1, int(round(window / dt)))
        weights = np.ones(window_samples, dtype=float) / window_samples
        moving_average = np.convolve(signal, weights, mode="valid")
        ma_times = timestamps[window_samples - 1:]
        return moving_average, ma_times


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate the conveyor belt tool example with the C3 controller."
        )
    )
    parser.add_argument(
        "--sim-time",
        type=float,
        default=7.0,
        help="Simulation horizon in seconds.",
    )
    parser.add_argument(
        "--settling-tolerance",
        type=float,
        default=5e-3,
        help="Settling tolerance for the error norm.",
    )
    parser.add_argument(
        "--settling-epsilon",
        type=float,
        default=2e-2,
        help="Slack added to the settling tolerance.",
    )
    parser.add_argument(
        "--settling-window",
        type=float,
        default=0.25,
        help="Duration (s) for the moving-average window.",
    )
    parser.add_argument(
        "--diagram-path",
        type=Path,
        default=DEFAULT_DIAGRAM_PATH,
        help="Optional path to save the system diagram Graphviz file.",
    )
    parser.add_argument(
        "--c3-overrides-path",
        type=Path,
        default=None,
        help="Optional JSON file with C3 option overrides.",
    )
    parser.add_argument(
        "--result-path",
        type=Path,
        default=None,
        help="Optional JSON file where metrics will be stored.",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Disable Meshcat visualization.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print plant state names.",
    )
    return parser.parse_args()


def _load_overrides_from_json(path: Optional[Path]) -> Optional[Dict[str, object]]:
    if path is None:
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("C3 overrides JSON must contain a dictionary.")
    return data


def main() -> None:
    args = parse_args()
    overrides = _load_overrides_from_json(args.c3_overrides_path)
    experiment = ConveyorToolExperiment(
        settling_tolerance=args.settling_tolerance,
        settling_epsilon=args.settling_epsilon,
        settling_window=args.settling_window,
        enable_visualization=not args.no_visualization,
        verbose=args.verbose,
    )
    result = experiment.run(
        sim_time=args.sim_time,
        c3_overrides=overrides,
        diagram_path=args.diagram_path,
    )
    summary = {
        "steady_state_error": result.steady_state_error,
        "settling_time": result.settling_time,
    }
    print(json.dumps(summary, indent=2))
    if args.result_path:
        args.result_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **summary,
            "target_state": result.target_state.tolist(),
            "c3_overrides": result.c3_overrides,
        }
        args.result_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
