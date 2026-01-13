#!/usr/bin/env python3
"""Optimize C3 controller gains using the Ax service API."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

try:
    from ax.service.ax_client import AxClient, ObjectiveProperties
except ImportError:  # pragma: no cover - ObjectiveProperties added in newer Ax
    from ax.service.ax_client import AxClient
    ObjectiveProperties = None  # type: ignore[assignment]

from conveyor_tool import ConveyorToolExperiment
from surface_velocity import SurfaceVelocityExperiment


REPO_ROOT = Path(__file__).resolve().parents[2]

FAILURE_PENALTY = 1e9
OBJECTIVE_NAME = "score"
TRACKING_METRICS = ("steady_state_error", "settling_time")

EXPERIMENT_CHOICES = {
    "conveyor": ConveyorToolExperiment,
    "surface": SurfaceVelocityExperiment,
}

ExperimentType = Type[Union[ConveyorToolExperiment, SurfaceVelocityExperiment]]


def _positive_bounds(value: float, factor: float = 10.0) -> Tuple[float, float]:
    base = max(value, 1e-6)
    lower = max(base / factor, 1e-6)
    upper = max(base * factor, lower * 1.1)
    return lower, upper


def build_param_space(experiment_cls: ExperimentType) -> List[Dict[str, object]]:
    """Constructs the Ax parameter space, mirroring the Ray search space."""
    parameters: List[Dict[str, object]] = []
    for name, values in experiment_cls.BASE_C3_OPTIONS.items():
        if not values:
            continue
        if len(values) == 1:
            low, high = _positive_bounds(values[0])
            parameters.append(
                {
                    "name": name,
                    "type": "range",
                    "bounds": [low, high],
                    "value_type": "float",
                    "log_scale": True,
                }
            )
            continue
        for idx, value in enumerate(values):
            low, high = _positive_bounds(value)
            parameters.append(
                {
                    "name": f"{name}_{idx}",
                    "type": "range",
                    "bounds": [low, high],
                    "value_type": "float",
                    "log_scale": True,
                }
            )
    return parameters


def config_to_c3_overrides(
    config: Dict[str, float],
    experiment_cls: ExperimentType,
) -> Dict[str, Union[List[float], float]]:
    """Translate Ax parameters back to the structure expected by C3."""
    overrides: Dict[str, List[float]] = {}
    for name, values in experiment_cls.BASE_C3_OPTIONS.items():
        if not values:
            continue
        if len(values) == 1:
            overrides[name] = [float(config[name])]
        else:
            overrides[name] = [
                float(config[f"{name}_{idx}"]) for idx in range(len(values))
            ]
    normalized: Dict[str, Union[List[float], float]] = {}
    for key, value in overrides.items():
        if key in experiment_cls.SCALAR_OPTION_KEYS:
            normalized[key] = value[0]
        else:
            normalized[key] = value
    return normalized


def evaluate_parameters(
    parameters: Dict[str, float],
    *,
    experiment_cls: ExperimentType,
    sim_time: float,
    settling_tolerance: float,
    settling_weight: float,
) -> Tuple[Dict[str, float], Optional[str]]:
    """Runs one simulation and returns the metrics along with a failure string."""
    if Path.cwd() != REPO_ROOT:
        os.chdir(REPO_ROOT)
    overrides = config_to_c3_overrides(parameters, experiment_cls)
    experiment = experiment_cls(
        settling_tolerance=settling_tolerance,
        enable_visualization=False,
    )
    try:
        run_kwargs = {"sim_time": sim_time, "c3_overrides": overrides}
        if isinstance(experiment, ConveyorToolExperiment):
            run_kwargs["diagram_path"] = None
        result = experiment.run(**run_kwargs)
    except Exception as exc:  # noqa: BLE001
        metrics = {
            "score": FAILURE_PENALTY,
            "steady_state_error": FAILURE_PENALTY,
            "settling_time": FAILURE_PENALTY,
        }
        return metrics, str(exc)

    score = result.steady_state_error + settling_weight * result.settling_time
    metrics = {
        "score": score,
        "steady_state_error": result.steady_state_error,
        "settling_time": result.settling_time,
    }
    return metrics, None


def save_best_result(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use Facebook Ax to minimize the sum of steady-state error and "
            "settling time for a selected C3-controlled experiment."
        )
    )
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENT_CHOICES.keys()),
        default="conveyor",
        help="Which experiment to optimize (surface or conveyor).",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=20,
        help="Total number of Ax trials to evaluate.",
    )
    parser.add_argument(
        "--sim-time",
        type=float,
        default=12.0,
        help="Simulation time horizon (seconds).",
    )
    parser.add_argument(
        "--settling-tolerance",
        type=float,
        default=5e-3,
        help="Tolerance value used for settling time computation.",
    )
    parser.add_argument(
        "--settling-weight",
        type=float,
        default=1.0,
        help="Weight applied to the settling time in the score.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed passed to Ax.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="c3_ax_optimize",
        help="Name assigned to the Ax experiment.",
    )
    parser.add_argument(
        "--best-result-path",
        type=Path,
        default=None,
        help="Optional JSON path where the best result will be stored.",
    )
    return parser.parse_args()


def _create_ax_experiment(
    ax_client: AxClient, *, name: str, parameters: List[Dict[str, object]]
) -> bool:
    """Handles API differences between legacy and modern Ax releases."""
    signature = inspect.signature(ax_client.create_experiment)
    kwargs: Dict[str, object] = {
        "name": name,
        "parameters": parameters,
    }
    params = signature.parameters
    if "objective_name" in params:
        kwargs["objective_name"] = OBJECTIVE_NAME
        if "minimize" in params:
            kwargs["minimize"] = True
    elif "objectives" in params:
        if ObjectiveProperties is None:
            raise RuntimeError(
                "The installed Ax version expects `objectives`, but "
                "ObjectiveProperties is unavailable. Please upgrade Ax."
            )
        kwargs["objectives"] = {
            OBJECTIVE_NAME: ObjectiveProperties(minimize=True)
        }
    else:
        raise RuntimeError(
            "Unsupported Ax version: cannot determine how to set the objective."
        )
    supports_tracking_metrics = False
    if "tracking_metric_names" in params:
        kwargs["tracking_metric_names"] = list(TRACKING_METRICS)
        supports_tracking_metrics = True
    ax_client.create_experiment(**kwargs)
    return supports_tracking_metrics


def main() -> None:
    args = parse_args()
    experiment_cls = EXPERIMENT_CHOICES[args.experiment]
    param_space = build_param_space(experiment_cls)

    ax_client = AxClient(random_seed=args.seed)
    supports_tracking_metrics = _create_ax_experiment(
        ax_client,
        name=args.run_name,
        parameters=param_space,
    )

    best_score = float("inf")
    best_parameters: Optional[Dict[str, float]] = None
    best_metrics: Optional[Dict[str, float]] = None

    for trial in range(args.num_trials):
        parameters, trial_index = ax_client.get_next_trial()
        metrics, failure = evaluate_parameters(
            parameters,
            experiment_cls=experiment_cls,
            sim_time=args.sim_time,
            settling_tolerance=args.settling_tolerance,
            settling_weight=args.settling_weight,
        )
        raw_data = {
            "score": (metrics["score"], 0.0),
        }
        if supports_tracking_metrics:
            raw_data.update(
                {
                    "steady_state_error": (metrics["steady_state_error"], 0.0),
                    "settling_time": (metrics["settling_time"], 0.0),
                }
            )
        ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)
        if failure:
            print(f"Trial {trial_index} failed: {failure}")
        else:
            print(
                f"Trial {trial_index}: score={metrics['score']:.4g}, "
                f"steady_state_error={metrics['steady_state_error']:.4g}, "
                f"settling_time={metrics['settling_time']:.4g}"
            )
        if metrics["score"] < best_score:
            best_score = metrics["score"]
            best_parameters = parameters.copy()
            best_metrics = metrics.copy()

    if best_parameters is None or best_metrics is None:
        raise RuntimeError("No successful trials were completed.")

    best_overrides = config_to_c3_overrides(best_parameters, experiment_cls)
    summary = {
        "score": best_metrics["score"],
        "steady_state_error": best_metrics["steady_state_error"],
        "settling_time": best_metrics["settling_time"],
        "c3_overrides": best_overrides,
    }

    if args.best_result_path is not None:
        save_best_result(summary, args.best_result_path)

    print("Best trial metrics:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
