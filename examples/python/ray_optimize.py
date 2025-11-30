#!/usr/bin/env python3
"""Optimize C3 controllers for either surface velocity or conveyor tool experiments using Ray Tune."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import ray
from ray import air, tune

from conveyor_tool import ConveyorToolExperiment
from surface_velocity import SurfaceVelocityExperiment


REPO_ROOT = Path(__file__).resolve().parents[2]

# Change this to ConveyorToolExperiment to optimize the conveyor tool system.
EXPERIMENT_CLASS = ConveyorToolExperiment

FAILURE_PENALTY = 1e9


def _positive_bounds(value: float, factor: float = 10.0) -> Tuple[float, float]:
    base = max(value, 1e-6)
    lower = max(base / factor, 1e-6)
    upper = max(base * factor, lower * 1.1)
    return lower, upper


def build_param_space() -> Dict[str, object]:
    """Creates the Ray search space using log-uniform sampling."""
    param_space: Dict[str, tune.search.sample.Domain] = {}
    for name, values in EXPERIMENT_CLASS.BASE_C3_OPTIONS.items():
        if not values:
            continue
        if len(values) == 1:
            low, high = _positive_bounds(values[0])
            param_space[name] = tune.loguniform(low, high)
            continue
        for idx, value in enumerate(values):
            low, high = _positive_bounds(value)
            param_space[f"{name}_{idx}"] = tune.loguniform(low, high)
    return param_space


def config_to_c3_overrides(
    config: Dict[str, float]
) -> Dict[str, Union[List[float], float]]:
    """Converts the Ray config dictionary back to C3 option overrides."""
    overrides: Dict[str, List[float]] = {}
    for name, values in EXPERIMENT_CLASS.BASE_C3_OPTIONS.items():
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
        if key in EXPERIMENT_CLASS.SCALAR_OPTION_KEYS:
            normalized[key] = value[0]
        else:
            normalized[key] = value
    return normalized


def optimization_objective(
    config: Dict[str, float],
    *,
    sim_time: float,
    settling_tolerance: float,
    settling_weight: float,
) -> None:
    """Ray Tune objective that evaluates one controller configuration."""
    if Path.cwd() != REPO_ROOT:
        os.chdir(REPO_ROOT)
    overrides = config_to_c3_overrides(config)
    experiment = EXPERIMENT_CLASS(
        settling_tolerance=settling_tolerance,
        enable_visualization=False,
    )
    try:
        run_kwargs = {"sim_time": sim_time, "c3_overrides": overrides}
        if isinstance(experiment, ConveyorToolExperiment):
            run_kwargs["diagram_path"] = None
        result = experiment.run(**run_kwargs)
    except Exception as exc:
        tune.report(
            {
                "score": FAILURE_PENALTY,
                "steady_state_error": FAILURE_PENALTY,
                "settling_time": FAILURE_PENALTY,
                "failure": str(exc),
            }
        )
        print(str(exc))
        return

    score = result.steady_state_error + settling_weight * result.settling_time
    tune.report(
        {
            "score": score,
            "steady_state_error": result.steady_state_error,
            "settling_time": result.settling_time,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use Ray Tune to minimize the steady-state error and settling time "
            "of a selected C3-controlled experiment."
        )
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of Ray Tune samples to evaluate.",
    )
    parser.add_argument(
        "--sim-time",
        type=float,
        default=12.0,
        help="Simulation horizon (seconds) for each evaluation.",
    )
    parser.add_argument(
        "--settling-tolerance",
        type=float,
        default=5e-3,
        help="Tolerance used to determine settling time.",
    )
    parser.add_argument(
        "--settling-weight",
        type=float,
        default=1,
        help="Weight applied to the settling time inside the objective score.",
    )
    parser.add_argument(
        "--cpus-per-trial",
        type=float,
        default=1.0,
        help="CPU resources reserved per Ray trial.",
    )
    parser.add_argument(
        "--max-concurrent-trials",
        type=int,
        default=2,
        help="Maximum number of concurrent Ray Tune trials.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ray_results"),
        help="Directory where Ray will store trial data.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="c3_experiment",
        help="Name assigned to the Ray run (used in the output directory).",
    )
    parser.add_argument(
        "--best-result-path",
        type=Path,
        default=None,
        help="Optional path to store the best configuration as JSON.",
    )
    return parser.parse_args()


def save_best_result(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_uri = output_dir.as_uri()
    ray.init(ignore_reinit_error=True)

    param_space = build_param_space()
    objective = tune.with_resources(
        tune.with_parameters(
            optimization_objective,
            sim_time=args.sim_time,
            settling_tolerance=args.settling_tolerance,
            settling_weight=args.settling_weight,
        ),
        resources={"cpu": args.cpus_per_trial},
    )

    tuner = tune.Tuner(
        objective,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric="score",
            mode="min",
            num_samples=args.num_samples,
            max_concurrent_trials=args.max_concurrent_trials,
        ),
        run_config=air.RunConfig(
            name=args.run_name,
            storage_path=storage_uri,
        ),
    )

    result_grid = tuner.fit()
    best_result = result_grid.get_best_result(metric="score", mode="min")
    best_overrides = config_to_c3_overrides(best_result.config)
    summary = {
        "score": best_result.metrics["score"],
        "steady_state_error": best_result.metrics["steady_state_error"],
        "settling_time": best_result.metrics["settling_time"],
        "c3_overrides": best_overrides,
    }

    if args.best_result_path:
        save_best_result(summary, args.best_result_path)

    print("Best trial metrics:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
