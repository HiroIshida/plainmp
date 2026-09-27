"""Compare OMPL RRTConnect and VampRRTC using the same collision checker.

Run: .venv/bin/python example/bench/vamp_rrtc.py --runs 1000 --output /tmp/rrtc.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from skrobot.model.primitives import Box, Cylinder

from plainmp.ompl_solver import (
    Algorithm,
    OMPLSolver,
    OMPLSolverConfig,
    VampRRTCSettings,
    set_log_level_none,
    set_random_seed,
)
from plainmp.problem import Problem
from plainmp.psdf import GroundSDF, UnionSDF
from plainmp.robot_spec import FetchSpec, PandaSpec
from plainmp.utils import primitive_to_plainmp_sdf


def make_problem(scene, validator):
    if scene == "fetch":
        spec = FetchSpec()
        table = Box([1.0, 2.0, 0.05])
        table.translate([0.95, 0, 0.8])
        obstacles = [primitive_to_plainmp_sdf(table), GroundSDF(0.0)]
        start = np.array([0, 1.32, 1.40, -0.20, 1.72, 0, 1.66, 0])
        goal = np.array([0.386, 0.205, 1.41, 0.308, -1.82, 0.245, 0.417, 6.01])
    else:
        spec = PandaSpec()
        ground = Box([2, 2, 0.05])
        ground.translate([0, 0, -0.05])
        pole1, pole2 = Cylinder(0.05, 1.0), Cylinder(0.05, 1.0)
        pole1.translate([0.3, 0.3, 0.5])
        pole2.translate([-0.3, -0.3, 0.5])
        primitives = [ground, pole1, pole2]
        if scene == "panda_difficult":
            ceiling = Box([2, 2, 0.05])
            ceiling.translate([0, 0, 1])
            primitives.append(ceiling)
        obstacles = [primitive_to_plainmp_sdf(p) for p in primitives]
        start = np.array([-1.54, 1.54, 0, -0.1, 0, 1.5, 0.81])
        goal = np.array([1.54, 1.54, 0, -0.1, 0, 1.5, 0.81])
    constraint = spec.create_collision_const()
    constraint.set_sdf(UnionSDF(obstacles))
    lb, ub = spec.angle_bounds()
    resolution = np.full(len(start), 0.05) if validator == "box" else 0.05
    return Problem(start, lb, ub, goal, constraint, None, resolution, validator_type=validator)


def benchmark(runs, compare_nn=False, algorithm_range=2.0, compare_samplers=False):
    rows = []
    for scene in ["fetch", "panda_easy", "panda_difficult"]:
        for validator in ["box", "euclidean"]:
            problem = make_problem(scene, validator)
            halton, uniform = VampRRTCSettings(), VampRRTCSettings()
            halton.use_kdtree = uniform.use_kdtree = True
            uniform.use_halton = False
            algorithms = ["RRTConnect", "vamp_rrtc_halton", "vamp_rrtc_uniform"]
            solvers = [
                OMPLSolver(OMPLSolverConfig(algorithm=Algorithm.RRTConnect, timeout=2.0)),
                OMPLSolver(
                    OMPLSolverConfig(
                        algorithm=Algorithm.VampRRTC, timeout=2.0, vamp_rrtc_settings=halton
                    )
                ),
                OMPLSolver(
                    OMPLSolverConfig(
                        algorithm=Algorithm.VampRRTC, timeout=2.0, vamp_rrtc_settings=uniform
                    )
                ),
            ]
            if compare_nn:
                linear = VampRRTCSettings()
                linear.use_kdtree = False
                algorithms = ["vamp_rrtc_kdtree", "vamp_rrtc_linear"]
                solvers = [
                    solvers[1],
                    OMPLSolver(
                        OMPLSolverConfig(
                            algorithm=Algorithm.VampRRTC, timeout=2.0, vamp_rrtc_settings=linear
                        )
                    ),
                ]
            elif compare_samplers:
                algorithms = algorithms[1:]
                solvers = solvers[1:]
            for solver in solvers:
                solver.config.algorithm_range = algorithm_range
            samples = [[] for _ in solvers]
            for run in range(runs + 10):
                # Different Halton subsequences, as in VAMP's multi-seed
                # benchmarks. A fresh default planner otherwise repeats the
                # exact same deterministic search on every trial.
                halton.halton_skip = run * 1000
                if compare_nn:
                    linear.halton_skip = halton.halton_skip
                for i in [(run + j) % len(solvers) for j in range(len(solvers))]:
                    ts = time.perf_counter_ns()
                    result = solvers[i].solve(problem)
                    wall_ms = (time.perf_counter_ns() - ts) / 1e6
                    if run >= 10:
                        samples[i].append(
                            {
                                "success": result.success,
                                "wall_ms": wall_ms,
                                "internal_ms": None
                                if result.ns_internal is None
                                else result.ns_internal / 1e6,
                                "calls": result.n_call,
                                "length": None if result.traj is None else result.traj.get_length(),
                            }
                        )
            for algorithm, data in zip(algorithms, samples):
                successful = [s for s in data if s["success"]]
                row = {
                    "scene": scene,
                    "validator": validator,
                    "algorithm": algorithm,
                    "successes": len(successful),
                    "runs": runs,
                    "median_ms": float(np.median([s["wall_ms"] for s in data])),
                    "p95_ms": float(np.percentile([s["wall_ms"] for s in data], 95)),
                    "mean_ms": float(np.mean([s["wall_ms"] for s in data])),
                    "median_calls": float(np.median([s["calls"] for s in successful])),
                    "median_length": float(np.median([s["length"] for s in successful])),
                    "samples": data,
                }
                rows.append(row)
                print({k: v for k, v in row.items() if k != "samples"}, flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--range", dest="algorithm_range", type=float, default=2.0)
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument(
        "--compare-nn",
        action="store_true",
        help="Compare KD-tree and linear scan with identical Halton samples",
    )
    comparison.add_argument(
        "--compare-samplers",
        action="store_true",
        help="Compare Halton and uniform random sampling, both using KD-tree",
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/vamp_rrtc_bench.json"))
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    if not np.isfinite(args.algorithm_range) or args.algorithm_range <= 0:
        parser.error("--range must be positive and finite")
    set_log_level_none()
    set_random_seed(args.seed)
    args.output.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "range": args.algorithm_range,
                "compare_nn": args.compare_nn,
                "compare_samplers": args.compare_samplers,
                "results": benchmark(
                    args.runs, args.compare_nn, args.algorithm_range, args.compare_samplers
                ),
            },
            indent=2,
        )
        + "\n"
    )
