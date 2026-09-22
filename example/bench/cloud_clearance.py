"""Benchmark point-cloud clearance reuse with identical seeded RRTConnect runs."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types


def load_native(path):
    """Load the requested build without using an editable installation."""
    package = types.ModuleType("plainmp")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "src/plainmp")]
    sys.modules["plainmp"] = package
    spec = importlib.util.spec_from_file_location("plainmp._plainmp", path)
    native = importlib.util.module_from_spec(spec)
    sys.modules["plainmp._plainmp"] = native
    spec.loader.exec_module(native)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--points", type=int, choices=[10000, 100000], default=10000)
    parser.add_argument("--seed", type=int, default=179424673)
    parser.add_argument("--plans", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    load_native(args.native)
    import numpy as np
    from plainmp.ompl_solver import OMPLSolver, OMPLSolverConfig, set_log_level_none, set_random_seed
    from plainmp.problem import Problem
    from plainmp.psdf import CloudSDF, UnionSDF
    from plainmp.robot_spec import FetchSpec

    set_log_level_none()
    set_random_seed(args.seed)
    robot = FetchSpec()
    cst = robot.create_collision_const(use_cache=False)
    points = np.random.RandomState(424242).rand(args.points, 3) * [1.0, 2.0, 0.05]
    points += np.array([0.95, 0, 0.8]) - np.array([0.5, 1.0, 0.025])
    cst.set_sdf(UnionSDF([CloudSDF(points, 0.002)]))
    lb, ub = robot.angle_bounds()
    start = np.array([0, 1.32, 1.40, -0.20, 1.72, 0, 1.66, 0])
    goal = np.array([0.386, 0.205, 1.41, 0.308, -1.82, 0.245, 0.417, 6.01])
    problem = Problem(start, lb, ub, goal, cst, None, 1 / 32, "euclidean")
    solver = OMPLSolver(OMPLSolverConfig(n_max_call=1000000))
    reset = getattr(cst, "reset_clearance_cache", lambda: None)
    for _ in range(30):
        reset()
        solver.solve(problem)
    result = dict(seed=args.seed, points=args.points, ns_internal=[], ns_wall=[], calls=[], paths=[])
    for _ in range(args.plans):
        start_time = time.perf_counter_ns()
        reset()
        ret = solver.solve(problem)
        result["ns_wall"].append(time.perf_counter_ns() - start_time)
        if not ret.success:
            raise RuntimeError("Planning failed")
        result["ns_internal"].append(ret.ns_internal)
        result["calls"].append(ret.n_call)
        result["paths"].append(hashlib.sha256(ret.traj.numpy().tobytes()).hexdigest())
    if args.reference:
        ref = json.loads(args.reference.read_text())
        assert (ref["seed"], ref["points"]) == (args.seed, args.points)
        assert ref["paths"] == result["paths"]
        assert ref["calls"] == result["calls"]
        print("Speedup:", np.median(ref["ns_internal"]) / np.median(result["ns_internal"]))
    args.output.write_text(json.dumps(result, indent=2))
    print("Median internal ms:", np.median(result["ns_internal"]) / 1e6)


if __name__ == "__main__":
    main()
