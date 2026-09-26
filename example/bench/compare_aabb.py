"""Run the three example scenes against separately built extension modules.

Run once per module/seed in a fresh process so OMPL's RNG starts identically.
The output includes per-solve times, query counts and exact trajectory hashes.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np


def load_extension(path):
    spec = importlib.util.spec_from_file_location("plainmp._plainmp", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def make_problem(case):
    from plainmp.problem import Problem
    from plainmp.psdf import BoxSDF, CylinderSDF, GroundSDF, Pose, UnionSDF
    from plainmp.robot_spec import FetchSpec, PandaSpec

    if case == "fetch":
        spec = FetchSpec()
        obstacles = [
            BoxSDF([1.0, 2.0, 0.05], Pose([0.95, 0.0, 0.8], np.eye(3))),
            GroundSDF(0.0),
        ]
        start = np.array([0.0, 1.32, 1.40, -0.20, 1.72, 0.0, 1.66, 0.0])
        goal = np.array([0.386, 0.205, 1.41, 0.308, -1.82, 0.245, 0.417, 6.01])
    else:
        spec = PandaSpec()
        obstacles = [
            BoxSDF([2.0, 2.0, 0.05], Pose([0.0, 0.0, -0.05], np.eye(3))),
            CylinderSDF(0.05, 1.0, Pose([0.3, 0.3, 0.5], np.eye(3))),
            CylinderSDF(0.05, 1.0, Pose([-0.3, -0.3, 0.5], np.eye(3))),
        ]
        if case == "panda_difficult":
            obstacles.append(BoxSDF([2.0, 2.0, 0.05], Pose([0.0, 0.0, 1.0], np.eye(3))))
        start = np.array([-1.54, 1.54, 0.0, -0.1, 0.0, 1.5, 0.81])
        goal = np.array([1.54, 1.54, 0.0, -0.1, 0.0, 1.5, 0.81])
    cst = spec.create_collision_const()
    cst.set_sdf(UnionSDF(obstacles))
    lb, ub = spec.angle_bounds()
    return Problem(start, lb, ub, goal, cst, None, np.full(len(start), 0.05))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--case", choices=["panda", "panda_difficult", "fetch"], required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--perf-control", help="perf stat --control command FIFO")
    parser.add_argument("--perf-ack", help="perf stat --control acknowledgement FIFO")
    args = parser.parse_args()
    if bool(args.perf_control) != bool(args.perf_ack):
        parser.error("--perf-control and --perf-ack must be provided together")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    load_extension(args.module.resolve())
    from plainmp.ompl_solver import OMPLSolver, set_log_level_none, set_random_seed

    set_log_level_none()
    set_random_seed(args.seed)
    np.random.seed(args.seed)
    problem = make_problem(args.case)
    solver = OMPLSolver()
    rows = []
    control = open(args.perf_control, "w", buffering=1) if args.perf_control else None
    ack = open(args.perf_ack) if args.perf_ack else None
    for i in range(args.warmup + args.samples):
        if control is not None and i == args.warmup:
            control.write("enable\n")
            assert ack.readline().lstrip("\0") == "ack\n"
        result = solver.solve(problem)
        assert result.success, (args.case, args.seed, i)
        if i >= args.warmup:
            path = np.asarray(result.traj.numpy(), dtype="<f8")
            rows.append(
                (
                    result.time_elapsed * 1000,
                    result.ns_internal / 1e6,
                    result.n_call,
                    hashlib.sha256(path.tobytes()).hexdigest(),
                )
            )
    if control is not None:
        control.write("disable\n")
        assert ack.readline().lstrip("\0") == "ack\n"
        control.close()
        ack.close()
    data = vars(args).copy()
    data.update(module=str(args.module.resolve()), output=str(args.output))
    data["columns"] = ["wall_ms", "internal_ms", "n_call", "path_sha256"]
    data["rows"] = rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data))
    times = np.array([row[0] for row in rows])
    print(
        f"{args.case} seed={args.seed} n={args.samples}: "
        f"mean={times.mean():.6f} ms median={np.median(times):.6f} ms",
        flush=True,
    )


if __name__ == "__main__":
    main()
