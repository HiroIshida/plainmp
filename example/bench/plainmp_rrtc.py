"""Compare existing RRTConnect and plainmp_rrtc (uniform sampling, range 2.0).

Both high-level solve (including construction) and persistent C++ planner reuse.
Report wall time and ns_internal (excluding output matrix conversion) separately.
Run: .venv/bin/python example/bench/plainmp_rrtc.py --runs 1000 --output /tmp/plainmp_rrtc.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from vamp_rrtc import make_problem

from plainmp.ompl_solver import (
    Algorithm,
    OMPLPlanner,
    OMPLSolver,
    OMPLSolverConfig,
    PlainmpRRTCPlanner,
    ValidatorConfig,
    ValidatorType,
    set_log_level_none,
    set_random_seed,
)


def benchmark(runs, step_range):
    rows = []
    for scene in ["fetch", "panda_easy", "panda_difficult"]:
        for validation in ["box", "euclidean"]:
            problem = make_problem(scene, validation)
            for mode in ["solver", "reuse"]:
                algorithms = [Algorithm.RRTConnect, Algorithm.PlainmpRRTC]
                if mode == "solver":
                    solvers = [
                        OMPLSolver(OMPLSolverConfig(algorithm=a, algorithm_range=step_range))
                        for a in algorithms
                    ]

                    def solve(i):
                        result = solvers[i].solve(problem)
                        return result.success, result.n_call, result.ns_internal

                else:
                    validator = ValidatorConfig()
                    validator.type = (
                        ValidatorType.BOX if validation == "box" else ValidatorType.EUCLIDEAN
                    )
                    validator.resolution = 0.05
                    validator.box_width = np.full(len(problem.start), 0.05)
                    args = (problem.lb, problem.ub, problem.global_ineq_const, 1000000, validator)
                    planners = [
                        OMPLPlanner(*args, "RRTConnect", step_range),
                        PlainmpRRTCPlanner(*args, step_range),
                    ]

                    def solve(i):
                        planner = planners[i]
                        path = planner.solve(problem.start, problem.goal_const, [])
                        return path is not None, planner.get_call_count(), planner.get_ns_internal()

                samples = [[], []]
                for trial in range(runs + 10):
                    for i in [trial % 2, 1 - trial % 2]:
                        ts = time.perf_counter_ns()
                        success, calls, ns_internal = solve(i)
                        elapsed = (time.perf_counter_ns() - ts) / 1e6
                        if trial >= 10:
                            samples[i].append(
                                dict(
                                    success=success,
                                    ms=elapsed,
                                    calls=calls,
                                    ns_internal=ns_internal,
                                )
                            )
                for algorithm, data in zip(algorithms, samples):
                    internal_ms = [
                        d["ns_internal"] / 1e6
                        for d in data
                        if d["success"] and d["ns_internal"] is not None
                    ]
                    row = dict(
                        scene=scene,
                        validator=validation,
                        mode=mode,
                        algorithm=algorithm.value,
                        runs=runs,
                        successes=sum(d["success"] for d in data),
                        median_ms=float(np.median([d["ms"] for d in data])),
                        mean_ms=float(np.mean([d["ms"] for d in data])),
                        p95_ms=float(np.percentile([d["ms"] for d in data], 95)),
                        internal_median_ms=float(np.median(internal_ms)) if internal_ms else None,
                        internal_mean_ms=float(np.mean(internal_ms)) if internal_ms else None,
                        internal_p95_ms=float(np.percentile(internal_ms, 95))
                        if internal_ms
                        else None,
                        median_calls=float(np.median([d["calls"] for d in data])),
                        samples=data,
                    )
                    rows.append(row)
                    print({k: v for k, v in row.items() if k != "samples"}, flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--range", dest="step_range", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("/tmp/plainmp_rrtc.json"))
    args = parser.parse_args()
    if args.runs <= 0 or not np.isfinite(args.step_range) or args.step_range <= 0:
        parser.error("runs and range must be positive and range must be finite")
    set_log_level_none()
    set_random_seed(args.seed)
    rows = benchmark(args.runs, args.step_range)
    args.output.write_text(
        json.dumps(dict(seed=args.seed, range=args.step_range, rows=rows), indent=2)
    )
