import argparse

import numpy as np
from matplotlib import pyplot as plt
from skrobot.model.primitives import Box, Cylinder

from plainmp.ompl_solver import OMPLSolver, set_log_level_none
from plainmp.problem import Problem
from plainmp.psdf import UnionSDF
from plainmp.robot_spec import PandaSpec
from plainmp.utils import primitive_to_plainmp_sdf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficult", action="store_true", help="difficult")
    args = parser.parse_args()

    # setup
    height = 1.0
    ground = Box([2.0, 2.0, 0.05])
    ground.translate([0, 0, -0.05])

    poll1 = Cylinder(0.05, height)
    poll1.translate([0.3, 0.3, 0.5 * height])

    poll2 = Cylinder(0.05, height)
    poll2.translate([-0.3, -0.3, 0.5 * height])

    primitives = [ground, poll1, poll2]

    if args.difficult:
        ceil = Box([2.0, 2.0, 0.05])
        ceil.translate([0, 0, height])
        primitives.append(ceil)

    sdf = UnionSDF([primitive_to_plainmp_sdf(p) for p in primitives])

    q0 = np.array([-1.54, 1.54, 0, -0.1, 0, 1.5, 0.81])
    q1 = np.array([1.54, 1.54, 0, -0.1, 0, 1.5, 0.81])
    spec = PandaSpec()
    cst = spec.create_collision_const()
    cst.set_sdf(sdf)

    lb, ub = spec.angle_bounds()
    resolution = np.ones(7) * 0.05
    problem = Problem(q0, lb, ub, q1, cst, None, resolution)
    solver = OMPLSolver()

    # bench
    set_log_level_none()
    time_list = []
    for _ in range(10000):
        ret = solver.solve(problem)
        time_list.append(ret.time_elapsed * 1000)  # ms
        assert ret.success

    # plot
    median = np.median(time_list)
    mean = np.mean(time_list)
    print(f"median: {median} ms")
    print(f"mean: {mean} ms")
    bins = np.logspace(np.log10(min(time_list)), np.log10(max(time_list)), 50)
    fig, ax = plt.subplots(figsize=(4, 3.2))
    ax.grid(which="both", axis="both", color="gray", linestyle="--", linewidth=0.5)
    ax.hist(time_list, bins=bins, alpha=0.7, color="blue", edgecolor="black")
    ax.axvline(median, color="red", linewidth=2, label=f"median: {median:.2f} ms")
    ax.axvline(mean, color="orange", linewidth=2, label=f"mean: {mean:.2f} ms")
    ax.set_xscale("log")
    ax.set_xlabel("planning time [ms]", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.set_xticks([0.1, 1, 10])
    ax.set_xticklabels(["0.1", "1", "10"], fontsize=14)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.7)
    plt.tight_layout()
    if args.difficult:
        plt.savefig("panda_plan_bench_difficult.png", dpi=100)
    else:
        plt.savefig("panda_plan_bench.png", dpi=100)
    plt.show()
