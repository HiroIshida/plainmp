import matplotlib.pyplot as plt
import numpy as np
from skrobot.model.primitives import Box

from plainmp.ompl_solver import OMPLSolver, set_log_level_none
from plainmp.problem import Problem
from plainmp.psdf import GroundSDF, UnionSDF
from plainmp.robot_spec import FetchSpec
from plainmp.utils import primitive_to_plainmp_sdf

if __name__ == "__main__":
    # setup
    fs = FetchSpec()
    cst = fs.create_collision_const()
    table = Box([1.0, 2.0, 0.05], face_colors=[100, 200, 100, 200])
    table.translate([0.95, 0.0, 0.8])
    table_sdf = primitive_to_plainmp_sdf(table)
    sdf = UnionSDF([table_sdf, GroundSDF(0.0)])
    cst.set_sdf(sdf)
    lb, ub = fs.angle_bounds()
    q_start = np.array([0.0, 1.32, 1.40, -0.20, 1.72, 0.0, 1.66, 0.0])
    q_goal = np.array([0.386, 0.205, 1.41, 0.308, -1.82, 0.245, 0.417, 6.01])
    resolution = np.ones(8) * 0.05
    problem = Problem(q_start, lb, ub, q_goal, cst, None, resolution)
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
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("fetch_plan_bench.png", dpi=100)
    plt.show()
