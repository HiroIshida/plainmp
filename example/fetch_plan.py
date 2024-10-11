import argparse
import time

import numpy as np
from skrobot.model.primitives import Box, PointCloudLink
from skrobot.models import Fetch
from skrobot.viewers import PyrenderViewer

from plainmp.ompl_solver import OMPLSolver
from plainmp.problem import Problem
from plainmp.psdf import CloudSDF, UnionSDF
from plainmp.robot_spec import FetchSpec
from plainmp.utils import set_robot_state, sksdf_to_cppsdf

try:
    from skmp.robot.fetch import FetchConfig
    from skmp.visualization.collision_visualizer import (
        CollisionSphereVisualizationManager,
    )

    SKMP_AVAILABLE = True
except ImportError:
    SKMP_AVAILABLE = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcloud", action="store_true", help="use point cloud for collision check")
    parser.add_argument("--visualize", action="store_true", help="visualize the result")
    args = parser.parse_args()

    fs = FetchSpec()
    cst = fs.create_collision_const()

    if args.pcloud:
        table_points = np.random.rand(10000, 3) * np.array([1.0, 2.0, 0.05]) - np.array(
            [0.5, 1.0, 0.025]
        )
        table_points += np.array([0.9, 0.0, 0.8])
        table = PointCloudLink(table_points)
        table_sdf = CloudSDF(table_points, 0.002)
    else:
        table = Box([1.0, 2.0, 0.05], with_sdf=True)
        table.translate([0.9, 0.0, 0.8])
        table_sdf = sksdf_to_cppsdf(table.sdf)

    ground = Box([2.0, 2.0, 0.05], with_sdf=True)
    sdf = UnionSDF([table_sdf, sksdf_to_cppsdf(ground.sdf)], False)
    cst.set_sdf(sdf)
    lb, ub = fs.angle_bounds()
    start = np.array([0.0, 1.31999949, 1.40000015, -0.20000077, 1.71999929, 0.0, 1.6600001, 0.0])
    goal = np.array([0.386, 0.20565, 1.41370, 0.30791, -1.82230, 0.24521, 0.41718, 6.01064])
    msbox = np.ones(8) * 0.05
    problem = Problem(start, lb, ub, goal, cst, None, msbox)
    solver = OMPLSolver()

    ret = solver.solve(problem)
    assert ret.traj is not None
    print(f"planning time {1000 * (ret.time_elapsed)} [ms]")

    if args.visualize:
        fetch = Fetch()
        set_robot_state(fetch, fs.control_joint_names, goal)
        v = PyrenderViewer()

        if SKMP_AVAILABLE:
            conf = FetchConfig()
            colkin = conf.get_collision_kin()
            colvis = CollisionSphereVisualizationManager(colkin, v)
            colvis.update(fetch)

        v.add(fetch)
        v.add(table)
        v.add(ground)
        v.show()

        time.sleep(1.0)
        for q in ret.traj.resample(20):
            set_robot_state(fetch, fs.control_joint_names, q)
            if SKMP_AVAILABLE:
                colvis.update(fetch)
            v.redraw()
            time.sleep(0.5)

        time.sleep(1000)
