import argparse
import time

import numpy as np
from skrobot.model.primitives import Box, PointCloudLink, Sphere
from skrobot.viewers import PyrenderViewer

from plainmp.ompl_solver import OMPLSolver, set_log_level_none
from plainmp.problem import Problem
from plainmp.psdf import CloudSDF, GroundSDF, UnionSDF
from plainmp.robot_spec import FetchSpec
from plainmp.utils import set_robot_state, sksdf_to_cppsdf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcloud", action="store_true", help="use point cloud for collision check")
    parser.add_argument("--visualize", action="store_true", help="visualize the result")
    args = parser.parse_args()

    set_log_level_none()
    fs = FetchSpec()
    cst = fs.create_collision_const()

    if args.pcloud:
        table_points = np.random.rand(10000, 3) * np.array([1.0, 2.0, 0.05]) - np.array(
            [0.5, 1.0, 0.025]
        )
        table_points += np.array([0.95, 0.0, 0.8])
        table = PointCloudLink(table_points)
        table_sdf = CloudSDF(table_points, 0.002)
    else:
        table = Box([1.0, 2.0, 0.05], face_colors=[100, 200, 100, 200], with_sdf=True)
        table.translate([0.95, 0.0, 0.8])
        table_sdf = sksdf_to_cppsdf(table.sdf)

    sdf = UnionSDF([table_sdf, GroundSDF(0.0)], False)
    cst.set_sdf(sdf)
    lb, ub = fs.angle_bounds()
    q_start = np.array([0.0, 1.32, 1.40, -0.20, 1.72, 0.0, 1.66, 0.0])
    q_goal = np.array([0.386, 0.205, 1.41, 0.308, -1.82, 0.245, 0.417, 6.01])
    resolution = np.ones(8) * 0.05
    problem = Problem(q_start, lb, ub, q_goal, cst, None, resolution)
    solver = OMPLSolver()
    ret = solver.solve(problem)
    print(f"solved in {ret.time_elapsed * 1000:.2f} ms")

    if args.visualize:
        robot_model = fs.get_robot_model(with_mesh=True)

        # get goal point
        set_robot_state(robot_model, fs.control_joint_names, q_goal)
        co = robot_model.gripper_link.copy_worldcoords()

        v = PyrenderViewer()
        v.add(Sphere(0.05, color=[255, 0, 0]).translate(co.worldpos()))
        v.add(robot_model)
        v.add(table)
        v.add(Box([2.0, 2.0, 0.05]))  # ground
        v.show()
        time.sleep(1.0)
        for q in ret.traj.resample(50):
            set_robot_state(robot_model, fs.control_joint_names, q)
            v.redraw()
            time.sleep(0.2)
        time.sleep(1000)
