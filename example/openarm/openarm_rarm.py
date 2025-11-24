import argparse
import time

import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis, Cylinder
from skrobot.viewers import PyrenderViewer

from plainmp.ik import solve_ik
from plainmp.ompl_solver import OMPLSolver, OMPLSolverConfig
from plainmp.problem import Problem
from plainmp.psdf import UnionSDF
from plainmp.robot_spec import OpenArmV10RarmSpec
from plainmp.utils import primitive_to_plainmp_sdf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="visualize the result or not")
    args = parser.parse_args()
    np.random.seed(0)

    s = OpenArmV10RarmSpec()

    target = Coordinates(pos=[0.4, -0.2, 0.62])
    cst = s.create_tcp_pose_const(target)

    # define collision constraint
    obstacle1 = Cylinder(pos=[0.2, -0.1, 0.4], radius=0.03, height=0.8, face_colors=[1, 0, 0, 0.4])
    obstacle2 = Cylinder(pos=[0.2, -0.3, 0.4], radius=0.03, height=0.8, face_colors=[1, 0, 0, 0.4])
    ineq_cst = s.create_collision_const(self_collision=True)
    sdf1 = primitive_to_plainmp_sdf(obstacle1)
    sdf2 = primitive_to_plainmp_sdf(obstacle2)
    ineq_cst.set_sdf(UnionSDF([sdf1, sdf2]))

    # lb and ub
    lb, ub = s.angle_bounds()

    # solve collision-aware ik
    result = solve_ik(cst, ineq_cst, lb, ub)
    assert result.success
    print(f"collision aware ik solved in {result.elapsed_time} sec")

    # solve motion plan
    resolution = np.ones(7) * 0.05
    problem = Problem(np.zeros(7), lb, ub, result.q, ineq_cst, None, resolution)
    solver = OMPLSolver(OMPLSolverConfig(shortcut=True))
    result = solver.solve(problem)
    print(f"motion plan solved in {result.time_elapsed} sec")

    if args.visualize:
        print("loading robot model to visualize...")
        model = s.get_robot_model(with_mesh=True)
        v = PyrenderViewer()
        v.add(obstacle1)
        v.add(obstacle2)
        v.add(Axis.from_coords(target))
        v.add(model)
        v.show()

        input("Press Enter to start animation...")

        for q in result.traj.resample(20):
            s.set_skrobot_model_state(model, q)
            time.sleep(0.5)
            v.redraw()
        time.sleep(1000)
