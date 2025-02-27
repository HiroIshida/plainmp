import argparse
import time

import numpy as np
from skrobot.model.primitives import Box, Cylinder, Sphere
from skrobot.viewers import PyrenderViewer

from plainmp.ompl_solver import OMPLSolver
from plainmp.problem import Problem
from plainmp.psdf import UnionSDF
from plainmp.robot_spec import PandaSpec
from plainmp.utils import primitive_to_plainmp_sdf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="visualize")
    parser.add_argument("--difficult", action="store_true", help="difficult")
    args = parser.parse_args()

    height = 1.0
    ground = Box([2.0, 2.0, 0.05])
    ground.translate([0, 0, -0.05])

    poll1 = Cylinder(0.05, height, face_colors=[100, 100, 200, 200])
    poll1.translate([0.3, 0.3, 0.5 * height])

    poll2 = Cylinder(0.05, height, face_colors=[100, 100, 200, 200])
    poll2.translate([-0.3, -0.3, 0.5 * height])

    primitives = [ground, poll1, poll2]

    if args.difficult:
        ceil = Box([2.0, 2.0, 0.05], face_colors=[100, 200, 100, 200])
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
    ret = solver.solve(problem)
    assert ret.success

    if args.visualize:
        v = PyrenderViewer()
        robot_model = spec.get_robot_model(with_mesh=True)
        spec.set_skrobot_model_state(robot_model, q1)

        co = robot_model.panda_hand.copy_worldcoords()
        v.add(Sphere(0.05, color=[255, 0, 0]).translate(co.worldpos()))

        v.add(robot_model)
        for p in primitives:
            v.add(p)
        v.show()
        input("Press Enter to replay the path")
        for q in ret.traj.resample(50):
            spec.set_skrobot_model_state(robot_model, q)
            v.redraw()
            time.sleep(0.2)
        time.sleep(1000)
