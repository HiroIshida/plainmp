import argparse
import time

import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis, Box
from skrobot.viewers import PyrenderViewer

from plainmp.kinematics import BaseType
from plainmp.ompl_solver import OMPLSolver, OMPLSolverConfig
from plainmp.problem import Problem
from plainmp.robot_spec import PR2RarmSpec
from plainmp.utils import primitive_to_plainmp_sdf, set_robot_state

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="visualize")
    parser.add_argument("--simplify", action="store_true", help="simplify the path")
    args = parser.parse_args()

    # environment
    table = Box([0.8, 2.0, 0.02], face_colors=[150, 100, 100, 200])
    table.translate([0.8, 0.0, 0.8])

    # common
    spec = PR2RarmSpec(base_type=BaseType.PLANAR)
    default_joint_positions = spec.default_joint_positions
    default_joint_positions["torso_lift_joint"] = 0.1
    spec.reflect_joint_positions(default_joint_positions)
    ineq_cst = spec.create_collision_const(self_collision=True)
    psdf = primitive_to_plainmp_sdf(table)
    ineq_cst.set_sdf(psdf)
    lb_joints, ub_joints = spec.angle_bounds()
    lb_base, ub_base = [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]
    lb = np.hstack([lb_joints, lb_base])
    ub = np.hstack([ub_joints, ub_base])

    # motion planning
    q_start = np.hstack([spec.q_default, [0, 0, 0]])
    q_goal = np.array([0.021, -0.348, -3.27, -0.76, 3.09, -0.413, 0.171, 0, 0, 0])
    resolution = np.ones(len(spec.control_joint_names) + 3) * 0.05
    problem = Problem(q_start, lb, ub, q_goal, ineq_cst, None, resolution)
    ompl_solver = OMPLSolver(OMPLSolverConfig(shortcut=args.simplify))
    mp_result = ompl_solver.solve(problem)
    assert mp_result.success
    print(f"elapsed time to solve RRTConnect: {mp_result.time_elapsed * 1000:.2f} [ms]")

    if args.visualize:
        viewer = PyrenderViewer()
        robot_model = spec.get_robot_model(with_mesh=True)
        for name, angle in default_joint_positions.items():
            robot_model.__dict__[name].joint_angle(angle)
        set_robot_state(robot_model, spec.control_joint_names, q_start, base_type=BaseType.PLANAR)
        co = Coordinates([0.9, -0.2, 0.9])
        viewer.add(Axis.from_coords(co))
        viewer.add(robot_model)
        viewer.add(table)
        viewer.show()
        input("Press Enter to show the planned path")
        for q in mp_result.traj.resample(50):
            set_robot_state(robot_model, spec.control_joint_names, q, base_type=BaseType.PLANAR)
            viewer.redraw()
            time.sleep(0.15)
        time.sleep(1000)
