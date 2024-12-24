import argparse
import time

import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis, Box
from skrobot.viewers import PyrenderViewer

from plainmp.ik import solve_ik
from plainmp.ompl_solver import OMPLSolver, OMPLSolverConfig
from plainmp.problem import Problem
from plainmp.robot_spec import PR2DualarmSpec
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
    spec = PR2DualarmSpec()
    default_joint_positions = spec.default_joint_positions
    default_joint_positions["torso_lift_joint"] = 0.1
    spec.reflect_joint_positions(default_joint_positions)  # very important!

    ineq_cst = spec.create_collision_const(self_collision=True)
    psdf = primitive_to_plainmp_sdf(table)
    ineq_cst.set_sdf(psdf)

    # solve ik (solving IK is rather hard for dual-arm PR2 than motion planning)
    # so we need to set max_trial to a large number
    target_pos_rarm = [0.9, -0.2, 0.9]
    target_pos_larm = [0.9, +0.2, 0.9]
    target_rot = [0, 0, 0]
    target_coords_rarm = Coordinates(pos=target_pos_rarm, rot=target_rot)
    target_coords_larm = Coordinates(pos=target_pos_larm, rot=target_rot)
    eq_cst = spec.create_gripper_pose_const(
        [target_pos_rarm + target_rot, target_pos_larm + target_rot]
    )
    lb, ub = spec.angle_bounds()
    ik_result = solve_ik(eq_cst, ineq_cst, lb, ub, max_trial=300)
    print(f"elapsed time to solve IK: {ik_result.elapsed_time * 1000:.2f} [ms]")
    assert ik_result.success

    # motion planning
    q_start = spec.q_default
    q_goal = ik_result.q
    resolution = np.ones(len(spec.control_joint_names)) * 0.05
    problem = Problem(q_start, lb, ub, q_goal, ineq_cst, None, resolution)
    ompl_solver = OMPLSolver(OMPLSolverConfig(shortcut=args.simplify))
    mp_result = ompl_solver.solve(problem)
    assert mp_result.success
    print(f"elapsed time to solve RRTConnect: {mp_result.time_elapsed * 1000:.2f} [ms]")

    if args.visualize:
        # visualization
        viewer = PyrenderViewer()
        robot_model = spec.get_robot_model(with_mesh=True)
        for name, angle in default_joint_positions.items():
            robot_model.__dict__[name].joint_angle(angle)
        set_robot_state(robot_model, spec.control_joint_names, ik_result.q)
        viewer.add(Axis.from_coords(target_coords_rarm))
        viewer.add(Axis.from_coords(target_coords_larm))
        viewer.add(robot_model)
        viewer.add(table)
        viewer.show()
        input("Press Enter to show the planned path")
        for q in mp_result.traj.resample(50):
            set_robot_state(robot_model, spec.control_joint_names, q)
            viewer.redraw()
            time.sleep(0.15)
        time.sleep(1000)
