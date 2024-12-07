import argparse
import time

import numpy as np
from skrobot.model.primitives import Box
from skrobot.viewers import PyrenderViewer

from plainmp.constraint import SphereAttachmentSpec
from plainmp.ik import IKConfig, solve_ik
from plainmp.ompl_solver import OMPLSolver, OMPLSolverConfig
from plainmp.problem import Problem
from plainmp.psdf import GroundSDF, UnionSDF
from plainmp.robot_spec import PandaSpec
from plainmp.utils import box_to_grid_poitns, primitive_to_plainmp_sdf, set_robot_state

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="visualize")
    args = parser.parse_args()

    # prepare environment
    shelf = Box([1.2, 0.3, 0.4], face_colors=[200, 200, 230, 200])
    shelf.translate([0.0, 0.4, 0.2])

    eps = 0.005  # to avoid collision of cardboard box vs ground
    cardboard_box = Box([0.2, 0.3, 0.2], face_colors=[139, 69, 19, 200])
    cardboard_box.translate([0.5, 0.0, 0.1 + eps])
    target_box_skeleton = Box([0.2, 0.3, 0.2], face_colors=[139, 69, 19, 100])
    target_box_skeleton.translate([0.0, 0.4, 0.5 + eps])

    # solve ik for start and goal
    ik_config = IKConfig(acceptable_error=1e-5)
    spec = PandaSpec()
    lb, ub = spec.angle_bounds()
    ineq_cst = spec.create_collision_const()
    sdf = UnionSDF([primitive_to_plainmp_sdf(p) for p in [shelf, target_box_skeleton]])
    ineq_cst.set_sdf(sdf)

    target_pos_goal = target_box_skeleton.worldpos() + np.array([0.0, 0.0, 0.23])
    pose_goal = np.hstack([target_pos_goal, [np.pi, 0, 0]])
    eq_cst = spec.create_pose_const(["panda_hand"], [pose_goal])
    ik_res = solve_ik(eq_cst, ineq_cst, lb, ub, config=ik_config)
    assert ik_res.success
    q_goal = ik_res.q
    print(
        f"elapsed time to determine goal ik: {ik_res.elapsed_time:.3f} sec, after {ik_res.n_trial} retries"
    )

    target_pos_start = cardboard_box.worldpos() + np.array([0.0, 0.0, 0.23])
    pose_start = np.hstack([target_pos_start, [np.pi, 0, 0]])
    eq_cst = spec.create_pose_const(["panda_hand"], [pose_start])
    sdf = UnionSDF(
        [primitive_to_plainmp_sdf(cardboard_box), primitive_to_plainmp_sdf(shelf), GroundSDF(0.0)]
    )
    ineq_cst.set_sdf(sdf)
    ik_res = solve_ik(eq_cst, ineq_cst, lb, ub, config=ik_config)
    assert ik_res.success
    q_start = ik_res.q
    print(
        f"elapsed time to determine start ik: {ik_res.elapsed_time:.3f} sec, after {ik_res.n_trial} retries"
    )

    # approximate cardboard box as grid points and create attachment
    points_wrt_world = box_to_grid_poitns(cardboard_box, 10)
    points_wrt_gripper = spec.transform_points_wrt_link(points_wrt_world, "panda_hand", q_start)
    radii = np.array([0.0] * len(points_wrt_gripper))
    att = SphereAttachmentSpec("panda_hand", points_wrt_gripper.transpose(), radii, False)

    # motion planning
    ineq_cst = spec.create_collision_const(attachements=[att])
    sdf = UnionSDF([primitive_to_plainmp_sdf(shelf), GroundSDF(0.0)])
    ineq_cst.set_sdf(sdf)
    problem = Problem(q_start, lb, ub, q_goal, ineq_cst, None, 0.03, "euclidean")
    result = OMPLSolver(OMPLSolverConfig(simplify=True)).solve(problem)
    print(f"elapsed time to solve motion planning: {result.time_elapsed:.3f} sec")

    if args.visualize:
        # visualization
        robot = spec.get_robot_model(with_mesh=True)
        set_robot_state(robot, spec.control_joint_names, ik_res.q)
        ground_vis = Box([2.0, 2.0, 0.03])
        v = PyrenderViewer()
        v.add(robot)
        v.add(ground_vis)
        v.add(shelf)
        v.add(cardboard_box)
        v.add(target_box_skeleton)
        v.show()

        robot.__dict__["panda_hand"].assoc(cardboard_box)

        for q in result.traj.resample(30):
            set_robot_state(robot, spec.control_joint_names, q)
            v.redraw()
            time.sleep(0.1)
        time.sleep(1000)
