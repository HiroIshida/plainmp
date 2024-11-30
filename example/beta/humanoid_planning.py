import argparse
import time

import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Box
from skrobot.viewers import PyrenderViewer

from plainmp.constraint import EqCompositeCst, IneqCompositeCst
from plainmp.ik import solve_ik
from plainmp.manifold_rrt.manifold_rrt_solver import ManiRRTConfig, ManiRRTConnectSolver
from plainmp.nlp_solver import SQPBasedSolver, SQPBasedSolverConfig
from plainmp.parallel import ParallelSolver
from plainmp.problem import Problem
from plainmp.robot_spec import JaxonSpec, RotType
from plainmp.utils import primitive_to_plainmp_sdf, set_robot_state

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="visualize the result")
    args = parser.parse_args()

    # common setup
    jspec = JaxonSpec(gripper_collision=True)
    com_const = jspec.create_default_com_const()

    table = Box([1.0, 3.0, 0.1], with_sdf=True)
    table.translate([0.8, 0.0, 0.8])

    coll_cst = jspec.create_collision_const()
    coll_cst.set_sdf(primitive_to_plainmp_sdf(table))
    ineq_cst = IneqCompositeCst([com_const, coll_cst])

    stance_cst = jspec.create_pose_const_from_coords(
        ["rleg_end_coords", "lleg_end_coords"],
        [Coordinates([0.0, -0.2, 0]), Coordinates([0.0, +0.2, 0])],
        [RotType.XYZW] * 2,
    )
    lb, ub = jspec.angle_bounds()

    # determine initial pose by solving IK
    init_arm_pose_cst = jspec.create_pose_const_from_coords(
        ["rarm_end_coords", "larm_end_coords"],
        [
            Coordinates([0.6, -0.6, 1.0], rot=[0, -0.5 * np.pi, 0]),
            Coordinates([0.6, 0.6, 1.0], rot=[0, -0.5 * np.pi, 0]),
        ],
        [RotType.XYZW] * 2,
    )
    print("setup eq")
    init_eq_eq_cst = EqCompositeCst([stance_cst, init_arm_pose_cst])
    ik_ret = solve_ik(init_eq_eq_cst, ineq_cst, lb, ub, q_seed=jspec.reset_manip_pose_q)
    assert ik_ret.success
    print(f"elapsed time to solve initial ik problem: {ik_ret.elapsed_time} [s]")

    # solve constrained-RRT
    reach_cst = jspec.create_pose_const(["rarm_end_coords"], [np.array([0.85, -0.6, 0.4])])
    goal_eq_cst = EqCompositeCst([stance_cst, reach_cst])
    ik_ret2 = solve_ik(goal_eq_cst, ineq_cst, lb, ub, q_seed=jspec.reset_manip_pose_q)
    assert ik_ret2.success
    print(f"elapsed time to solve second ik problem: {ik_ret2.elapsed_time} [s]")

    problem = Problem(ik_ret.q, lb, ub, ik_ret2.q, ineq_cst, stance_cst, np.ones(37) * 0.1)
    solver = ManiRRTConnectSolver(ManiRRTConfig(10000))
    psolver = ParallelSolver(solver, 4)
    ret = psolver.solve(problem)
    assert ret.traj is not None
    print(f"elapsed time to solve constrained motion planning: {ret.time_elapsed} [s]")

    # solve trajectory optimization using RRT result as initial guess
    smoother = SQPBasedSolver(
        SQPBasedSolverConfig(50, 37, ctol_eq=1e-3, ctol_ineq=1e-3, ineq_tighten_coef=0.0)
    )
    ret = smoother.solve(problem, ret.traj)
    assert ret.traj is not None
    print(f"elapsed time to smoothing by sqp: {ret.time_elapsed} [s]")
    print(f"number of iterations: {ret.n_call}")

    if args.visualize:
        v = PyrenderViewer()
        robot = jspec.get_robot_model(with_mesh=True)
        set_robot_state(robot, jspec.control_joint_names, ik_ret.q, True)
        ground = Box([2, 2, 0.01])
        v.add(ground)
        v.add(table)
        v.add(robot)
        v.show()
        time.sleep(2)
        for q in ret.traj:
            set_robot_state(robot, jspec.control_joint_names, q, True)
            v.redraw()
            time.sleep(0.15)
        time.sleep(20)
