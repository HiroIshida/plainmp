import time

import numpy as np
import pytest
from skrobot.model.primitives import Box

from plainmp.ompl_solver import Algorithm, OMPLSolver, OMPLSolverConfig
from plainmp.problem import Problem
from plainmp.psdf import UnionSDF
from plainmp.robot_spec import FetchSpec
from plainmp.utils import primitive_to_plainmp_sdf

algos = (Algorithm.RRTConnect, Algorithm.KPIECE1)
test_conditions = [(True, algo, False) for algo in algos] + [(False, algo, False) for algo in algos]
test_conditions.append((True, Algorithm.KPIECE1, True))
test_conditions.append((True, Algorithm.RRT, True))


@pytest.mark.parametrize("goal_is_pose,algo,use_goal_sampler", test_conditions)
def test_ompl_solver(goal_is_pose: bool, algo: Algorithm, use_goal_sampler: bool):
    fetch = FetchSpec()
    cst = fetch.create_collision_const()

    table = Box([1.0, 2.0, 0.05], with_sdf=True)
    table.translate([1.0, 0.0, 0.8])
    ground = Box([2.0, 2.0, 0.05], with_sdf=True)
    sdf = UnionSDF([primitive_to_plainmp_sdf(table), primitive_to_plainmp_sdf(ground)])
    cst.set_sdf(sdf)
    lb, ub = fetch.angle_bounds()
    start = np.array([0.0, 1.31999949, 1.40000015, -0.20000077, 1.71999929, 0.0, 1.6600001, 0.0])
    if goal_is_pose:
        goal_cst = fetch.create_gripper_pose_const(np.array([0.7, 0.0, 0.9, 0.0, 0.0, 0.0]))
    else:
        goal_cst = np.array([0.386, 0.20565, 1.41370, 0.30791, -1.82230, 0.24521, 0.41718, 6.01064])
    msbox = np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.2, 0.2])
    problem = Problem(start, lb, ub, goal_cst, cst, None, msbox)
    config = OMPLSolverConfig(algorithm=algo, use_goal_sampler=use_goal_sampler)

    for _ in range(20):
        solver = OMPLSolver(config)
        ret = solver.solve(problem)
        assert ret.traj is not None

        for q in ret.traj.numpy():
            assert cst.is_valid(q)

        # using the previous planning result, re-plan
        conf = OMPLSolverConfig(n_max_ik_trial=1)
        solver = OMPLSolver(conf)
        ret_replan = solver.solve(problem, guess=ret.traj)
        for q in ret_replan.traj.numpy():
            assert cst.is_valid(q)
        assert ret_replan.n_call < ret.n_call  # re-planning should be faster
        print(f"n_call: {ret.n_call} -> {ret_replan.n_call}")


def test_timeout():
    fetch = FetchSpec()
    cst = fetch.create_collision_const()
    obstacle = Box([0.1, 0.1, 0.1], with_sdf=True)
    obstacle.translate([0.7, 0.0, 0.9])  # overlap with the goal to make problem infeasible
    sdf = UnionSDF([primitive_to_plainmp_sdf(obstacle)])
    cst.set_sdf(sdf)
    lb, ub = fetch.angle_bounds()
    start = np.array([0.0, 1.31999949, 1.40000015, -0.20000077, 1.71999929, 0.0, 1.6600001, 0.0])
    goal_cst = fetch.create_gripper_pose_const(np.array([0.7, 0.0, 0.9, 0.0, 0.0, 0.0]))
    msbox = np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.2, 0.2])
    problem = Problem(start, lb, ub, goal_cst, cst, None, msbox)
    conf = OMPLSolverConfig(timeout=3.0, n_max_ik_trial=10000000000, n_max_call=10000000000)
    solver = OMPLSolver(conf)
    ts = time.time()
    ret = solver.solve(problem)
    elapsed = time.time() - ts
    assert 2.9 < elapsed < 3.1
    assert ret.traj is None
