import time
from typing import Sequence

import numpy as np
import pytest
from skrobot.model.primitives import Box

from plainmp.ompl_solver import (
    Algorithm,
    OMPLSolver,
    OMPLSolverConfig,
    RefineType,
    simplify_path,
)
from plainmp.problem import Problem
from plainmp.psdf import UnionSDF
from plainmp.robot_spec import FetchSpec
from plainmp.utils import primitive_to_plainmp_sdf

algos = (Algorithm.RRTConnect, Algorithm.KPIECE1)
test_conditions = []
for goal_is_pose in [True, False]:
    for algo in algos:
        for refine_seq in [
            tuple(),
            (RefineType.SHORTCUT,),
            (RefineType.BSPLINE,),
            (RefineType.SHORTCUT, RefineType.BSPLINE),
            (RefineType.SHORTCUT, RefineType.BSPLINE, RefineType.SHORTCUT, RefineType.BSPLINE),
        ]:
            use_goal_sampler = False
            test_conditions.append((goal_is_pose, algo, refine_seq, use_goal_sampler))
test_conditions.append((True, Algorithm.KPIECE1, tuple(), True))
test_conditions.append((True, Algorithm.RRT, tuple(), True))


def create_test_problem(goal_is_pose: bool):
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
    return problem


@pytest.mark.parametrize("goal_is_pose,algo,refine_seq,use_goal_sampler", test_conditions)
def test_ompl_solver(
    goal_is_pose: bool, algo: Algorithm, refine_seq: Sequence[RefineType], use_goal_sampler: bool
):
    problem = create_test_problem(goal_is_pose)
    config = OMPLSolverConfig(
        algorithm=algo, use_goal_sampler=use_goal_sampler, refine_seq=refine_seq
    )

    for _ in range(20):
        solver = OMPLSolver(config)
        ret = solver.solve(problem)
        assert ret.traj is not None

        for q in ret.traj.numpy():
            assert np.all(problem.lb <= q) and np.all(q <= problem.ub)
            if RefineType.BSPLINE in refine_seq:
                # NOTE: bspline may make the trajectory slightly invalid
                # so we use a relaxed threshold
                value = problem.global_ineq_const.evaluate(q)[0]
                assert (value > -1e3).all()
            else:
                assert problem.global_ineq_const.is_valid(q)

        # using the previous planning result, re-plan
        conf = OMPLSolverConfig(n_max_ik_trial=1)
        solver = OMPLSolver(conf)
        ret_replan = solver.solve(problem, guess=ret.traj)
        for q in ret_replan.traj.numpy():
            assert np.all(problem.lb <= q) and np.all(q <= problem.ub)
            if RefineType.BSPLINE in refine_seq:
                # NOTE: bspline may make the trajectory slightly invalid
                # so we use a relaxed threshold
                value = problem.global_ineq_const.evaluate(q)[0]
                assert (value > -1e3).all()
            else:
                assert problem.global_ineq_const.is_valid(q)

        if len(refine_seq) == 0:
            # NOTE: this test does not work with refine_seq
            assert ret_replan.n_call < ret.n_call  # re-planning should be faster
            print(f"n_call: {ret.n_call} -> {ret_replan.n_call}")


def create_infeasible_problem() -> Problem:
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
    return problem


def test_timeout():
    problem = create_infeasible_problem()
    conf = OMPLSolverConfig(timeout=2.0, n_max_ik_trial=10000000000, n_max_call=10000000000)
    solver = OMPLSolver(conf)
    ts = time.time()
    ret = solver.solve(problem)
    elapsed = time.time() - ts
    assert 1.9 < elapsed < 2.1
    assert ret.traj is None


def test_timeout_many():
    # NOTE: ompl solver internally calls ik solver. Because, we need to mange timeout for
    # both ompl and ik solver, with a pour timeout management, timeout exception race condition
    # may happen (and actually happens until commit 5aadac6).
    problem = create_infeasible_problem()
    conf = OMPLSolverConfig(timeout=0.01, n_max_ik_trial=10000000000, n_max_call=10000000000)
    solver = OMPLSolver(conf)
    for _ in range(300):
        solver.solve(problem)


def test_simplifier():
    problem = create_test_problem(goal_is_pose=True)
    config = OMPLSolverConfig()
    solver = OMPLSolver(config)

    for _ in range(10):
        ret = solver.solve(problem)
        assert ret.traj is not None

        original_length = ret.traj.get_length()
        ret = simplify_path(
            ret.traj,
            problem.lb,
            problem.ub,
            problem.global_ineq_const,
            problem.resolution,
            problem.validator_type,
        )
        post_simplified_length = ret.get_length()
        assert post_simplified_length < original_length


if __name__ == "__main__":
    test_timeout()
