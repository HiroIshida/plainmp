import numpy as np
import pytest
from skrobot.model.primitives import Box

from plainmp.ompl_solver import OMPLSolver
from plainmp.problem import Problem
from plainmp.psdf import UnionSDF
from plainmp.robot_spec import FetchSpec
from plainmp.utils import sksdf_to_cppsdf


@pytest.mark.parametrize("goal_is_pose", [True, False])
def test_ompl_solver(goal_is_pose: bool):
    fetch = FetchSpec()
    cst = fetch.create_collision_const()

    table = Box([1.0, 2.0, 0.05], with_sdf=True)
    table.translate([1.0, 0.0, 0.8])
    ground = Box([2.0, 2.0, 0.05], with_sdf=True)
    sdf = UnionSDF([sksdf_to_cppsdf(table.sdf), sksdf_to_cppsdf(ground.sdf)], False)
    cst.set_sdf(sdf)
    lb, ub = fetch.angle_bounds()
    start = np.array([0.0, 1.31999949, 1.40000015, -0.20000077, 1.71999929, 0.0, 1.6600001, 0.0])
    if goal_is_pose:
        goal_cst = fetch.create_gripper_pose_const(np.array([0.7, 0.0, 0.9, 0.0, 0.0, 0.0]))
    else:
        goal_cst = np.array([0.386, 0.20565, 1.41370, 0.30791, -1.82230, 0.24521, 0.41718, 6.01064])
    msbox = np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.2, 0.2])
    problem = Problem(start, lb, ub, goal_cst, cst, None, msbox)
    solver = OMPLSolver()
    ret = solver.solve(problem)
    assert ret.traj is not None

    for q in ret.traj.numpy():
        assert cst.is_valid(q)
