import pytest

from plainmp.ik import solve_ik, solve_ik_srinv_experimental
from plainmp.robot_spec import FetchSpec


def _test_ik(with_rot: bool, with_self_collision: bool):
    fs = FetchSpec()
    if with_rot:
        eq_cst = fs.create_gripper_pose_const([0.7, +0.2, 0.8, 0.0, 0, 0.0])  # xyzrpy
    else:
        eq_cst = fs.create_gripper_pose_const([0.7, +0.2, 0.8])

    if with_self_collision:
        ineq_cst = fs.create_collision_const(True)
    else:
        ineq_cst = None

    lb, ub = fs.angle_bounds()
    ret = solve_ik(eq_cst, ineq_cst, lb, ub)
    assert ret.success


test_cases = [[False, False], [False, True], [True, False], [True, True]]


@pytest.mark.parametrize("with_rot, with_self_collision", test_cases)
def test_ik(with_rot, with_self_collision):
    _test_ik(with_rot, with_self_collision)


def _test_ik_srinv(with_rot: bool):
    fs = FetchSpec()
    if with_rot:
        link_pose_cst = fs.create_gripper_pose_const([0.7, +0.2, 0.8, 0.0, 0, 0.0])  # xyzrpy
    else:
        link_pose_cst = fs.create_gripper_pose_const([0.7, +0.2, 0.8])

    lb, ub = fs.angle_bounds()
    ret = solve_ik_srinv_experimental(link_pose_cst, lb, ub)
    assert ret.success


@pytest.mark.parametrize("with_rot", [False, True])
def test_ik_srinv(with_rot):
    _test_ik_srinv(with_rot)
