import copy

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from plainmp.constraint import (
    AppliedForceSpec,
    ComInPolytopeCst,
    EqCompositeCst,
    SequentialCst,
)
from plainmp.psdf import BoxSDF, Pose
from plainmp.robot_spec import FetchSpec


def jac_numerical(const, q0: np.ndarray, eps: float) -> np.ndarray:
    f0, _ = const.evaluate(q0)
    dim_domain = len(q0)
    dim_codomain = len(f0)

    jac = np.zeros((dim_codomain, dim_domain))
    for i in range(dim_domain):
        q1 = copy.deepcopy(q0)
        q1[i] += eps
        f1, _ = const.evaluate(q1)
        jac[:, i] = (f1 - f0) / eps
    return jac


def check_jacobian(const, dim: int, eps: float = 1e-7, decimal: int = 4, std: float = 1.0):
    # check single jacobian
    for _ in range(10):
        q_test = np.random.randn(dim) * std
        _, jac_anal = const.evaluate(q_test)
        if isinstance(jac_anal, csc_matrix):
            jac_anal = jac_anal.todense()
        jac_numel = jac_numerical(const, q_test, eps)
        np.testing.assert_almost_equal(jac_anal, jac_numel, decimal=decimal)


@pytest.mark.parametrize("with_base", [False, True])
def test_config_pose_constraint(with_base: bool):
    fs = FetchSpec(with_base=with_base)
    dof = (8 + 6) if with_base else 8
    q = np.random.randn(dof)
    cst = fs.create_config_point_const(q)
    if with_base:
        check_jacobian(cst, dof, std=0.1)
    else:
        check_jacobian(cst, dof)


@pytest.mark.parametrize("with_base", [False, True])
@pytest.mark.parametrize("with_rpy", [False, True])
def test_link_pose_constraint(with_base: bool, with_rpy: bool):
    if with_rpy:
        pose = [0.7, 0.0, 0.7, 0.0, 0.0, 0.0]
    else:
        pose = [0.7, 0.0, 0.7]

    fs = FetchSpec(with_base=with_base)
    cst = fs.create_gripper_pose_const(pose)
    if with_base:
        check_jacobian(cst, 8 + 6, std=0.1)
    else:
        check_jacobian(cst, 8)


@pytest.mark.parametrize("with_base", [False, True])
def test_link_pose_constraint_multi_link(with_base):
    pose1 = [0.7, 0.0, 0.7, 0.0, 0.0, 0.0]
    pose2 = [0.7, 0.0, 0.7]
    fs = FetchSpec(with_base=with_base)
    cst = fs.create_pose_const(["gripper_link", "wrist_roll_link"], [pose1, pose2])
    if with_base:
        check_jacobian(cst, 8 + 6, std=0.1)
    else:
        check_jacobian(cst, 8)


@pytest.mark.parametrize("with_base", [False, True])
def test_relative_pose_constraint(with_base):
    fs = FetchSpec(with_base=with_base)
    cst = fs.create_relative_pose_const("head_pan_link", "gripper_link", np.ones(3))
    if with_base:
        check_jacobian(cst, 8 + 6, std=0.1)
    else:
        check_jacobian(cst, 8)


@pytest.mark.parametrize("with_base", [False, True])
def test_collision_free_constraint(with_base):
    sdf = BoxSDF([1, 1, 1], Pose([0.5, 0.5, 0.5], np.eye(3)))
    for self_collision in [False, True]:
        fs = FetchSpec(with_base=with_base)
        cst = fs.create_collision_const(self_collision)
        cst.set_sdf(sdf)
        if with_base:
            check_jacobian(cst, 8 + 6, std=0.1)
        else:
            check_jacobian(cst, 8)


@pytest.mark.parametrize("with_base", [False, True])
@pytest.mark.parametrize("with_force", [False, True])
def test_com_in_polytope_constraint(with_base, with_force: bool):
    fs = FetchSpec(with_base=with_base)
    sdf = BoxSDF([0.3, 0.3, 0], Pose([0.0, 0.0, 0.0], np.eye(3)))
    afspecs = []
    if with_force:
        afspecs.append(AppliedForceSpec("gripper_link", 2.0))
    cst = ComInPolytopeCst(fs.get_kin(), fs.control_joint_names, with_base, sdf, afspecs)
    if with_base:
        check_jacobian(cst, 8 + 6, std=0.1)
    else:
        check_jacobian(cst, 8)


def test_eq_composite_constraint():
    fs = FetchSpec()
    cst1 = fs.create_gripper_pose_const([0.7, 0.0, 0.7])
    cst2 = fs.create_pose_const(
        ["gripper_link", "wrist_roll_link", "torso_lift_link"],
        [[0.7, 0.0, 0.7], [0.7, 0.0, 0.7, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
    )
    cst = EqCompositeCst([cst1, cst2])
    check_jacobian(cst, 8)


def test_sequntial_constraint():
    fs = FetchSpec()
    cst1 = fs.create_gripper_pose_const([0.7, 0.0, 0.7])
    cst2 = fs.create_pose_const(
        ["gripper_link", "wrist_roll_link", "torso_lift_link"],
        [[0.7, 0.0, 0.7], [0.7, 0.0, 0.7, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
    )
    T = 4
    cst = SequentialCst(T)
    cst.add_globally(cst1)
    cst.add_at(cst2, 0)
    cst.add_at(cst2, 2)
    cst.add_fixed_point_at(np.zeros(8), 0)

    # msbox is ineq constraint so it is quite strange to mix with eq constraint
    # but only for testing purpose
    msbox = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    cst.add_motion_step_box_constraint(msbox)
    cst.finalize()

    # check cst dim
    cst1_dim = 3
    cst2_dim = 3 + 6 + 7
    fixed_cst_dim = 8
    msbox_cst_dim = 8 * 2 * (T - 1)
    assert cst.cst_dim() == cst1_dim * (T - 1) + cst2_dim * (2 - 1) + fixed_cst_dim + msbox_cst_dim
    # here (T - 1) and (2 - 1) because, we set fixed point, so these cst are removed in finalize()

    # check jacobian
    check_jacobian(cst, 8 * T)

    # check motion step box constraint
    # ok case
    q1 = np.zeros(8)
    q2 = q1 + msbox * 0.5
    q3 = q2 + msbox * 0.5
    q4 = q3 + msbox * 0.5
    x = np.concatenate([q1, q2, q3, q4])
    values = cst.evaluate(x)[0]
    values_here = values[-8 * 2 * (T - 1) :]
    assert np.all(values_here >= 0)

    # ng case
    q1 = np.zeros(8)
    q2 = q1 + msbox * 1.1
    q3 = q2 + msbox * 1.1
    q4 = q3 + msbox * 1.1
    x = np.concatenate([q1, q2, q3, q4])
    values = cst.evaluate(x)[0]
    values_here = values[-8 * 2 * (T - 1) :]
    # half of the values should be negative
    assert np.sum(values_here < 0) == 8 * (T - 1)
    # half of the values should be positive
    assert np.sum(values_here > 0) == 8 * (T - 1)


if __name__ == "__main__":
    test_sequntial_constraint()
