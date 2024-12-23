import copy
import pickle
from hashlib import md5
from typing import Union

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from plainmp.constraint import (
    AppliedForceSpec,
    ComInPolytopeCst,
    EqCompositeCst,
    IneqCompositeCst,
    SequentialCst,
)
from plainmp.kinematics import BaseType
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
    for _ in range(100):
        q_test = np.random.randn(dim) * std
        _, jac_anal = const.evaluate(q_test)
        if isinstance(jac_anal, csc_matrix):
            jac_anal = jac_anal.todense()
        jac_numel = jac_numerical(const, q_test, eps)
        np.testing.assert_almost_equal(jac_anal, jac_numel, decimal=decimal)


def check_eval_is_valid_consistency(const, dim: int, std: float = 1.0):
    for _ in range(100):
        q_test = np.random.randn(dim) * std
        values, _ = const.evaluate(q_test)
        is_valid_by_evaluation = np.all(values > 0)
        is_valid_by_is_valid = const.is_valid(q_test)
        assert is_valid_by_evaluation == is_valid_by_is_valid


def check_sparse_structure(
    const: Union[EqCompositeCst, IneqCompositeCst], dim: int, std: float = 1.0
):
    hash_values = []
    for _ in range(10):
        q_test = np.random.randn(dim) * std
        _, jac = const.evaluate(q_test)
        assert 2 * jac.nnz < jac.shape[0] * jac.shape[1]
        hash_values.append(md5(pickle.dumps(jac.nonzero())).hexdigest())
    # all hash values should be the same
    assert len(set(hash_values)) == 1


base_types = (BaseType.FIXED, BaseType.FLOATING, BaseType.PLANAR)


def base_type_to_fetch_dof(base_type):
    if base_type == BaseType.FIXED:
        return 8
    elif base_type == BaseType.PLANAR:
        return 8 + 3
    elif base_type == BaseType.FLOATING:
        return 8 + 6
    else:
        assert False


@pytest.mark.parametrize("base_type", base_types)
def test_config_pose_constraint(base_type):
    fs = FetchSpec(base_type=base_type)
    dof = base_type_to_fetch_dof(base_type)
    q = np.random.randn(dof)
    cst = fs.create_config_point_const(q)
    if base_type != BaseType.FIXED:
        check_jacobian(cst, dof, std=0.1)
    else:
        check_jacobian(cst, dof)


@pytest.mark.parametrize("base_type", base_types)
@pytest.mark.parametrize("with_rpy", [False, True])
def test_link_pose_constraint(base_type: BaseType, with_rpy: bool):
    if with_rpy:
        pose = [0.7, 0.0, 0.7, 0.0, 0.0, 0.0]
    else:
        pose = [0.7, 0.0, 0.7]

    fs = FetchSpec(base_type=base_type)
    cst = fs.create_gripper_pose_const(pose)
    dof = base_type_to_fetch_dof(base_type)
    if base_type != BaseType.FIXED:
        check_jacobian(cst, dof, std=0.1)
    else:
        check_jacobian(cst, dof)


@pytest.mark.parametrize("base_type", base_types)
def test_link_pose_constraint_multi_link(base_type: BaseType):
    pose1 = [0.7, 0.0, 0.7, 0.0, 0.0, 0.0]
    pose2 = [0.7, 0.0, 0.7]
    fs = FetchSpec(base_type=base_type)
    cst = fs.create_pose_const(["gripper_link", "wrist_roll_link"], [pose1, pose2])
    dof = base_type_to_fetch_dof(base_type)
    if base_type != BaseType.FIXED:
        check_jacobian(cst, dof, std=0.1)
    else:
        check_jacobian(cst, dof)


@pytest.mark.parametrize("base_type", base_types)
def test_relative_pose_constraint(base_type: BaseType):
    fs = FetchSpec(base_type=base_type)
    cst = fs.create_relative_pose_const("head_pan_link", "gripper_link", np.ones(3))
    dof = base_type_to_fetch_dof(base_type)
    if base_type != BaseType.FIXED:
        check_jacobian(cst, dof, std=0.1)
    else:
        check_jacobian(cst, dof)


@pytest.mark.parametrize("base_type", base_types)
def test_fixed_z_axis_constraint(base_type: BaseType):
    fs = FetchSpec(base_type=base_type)
    cst = fs.create_fixed_zaxis_const("gripper_link")
    dof = base_type_to_fetch_dof(base_type)
    if base_type != BaseType.FIXED:
        check_jacobian(cst, dof, std=0.1)
    else:
        check_jacobian(cst, dof)


@pytest.mark.parametrize("base_type", base_types)
def test_collision_free_constraint(base_type: BaseType):
    sdf = BoxSDF([1, 1, 1], Pose([0.5, 0.5, 0.5], np.eye(3)))
    for self_collision in [False, True]:
        fs = FetchSpec(base_type=base_type)
        cst = fs.create_collision_const(self_collision)
        cst.set_sdf(sdf)
        dof = base_type_to_fetch_dof(base_type)
        if base_type != BaseType.FIXED:
            check_jacobian(cst, dof, std=0.1)
            check_eval_is_valid_consistency(cst, dof, std=0.1)
        else:
            check_jacobian(cst, dof)
            check_eval_is_valid_consistency(cst, dof)


@pytest.mark.parametrize("base_type", base_types)
@pytest.mark.parametrize("with_force", [False, True])
def test_com_in_polytope_constraint(base_type, with_force: bool):
    fs = FetchSpec(base_type=base_type)
    sdf = BoxSDF([0.3, 0.3, 0], Pose([0.0, 0.0, 0.0], np.eye(3)))
    afspecs = []
    if with_force:
        afspecs.append(AppliedForceSpec("gripper_link", 2.0))
    cst = ComInPolytopeCst(fs.get_kin(), fs.control_joint_names, base_type, sdf, afspecs)
    dof = base_type_to_fetch_dof(base_type)
    if base_type != BaseType.FIXED:
        check_jacobian(cst, dof, std=0.1)
        check_eval_is_valid_consistency(cst, dof, std=0.1)
    else:
        check_jacobian(cst, dof)
        check_eval_is_valid_consistency(cst, dof)


def test_eq_composite_constraint():
    fs = FetchSpec()
    cst1 = fs.create_gripper_pose_const([0.7, 0.0, 0.7])
    cst2 = fs.create_pose_const(
        ["gripper_link", "wrist_roll_link", "torso_lift_link"],
        [[0.7, 0.0, 0.7], [0.7, 0.0, 0.7, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
    )
    cst = EqCompositeCst([cst1, cst2])
    check_jacobian(cst, 8)


@pytest.mark.parametrize("with_msbox", [False, True])
@pytest.mark.parametrize("with_fixed_point", [False, True])
def test_sequntial_constraint(with_msbox: bool, with_fixed_point: bool):
    fs = FetchSpec()
    cst1 = fs.create_gripper_pose_const([0.7, 0.0, 0.7])
    cst2 = fs.create_pose_const(
        ["gripper_link", "wrist_roll_link", "torso_lift_link"],
        [[0.7, 0.0, 0.7], [0.7, 0.0, 0.7, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
    )
    T = 4
    cst = SequentialCst(T, 8)
    cst.add_globally(cst1)
    cst.add_at(cst2, 0)
    cst.add_at(cst2, 2)
    if with_fixed_point:
        cst.add_fixed_point_at(np.zeros(8), 0)

    # msbox is ineq constraint so it is quite strange to mix with eq constraint
    # but only for testing purpose
    if with_msbox:
        msbox = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        cst.add_motion_step_box_constraint(msbox)
    cst.finalize()

    # check cst dim
    cst1_dim = 3
    cst2_dim = 3 + 6 + 7
    fixed_cst_dim = 8
    msbox_cst_dim = 8 * 2 * (T - 1)

    total_expected = cst1_dim * T + cst2_dim * 2
    if with_msbox:
        total_expected += msbox_cst_dim

    if with_fixed_point:
        total_expected += fixed_cst_dim
        total_expected -= cst1_dim + cst2_dim

    assert cst.cst_dim() == total_expected

    # check jacobian
    check_jacobian(cst, 8 * T)

    # check spase-structure consistency
    check_sparse_structure(cst, 8 * T)

    if with_msbox:
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
    with_rpy = False
    if with_rpy:
        pose = [0.7, 0.0, 0.7, 0.0, 0.0, 0.0]
    else:
        pose = [0.7, 0.0, 0.7]
    base_type = BaseType.FLOATING
    fs = FetchSpec(base_type=base_type)
    cst = fs.create_gripper_pose_const(pose)
    dof = base_type_to_fetch_dof(base_type)
    if base_type != BaseType.FIXED:
        check_jacobian(cst, dof, std=0.1)
    else:
        check_jacobian(cst, dof)
