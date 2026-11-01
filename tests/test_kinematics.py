import numpy as np
import pytest
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import (
    quaternion2matrix,
    rotation_matrix_from_quat,
    xyzw2wxyz,
)

from plainmp.constraint import AppliedForceSpec, ComInPolytopeCst
from plainmp.kinematics import BaseType, KinematicModel
from plainmp.psdf import BoxSDF, Pose
from plainmp.robot_spec import (
    FetchSpec,
    PandaSpec,
    PR2DualarmSpec,
    PR2LarmSpec,
    PR2RarmSpec,
    RobotSpec,
)


@pytest.fixture
def mass_kin():
    # The massless spacer makes the arm's link ID differ from its mass index.
    return KinematicModel(
        """
        <robot name="mass_test">
          <link name="base">
            <inertial>
              <mass value="1"/>
              <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
            </inertial>
          </link>
          <link name="spacer"/>
          <link name="arm">
            <inertial>
              <origin xyz="1 0 0"/>
              <mass value="2"/>
              <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
            </inertial>
          </link>
          <joint name="fixed" type="fixed">
            <parent link="base"/><child link="spacer"/>
          </joint>
          <joint name="joint" type="continuous">
            <parent link="spacer"/><child link="arm"/><axis xyz="0 1 0"/>
          </joint>
        </robot>
    """
    )


@pytest.mark.parametrize("with_force", [False, True])
def test_link_mass(mass_kin, with_force):
    kin = mass_kin
    base_id, arm_id = kin.get_link_ids(["base", "arm"])
    joint_ids = kin.get_joint_ids(["joint"])
    assert kin.get_link_mass(arm_id) == 2.0
    sdf = BoxSDF([4, 4, 0], Pose([0, 0, 0], np.eye(3)))
    # Applied forces are normalized by total_mass_, so this catches mass changes
    # that update link_masses_ but leave total_mass_ stale. Gravity and COM
    # calculations without applied forces use link_masses_ directly and would
    # miss that bug.
    forces = [AppliedForceSpec("base", 1.0)] if with_force else []
    cst = ComInPolytopeCst(kin, ["joint"], BaseType.FIXED, sdf, forces)
    q = np.array([0.3])
    kin.set_joint_positions(joint_ids, q)
    kin.get_gravity_term(joint_ids)  # Populate FK caches before changing mass.
    for mass in (4.0, 0.0, 2.0):
        kin.set_link_mass(link_id=arm_id, mass=mass)
        assert kin.get_link_mass(arm_id) == mass
        assert kin.get_link_mass(base_id) == 1.0

        np.testing.assert_allclose(kin.get_gravity_term(joint_ids), mass * 9.80665 * np.cos(q))
        values, jac = cst.evaluate(q)
        weight = mass / (1.0 + mass + float(with_force))
        np.testing.assert_allclose(values, 2.0 - weight * np.cos(q))
        np.testing.assert_allclose(jac, [[weight * np.sin(q[0])]], atol=1e-8)


def test_link_mass_invalid_input(mass_kin):
    kin = mass_kin
    arm_id, spacer_id = kin.get_link_ids(["arm", "spacer"])
    for mass in (-1.0, np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite and non-negative"):
            kin.set_link_mass(arm_id, mass)
        assert kin.get_link_mass(arm_id) == 2.0
    for link_id in (spacer_id, 999):
        with pytest.raises(ValueError, match="no mass data"):
            kin.get_link_mass(link_id)
        with pytest.raises(ValueError, match="no mass data"):
            kin.set_link_mass(link_id, 1.0)


def test_model_add_new_link():
    fs = FetchSpec()
    # NOTE: to understand this test, you need to read c++ side KinematicModel.add_new_link
    # because create collision const create link with hashval name
    # we can create two same constraints which are different object
    # but shared the kinematic chain
    cst1 = fs.create_collision_const()  # noqa
    cst2 = fs.create_collision_const()  # noqa

    # However, for link_name explicitly specified, we can not add the same link twice
    kin = fs.get_kin()
    consider_rotation = True
    kin.add_new_link("hahahahah", "base_link", [0, 0, 0], [0, 0, 0], consider_rotation)
    with pytest.raises(RuntimeError):
        kin.add_new_link("hahahahah", "base_link", [0, 0, 0], [0, 0, 0], consider_rotation)


def _test_get_link_pose(spec: RobotSpec):
    # by comparing the link pose with the one by scikit-robot, which is well tested by long time
    # and has very different implementation from the one in plainmp. Thus, by comparing the two
    # implementation, we can verify the correctness of the kinematic model in plainmp
    kin = spec.get_kin()
    lb, ub = spec.angle_bounds()
    control_joint_names = spec.control_joint_names
    kin_joint_ids = kin.get_joint_ids(control_joint_names)
    robot_model = spec.get_robot_model()
    link_names = [l.name for l in robot_model.link_list]

    for _ in range(30):
        q = np.random.uniform(lb, ub)
        kin.set_joint_positions(kin_joint_ids, q)
        spec.reflect_kin_to_skrobot_model(robot_model)
        for link_name in link_names:
            print(link_name)
            pose = kin.debug_get_link_pose(link_name)
            pos = pose[:3]
            rot = quaternion2matrix(xyzw2wxyz(pose[3:]))
            pos_ref = robot_model.__dict__[link_name].worldpos()
            rot_ref = robot_model.__dict__[link_name].worldrot()
            np.testing.assert_allclose(pos, pos_ref, atol=1e-3)
            np.testing.assert_allclose(rot, rot_ref, atol=1e-3)


@pytest.mark.parametrize(
    "spec", [FetchSpec(), PandaSpec(), PR2RarmSpec(), PR2LarmSpec(), PR2DualarmSpec()]
)
def test_get_link_pose(spec):
    _test_get_link_pose(spec)


def test_skrobot_kin_base_mirror():
    spec = FetchSpec()
    model = spec.get_robot_model()
    kin = spec.get_kin()

    # skrobot => kin
    co = Coordinates([1, 2, 3], [0.1, 0.2, 0.3])
    model.newcoords(co)
    spec.reflect_skrobot_model_to_kin(model)
    pose_vec = kin.get_base_pose()
    pos, quat = pose_vec[:3], pose_vec[3:]
    mat = rotation_matrix_from_quat(xyzw2wxyz(quat))
    np.testing.assert_allclose(pos, co.worldpos(), atol=1e-3)
    np.testing.assert_allclose(mat, co.worldrot(), atol=1e-3)

    # kin => skrobot
    kin.set_base_pose([4, 5, 6, 0, 0, 0, 1.0])  # whatever
    spec.reflect_kin_to_skrobot_model(model)
    pos, rotmat = model.worldpos(), model.worldrot()
    np.testing.assert_allclose(pos, [4, 5, 6], atol=1e-3)
    np.testing.assert_allclose(rotmat, np.eye(3), atol=1e-3)
