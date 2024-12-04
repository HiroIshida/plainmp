import numpy as np
import pytest
from skrobot.coordinates.math import quaternion2matrix, xyzw2wxyz

from plainmp.robot_spec import (
    FetchSpec,
    PandaSpec,
    PR2DualarmSpec,
    PR2LarmSpec,
    PR2RarmSpec,
    RobotSpec,
)


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
