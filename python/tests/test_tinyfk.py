import pytest

from plainmp.robot_spec import FetchSpec


def test_tinyfk_kinematic_mdoel_add_new_link():
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
