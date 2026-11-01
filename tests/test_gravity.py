import xml.etree.ElementTree as ET

import numpy as np
import pytest

from plainmp.kinematics import BaseType, KinematicModel
from plainmp.robot_spec import FetchSpec, OpenArmV10RarmSpec


def _rotate(quaternion, vector):
    # Eigen's quaternion action, including the non-unit quaternions produced
    # by accurate=False. Libraries which normalize would hide regressions.
    xyz, w = quaternion[:3], quaternion[3]
    return vector + 2 * np.cross(xyz, np.cross(xyz, vector) + w * vector)


def _legacy_gravity(kin, xml, names):
    robot = ET.fromstring(xml)
    joints = {j.attrib["name"]: j for j in robot.findall("joint")}
    parents = {
        j.find("child").attrib["link"]: j.find("parent").attrib["link"] for j in joints.values()
    }
    result = np.zeros(len(names))
    for link in robot.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        name = link.attrib["name"]
        pose = np.asarray(kin.debug_get_link_pose(name))
        origin = inertial.find("origin")
        com = np.fromstring(
            origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" "
        )
        com = pose[:3] + _rotate(pose[3:], com)
        force = np.array([0.0, 0.0, -9.80665 * float(inertial.find("mass").attrib["value"])])
        ancestors = {name}
        while name in parents:
            name = parents[name]
            ancestors.add(name)
        for i, joint_name in enumerate(names):
            joint = joints[joint_name]
            child = joint.find("child").attrib["link"]
            if child not in ancestors:
                continue
            jp = np.asarray(kin.debug_get_link_pose(child))
            axis = _rotate(jp[3:], np.fromstring(joint.find("axis").attrib["xyz"], sep=" "))
            result[i] += (
                force @ axis
                if joint.attrib["type"] == "prismatic"
                else np.cross(com - jp[:3], force) @ axis
            )
    return result


@pytest.mark.parametrize("spec_type", [FetchSpec, OpenArmV10RarmSpec])
@pytest.mark.parametrize("accurate", [True, False])
def test_gravity_matches_original_formula(spec_type, accurate):
    spec = spec_type()
    kin = spec.get_kin()
    xml = spec.urdf_path.read_text()
    names = spec.control_joint_names
    ids = kin.get_joint_ids(names)
    rng = np.random.default_rng(42)
    for q in rng.uniform(*spec.angle_bounds(), (8, len(ids))):
        actual = kin.get_gravity_term2(ids, q, accurate=accurate)
        expected = _legacy_gravity(kin, xml, names)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.fixture
def gravity_arm():
    xml = """
    <robot name="gravity">
      <link name="base"/>
      <link name="a"><inertial><origin xyz="0.4 0.1 0"/><mass value="2"/>
        <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial></link>
      <link name="spacer"/>
      <link name="b"><inertial><origin xyz="0.3 0.2 0.1"/><mass value="3"/>
        <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial></link>
      <link name="side"><inertial><origin xyz="0.2 0 0"/><mass value="1"/>
        <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial></link>
      <joint name="first" type="continuous"><parent link="base"/><child link="a"/>
        <axis xyz="0 1 0"/></joint>
      <joint name="fixed" type="fixed"><parent link="a"/><child link="spacer"/>
        <origin xyz="0.8 0 0.1" rpy="0.2 0.3 0.4"/></joint>
      <joint name="second" type="continuous"><parent link="spacer"/><child link="b"/>
        <axis xyz="1 0 0"/></joint>
      <joint name="side_joint" type="continuous"><parent link="base"/><child link="side"/>
        <axis xyz="0 1 0"/></joint>
    </robot>
    """
    return KinematicModel(xml), xml


def test_gravity_selection_order_and_locked_joint_updates(gravity_arm):
    kin, xml = gravity_arm
    names = ["first", "second", "side_joint"]
    ids = kin.get_joint_ids(names)
    q = np.array([0.3, -0.7, 0.4])
    full = kin.get_gravity_term2(ids, q)
    np.testing.assert_allclose(kin.get_gravity_term(ids[::-1]), full[::-1], atol=1e-12)
    np.testing.assert_allclose(
        kin.get_gravity_term([ids[0], ids[0]]), [full[0], full[0]], atol=1e-12
    )
    for value in [0.1, 1.7, -2.4]:
        kin.set_joint_positions([ids[1]], np.array([value]))
        actual = kin.get_gravity_term([ids[0]])
        np.testing.assert_allclose(actual, _legacy_gravity(kin, xml, ["first"]), atol=1e-12)
    for mass in [0.0, 5.0, 3.0]:
        link_id = kin.get_link_ids(["b"])[0]
        kin.set_link_mass(link_id, mass)
        changed = ET.fromstring(xml)
        changed.find("./link[@name='b']/inertial/mass").set("value", str(mass))
        np.testing.assert_allclose(
            kin.get_gravity_term([ids[0]]),
            _legacy_gravity(kin, ET.tostring(changed), ["first"]),
            atol=1e-12,
        )


def test_gravity_approximate_fk_with_locked_rotations(gravity_arm):
    kin, xml = gravity_arm
    first, second = kin.get_joint_ids(["first", "second"])
    kin.set_joint_positions([second], np.array([0.7]))
    actual = kin.get_gravity_term2([first], np.array([3.0]), accurate=False)
    np.testing.assert_allclose(actual, _legacy_gravity(kin, xml, ["first"]), atol=1e-11, rtol=1e-12)
    kin.set_joint_positions([second], np.array([3.0]), accurate=False)
    for accurate in [True, False, True]:
        actual = kin.get_gravity_term2([first], np.array([0.4]), accurate=accurate)
        np.testing.assert_allclose(
            actual, _legacy_gravity(kin, xml, ["first"]), atol=1e-11, rtol=1e-12
        )


def test_gravity_base_pose_and_added_links(gravity_arm):
    kin, xml = gravity_arm
    names = ["first", "second", "side_joint"]
    ids = kin.get_joint_ids(names)
    q = np.array([0.3, -0.5, 0.7])
    kin.get_gravity_term2(ids, q)
    quaternion = np.array([0.2, 0.3, -0.1, 0.9])
    quaternion /= np.linalg.norm(quaternion)
    kin.set_base_pose(np.r_[[0.5, -0.2, 0.8], quaternion])
    np.testing.assert_allclose(
        kin.get_gravity_term(ids), _legacy_gravity(kin, xml, names), atol=1e-11
    )
    kin.add_new_link("aux", "b", [0.2, 0.1, 0.4], [0.1, 0.3, 0.2], False)
    np.testing.assert_allclose(
        kin.get_gravity_term2(ids, q), _legacy_gravity(kin, xml, names), atol=1e-11
    )


@pytest.mark.parametrize(
    "layout", ["strided", "reversed", "float32", "readonly", "unaligned", "big_endian"]
)
def test_gravity_array_layouts_and_ownership(gravity_arm, layout):
    kin, _ = gravity_arm
    ids = kin.get_joint_ids(["first", "second"])
    q = np.array([0.3, -0.7])
    if layout == "strided":
        q = np.repeat(q, 2)[::2]
    elif layout == "reversed":
        q = q[::-1].copy()[::-1]
    elif layout == "float32":
        q = q.astype(np.float32)
    elif layout == "readonly":
        q.flags.writeable = False
    elif layout == "big_endian":
        q = q.astype(">f8")
    else:
        unaligned = np.ndarray(q.shape, dtype=float, buffer=bytearray(q.nbytes + 1), offset=1)
        unaligned[:] = q
        q = unaligned
    expected = kin.get_gravity_term2(ids, np.array(q, dtype=float))
    actual = kin.get_gravity_term2(ids, q)
    saved = kin.get_gravity_term(ids)
    kin.get_gravity_term2(ids, np.zeros(2))
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(saved, expected, atol=1e-12)


def test_gravity_invalid_input_does_not_change_positions(gravity_arm):
    kin, _ = gravity_arm
    ids = kin.get_joint_ids(["first", "second"])
    q = np.array([0.3, -0.7])
    kin.get_gravity_term2(ids, q)
    with pytest.raises(ValueError, match="match joint_ids"):
        kin.get_gravity_term2(ids, np.zeros(1))
    with pytest.raises(ValueError, match="invalid joint id"):
        kin.get_gravity_term2([999], np.zeros(1))
    with pytest.raises(ValueError, match="invalid joint id"):
        kin.get_gravity_term([999])
    np.testing.assert_array_equal(kin.get_joint_positions(ids), q)


@pytest.mark.parametrize(
    "base_type,extra", [(BaseType.FIXED, 0), (BaseType.PLANAR, 3), (BaseType.FLOATING, 6)]
)
def test_gravity_legacy_base_output_shape(gravity_arm, base_type, extra):
    kin, xml = gravity_arm
    ids = kin.get_joint_ids(["first", "second"])
    actual = kin.get_gravity_term2(ids, np.array([0.2, -0.4]), base_type)
    assert actual.shape == (2 + extra,)
    np.testing.assert_allclose(
        actual[:2], _legacy_gravity(kin, xml, ["first", "second"]), atol=1e-12
    )
    np.testing.assert_array_equal(actual[2:], np.zeros(extra))
    np.testing.assert_array_equal(kin.get_gravity_term([], base_type), np.zeros(extra))
