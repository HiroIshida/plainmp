import copy
import time

import numpy as np
import pytest
from skrobot.coordinates import Coordinates, rpy_matrix
from skrobot.sdf import BoxSDF, CylinderSDF, SignedDistanceFunction, SphereSDF, UnionSDF

import plainmp.psdf as psdf


def convert(sksdf: SignedDistanceFunction) -> psdf.SDFBase:
    # get xyz and rotation matrix from sksdf and create Pose
    pose = psdf.Pose(sksdf.worldpos(), sksdf.worldrot())
    if isinstance(sksdf, BoxSDF):
        return psdf.BoxSDF(sksdf._width, pose)
    elif isinstance(sksdf, SphereSDF):
        return psdf.SphereSDF(sksdf._radius, pose)
    elif isinstance(sksdf, CylinderSDF):
        return psdf.CylinderSDF(sksdf._radius, sksdf._height, pose)
    elif isinstance(sksdf, UnionSDF):
        return psdf.UnionSDF([convert(s) for s in sksdf.sdf_list])
    else:
        raise ValueError("Unknown SDF type")


def test_pose_axis_align():
    trans = np.zeros(3)
    p1 = psdf.Pose(trans, np.eye(3))
    assert p1.axis_aligned
    assert p1.z_axis_aligned

    p2 = psdf.Pose(trans, rpy_matrix(0.3, 0.0, 0.0))
    assert not p2.axis_aligned
    assert p2.z_axis_aligned

    p3 = psdf.Pose(trans, rpy_matrix(0.3, 0.3, 0.3))
    assert not p3.axis_aligned
    assert not p3.z_axis_aligned


def test_copy_pose():
    for copy_method in [copy.copy, copy.deepcopy]:
        p1 = psdf.Pose(np.zeros(3), np.eye(3))
        p2 = copy_method(p1)
        p2.translate(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(p1.position, np.zeros(3))
        np.testing.assert_allclose(p2.position, np.array([1.0, 2.0, 3.0]))


def test_pose_transform():
    p = psdf.Pose(np.zeros(3), np.eye(3))
    p.translate(np.array([1.0, 2.0, 3.0]))
    p.rotate_z(0.3)
    rotmat_expected = rpy_matrix(0.3, 0.0, 0.0)
    np.testing.assert_allclose(p.position, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(p.rotation, rotmat_expected)


def test_clone_sdf():
    points = np.random.randn(300, 3) * 2

    def is_same_sdf(sdf1, sdf2):
        values1 = sdf1.evaluate_batch(points.T)
        values2 = sdf2.evaluate_batch(points.T)
        return np.allclose(values1, values2)

    # single case
    box_sdf = psdf.BoxSDF([1, 1, 1], psdf.Pose())
    box_sdf2 = box_sdf.clone()
    assert is_same_sdf(box_sdf, box_sdf2)
    box_sdf2.translate(np.array([1.0, 2.0, 3.0]))
    assert not is_same_sdf(box_sdf, box_sdf2)
    np.testing.assert_allclose(box_sdf.pose.position, np.zeros(3))
    np.testing.assert_allclose(box_sdf2.pose.position, np.array([1.0, 2.0, 3.0]))

    # composite case
    union_sdf = psdf.UnionSDF([box_sdf])
    union_sdf2 = union_sdf.clone()
    assert is_same_sdf(union_sdf, union_sdf2)
    union_sdf2.translate(np.array([1.0, 2.0, 3.0]))
    assert not is_same_sdf(union_sdf, union_sdf2)
    np.testing.assert_allclose(box_sdf.pose.position, np.zeros(3))
    union_sdf.translate(np.array([1.0, 2.0, 3.0]))
    assert is_same_sdf(union_sdf, union_sdf2)
    np.testing.assert_allclose(box_sdf.pose.position, np.array([1.0, 2.0, 3.0]))


@pytest.mark.parametrize("sksdf", [BoxSDF([0.5, 0.3, 0.6]), CylinderSDF(0.7, 0.2), SphereSDF(0.5)])
def test_consistency_with_skrobot(sksdf):
    for i in range(100):
        trans = np.random.randn(3) * 0.5
        if i == 0:
            yaw = pitch = roll = 0.0
        elif i == 1:
            yaw = 0.3
            pitch = 0.0
            roll = 0.0
        else:
            yaw, pitch, roll = np.random.randn(3)
        co = Coordinates(trans, rot=rpy_matrix(yaw, pitch, roll))
        sksdf.newcoords(co)
        sdf = convert(sksdf)

        if np.random.rand() < 0.5:
            r = 0.0
        else:
            r = np.random.uniform(0, 0.3)
        points = np.random.randn(1000, 3)
        sk_outside = sksdf(points) > r
        outside = np.array([sdf.is_outside(p, r) for p in points])
        assert np.all(sk_outside == outside)

        values = sksdf(points)
        cpp_values = sdf.evaluate_batch(points.transpose())
        assert np.allclose(values, cpp_values)

        # after translation and rotation
        trans = np.random.randn(3)
        yaw = np.random.randn()

        sksdf.translate(trans, wrt="world")
        sksdf.rotate(yaw, "z", wrt="world")

        sdf.translate(trans)
        sdf.rotate_z(yaw)

        sk_outside = sksdf(points) > r
        outside = np.array([sdf.is_outside(p, r) for p in points])
        assert np.all(sk_outside == outside)

        values = sksdf(points)
        cpp_values = sdf.evaluate_batch(points.transpose())
        np.abs(values - cpp_values)
        assert np.allclose(values, cpp_values)

        # after clone
        sdf_clone = sdf.clone()
        cpp_values_clone = sdf_clone.evaluate_batch(points.transpose())
        assert np.allclose(cpp_values, cpp_values_clone)


def check_single_batch_consistency(cppsdf: psdf.SDFBase, points):
    values = [cppsdf.evaluate(p) for p in points]
    values_batch = cppsdf.evaluate_batch(points.T)
    assert np.allclose(values, values_batch)


def check_is_outside_consistency(cppsdf: psdf.SDFBase, points):
    for r in np.linspace(0.0, 2.0, 10):
        values = [cppsdf.is_outside(p, r) for p in points]
        values_batch = cppsdf.evaluate_batch(points.T) > r
        assert np.allclose(values, values_batch)


sksdfs = [
    BoxSDF([1, 1, 1]),
    SphereSDF(1),
    CylinderSDF(1, 1),
]


@pytest.mark.parametrize("sksdf", sksdfs)
def test_closed_primitive_sdfs(sksdf):
    for i in range(10):
        xyz = np.random.randn(3)
        if i == 0:
            ypr = np.zeros(3)
        elif i == 1:
            ypr = np.array([0.3, 0.0, 0.0])
        else:
            ypr = np.random.randn(3)
        sksdf.newcoords(Coordinates(xyz, ypr))
        cppsdf = convert(sksdf)

        points = np.random.randn(100, 3) * 2
        sk_dist = sksdf(points)
        dist = cppsdf.evaluate_batch(points.T)
        assert np.allclose(sk_dist, dist)

        check_single_batch_consistency(cppsdf, points)
        check_is_outside_consistency(cppsdf, points)


def test_ground_sdf():
    sdf = psdf.GroundSDF(1.0)
    assert sdf.evaluate(np.array([0.0, 0.0, 0.0])) == 1.0
    assert sdf.evaluate(np.array([1.0, 0.0, 0.0])) == 1.0

    check_single_batch_consistency(sdf, np.random.randn(100, 3) * 3)
    check_is_outside_consistency(sdf, np.random.randn(100, 3) * 3)


def test_cloud_sdf():
    obstacles = np.random.rand(100, 3)
    sdf = psdf.CloudSDF(obstacles, 0.1)
    eval_points = np.random.randn(1000, 3) * 1.5
    check_single_batch_consistency(sdf, eval_points)
    check_is_outside_consistency(sdf, eval_points)


def test_union_sdf():

    for _ in range(10):
        sdf1 = BoxSDF([1, 1, 1])
        xyz = np.random.randn(3)
        ypr = np.random.randn(3)
        sdf1.newcoords(Coordinates(xyz, ypr))
        sdf2 = SphereSDF(1)
        sksdf = UnionSDF([sdf1, sdf2])
        cppsdf = convert(sksdf)

        points = np.random.randn(100, 3) * 2
        sk_dist = sksdf(points)
        dist = cppsdf.evaluate_batch(points.T)
        assert np.allclose(sk_dist, dist)

        check_single_batch_consistency(cppsdf, points)
        check_is_outside_consistency(cppsdf, points)

        # after translation and rotation
        trans = np.random.randn(3)
        yaw = np.random.randn()
        cppsdf.translate(trans)
        cppsdf.rotate_z(yaw)

        sdf1.translate(trans, wrt="world")
        sdf1.rotate(yaw, "z", wrt="world")
        sdf2.translate(trans, wrt="world")
        sdf2.rotate(yaw, "z", wrt="world")
        sk_dist = sksdf(points)
        dist = cppsdf.evaluate_batch(points.T)
        assert np.allclose(sk_dist, dist)

        # after clone
        cppsdf_clone = cppsdf.clone()
        dist_clone = cppsdf_clone.evaluate_batch(points.T)
        assert np.allclose(dist, dist_clone)


def test_speed():
    sdf1 = BoxSDF([1, 1, 1])
    xyz = np.random.randn(3)
    ypr = np.random.randn(3)
    sdf1.newcoords(Coordinates(xyz, ypr))
    sdf2 = SphereSDF(1)
    sksdf = UnionSDF([sdf1, sdf2])
    cppsdf = convert(sksdf)

    points = np.random.randn(100, 3)
    ts = time.time()
    for _ in range(10000):
        sksdf(points)
    skrobot_time = time.time() - ts
    ts = time.time()
    for _ in range(10000):
        cppsdf.evaluate_batch(points.T)
    cppsdf_time = time.time() - ts
    print(f"skrobot_time: {skrobot_time}, cppsdf_time: {cppsdf_time}")
    assert cppsdf_time < skrobot_time * 0.1


if __name__ == "__main__":
    pass
