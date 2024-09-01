import time

import numpy as np
import pytest
from skrobot.coordinates import Coordinates, rpy_matrix
from skrobot.sdf import BoxSDF, CylinderSDF, SignedDistanceFunction, SphereSDF, UnionSDF

import plainmp.psdf as psdf
from plainmp.utils import sksdf_to_cppsdf


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
        sdf = sksdf_to_cppsdf(sksdf)

        if np.random.rand() < 0.5:
            r = 0.0
        else:
            r = np.random.uniform(0, 0.3)
        points = np.random.randn(1000, 3)
        sk_outside = sksdf(points) > r
        inside = np.array([sdf.is_outside(p, r) for p in points])
        assert np.all(sk_outside == inside)

        values = sksdf(points)
        cpp_values = sdf.evaluate_batch(points.transpose())
        assert np.allclose(values, cpp_values)


def convert(sksdf: SignedDistanceFunction, create_bvh: bool = False) -> psdf.SDFBase:
    # get xyz and rotation matrix from sksdf and create Pose
    pose = psdf.Pose(sksdf.worldpos(), sksdf.worldrot())
    if isinstance(sksdf, BoxSDF):
        return psdf.BoxSDF(sksdf._width, pose)
    elif isinstance(sksdf, SphereSDF):
        return psdf.SphereSDF(sksdf._radius, pose)
    elif isinstance(sksdf, CylinderSDF):
        return psdf.CylinderSDF(sksdf._radius, sksdf._height, pose)
    elif isinstance(sksdf, UnionSDF):
        return psdf.UnionSDF([convert(s) for s in sksdf.sdf_list], create_bvh)
    else:
        raise ValueError("Unknown SDF type")


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
