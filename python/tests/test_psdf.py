import time

import numpy as np
import pytest
from skrobot.coordinates import Coordinates
from skrobot.sdf import BoxSDF, CylinderSDF, SignedDistanceFunction, SphereSDF, UnionSDF

import plainmp.psdf as psdf


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
    values = [cppsdf.is_outside(p, 0.0) for p in points]
    values_batch = cppsdf.evaluate_batch(points.T) > 0.0
    assert np.allclose(values, values_batch)


sksdfs = [
    BoxSDF([1, 1, 1]),
    SphereSDF(1),
    CylinderSDF(1, 1),
]


@pytest.mark.parametrize("sksdf", sksdfs)
def test_primitive_sdfs(sksdf):
    for _ in range(10):
        xyz = np.random.randn(3)
        ypr = np.random.randn(3)
        sksdf.newcoords(Coordinates(xyz, ypr))
        cppsdf = convert(sksdf)

        points = np.random.randn(100, 3) * 2
        sk_dist = sksdf(points)
        dist = cppsdf.evaluate_batch(points.T)
        assert np.allclose(sk_dist, dist)

        check_single_batch_consistency(cppsdf, points)
        check_is_outside_consistency(cppsdf, points)


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


def test_aabb_boxsdf():
    sdf = BoxSDF([1, 2, 3])
    sdf.translate(np.ones(3) * 0.5)
    cpp_sdf = convert(sdf)
    aabb = cpp_sdf.get_aabb()
    assert np.allclose(aabb.lb, [0.0, -0.5, -1.0])
    assert np.allclose(aabb.ub, [1.0, 1.5, 2.0])

    sdf = BoxSDF([1, 1, 1])
    sdf.rotate(np.pi / 6, "z")
    cpp_sdf = convert(sdf)
    aabb = cpp_sdf.get_aabb()
    w = 0.5 * (np.cos(np.pi / 6) + np.sin(np.pi / 6))
    assert np.allclose(aabb.lb, [-w, -w, -0.5])
    assert np.allclose(aabb.ub, [+w, +w, 0.5])


def test_aabb_spheresdf():
    sdf = SphereSDF(1)
    sdf.translate(np.ones(3) * 0.5)
    cpp_sdf = convert(sdf)
    aabb = cpp_sdf.get_aabb()
    assert np.allclose(aabb.lb, [-0.5, -0.5, -0.5])
    assert np.allclose(aabb.ub, [1.5, 1.5, 1.5])


def test_aabb_cylindersdf():
    sdf = CylinderSDF(2, 1)
    sdf.translate(np.ones(3) * 0.5)
    cpp_sdf = convert(sdf)
    aabb = cpp_sdf.get_aabb()
    assert np.allclose(aabb.lb, [-0.5, -0.5, -0.5])
    assert np.allclose(aabb.ub, [1.5, 1.5, 1.5])


def test_bvh():
    return  # skip this test for now
    for _ in range(100):
        sdf1 = BoxSDF([1, 1, 1])
        sdf1.translate(np.random.rand(3) * 3)
        sdf2 = SphereSDF(1)
        sdf2.translate(np.random.rand(3) * 3)
        sksdf = UnionSDF([sdf1, sdf2])
        cppsdf = convert(sksdf, create_bvh=True)

        sdf3 = CylinderSDF(1, 1)
        sdf3.translate(np.random.rand(3) * 3)
        cppsdf_total_bvh = psdf.UnionSDF([cppsdf, convert(sdf3)], True)
        cppsdf_total_naive = convert(UnionSDF([sdf1, sdf2, sdf3]), create_bvh=False)

        points = np.random.rand(1000, 3) * 3
        rs = np.random.rand(1000) * 0.2

        ts = time.time()
        values1 = []
        for p, r in zip(points, rs):
            values1.append(cppsdf_total_bvh.is_outside(p, r))
        time.time() - ts

        ts = time.time()
        values2 = []
        for p, r in zip(points, rs):
            values2.append(cppsdf_total_naive.is_outside(p, r))
        time.time() - ts
        assert np.allclose(values1, values2)


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
