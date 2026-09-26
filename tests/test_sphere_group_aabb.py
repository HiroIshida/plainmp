import numpy as np
import pytest

from plainmp.constraint import SphereAttachmentSpec, SphereCollisionCst
from plainmp.kinematics import BaseType, KinematicModel
from plainmp.psdf import BoxSDF, GroundSDF, Pose, SphereSDF, UnionSDF


def make_constraint(positions=None, radii=None):
    kin = KinematicModel('<robot name="test"><link name="base"/></robot>')
    if positions is None:
        positions = np.array([[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    if radii is None:
        radii = np.array([0.05, 0.1, 0.075])
    attachment = SphereAttachmentSpec("base", positions, radii, False)
    return SphereCollisionCst(kin, [], BaseType.FLOATING, [attachment], [], None, False)


def old_is_valid(cst, obstacles):
    """Original main-sphere / sub-sphere order, using the same FK results."""

    def outside_aabb(sdf, p, r):
        return np.any(p < sdf.lb - r) or np.any(p > sdf.ub + r)

    [(center, radius)] = cst.get_group_spheres()
    spheres = cst.get_all_spheres()
    for sdf in obstacles:
        if outside_aabb(sdf, center, radius) or sdf.is_outside(center, radius):
            continue
        for p, r in spheres:
            if not outside_aabb(sdf, p, r) and not sdf.is_outside(p, r):
                return False
    return True


def test_aabb_lifetime_and_later_obstacle_collision():
    cst = make_constraint()
    # The first box intersects the main sphere but misses the thin group.
    # The second box hits the LAST sphere after the AABB has been cached.
    obstacles = [
        BoxSDF([0.1, 0.1, 0.1], Pose([0.0, 0.5, 0.0], np.eye(3))),
        BoxSDF([0.1, 0.1, 0.1], Pose([1.0, 0.0, 0.0], np.eye(3))),
    ]
    cst.set_sdf(UnionSDF(obstacles))
    q_free = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0])
    for q, expected in [(np.zeros(6), False), (q_free, True), (np.zeros(6), False)]:
        assert cst.is_valid(q) == expected
        assert old_is_valid(cst, obstacles) == expected
        # Computing gradient-side positions must not leave stale AABB bounds.
        cst.evaluate(q_free)


@pytest.mark.parametrize("offset", [0.0, 1e12, -1e12])
def test_aabb_near_contact_and_large_coordinates(offset):
    cst = make_constraint()
    q = np.array([offset, offset, offset, 0.0, 0.0, 0.0])
    first = BoxSDF([0.02, 0.02, 0.02], Pose([offset, offset + 0.5, offset], np.eye(3)))
    # Vary the second box by one representable step on either side of contact.
    touching = offset + 0.15
    for y in [np.nextafter(touching, -np.inf), touching, np.nextafter(touching, np.inf)]:
        obstacles = [first, BoxSDF([0.1, 0.1, 0.1], Pose([offset, y, offset], np.eye(3)))]
        cst.set_sdf(UnionSDF(obstacles))
        assert cst.is_valid(q) == old_is_valid(cst, obstacles)


def test_aabb_rotations_and_infinite_ground_bounds():
    cst = make_constraint()
    rng = np.random.RandomState(42)
    obstacles = [
        BoxSDF([0.1, 0.1, 0.1], Pose([0.0, 0.5, 0.0], np.eye(3))),
        SphereSDF(0.2, Pose([0.5, 0.3, 0.2], np.eye(3))),
        GroundSDF(0.0),
    ]
    cst.set_sdf(UnionSDF(obstacles))
    for _ in range(1000):
        q = np.r_[rng.uniform(-0.5, 0.5, 3), rng.uniform(-np.pi, np.pi, 3)]
        assert cst.is_valid(q) == old_is_valid(cst, obstacles)


def test_aabb_overlapping_boxes_with_noncolliding_spheres():
    # The group AABB overlaps the box, but the box lies between the spheres.
    cst = make_constraint(np.array([[-1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]), np.full(2, 0.05))
    obstacles = [BoxSDF([0.1, 0.1, 0.1], Pose())]
    cst.set_sdf(UnionSDF(obstacles))
    assert cst.is_valid(np.zeros(6))
    assert old_is_valid(cst, obstacles)


def test_aabb_cancellation_at_contact():
    # p + r rounds to zero, while object.lb - r rounds back to p. Expanding
    # just nextafter(p + r) would incorrectly discard the touching sphere.
    positions = np.array([[-1.0, -2.0], [0.0, 0.0], [0.0, 0.0]])
    cst = make_constraint(positions, np.array([1.0, 0.1]))
    obstacles = [
        BoxSDF([0.01, 0.01, 0.01], Pose([-1.5, 1.25, 0.0], np.eye(3))),
        BoxSDF([0.0, 0.0, 0.0], Pose([1e-300, 0.0, 0.0], np.eye(3))),
    ]
    cst.set_sdf(UnionSDF(obstacles))
    assert not cst.is_valid(np.zeros(6))
    assert not old_is_valid(cst, obstacles)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf, -0.05])
def test_aabb_nonfinite_input_falls_back(bad_value):
    cst = make_constraint(radii=np.array([0.05, bad_value, 0.075]))
    obstacles = [BoxSDF([0.1, 0.1, 0.1], Pose([0.0, 0.5, 0.0], np.eye(3)))]
    cst.set_sdf(UnionSDF(obstacles))
    assert cst.is_valid(np.zeros(6)) == old_is_valid(cst, obstacles)


def test_aabb_empty_group_without_reordering():
    cst = make_constraint(np.empty((3, 0)), np.empty(0))
    obstacles = [BoxSDF([1.0, 1.0, 1.0], Pose())]
    cst.set_sdf(UnionSDF(obstacles))
    assert cst.is_valid(np.zeros(6))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_aabb_nonfinite_center_is_not_hidden_by_later_spheres(bad_value, index):
    positions = np.array([[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    positions[0, index] = bad_value
    cst = make_constraint(positions, np.full(3, 0.05))
    obstacles = [BoxSDF([0.1, 0.1, 0.1], Pose([0.0, 0.5, 0.0], np.eye(3)))]
    cst.set_sdf(UnionSDF(obstacles))
    assert cst.is_valid(np.zeros(6)) == old_is_valid(cst, obstacles)


@pytest.mark.parametrize("radius", [0.0, 0.05, 0.1])
def test_aabb_uniform_radius_contact(radius):
    cst = make_constraint(radii=np.full(3, radius))
    touching = radius + 0.05
    for y in [np.nextafter(touching, -np.inf), touching, np.nextafter(touching, np.inf)]:
        obstacles = [
            BoxSDF([0.1, 0.1, 0.1], Pose([0.0, 0.5, 0.0], np.eye(3))),
            BoxSDF([0.1, 0.1, 0.1], Pose([1.0, y, 0.0], np.eye(3))),
        ]
        cst.set_sdf(UnionSDF(obstacles))
        assert cst.is_valid(np.zeros(6)) == old_is_valid(cst, obstacles)


@pytest.mark.parametrize("offset", [1e308, -1e308])
def test_aabb_finite_centers_with_overflowing_sum(offset):
    cst = make_constraint(radii=np.full(3, 0.05))
    obstacles = [BoxSDF([0.1, 0.1, 0.1], Pose([offset, 0.5, 0.0], np.eye(3)))]
    cst.set_sdf(UnionSDF(obstacles))
    q = np.array([offset, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert cst.is_valid(q)
    assert old_is_valid(cst, obstacles)
