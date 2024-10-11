import numpy as np

from plainmp.kdtree import KDTree


def test_kdtree():
    points = np.random.randn(1000, 3)
    tree = KDTree(points, 0.0)

    def blute_force(points, query):
        idx = np.argmin(np.sum((points - query) ** 2, axis=1))
        return points[idx]

    for _ in range(1000):
        query = np.random.randn(3)
        nearest = tree.query(query)
        assert np.allclose(nearest, blute_force(points, query))

        gt_sqdist_to_nearest = np.sum((nearest - query) ** 2)
        assert np.allclose(gt_sqdist_to_nearest, tree.sqdist(query))


def test_kdtree_collision():
    margin = 0.1
    sphere_radius = 0.1
    for _ in range(10):
        points = np.random.rand(50, 3)
        tree = KDTree(points, margin)

        for _ in range(100):
            query = np.random.rand(3)
            collide = tree.check_point_collision(query)
            assert collide == np.any(np.sum((points - query) ** 2, axis=1) < margin**2)

            collide = tree.check_sphere_collision(query, sphere_radius)
            assert collide == np.any(
                np.sum((points - query) ** 2, axis=1) < (margin + sphere_radius) ** 2
            )
