import numpy as np

from plainmp.kdtree import KDTree


def test_kdtree():
    points = np.random.randn(1000, 3)
    tree = KDTree(points)

    def blute_force(points, query):
        idx = np.argmin(np.sum((points - query) ** 2, axis=1))
        return points[idx]

    for _ in range(1000):
        query = np.random.randn(3)
        nearest = tree.query(query)
        assert np.allclose(nearest, blute_force(points, query))

        gt_sqdist_to_nearest = np.sum((nearest - query) ** 2)
        assert np.allclose(gt_sqdist_to_nearest, tree.sqdist(query))
