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
        assert np.allclose(tree.query(query), blute_force(points, query))
