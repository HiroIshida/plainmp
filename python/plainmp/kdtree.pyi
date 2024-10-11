import numpy as np

class KDTree:
    def __init__(self, data: np.ndarray, margin: float):
        """margin is used for collision detection"""
        ...
    def query(self, query: np.ndarray) -> np.ndarray:
        """Find the nearest neighbor of the query point."""
        ...
    def sqdist(self, query: np.ndarray) -> float:
        """Find the squared distance to the nearest neighbor of the query point."""
        ...
    def check_point_collision(self, point: np.ndarray) -> bool:
        """Check if the point is within the margin of any point in the tree."""
        ...
    def check_sphere_collision(self, center: np.ndarray, radius: float) -> bool:
        """Check if the sphere is within the margin of any point in the tree."""
        ...
