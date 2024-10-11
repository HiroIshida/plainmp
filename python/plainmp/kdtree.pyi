import numpy as np

class KDTree:
    def __init__(self, data: np.ndarray): ...
    def query(self, query: np.ndarray) -> np.ndarray:
        """Find the nearest neighbor of the query point."""
        ...
    def sqdist(self, query: np.ndarray) -> float:
        """Find the squared distance to the nearest neighbor of the query point."""
        ...
