from typing import Any, List

import numpy as np

class Pose:
    def __init__(self, translation: np.ndarray, rotation: np.ndarray) -> None: ...

class SDFBase:
    def evaluate(self, point: np.ndarray) -> np.ndarray:
        """Evaluate the SDF at the given points.
        Args:
            point: The (3,) point to evaluate the SDF at.
        Returns:
            The signed distance at
        """
        ...
    def evaluate_batch(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the SDF at the given points.
        Args:
            points: The (N, 3) points to evaluate the SDF at.
        Returns:
            The signed distances at the given points.
        """
        ...

class UnionSDF(SDFBase):
    def __init__(self, sdf_list: List[SDFBase], create_bvh: bool) -> None: ...

class PrimitiveSDFBase(SDFBase): ...

class BoxSDF(PrimitiveSDFBase):
    def __init__(self, size: np.ndarray, pose: Pose) -> None: ...

class CylinderSDF(PrimitiveSDFBase):
    def __init__(self, radius: float, height: float, pose: Pose) -> None: ...

class SphereSDF(PrimitiveSDFBase):
    def __init__(self, radius: float, pose: Pose) -> None: ...
