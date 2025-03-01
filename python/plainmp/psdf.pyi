from typing import Any, List

import numpy as np

class Pose:
    def __init__(self, translation: np.ndarray, rotation: np.ndarray) -> None:
        """Create a new Pose.
        Args:
            translation: The (3,) translation vector. Defaults to [0, 0, 0].
            rotation: The (3,) rotation vector. Defaults to identity.
        """
        ...
    @property
    def axis_aligned(self) -> bool: ...
    @property
    def z_axis_aligned(self) -> bool: ...
    @property
    def position(self) -> np.ndarray: ...
    @property
    def rotation(self) -> np.ndarray: ...
    def translate(self, translation: np.ndarray) -> None: ...
    def rotate_z(self, angle: float) -> None: ...

class SDFBase:
    def evaluate(self, point: np.ndarray) -> float:
        """Evaluate the SDF at the given points.
        Args:
            point: The (3,) point to evaluate the SDF at.
        Returns:
            The signed distance at
        """
        ...
    def evaluate_batch(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the SDF at the given points (Note 3xN, not Nx3).
        Args:
            points: The (3, N) points to evaluate the SDF at.
        Returns:
            The signed distances at the given points.
        """
        ...
    def is_outside(self, point: np.ndarray, radius: float) -> bool:
        """Check if the point is outside the SDF.
        Args:
            point: The (3,) point to check.
            radius: The radius of the point.
        Returns:
            True if the point is outside the SDF, False otherwise.
        """
        ...

class UnionSDF(SDFBase):
    def __init__(self, sdf_list: List[SDFBase]) -> None: ...

class PrimitiveSDFBase(SDFBase): ...

class GroundSDF(PrimitiveSDFBase):
    def __init__(self, height: float) -> None: ...

class ClosedPrimitiveSDFBase(PrimitiveSDFBase): ...

class BoxSDF(ClosedPrimitiveSDFBase):
    def __init__(self, size: np.ndarray, pose: Pose) -> None: ...

class CylinderSDF(ClosedPrimitiveSDFBase):
    def __init__(self, radius: float, height: float, pose: Pose) -> None: ...

class SphereSDF(ClosedPrimitiveSDFBase):
    def __init__(self, radius: float, pose: Pose) -> None: ...

class CloudSDF(PrimitiveSDFBase):
    def __init__(self, points: np.ndarray, radius: float) -> None: ...
