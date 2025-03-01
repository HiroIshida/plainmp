from typing import Any, List, TypeVar

import numpy as np

class Pose:
    def __init__(self, translation: np.ndarray, rotation: np.ndarray) -> None:
        """Create a new Pose.
        Args:
            translation: The (3,) translation vector. Defaults to [0, 0, 0].
            rotation: The (3, 3) rotation matrix. Defaults to identity.
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
    def translate(self, translation: np.ndarray) -> None:
        """Translate the pose wrt world frame.
        Args:
            translation: The (3,) translation vector.
        """
        ...
    def rotate_z(self, angle: float) -> None:
        """Rotate the pose wrt world frame.
        Args:
            angle: The angle to rotate by.
        """
        ...

_T = TypeVar("T")

class SDFBase:
    def clone(self: _T) -> _T: ...
    def translate(self, translation: np.ndarray) -> None:
        """Translate the SDF wrt world frame.
        Args:
            translation: The (3,) translation vector.
        """
        ...
    def rotate_z(self, angle: float) -> None:
        """Rotate the SDF wrt world frame.
        Args:
            angle: The angle to rotate by.
        """
        ...
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

class _HasPose:
    @property
    def pose(self) -> Pose: ...

class UnionSDF(SDFBase):
    def __init__(self, sdf_list: List[SDFBase]) -> None: ...
    def merge(self, other: UnionSDF, clone: bool = False) -> None: ...
    def add(self, sdf: SDFBase, clone: bool = False) -> None: ...

class PrimitiveSDFBase(SDFBase): ...

class GroundSDF(PrimitiveSDFBase):
    def __init__(self, height: float) -> None: ...

class ClosedPrimitiveSDFBase(PrimitiveSDFBase): ...

class BoxSDF(ClosedPrimitiveSDFBase, _HasPose):
    def __init__(self, size: np.ndarray, pose: Pose) -> None: ...

class CylinderSDF(ClosedPrimitiveSDFBase, _HasPose):
    def __init__(self, radius: float, height: float, pose: Pose) -> None: ...

class SphereSDF(ClosedPrimitiveSDFBase, _HasPose):
    def __init__(self, radius: float, pose: Pose) -> None: ...

class CloudSDF(PrimitiveSDFBase):
    def __init__(self, points: np.ndarray, radius: float) -> None: ...
