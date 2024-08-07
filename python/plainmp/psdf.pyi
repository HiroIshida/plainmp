from typing import Any, List

import numpy as np

class Pose:
    def __init__(self, translation: np.ndarray, rotation: np.ndarray) -> None: ...

class SDFBase:
    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the SDF at the given points.
        Args:
            points: The (3, n_pts) points to evaluate the SDF at.
                TODO: Should this be a (n_pts, 3) array for consistency with numpy?
                This would require changing Eigen's default storage order to be row-major.
        Returns:
            The signed distance at each point.
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
