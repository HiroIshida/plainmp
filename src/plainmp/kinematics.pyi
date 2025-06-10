from enum import Enum
from typing import List, Tuple

import numpy as np

class BaseType(Enum):
    FIXED = 0
    FLOATING = 1
    PLANAR = 2

class KinematicModel:
    def __init__(self, urdf_string: str) -> None: ...
    # NOTE: urdf_string is not a file path, but the actual content of the URDF file

    def add_new_link(
        self,
        link_name: str,
        parent_link_name: str,
        position: np.ndarray,
        rpy: np.ndarray,
        consider_rotation: bool,
    ) -> None: ...
    def debug_get_link_pose(self, link_name: str) -> List[float]:
        """Return the pose of the link in world frame
        Args:
            link_name: name of the link
        Returns:
            [x, y, z, qx, qy, qz, qw]
        Note:
            use this method mainly for debugging because it is not efficient
        """
        ...
    def get_joint_position_limits(self, joint_ids: List[int]) -> List[Tuple[float, float]]: ...
    def set_joint_positions(self, joint_ids: List[int], positions: np.ndarray) -> None: ...
    def get_joint_positions(self, joint_ids: List[int]) -> List[float]: ...
    def set_base_pose(self, pose_vec: np.ndarray) -> None:
        """Set the base pose of the robot
        Args:
            pose_vec: [x, y, z, qx, qy, qz, qw]
        """
        ...
    def get_base_pose(self) -> np.ndarray:
        """Get the base pose of the robot
        Returns:
            [x, y, z, qx, qy, qz, qw]
        """
        ...
    def get_joint_ids(self, joint_names: List[str]) -> List[int]: ...
