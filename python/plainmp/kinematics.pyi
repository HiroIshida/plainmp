from typing import List, Tuple

import numpy as np

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
    def get_joint_position_limits(self, joint_ids: List[int]) -> List[Tuple[float, float]]: ...
    def set_joint_positions(self, joint_ids: List[int], positions: List[float]) -> None: ...
    def get_joint_positions(self, joint_ids: List[int]) -> List[float]: ...
    def get_joint_ids(self, joint_names: List[str]) -> List[int]: ...
