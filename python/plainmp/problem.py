from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import numpy as np

from plainmp.constraint import EqConstraintBase, IneqConstraintBase


@dataclass
class Problem:
    """
    The resolution here is the euclidean distance in C-space.
    Currently we assume that if you set validator_type to "euclidean", the resolution is a float.
    NOTE: currently validator_type = "euclidean" is only be used for SBMP, an not supported by
    optimization-based planners.
    """

    start: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    goal_const: Union[EqConstraintBase, np.ndarray]
    global_ineq_const: Optional[IneqConstraintBase]
    global_eq_const: Optional[EqConstraintBase]
    resolution: Union[float, np.ndarray]
    validator_type: Literal["euclidean", "box"] = "box"

    def __post_init__(self):
        # In current implementation (but maybe extended in the future)
        # if you set validator_type to "box", the resolution is a numpy array.
        # Box validator, discretizes the straight line into waypoints such that the distance between
        # two consecutive waypoints is inside the box.
        # Default is "box", because it can be easily handled both by SBMP and optimization-based planners.
        if self.validator_type == "euclidean":
            assert isinstance(self.resolution, float), "not implemented yet"
        elif self.validator_type == "box":
            if isinstance(self.resolution, (List, Tuple)):
                self.resolution = np.array(self.resolution)
            assert isinstance(self.resolution, np.ndarray), "not implemented yet"
        else:
            raise ValueError(f"Unknown validator type: {self.validator_type}")

    def check_init_feasibility(self) -> Tuple[bool, str]:
        if not (np.all(self.lb <= self.start) and np.all(self.start <= self.ub)):
            return False, "Start point is out of bounds"
        if self.global_ineq_const is not None:
            if not self.global_ineq_const.is_valid(self.start):
                return False, "Start point violates global inequality constraints"
        return True, ""
