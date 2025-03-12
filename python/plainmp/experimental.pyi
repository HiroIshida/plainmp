from typing import List

import numpy as np

from plainmp.constraint import IneqConstraintBase

class MultiGoalRRT:
    def __init__(
        self,
        start: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        ineq: IneqConstraintBase,
        max_nodes: int,
    ): ...
    def get_debug_states(self) -> np.ndarray:
        """Returns the states of the tree (D, N) <- reversed!"""
        ...
    def get_debug_parents(self) -> List[int]:
        """Returns the parent indices of the tree (N,)"""
        ...
    def is_reachable(self, goal: np.ndarray, search_radius: float) -> bool: ...
    def is_reachable_batch(self, goals: np.ndarray, search_radius: float) -> List[bool]: ...
