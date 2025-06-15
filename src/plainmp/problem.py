from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import numpy as np

from plainmp.constraint import EqConstraintBase, IneqConstraintBase


@dataclass
class Problem:
    """Motion planning problem specification.

    This class encapsulates all the information needed to define a motion planning
    problem, including start/goal configurations, bounds, constraints, and resolution
    settings for collision checking.

    Parameters
    ----------
    start : np.ndarray
        Initial configuration (joint angles).
    lb : np.ndarray
        Lower bounds for joint angles.
    ub : np.ndarray
        Upper bounds for joint angles.
    goal_const : Union[EqConstraintBase, np.ndarray]
        Goal specification - either target joint angles or equality constraint.
    global_ineq_const : IneqConstraintBase, optional
        Global inequality constraints (e.g., collision avoidance).
    global_eq_const : EqConstraintBase, optional
        Global equality constraints.
    resolution : Union[float, np.ndarray]
        Motion validation resolution. If validator_type is "euclidean", use float.
        If "box", use array with per-joint resolution.
    validator_type : {"euclidean", "box"}, default="box"
        Type of motion validator for collision checking.
    goal_ineq_const : IneqConstraintBase, optional
        Goal-specific inequality constraints.
    goal_lb : np.ndarray, optional
        Goal-specific lower bounds.
    goal_ub : np.ndarray, optional
        Goal-specific upper bounds.

    Examples
    --------
    >>> # Point-to-point planning
    >>> problem = Problem(
    ...     start=q_start,
    ...     lb=joint_lb, ub=joint_ub,
    ...     goal_const=q_goal,
    ...     global_ineq_const=collision_constraint,
    ...     global_eq_const=None,
    ...     resolution=np.ones(7) * 0.05
    ... )
    >>>
    >>> # Planning to pose constraint
    >>> pose_constraint = robot_spec.create_gripper_pose_const([0.7, 0.2, 0.95, 0, 0, 0])
    >>> problem = Problem(
    ...     start=q_start,
    ...     lb=joint_lb, ub=joint_ub,
    ...     goal_const=pose_constraint,
    ...     global_ineq_const=collision_constraint,
    ...     global_eq_const=None,
    ...     resolution=np.ones(7) * 0.05
    ... )

    Notes
    -----
    The resolution parameter controls motion validation during planning:
    - "euclidean": Uses Euclidean distance in configuration space (float resolution)
    - "box": Uses per-joint resolution for more precise control (array resolution)

    Currently, "euclidean" validator is only supported by SBMP planners, not
    optimization-based planners.
    """

    # core specification
    start: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    goal_const: Union[EqConstraintBase, np.ndarray]
    global_ineq_const: Optional[IneqConstraintBase]
    global_eq_const: Optional[EqConstraintBase]
    resolution: Union[float, np.ndarray]
    validator_type: Literal["euclidean", "box"] = "box"

    # experimental features (not supported by all planners)
    goal_ineq_const: Optional[IneqConstraintBase] = None
    goal_lb: Optional[
        np.ndarray
    ] = None  # lb for goal (useful for ensuring final state manipulatability)
    goal_ub: Optional[np.ndarray] = None  # ub for goal

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
