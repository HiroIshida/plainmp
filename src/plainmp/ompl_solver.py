import time
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Sequence, TypeVar, Union

import numpy as np

from plainmp.constraint import IneqCompositeCst
from plainmp.ik import IKConfig, IKResult, solve_ik
from plainmp.problem import Problem
from plainmp.trajectory import Trajectory

from ._plainmp.ompl import (  # noqa: F401
    ERTConnectPlanner,
    OMPLPlanner,
    RefineType,
    ValidatorConfig,
    ValidatorType,
    set_log_level_none,
    set_random_seed,
    simplify,
)


class Algorithm(Enum):
    BKPIECE1 = "BKPIECE1"
    KPIECE1 = "KPIECE1"
    LBKPIECE1 = "LBKPIECE1"
    RRTConnect = "RRTConnect"
    RRT = "RRT"
    RRTstar = "RRTstar"
    EST = "EST"
    BiEST = "BiEST"

    def is_unidirectional(self) -> bool:
        return self in [Algorithm.RRT, Algorithm.KPIECE1, Algorithm.LBKPIECE1]


@dataclass
class OMPLSolverConfig:
    n_max_call: int = 1000000
    n_max_ik_trial: int = 100
    algorithm: Algorithm = Algorithm.RRTConnect
    algorithm_range: Optional[float] = 2.0
    refine_seq: Sequence[RefineType] = tuple()
    shortcut: bool = False
    bspline: bool = False
    # these three parameters are used only when ERTConnect is selected
    # NOTE: the default values are the same as the default values in the paper
    ertconnect_omega_min: float = 0.05
    ertconnect_omega_max: float = 0.1
    ertconnect_eps: float = 5.0
    timeout: Optional[float] = None
    use_goal_sampler: bool = (
        False  # use goal sampler in unidirectional planner. Use only when the goal is not a point
    )
    max_goal_sampler_count: int = 100

    def __post_init__(self):
        if len(self.refine_seq) > 0:
            return
        # keep this for backward compatibility. This is deprecated.
        refine_seq = []
        if self.bspline:
            refine_seq.append(RefineType.BSPLINE)
        if self.shortcut:
            refine_seq.append(RefineType.SHORTCUT)
        self.refine_seq = refine_seq


class TerminateState(Enum):
    SUCCESS = 1
    FAIL_SATISFACTION = 2
    FAIL_PLANNING = 3


@dataclass
class OMPLSolverResult:
    traj: Optional[Trajectory]
    time_elapsed: Optional[float]
    n_call: int
    terminate_state: TerminateState
    ns_internal: Optional[int] = None  # internal measurement in nanoseconds

    @property
    def success(self) -> bool:
        return self.traj is not None


OMPLSolverT = TypeVar("OMPLSolverT", bound="OMPLSolver")


class OMPLSolver:
    """Motion planning solver using OMPL (Open Motion Planning Library).

    This solver provides sampling-based motion planning algorithms for robot path planning.
    It supports various algorithms like RRTConnect, RRT*, and custom variants with
    collision checking and path optimization.

    Parameters
    ----------
    config : OMPLSolverConfig, optional
        Configuration parameters for the solver. If None, default settings are used.

    Examples
    --------
    >>> from plainmp.ompl_solver import OMPLSolver, OMPLSolverConfig
    >>> from plainmp.problem import Problem
    >>>
    >>> config = OMPLSolverConfig(algorithm=Algorithm.RRTConnect, n_max_call=1000000)
    >>> solver = OMPLSolver(config)
    >>> problem = Problem(q_start, lb, ub, q_goal, collision_cst, None, resolution)
    >>> result = solver.solve(problem)
    >>> if result.success:
    ...     print(f"Path found in {result.time_elapsed:.3f} seconds")
    ...     trajectory = result.traj
    """

    config: OMPLSolverConfig

    def __init__(self, config: Optional[OMPLSolverConfig] = None):
        if config is None:
            config = OMPLSolverConfig()
        self.config = config

    def solve_ik(self, problem: Problem, guess: Optional[Trajectory] = None) -> IKResult:
        # IK is supposed to stop within the timeout but somehow it does not work well
        # so we set...
        config = IKConfig(timeout=self.config.timeout)

        # determine the inequality constraint
        if problem.goal_ineq_const is None and problem.global_ineq_const is None:
            ineq_const = None
        elif problem.goal_ineq_const is None:
            ineq_const = problem.global_ineq_const
        elif problem.global_ineq_const is None:
            ineq_const = problem.goal_ineq_const
        else:
            ineq_const = IneqCompositeCst([problem.goal_ineq_const, problem.global_ineq_const])

        # determine bounds
        lb = problem.lb if problem.goal_lb is None else problem.goal_lb
        ub = problem.ub if problem.goal_ub is None else problem.goal_ub

        if guess is not None:
            # If guess is provided, use the last element of the trajectory as the initial guess
            q_guess = guess.numpy()[-1]
            ret = solve_ik(
                problem.goal_const,
                ineq_const,
                lb,
                ub,
                q_seed=q_guess,
                max_trial=self.config.n_max_ik_trial,
                config=config,
            )
            return ret
        else:
            ret = solve_ik(
                problem.goal_const,
                ineq_const,
                lb,
                ub,
                max_trial=self.config.n_max_ik_trial,
                config=config,
            )
            if ret.success:
                return ret
            return ret  # type: ignore

    def solve(self, problem: Problem, guess: Optional[Trajectory] = None) -> OMPLSolverResult:
        """Solve a motion planning problem.

        This method solves the given motion planning problem using the configured
        sampling-based algorithm. It handles goal configuration generation via IK
        if needed, collision checking, and path simplification if specified in the config.

        Parameters
        ----------
        problem : Problem
            Motion planning problem containing start/goal, bounds, and constraints.
        guess : Trajectory, optional
            Initial trajectory guess for warm-starting the planner (uses ERTConnect).

        Returns
        -------
        OMPLSolverResult
            Result containing the planned trajectory, timing information, and success status.

        Examples
        --------
        >>> solver = OMPLSolver()
        >>> problem = Problem(q_start, lb, ub, q_goal, collision_cst, None, resolution)
        >>> result = solver.solve(problem)
        >>> if result.success:
        ...     for q in result.traj.resample(50):
        ...         # Execute or visualize each waypoint
        ...         pass
        """

        ts = time.time()
        assert problem.global_eq_const is None, "not supported by OMPL"
        if isinstance(problem.goal_const, np.ndarray):
            q_goal = problem.goal_const
            goal_sampler = None
        elif self.config.use_goal_sampler:
            assert (
                self.config.algorithm.is_unidirectional()
            ), "goal sampler is used only for unidirectional planner"
            assert guess is None, "goal sampler is used only when guess is None"
            q_goal = None

            def goal_sampler():
                return self.solve_ik(problem).q

        else:
            ik_ret = self.solve_ik(problem, guess)
            if not ik_ret.success:
                return OMPLSolverResult(None, None, -1, TerminateState.FAIL_SATISFACTION)
            q_goal = ik_ret.q
            goal_sampler = None

        vconfig = ValidatorConfig()
        if problem.validator_type == "box":
            vconfig.type = ValidatorType.BOX
            vconfig.box_width = problem.resolution
        elif problem.validator_type == "euclidean":
            vconfig.type = ValidatorType.EUCLIDEAN
            vconfig.resolution = problem.resolution
        else:
            raise ValueError(
                f"Unknown validator type: {problem.motion_validation_resolution.validator_type}"
            )

        if guess is not None:
            planner = ERTConnectPlanner(
                problem.lb,
                problem.ub,
                problem.global_ineq_const,
                self.config.n_max_call,
                vconfig,
            )
            planner.set_heuristic(guess.numpy())
            planner.set_parameters(
                self.config.ertconnect_omega_min,
                self.config.ertconnect_omega_max,
                self.config.ertconnect_eps,
            )
        else:
            planner = OMPLPlanner(
                problem.lb,
                problem.ub,
                problem.global_ineq_const,
                self.config.n_max_call,
                vconfig,
                self.config.algorithm.value,
                self.config.algorithm_range,
            )
        timeout_remain = (
            None if (self.config.timeout is None) else self.config.timeout - (time.time() - ts)
        )
        assert timeout_remain is None or timeout_remain > 0
        result = planner.solve(
            problem.start,
            q_goal,
            self.config.refine_seq,
            timeout_remain,
            goal_sampler,
            self.config.max_goal_sampler_count,
        )
        if result is None:
            return OMPLSolverResult(None, None, -1, TerminateState.FAIL_PLANNING)
        else:
            n_call = planner.get_call_count()
            ns_internal = planner.get_ns_internal()
            return OMPLSolverResult(
                Trajectory(list(result)),
                time.time() - ts,
                n_call,
                TerminateState.SUCCESS,
                ns_internal,
            )


def simplify_path(
    traj: Trajectory,
    lb: np.ndarray,
    ub: np.ndarray,
    ineq_cst: IneqCompositeCst,
    resolution: Union[float, np.ndarray],  # see problem.Problem for definition
    validator_type: Literal["euclidean", "box"] = "box",  # see problem.Problem for definition
    n_max_call: int = 1000000,
    refine_seq: Sequence[RefineType] = (RefineType.SHORTCUT, RefineType.BSPLINE),
) -> Trajectory:
    """Simplify and optimize a robot trajectory.

    This function post-processes a planned trajectory to reduce path length,
    smooth the motion, and improve overall quality while maintaining feasibility.

    Parameters
    ----------
    traj : Trajectory
        Input trajectory to simplify.
    lb : np.ndarray
        Lower bounds for joint angles.
    ub : np.ndarray
        Upper bounds for joint angles.
    ineq_cst : IneqCompositeCst
        Inequality constraints for feasibility checking.
    resolution : Union[float, np.ndarray]
        Motion validation resolution (see Problem class for details).
    validator_type : {"euclidean", "box"}, default="box"
        Type of motion validator.
    n_max_call : int, default=1000000
        Maximum number of optimization iterations.
    refine_seq : Sequence[RefineType], default=(SHORTCUT, BSPLINE)
        Sequence of refinement operations to apply.

    Returns
    -------
    Trajectory
        Simplified and optimized trajectory.

    Examples
    --------
    >>> # Simplify a planned trajectory
    >>> simplified = simplify_path(
    ...     result.traj, lb, ub, collision_cst, resolution,
    ...     refine_seq=[RefineType.SHORTCUT, RefineType.BSPLINE]
    ... )
    """

    vconfig = ValidatorConfig()
    if validator_type == "box":
        vconfig.type = ValidatorType.BOX
        vconfig.box_width = resolution
    elif validator_type == "euclidean":
        vconfig.type = ValidatorType.EUCLIDEAN
        vconfig.resolution = resolution
    else:
        raise ValueError(f"Unknown validator type: {validator_type}")

    ret = simplify(traj.numpy(), refine_seq, lb, ub, ineq_cst, n_max_call, vconfig)
    return Trajectory(list(ret))
