import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, TypeVar

import numpy as np

from plainmp.ik import IKResult, solve_ik
from plainmp.problem import Problem
from plainmp.trajectory import Trajectory

from ._plainmp.ompl import ERTConnectPlanner, OMPLPlanner


class Algorithm(Enum):
    BKPIECE1 = "BKPIECE1"
    KPIECE1 = "KPIECE1"
    LBKPIECE1 = "LBKPIECE1"
    RRTConnect = "RRTConnect"
    RRT = "RRT"
    RRTstar = "RRTstar"


@dataclass
class OMPLSolverConfig:
    n_max_call: int = 100000
    n_max_ik_trial: int = 100
    algorithm: Algorithm = Algorithm.RRTConnect
    algorithm_range: Optional[float] = 2.0
    simplify: bool = False
    ertconnect_eps: float = 5.0  # used only when ertconnect is selected


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


OMPLSolverT = TypeVar("OMPLSolverT", bound="OMPLSolver")


class OMPLSolver:
    config: OMPLSolverConfig

    def __init__(self, config: Optional[OMPLSolverConfig] = None):
        if config is None:
            config = OMPLSolverConfig()
        self.config = config

    def solve_ik(self, problem: Problem, guess: Optional[Trajectory] = None) -> IKResult:
        if guess is not None:
            assert (
                self.config.n_max_ik_trial == 1
            ), "not supported. please configure n_max_ik_trial=1"
            # If guess is provided, use the last element of the trajectory as the initial guess
            q_guess = guess.numpy()[-1]
            ret = solve_ik(
                problem.goal_const,
                problem.global_ineq_const,
                problem.lb,
                problem.ub,
                q_seed=q_guess,
            )
            return ret
        else:
            ret = solve_ik(
                problem.goal_const,
                problem.global_ineq_const,
                problem.lb,
                problem.ub,
                max_trial=self.config.n_max_ik_trial,
            )
            if ret.success:
                return ret
            return ret  # type: ignore

    def solve(self, problem: Problem, guess: Optional[Trajectory] = None) -> OMPLSolverResult:
        ts = time.time()
        assert problem.global_eq_const is None, "not supported by OMPL"
        if isinstance(problem.goal_const, np.ndarray):
            q_goal = problem.goal_const
        else:
            ik_ret = self.solve_ik(problem, guess)
            if not ik_ret.success:
                return OMPLSolverResult(None, None, -1, TerminateState.FAIL_SATISFACTION)
            q_goal = ik_ret.q

        if guess is not None:
            planner = ERTConnectPlanner(
                problem.lb,
                problem.ub,
                problem.global_ineq_const,
                self.config.n_max_call,
                problem.motion_step_box,
            )
            planner.set_heuristic(guess.numpy())
        else:
            planner = OMPLPlanner(
                problem.lb,
                problem.ub,
                problem.global_ineq_const,
                self.config.n_max_call,
                problem.motion_step_box,
                self.config.algorithm.value,
                self.config.algorithm_range,
            )

        result = planner.solve(problem.start, q_goal, self.config.simplify)
        if result is None:
            return OMPLSolverResult(None, None, -1, TerminateState.FAIL_PLANNING)
        else:
            for i in range(len(result)):
                result[i] = np.array(result[i])
            n_call = planner.get_call_count()
            return OMPLSolverResult(
                Trajectory(result), time.time() - ts, n_call, TerminateState.SUCCESS
            )
