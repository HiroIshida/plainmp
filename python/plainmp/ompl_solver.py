import time
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, TypeVar

import numpy as np
from ompl import Algorithm, ConstStateType, ERTConnectPlanner, Planner, RepairPlanner

from plainmp.ik import IKResult, solve_ik
from plainmp.problem import Problem
from plainmp.trajectory import Trajectory


@dataclass
class OMPLSolverConfig:
    n_max_call: int = 100000
    n_max_satisfaction_trial: int = 100
    algorithm: Algorithm = Algorithm.RRTConnect
    algorithm_range: Optional[float] = None
    simplify: bool = False
    expbased_planner_backend: Literal["ertconnect", "lightning"] = "lightning"
    ertconnect_eps: float = 5.0  # used only when ertconnect is selected
    const_state_type: ConstStateType = ConstStateType.PROJECTION
    timeout: Optional[float] = None


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
            assert self.config.n_max_satisfaction_trial == 1, "not supported"
            # If guess is provided, use the last element of the trajectory as the initial guess
            q_guess = guess.numpy()[-1]
            ret = solve_ik(
                problem.goal_const, problem.global_ineq_const, problem.lb, problem.ub, q_guess
            )
            return ret
        else:
            for _ in range(self.config.n_max_satisfaction_trial):
                ret = solve_ik(
                    problem.goal_const, problem.global_ineq_const, problem.lb, problem.ub
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

        n_count = [0]

        def is_valid(q: np.ndarray) -> bool:
            n_count[0] += 1
            if problem.global_ineq_const is None:
                return True
            return problem.global_ineq_const.is_valid(q)

        if guess is not None:
            if self.config.expbased_planner_backend == "ertconnect":
                planner_t = ERTConnectPlanner
            elif self.config.expbased_planner_backend == "lightning":
                planner_t = RepairPlanner  # type: ignore
            else:
                assert False
        else:
            planner_t = Planner

        planner = planner_t(
            problem.lb,
            problem.ub,
            is_valid,
            self.config.n_max_call,
            problem.motion_step_box,
            self.config.algorithm,
            self.config.algorithm_range,
        )

        if guess is not None:
            assert not isinstance(planner, Planner)
            planner.set_heuristic(guess.numpy())

        result = planner.solve(problem.start, q_goal, self.config.simplify)

        if result is None:
            return OMPLSolverResult(None, None, -1, TerminateState.FAIL_PLANNING)
        else:
            return OMPLSolverResult(
                Trajectory(result), time.time() - ts, n_count[0], TerminateState.SUCCESS
            )
