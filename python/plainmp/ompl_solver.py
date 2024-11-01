import signal
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, TypeVar

import numpy as np

from plainmp.ik import IKConfig, IKResult, solve_ik
from plainmp.problem import Problem
from plainmp.trajectory import Trajectory

from ._plainmp.ompl import (  # noqa: F401
    ERTConnectPlanner,
    OMPLPlanner,
    set_log_level_none,
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
    BITstar = "BITstar"
    BITstarStop = "BITstarStop"  # stop after first solution

    def is_unidirectional(self) -> bool:
        return self in [Algorithm.RRT, Algorithm.KPIECE1, Algorithm.LBKPIECE1]


@dataclass
class OMPLSolverConfig:
    n_max_call: int = 1000000
    n_max_ik_trial: int = 100
    algorithm: Algorithm = Algorithm.RRTConnect
    algorithm_range: Optional[float] = 2.0
    simplify: bool = False
    ertconnect_eps: float = 5.0  # used only when ertconnect is selected
    timeout: Optional[float] = None
    use_goal_sampler: bool = (
        False  # use goal sampler in unidirectional planner. Use only when the goal is not a point
    )
    max_goal_sampler_count: int = 100


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
        # IK is supposed to stop within the timeout but somehow it does not work well
        # so we set...
        config = IKConfig(timeout=self.config.timeout)

        if guess is not None:
            # If guess is provided, use the last element of the trajectory as the initial guess
            q_guess = guess.numpy()[-1]
            ret = solve_ik(
                problem.goal_const,
                problem.global_ineq_const,
                problem.lb,
                problem.ub,
                q_seed=q_guess,
                max_trial=self.config.n_max_ik_trial,
                config=config,
            )
            return ret
        else:
            ret = solve_ik(
                problem.goal_const,
                problem.global_ineq_const,
                problem.lb,
                problem.ub,
                max_trial=self.config.n_max_ik_trial,
                config=config,
            )
            if ret.success:
                return ret
            return ret  # type: ignore

    def solve(
        self, problem: Problem, guess: Optional[Trajectory] = None, bench: bool = False
    ) -> OMPLSolverResult:

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
            if self.config.timeout is not None:

                def handler(sig, frame):
                    raise TimeoutError

                signal.signal(signal.SIGALRM, handler)
                signal.setitimer(signal.ITIMER_REAL, self.config.timeout)
            try:
                ik_ret = self.solve_ik(problem, guess)
            except TimeoutError:
                return OMPLSolverResult(None, None, -1, TerminateState.FAIL_SATISFACTION)
            finally:
                if self.config.timeout is not None:
                    signal.setitimer(signal.ITIMER_REAL, 0)

            if not ik_ret.success:
                return OMPLSolverResult(None, None, -1, TerminateState.FAIL_SATISFACTION)
            q_goal = ik_ret.q
            goal_sampler = None

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

        if bench:
            ts = time.time()
            set_log_level_none()
            for _ in range(10000):
                planner.solve(problem.start, q_goal, self.config.simplify)
            print(f"ms per solve() = {(time.time() - ts) / 10000 * 1000:.3f} ms")

        timeout_remain = (
            None if (self.config.timeout is None) else self.config.timeout - (time.time() - ts)
        )
        result = planner.solve(
            problem.start,
            q_goal,
            self.config.simplify,
            timeout_remain,
            goal_sampler,
            self.config.max_goal_sampler_count,
        )
        if result is None:
            return OMPLSolverResult(None, None, -1, TerminateState.FAIL_PLANNING)
        else:
            for i in range(len(result)):
                result[i] = np.array(result[i])
            n_call = planner.get_call_count()
            return OMPLSolverResult(
                Trajectory(result), time.time() - ts, n_call, TerminateState.SUCCESS
            )
