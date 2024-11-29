import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from plainmp.ik import IKConfig, IKResult, solve_ik
from plainmp.manifold_rrt._manifold_rrt_algorithms import (
    InvalidStartPosition,
    ManifoldRRTConfig,
    ManifoldRRTConnect,
)
from plainmp.problem import Problem
from plainmp.trajectory import Trajectory


@dataclass
class ManiRRTConfig:
    n_max_call: int
    n_max_satisfaction_trial: int = 100
    ik_conf: Optional[IKConfig] = IKConfig()
    n_subgoal: int = 4  # used only when init_traj is specified
    timeout: Optional[float] = None

    @property
    def sample_goal_first(self) -> bool:
        return self.ik_conf is not None


class TerminateState(Enum):
    SUCCESS = 1
    FAIL_SATISFACTION = 2
    FAIL_PLANNING = 3


@dataclass
class MyRRTResult:
    traj: Optional[Trajectory]
    time_elapsed: Optional[float]
    n_call: int
    terminate_state: TerminateState

    @classmethod
    def abnormal(cls) -> "MyRRTResult":
        return cls(None, None, -1, TerminateState.FAIL_SATISFACTION)


class ManiRRTConnectSolver:
    def __init__(self, config: ManiRRTConfig):
        self.config = config

    def solve(self, problem: Problem, guess: Optional[Trajectory] = None) -> MyRRTResult:
        """solve problem with maybe a solution guess"""
        ts = time.time()
        assert guess is None, "don't support replanning"

        assert self.config.sample_goal_first, "goal must be sampled before in rrt-connect"
        if isinstance(problem.goal_const, np.ndarray):
            q_goal = problem.goal_const
        else:
            ik_result: IKResult
            for _ in range(self.config.n_max_satisfaction_trial):
                ik_result = solve_ik(
                    problem.goal_const,
                    problem.global_ineq_const,
                    problem.lb,
                    problem.ub,
                )
                if ik_result.success:
                    break
            if not ik_result.success:
                return MyRRTResult.abnormal()
            q_goal = ik_result.q

        conf = ManifoldRRTConfig(self.config.n_max_call)

        def project(q: np.ndarray, collision_aware: bool = False) -> Optional[np.ndarray]:
            assert problem is not None

            if problem.global_eq_const is None:
                return q
            else:
                ineq_const = problem.global_ineq_const if collision_aware else None
                res = solve_ik(
                    problem.global_eq_const, ineq_const, problem.lb, problem.ub, q_seed=q
                )
                if res.success:
                    return res.q
                else:
                    return None

        def is_valid(q: np.ndarray) -> bool:
            assert problem is not None
            if problem.global_ineq_const is None:
                return True
            return problem.global_ineq_const.is_valid(q)

        assert problem.validator_type == "box", "currenlty only box validator is supported"
        rrtconnect = ManifoldRRTConnect(
            problem.start,
            q_goal,  # type: ignore
            problem.lb,
            problem.ub,
            problem.resolution,
            project,
            is_valid,
            config=conf,
        )
        try:
            is_success = rrtconnect.solve()
        except InvalidStartPosition:
            return MyRRTResult.abnormal()

        if is_success:
            traj = Trajectory(list(rrtconnect.get_solution()))
            return MyRRTResult(
                traj, time.time() - ts, rrtconnect.n_extension_trial, TerminateState.SUCCESS
            )
        else:
            return MyRRTResult(
                None, time.time() - ts, self.config.n_max_call, TerminateState.FAIL_PLANNING
            )
