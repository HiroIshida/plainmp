import copy
import time
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Union

import numpy as np
from scipy.sparse import csc_matrix

from plainmp.constraint import (
    ConstraintBase,
    EqCompositeCst,
    IneqCompositeCst,
    SequentialCst,
)
from plainmp.ik import IKResult, solve_ik
from plainmp.nlp_solver.osqp_sqp import (
    OsqpSqpConfig,
    OsqpSqpResult,
    OsqpSqpSolver,
    sparsify,
)
from plainmp.problem import Problem
from plainmp.trajectory import Trajectory


def add_at_workaround(
    seq_cst: SequentialCst, cst: Union[IneqCompositeCst, EqCompositeCst, ConstraintBase], idx: int
) -> None:
    # TODO: seq_cst should accept composite constraint
    if isinstance(cst, (IneqCompositeCst, EqCompositeCst)):
        for sub_cst in cst.constraints:
            seq_cst.add_at(sub_cst, idx)
    else:
        seq_cst.add_at(cst, idx)


def add_globally_workaround(
    seq_cst: SequentialCst, cst: Union[IneqCompositeCst, EqCompositeCst, ConstraintBase]
) -> None:
    # TODO: seq_cst should accept composite constraint
    if isinstance(cst, (IneqCompositeCst, EqCompositeCst)):
        for sub_cst in cst.constraints:
            seq_cst.add_globally(sub_cst)
    else:
        seq_cst.add_globally(cst)


def translate(problem: Problem, n_wp: int) -> Tuple[SequentialCst, SequentialCst]:
    n_dof = len(problem.start)
    # equality
    seq_eq_const = SequentialCst(n_wp, n_dof)
    seq_eq_const.add_fixed_point_at(problem.start, 0)
    if isinstance(problem.goal_const, np.ndarray):
        seq_eq_const.add_fixed_point_at(problem.goal_const, n_wp - 1)
    else:
        # seq_eq_const.add_at(problem.goal_const, n_wp - 1)
        add_at_workaround(seq_eq_const, problem.goal_const, n_wp - 1)
    if problem.global_eq_const is not None:
        # seq_eq_const.add_globally(problem.global_eq_const)
        add_globally_workaround(seq_eq_const, problem.global_eq_const)
    seq_eq_const.finalize()

    # inequality
    seq_ineq_const = SequentialCst(n_wp, n_dof)
    if problem.global_ineq_const is not None:
        # seq_ineq_const.add_globally(problem.global_ineq_const)
        add_globally_workaround(seq_ineq_const, problem.global_ineq_const)
    assert problem.validator_type == "box", "currently only box validator is supported"
    seq_ineq_const.add_motion_step_box_constraint(problem.resolution)
    seq_ineq_const.finalize()
    return seq_eq_const, seq_ineq_const


def smoothcost_fullmat(n_dof: int, n_wp: int, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute A of eq. (17) of IJRR-version (2013) of CHOMP"""

    def construct_smoothcost_mat(n_wp):
        # In CHOMP (2013), squared sum of velocity is computed.
        # In this implementation we compute squared sum of acceralation
        # if you set acc_block * 0.0, vel_block * 1.0, then the trajectory
        # cost is same as the CHOMP one.
        acc_block = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
        vel_block = np.array([[1, -1], [-1, 1]])
        A_ = np.zeros((n_wp, n_wp))
        for i in [1 + i for i in range(n_wp - 2)]:
            A_[i - 1 : i + 2, i - 1 : i + 2] += acc_block * 1.0
            A_[i - 1 : i + 1, i - 1 : i + 1] += vel_block * 0.0  # do nothing
        return A_

    if weights is None:
        weights = np.ones(n_dof)

    w_mat = np.diag(weights)
    A_ = construct_smoothcost_mat(n_wp)
    A = np.kron(A_, w_mat**2)
    return A


@dataclass
class SQPBasedSolverConfig:
    """
    motion_step_satisfaction: either of "implicit", "explicit", "post"

    NOTE: choice motion_step_satisfaction affects performance a lot.

    In general, the following inequality is observed.
    solvability: implicit > explicit >> post
    speed: post >> explicit > implicit
    when you choose good n_wp, the solvability order will be
    solvability: explicit > post ~ implicit
    """

    n_wp: int
    n_dof: int
    n_max_call: int = 30
    motion_step_satisfaction: Literal["implicit", "explicit", "post", "debug_ignore"] = "implicit"
    force_deterministic: bool = False
    osqp_verbose: bool = False
    verbose: bool = False
    n_max_satisfaction_trial: int = 100  # only used if init traj is not satisfied
    ctol_eq: float = 1e-4
    ctol_ineq: float = 1e-3
    ineq_tighten_coef: float = (
        2.0  # NOTE: in some large problem like humanoid planning, this value should be zero
    )
    step_box: Optional[np.ndarray] = None
    _osqpsqp_config: OsqpSqpConfig = field(
        default_factory=OsqpSqpConfig
    )  # don't directly access this
    timeout: Optional[float] = None
    return_osqp_result: bool = False  # helpful for debugging but memory footprint is large
    step_size_init: float = 1.0
    step_size_step: float = 0.0

    @property
    def osqpsqp_config(self) -> OsqpSqpConfig:
        osqpsqp_config = copy.deepcopy(self._osqpsqp_config)
        osqpsqp_config.n_max_eval = self.n_max_call
        osqpsqp_config.force_deterministic = self.force_deterministic
        osqpsqp_config.verbose = self.verbose
        osqpsqp_config.osqp_verbose = self.osqp_verbose
        osqpsqp_config.ctol_eq = self.ctol_eq
        osqpsqp_config.ctol_ineq = self.ctol_ineq
        osqpsqp_config.step_size_init = self.step_size_init
        osqpsqp_config.step_size_step = self.step_size_step
        if self.step_box is not None:
            # self.step_box is for single waypont
            # thus we need to scale it to n_wp
            step_box_stacked = np.tile(self.step_box, self.n_wp)
            osqpsqp_config.step_box = step_box_stacked
        return osqpsqp_config


@dataclass
class SQPBasedSolverResult:
    traj: Optional[Trajectory]
    time_elapsed: Optional[float]
    n_call: int
    osqpsqp_raw_result: Optional[OsqpSqpResult]

    @classmethod
    def abnormal(cls) -> "SQPBasedSolverResult":
        return cls(None, None, -1, None)


class SQPBasedSolver:
    config: SQPBasedSolverConfig
    smooth_mat: csc_matrix

    def __init__(self, config: SQPBasedSolverConfig) -> None:
        smooth_mat = smoothcost_fullmat(config.n_dof, config.n_wp)
        self.smooth_mat = sparsify(smooth_mat)
        self.config = config

    def solve(self, problem: Problem, guess: Optional[Trajectory] = None) -> SQPBasedSolverResult:
        ts = time.time()
        config = self.config
        seq_eq_const, seq_ineq_const = translate(problem, config.n_wp)

        lb_stacked = np.tile(problem.lb, config.n_wp)
        ub_stacked = np.tile(problem.ub, config.n_wp)
        ctol_ineq = config.osqpsqp_config.ctol_ineq

        def ineq_tighten(x):
            # somehow, osqp-sqp result has some ineq error
            # thus to compensate that, we tighten the ineq constraint here
            f, jac = seq_ineq_const.evaluate(x)
            return f - ctol_ineq * config.ineq_tighten_coef, jac

        solver = OsqpSqpSolver(
            self.smooth_mat,
            lambda x: seq_eq_const.evaluate(x),
            ineq_tighten,
            lb_stacked,
            ub_stacked,
        )

        if guess is None:
            if isinstance(problem.goal_const, np.ndarray):
                q_goal = problem.goal_const
            else:
                ik_result: Optional[IKResult] = None
                for _ in range(self.config.n_max_satisfaction_trial):
                    ik_result = solve_ik(
                        problem.goal_const,
                        problem.global_ineq_const,
                        problem.lb,
                        problem.ub,
                        q_seed=None,
                    )
                    if ik_result.success:
                        break
                assert ik_result is not None
                if not ik_result.success:
                    return SQPBasedSolverResult(None, None, -1, None)

                q_goal = ik_result.q
            guess = Trajectory.from_two_points(problem.start, q_goal, self.config.n_wp)

        x_init = guess.resample(self.config.n_wp).numpy().flatten()
        raw_result = solver.solve(x_init, config=self.config.osqpsqp_config)

        success = raw_result.success

        traj_solution: Optional[Trajectory] = None
        if success:
            traj_solution = Trajectory(list(raw_result.x.reshape(self.config.n_wp, -1)))

        if self.config.return_osqp_result:
            return SQPBasedSolverResult(traj_solution, time.time() - ts, raw_result.nit, raw_result)
        else:
            return SQPBasedSolverResult(traj_solution, time.time() - ts, raw_result.nit, None)
