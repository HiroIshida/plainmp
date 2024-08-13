import copy
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, Tuple, Union

import numpy as np
import osqp
import scipy.sparse as sparse
from scipy.sparse import csc_matrix


class Differentiable(Protocol):
    def __call__(self, __x: np.ndarray) -> Tuple[np.ndarray, Union[np.ndarray, csc_matrix]]:
        """return value and jacobian (or grad)"""
        ...


def sparsify(m: Union[np.ndarray, csc_matrix]) -> csc_matrix:
    if isinstance(m, csc_matrix):
        return m
    return csc_matrix(m)


@dataclass
class OsqpSqpConfig:
    ftol: float = 1e-3
    ctol_eq: float = 1e-4  # constraint tolerance
    ctol_ineq: float = 1e-3  # constraint tolerance
    n_max_eval: int = 10
    maxrelax: int = 10
    trust_box_init_size: float = 0.5
    osqp_verbose: bool = False
    verbose: bool = False
    relax_step_convex: float = 0.1
    force_deterministic: bool = False
    step_box: Optional[np.ndarray] = None
    step_size_init: float = 1.0
    step_size_step: float = 0.0


class OsqpSqpExitMode(Enum):
    SOLVED = 0
    REACH_LIMIT = 1
    OSQP_FAIL = 2
    UNKNOWN = 3


@dataclass
class OsqpSqpResult:
    x: np.ndarray
    nit: int
    fobj: float
    feq: np.ndarray
    fineq: np.ndarray
    success: bool = False
    status: OsqpSqpExitMode = OsqpSqpExitMode.UNKNOWN


@dataclass
class OsqpSqpSolver:
    P: Union[np.ndarray, csc_matrix]
    cons_eq: Differentiable
    cons_ineq: Differentiable
    lb: np.ndarray
    ub: np.ndarray

    @dataclass
    class EvaluateCache:
        val_eq: np.ndarray
        J_eq: csc_matrix
        Jx_eq: np.ndarray
        val_ineq: np.ndarray
        J_ineq: csc_matrix
        Jx_ineq: np.ndarray
        val_obj: Optional[float] = None
        grad_obj: Optional[np.ndarray] = None

    def __post_init__(self):
        self.P = sparsify(self.P)

    def solve_convex_subproblem(
        self,
        x_guess: np.ndarray,
        eval_cache: EvaluateCache,
        relax_step: float,
        max_relax: int,
        verbose: bool,
        osqp_verbose: bool,
        force_deterministic: bool = False,
    ) -> Optional[np.ndarray]:

        if force_deterministic:
            # see here:
            # https://osqp.discourse.group/t/what-settings-to-choose-to-ensure-reproducibility/64/4
            # but I'm not sure how this parameter affects the performance
            # so, this option is supposed to be used in test and debugging
            osqp_kwargs = {"adaptive_rho_interval": 25}
        else:
            osqp_kwargs = {}

        success_status = osqp.constant("OSQP_SOLVED")
        dim_codomain_ineq = len(eval_cache.val_ineq)
        dim_x = eval_cache.J_eq.shape[1]
        E = sparse.eye(dim_x)
        A = sparse.vstack((eval_cache.J_eq, eval_cache.J_ineq, E))
        # inequality gradual relaxing
        # in some problem, convexified problem is infeasible and cannot be solved by osqp.
        counter = 0
        copy.deepcopy(eval_cache.val_ineq)
        prob: Optional[osqp.OSQP] = None

        while True:
            relax_step_now = relax_step * counter

            l = np.hstack(
                (
                    eval_cache.Jx_eq - eval_cache.val_eq,
                    eval_cache.Jx_ineq - eval_cache.val_ineq,
                    self.lb,
                )
            )
            u = np.hstack(
                (eval_cache.Jx_eq - eval_cache.val_eq, np.inf * np.ones(dim_codomain_ineq), self.ub)
            )
            # osqp fails with primal-infeasible error some-times
            # with slight break of inequality constraint
            # TODO(HiroIshida): this should be implemented in higher layer?
            l -= 1e-7
            u += 1e-7

            if prob is None:
                prob = osqp.OSQP()
                prob.setup(
                    P=self.P,
                    q=None,
                    l=l,
                    u=u,
                    A=A,
                    warm_start=True,
                    verbose=osqp_verbose,
                    **osqp_kwargs,
                )
            else:
                prob.update(l=l - relax_step_now, u=u + relax_step_now)
            prob.warm_start(x=x_guess)
            raw_result = prob.solve()
            if raw_result.info.status_val == success_status:
                break

            counter += 1
            if counter > max_relax:
                return None
            if verbose:
                print("relax inequality!")

        x_guess = raw_result.x
        return x_guess

    def solve(
        self,
        x_guess: np.ndarray,
        config: OsqpSqpConfig = OsqpSqpConfig(),
    ) -> OsqpSqpResult:

        val_objective_previous = np.inf
        step_size = config.step_size_init

        result: Optional[OsqpSqpResult] = None
        for idx_iter in range(config.n_max_eval):
            if config.verbose:
                print("iteration num: {}".format(idx_iter))
            # equality
            val_eq, J_eq = self.cons_eq(x_guess)
            J_eq = sparsify(J_eq)
            Jx_eq = J_eq.dot(x_guess)

            # inequality
            val_ineq, J_ineq = self.cons_ineq(x_guess)
            J_ineq = sparsify(J_ineq)
            Jx_ineq = J_ineq.dot(x_guess)

            eval_cache = self.EvaluateCache(val_eq, J_eq, Jx_eq, val_ineq, J_ineq, Jx_ineq)

            # check break condition satisfied
            eq_const_satisfied = np.all(np.abs(val_eq) < config.ctol_eq)
            ineq_const_satisfied = np.all(val_ineq > -config.ctol_ineq)
            objective = self.P.dot(x_guess).dot(x_guess)
            if config.verbose:
                print("iter: {}".format(idx_iter))

                def to_exp_notation(val: float) -> str:
                    return np.format_float_scientific(val, precision=3, exp_digits=2)

                print(
                    "obj: {}, eq satisfied?: {}, ineq satisfied?: {}".format(
                        to_exp_notation(objective), eq_const_satisfied, ineq_const_satisfied
                    )
                )
                n_eq_violation = sum(np.abs(val_eq) > config.ctol_eq)
                eq_max_vilation = np.max(np.abs(val_eq))
                n_ineq_violation = sum(val_ineq < -config.ctol_ineq)
                ineq_max_violation = np.min(val_ineq)
                print(
                    "eq max violation: {}, ineq max violation: {}".format(
                        to_exp_notation(eq_max_vilation), to_exp_notation(ineq_max_violation)
                    )
                )
                print(
                    "eq num violation: {}, ineq num violation: {}".format(
                        n_eq_violation, n_ineq_violation
                    )
                )

            result = OsqpSqpResult(x_guess, idx_iter, objective, val_eq, val_ineq)

            if eq_const_satisfied and ineq_const_satisfied:
                diff_objective = val_objective_previous - objective
                if abs(diff_objective) < config.ftol:
                    result.success = True
                    result.status = OsqpSqpExitMode.SOLVED
                    return result

            val_objective_previous = objective

            if idx_iter == 0:
                subproblem_result = self.solve_convex_subproblem(
                    x_guess,
                    eval_cache,
                    config.relax_step_convex,
                    config.maxrelax,
                    config.verbose,
                    config.osqp_verbose,
                    config.force_deterministic,
                )
            else:
                subproblem_result = self.solve_convex_subproblem(
                    x_guess,
                    eval_cache,
                    config.relax_step_convex,
                    0,
                    config.verbose,
                    config.osqp_verbose,
                    config.force_deterministic,
                )

            if subproblem_result is None:
                result.success = False
                result.status = OsqpSqpExitMode.OSQP_FAIL
                return result

            if config.step_box is not None:
                # ignore step size
                lb_clamp = x_guess - config.step_box
                ub_clamp = x_guess + config.step_box
                x_guess = np.clip(subproblem_result, lb_clamp, ub_clamp)
            else:
                x_guess = x_guess + step_size * (subproblem_result - x_guess)
            step_size = min(1.0, step_size + config.step_size_step)

        assert result is not None
        result.success = False
        result.status = OsqpSqpExitMode.REACH_LIMIT
        return result
