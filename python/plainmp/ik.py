import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, minimize

from plainmp.constraint import EqConstraintBase, IneqConstraintBase


def scipinize(fun: Callable) -> Tuple[Callable, Callable]:
    closure_member = {"jac_cache": None}

    def fun_scipinized(x):
        f, jac = fun(x)
        closure_member["jac_cache"] = jac
        return f

    def fun_scipinized_jac(x):
        return closure_member["jac_cache"]

    return fun_scipinized, fun_scipinized_jac


@dataclass
class IKConfig:
    ftol: float = 1e-6
    disp: bool = False
    n_max_eval: int = 200
    acceptable_error: float = 1e-6


@dataclass
class IKResult:
    q: np.ndarray
    elapsed_time: float
    success: bool
    n_trial: int


def solve_ik(
    eq_const: EqConstraintBase,
    ineq_const: Optional[IneqConstraintBase],
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    q_seed: Optional[np.ndarray] = None,
    config: Optional[IKConfig] = None,
    max_trial: int = 100,
) -> IKResult:
    ts = time.time()
    if config is None:
        config = IKConfig()

    def objective_fun(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vals, jac = eq_const.evaluate(q)
        f = vals.dot(vals)
        grad = 2 * vals.dot(jac)
        return f, grad

    f, jac = scipinize(objective_fun)

    # define constraint
    constraints = []
    if ineq_const is not None:

        def fun_ineq(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            val, jac = ineq_const.evaluate(q)
            margin_numerical = 1e-6
            return val - margin_numerical, jac

        ineq_const_scipy, ineq_const_jac_scipy = scipinize(fun_ineq)
        ineq_dict = {"type": "ineq", "fun": ineq_const_scipy, "jac": ineq_const_jac_scipy}
        constraints.append(ineq_dict)

    bounds = Bounds(lb, ub, keep_feasible=True)  # type: ignore

    if q_seed is None:
        q_seed = np.random.uniform(lb, ub)

    slsqp_option: Dict = {
        "ftol": config.ftol,
        "disp": config.disp,
        "maxiter": config.n_max_eval - 1,  # somehome scipy iterate +1 more time
    }

    for i in range(max_trial):
        res = minimize(
            f,
            q_seed,
            method="SLSQP",
            jac=jac,
            bounds=bounds,
            constraints=constraints,
            options=slsqp_option,
        )

        # check additional success condition
        if eq_const is not None:
            is_ik_actually_solved = res.fun < config.acceptable_error  # ensure not in local optima
            if is_ik_actually_solved:
                return IKResult(res.x, time.time() - ts, res.success, i + 1)
        # if not solved, try again without any initial guess
        q_seed = np.random.uniform(lb, ub)
    return IKResult(np.empty([0]), time.time() - ts, False, max_trial)
