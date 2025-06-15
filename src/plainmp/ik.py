import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, minimize

from plainmp.constraint import EqConstraintBase, IneqConstraintBase, LinkPoseCst


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
    timeout: Optional[float] = 10.0


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
    """Solve inverse kinematics problem using nonlinear optimization.

    This function solves IK problems by formulating them as constrained optimization
    problems and using sequential least squares programming (SLSQP) solver.

    Parameters
    ----------
    eq_const : EqConstraintBase
        Equality constraint representing the desired pose/configuration.
    ineq_const : IneqConstraintBase, optional
        Inequality constraints (e.g., collision avoidance constraints).
    lb : np.ndarray
        Lower bounds for joint angles.
    ub : np.ndarray
        Upper bounds for joint angles.
    q_seed : np.ndarray, optional
        Initial guess for joint angles. If None, random values within bounds are used.
    config : IKConfig, optional
        Configuration parameters for the IK solver.
    max_trial : int, default=100
        Maximum number of random restarts to attempt if solution fails.

    Returns
    -------
    IKResult
        Result containing solution joint angles, elapsed time, success status, and trial count.

    Examples
    --------
    >>> fs = FetchSpec()
    >>> eq_cst = fs.create_gripper_pose_const([0.7, 0.2, 0.95, 0, 0, 0])
    >>> ineq_cst = fs.create_collision_const()
    >>> lb, ub = fs.angle_bounds()
    >>> result = solve_ik(eq_cst, ineq_cst, lb, ub)
    >>> if result.success:
    ...     print(f"Solution found: {result.q}")
    """
    ts = time.time()

    if config is None:
        config = IKConfig()

    def objective_fun(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vals, jac = eq_const.evaluate(q)
        f = vals.dot(vals)
        grad = 2 * vals.dot(jac)
        elapsed_time = time.time() - ts
        if config.timeout is not None and elapsed_time > config.timeout:
            raise TimeoutError("IK solver timeout")
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

    try:
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

            # the following is to ignore local minima
            solved = True
            if eq_const is not None:
                if res.fun > config.acceptable_error:
                    solved = False
            if ineq_const is not None:
                if not ineq_const.is_valid(res.x):
                    solved = False
            if solved:
                return IKResult(res.x, time.time() - ts, res.success, i + 1)
            q_seed = np.random.uniform(lb, ub)
    except TimeoutError:
        return IKResult(np.empty([0]), time.time() - ts, False, i + 1)
    return IKResult(np.empty([0]), time.time() - ts, False, max_trial)


def solve_ik_srinv_experimental(
    link_pose_cst: LinkPoseCst,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    q_seed: Optional[np.ndarray] = None,
    config: Optional[IKConfig] = None,
    max_trial: int = 100,
) -> IKResult:
    """Solve inverse kinematics using SR-inverse (damped least squares) method.

    This function uses the singular-robust inverse (SR-inverse) to solve IK problems.

    Parameters
    ----------
    link_pose_cst : LinkPoseCst
        Link pose constraint specifying target poses for robot links.
    lb : np.ndarray
        Lower bounds for joint angles.
    ub : np.ndarray
        Upper bounds for joint angles.
    q_seed : np.ndarray, optional
        Initial guess for joint angles. If None, random values within bounds are used.
    config : IKConfig, optional
        Configuration parameters for the IK solver.
    max_trial : int, default=100
        Maximum number of random restarts to attempt if solution fails.

    Returns
    -------
    IKResult
        Result containing solution joint angles, elapsed time, success status, and trial count.
    """
    ts = time.time()

    if config is None:
        config = IKConfig()
        # SQP includes line search, but we do not use it here
        # so we can reduce the number of evaluations
        config.n_max_eval = 50

    if q_seed is None:
        q_seed = np.random.uniform(lb, ub)

    for trial in range(max_trial):
        q = q_seed.copy()

        for iteration in range(config.n_max_eval):
            elapsed_time = time.time() - ts
            if config.timeout is not None and elapsed_time > config.timeout:
                return IKResult(np.empty([0]), elapsed_time, False, trial + 1)

            vals, jac = link_pose_cst.evaluate(q)

            error = np.linalg.norm(vals)
            if error < config.acceptable_error:
                return IKResult(q, time.time() - ts, True, trial + 1)

            # Compute SR-inverse (damped least squares)
            damping = 1e-6 + error * 1e-3  # Adaptive damping
            A = jac @ jac.T + damping * np.eye(jac.shape[0])
            try:
                dq = jac.T @ np.linalg.solve(A, vals)
            except np.linalg.LinAlgError:
                damping = 1e-3
                A = jac @ jac.T + damping * np.eye(jac.shape[0])
                dq = jac.T @ np.linalg.solve(A, vals)

            # Update joint angles with step size control
            step_size = 0.5
            q_new = q - step_size * dq

            # Apply joint limits
            q_new = np.clip(q_new, lb, ub)

            # Check for convergence in joint space
            if np.linalg.norm(q_new - q) < config.ftol:
                q = q_new
                break

            q = q_new

        # Try with new random seed if failed
        q_seed = np.random.uniform(lb, ub)

    return IKResult(np.empty([0]), time.time() - ts, False, max_trial)
