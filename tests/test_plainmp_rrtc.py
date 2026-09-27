import numpy as np
import pytest
from test_ompl_solver import create_test_problem

from plainmp.ompl_solver import (
    Algorithm,
    OMPLPlanner,
    OMPLSolver,
    OMPLSolverConfig,
    PlainmpRRTCPlanner,
    PlainmpRRTCSettings,
    RefineType,
    ValidatorConfig,
    ValidatorType,
)


def check_path(path, problem, goal, resolution=0.01):
    np.testing.assert_array_equal(path[0], problem.start)
    np.testing.assert_array_equal(path[-1], goal)
    assert np.all(path >= problem.lb)
    assert np.all(path <= problem.ub)
    # Independently recheck interiors much more densely than the planner.
    for a, b in zip(path[:-1], path[1:]):
        count = max(1, int(np.ceil(np.linalg.norm(b - a) / resolution)))
        for t in np.linspace(0, 1, count + 1):
            assert problem.global_ineq_const.is_valid(a + t * (b - a))


@pytest.fixture
def problem():
    problem = create_test_problem(False)
    # Dense auditing needs a finer validation grid than the coarse example.
    problem.resolution = np.full(len(problem.start), 0.005)
    return problem


def planner_for(problem, capacity=100000, budget=1000000, euclidean=False, step_range=2.0):
    validator = ValidatorConfig()
    validator.type = ValidatorType.EUCLIDEAN if euclidean else ValidatorType.BOX
    validator.resolution = 0.005
    validator.box_width = problem.resolution
    settings = PlainmpRRTCSettings()
    settings.max_samples = capacity
    return PlainmpRRTCPlanner(
        problem.lb, problem.ub, problem.global_ineq_const, budget, validator, step_range, settings
    )


@pytest.mark.parametrize("euclidean", [False, True])
@pytest.mark.parametrize("step_range", [0.5, 2.0, None])
def test_obstacle_and_reuse(problem, euclidean, step_range):
    planner = planner_for(problem, euclidean=euclidean, step_range=step_range)
    for _ in range(3):
        path = planner.solve(problem.start, problem.goal_const, [], 2.0)
        assert path is not None
        assert planner.get_ns_internal() > 0
        assert planner.get_node_count() >= len(path)
        check_path(path, problem, problem.goal_const)
        if step_range is not None:
            assert np.all(np.linalg.norm(np.diff(path, axis=0), axis=1) <= step_range + 1e-12)
    # A failed query must not expose a previous result and must reset its pools.
    assert planner.solve(problem.start, problem.goal_const, [], 0.0) is None
    assert planner.get_node_count() == 0
    assert planner.get_call_count() == 0
    assert planner.solve(problem.start, problem.goal_const, [], 2.0) is not None


def test_identical_endpoints(problem):
    planner = planner_for(problem)
    path = planner.solve(problem.start, problem.start, [], 2.0)
    assert path is not None
    check_path(path, problem, problem.start)


@pytest.mark.parametrize("capacity", [2, 3, 4])
def test_capacity(problem, capacity):
    planner = planner_for(problem, capacity=capacity)
    assert planner.solve(problem.start, problem.goal_const, [], 1.0) is None
    assert planner.get_node_count() <= capacity


@pytest.mark.parametrize("budget", [0, 1, 2, 5, 20])
def test_call_limit(problem, budget):
    planner = planner_for(problem, budget=budget)
    assert planner.solve(problem.start, problem.goal_const, [], 1.0) is None
    assert planner.get_call_count() <= budget


@pytest.mark.parametrize("step_range", [0.0, -1.0, np.nan, np.inf])
def test_invalid_range(problem, step_range):
    with pytest.raises(ValueError):
        planner_for(problem, step_range=step_range)


@pytest.mark.parametrize("capacity", [0, 1, 2**32 - 1])
def test_invalid_capacity(problem, capacity):
    with pytest.raises(ValueError):
        planner_for(problem, capacity=capacity)


def test_invalid_inputs(problem):
    planner = planner_for(problem)
    with pytest.raises(ValueError):
        planner.solve(problem.start[:-1], problem.goal_const, [])
    bad = problem.start.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError):
        planner.solve(bad, problem.goal_const, [])
    bad[0] = problem.ub[0] + 1
    assert planner.solve(bad, problem.goal_const, []) is None
    assert planner.solve(problem.start, bad, []) is None
    for timeout in [-1, np.nan, np.inf]:
        with pytest.raises(ValueError):
            planner.solve(problem.start, problem.goal_const, [], timeout)
    with pytest.raises(ValueError):
        planner.solve(problem.start, None, [])


@pytest.mark.parametrize("pose_goal", [False, True])
@pytest.mark.parametrize("refine", [(), (RefineType.SHORTCUT,), (RefineType.BSPLINE,)])
def test_solver_interface(pose_goal, refine):
    problem = create_test_problem(pose_goal)
    problem.resolution = np.full(len(problem.start), 0.005)
    config = OMPLSolverConfig(algorithm=Algorithm.PlainmpRRTC, refine_seq=refine)
    result = OMPLSolver(config).solve(problem)
    assert result.success
    assert result.n_call > 0 and result.ns_internal > 0
    path = result.traj.numpy()
    check_path(path, problem, path[-1] if pose_goal else problem.goal_const)


def test_standard_ompl_registration(problem):
    validator = ValidatorConfig()
    validator.type = ValidatorType.BOX
    validator.box_width = problem.resolution
    planner = OMPLPlanner(
        problem.lb, problem.ub, problem.global_ineq_const, 100000, validator, "plainmp_rrtc", 2.0
    )
    path = planner.solve(problem.start, problem.goal_const, [], 2.0)
    assert path is not None
    check_path(path, problem, problem.goal_const)
    assert OMPLSolverConfig().algorithm == Algorithm.PlainmpRRTC
    assert OMPLSolverConfig().algorithm_range == 2.0
