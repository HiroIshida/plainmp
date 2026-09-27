import numpy as np
import pytest
from test_ompl_solver import create_test_problem

from plainmp.ompl_solver import (
    Algorithm,
    OMPLSolver,
    OMPLSolverConfig,
    RefineType,
    ValidatorConfig,
    ValidatorType,
    VampRRTCPlanner,
    VampRRTCSettings,
)


@pytest.fixture
def problem():
    return create_test_problem(False)


def planner_for(problem, settings=None, budget=100000, euclidean=False):
    validator = ValidatorConfig()
    validator.type = ValidatorType.EUCLIDEAN if euclidean else ValidatorType.BOX
    validator.resolution = 0.025
    validator.box_width = problem.resolution
    return VampRRTCPlanner(
        problem.lb,
        problem.ub,
        problem.global_ineq_const,
        budget,
        validator,
        2.0,
        settings,
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


@pytest.mark.parametrize("euclidean", [False, True])
@pytest.mark.parametrize(
    "start_first,dynamic_domain,balance",
    [(True, True, True), (False, True, True), (False, False, False)],
)
def test_obstacle_and_reuse(problem, euclidean, start_first, dynamic_domain, balance):
    settings = VampRRTCSettings()
    settings.start_tree_first = start_first
    settings.dynamic_domain = dynamic_domain
    settings.balance = balance
    planner = planner_for(problem, settings, euclidean=euclidean)
    for _ in range(3):
        path = planner.solve(problem.start, problem.goal_const, [], 2.0)
        assert path is not None
        assert planner.get_ns_internal() > 0
        check_path(path, problem, problem.goal_const)


def test_direct_and_identical(problem):
    planner = planner_for(problem)
    for delta in [0.0, 0.01]:
        goal = problem.start.copy()
        goal[1] += delta
        path = planner.solve(problem.start, goal, [])
        assert len(path) == 2
        check_path(path, problem, goal)


@pytest.mark.parametrize("capacity", [2, 3, 4, 128, 129])
def test_capacity_and_iterations(problem, capacity):
    settings = VampRRTCSettings()
    settings.max_samples = capacity
    settings.max_iterations = 0
    planner = planner_for(problem, settings)
    assert planner.solve(problem.start, problem.goal_const, []) is None
    # Pool limits must also apply when CONNECT would insert multiple nodes.
    settings.max_iterations = 10000
    settings.max_samples = min(capacity, 4)
    planner = planner_for(problem, settings)
    assert planner.solve(problem.start, problem.goal_const, [], 1.0) is None


@pytest.mark.parametrize("budget", [0, 1, 2, 5, 20])
def test_call_limit_and_timeout(problem, budget):
    planner = planner_for(problem, budget=budget)
    assert planner.solve(problem.start, problem.goal_const, [], 1.0) is None
    assert planner.get_call_count() <= budget
    planner = planner_for(problem)
    assert planner.solve(problem.start, problem.goal_const, [], 0.0) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_samples", 1),
        ("radius", 0),
        ("alpha", 1.0),
        ("alpha", float("nan")),
        ("tree_ratio", 0),
        ("min_radius", -1),
    ],
)
def test_invalid_settings(problem, field, value):
    settings = VampRRTCSettings()
    setattr(settings, field, value)
    with pytest.raises(ValueError):
        planner_for(problem, settings)


def test_invalid_endpoints(problem):
    planner = planner_for(problem)
    with pytest.raises(ValueError):
        planner.solve(problem.start[:-1], problem.goal_const, [])
    invalid = problem.start.copy()
    invalid[0] = np.nan
    with pytest.raises(ValueError):
        planner.solve(invalid, problem.goal_const, [])
    invalid[0] = problem.ub[0] + 1
    assert planner.solve(invalid, problem.goal_const, []) is None


def test_range_and_halton_reproducibility(problem):
    assert OMPLSolverConfig().algorithm_range == 2.0
    settings = VampRRTCSettings()
    settings.halton_skip = 1000
    planner = planner_for(problem, settings)
    first = planner.solve(problem.start, problem.goal_const, [])
    second = planner.solve(problem.start, problem.goal_const, [])
    assert first is not None
    np.testing.assert_array_equal(first, second)
    settings.use_halton = False
    uniform = planner_for(problem, settings)
    assert uniform.solve(problem.start, problem.goal_const, [], 2.0) is not None


@pytest.mark.parametrize("euclidean", [False, True])
def test_linear_matches_kdtree(problem, euclidean):
    for skip in [0, 1000, 10000]:
        settings = VampRRTCSettings()
        settings.halton_skip = skip
        kd = planner_for(problem, settings, euclidean=euclidean)
        settings.use_kdtree = False
        linear = planner_for(problem, settings, euclidean=euclidean)
        for _ in range(2):
            expected = kd.solve(problem.start, problem.goal_const, [])
            actual = linear.solve(problem.start, problem.goal_const, [])
            assert expected is not None
            np.testing.assert_array_equal(actual, expected)
            assert kd.get_call_count() == linear.get_call_count()


@pytest.mark.parametrize("pose_goal", [False, True])
@pytest.mark.parametrize("refine", [(), (RefineType.SHORTCUT,), (RefineType.BSPLINE,)])
def test_solver_interface(pose_goal, refine):
    problem = create_test_problem(pose_goal)
    # Smoothing uses the same discrete validator as the existing OMPL
    # planners; use a fine grid before doing a dense independent audit.
    problem.resolution = np.full(len(problem.start), 0.005)
    config = OMPLSolverConfig(algorithm=Algorithm.VampRRTC, refine_seq=refine)
    result = OMPLSolver(config).solve(problem)
    assert result.success
    assert result.n_call > 0
    assert result.ns_internal > 0
    path = result.traj.numpy()
    goal = path[-1] if pose_goal else problem.goal_const
    check_path(path, problem, goal)
