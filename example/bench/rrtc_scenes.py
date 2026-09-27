"""Shared fixed-goal Fetch/Panda scenes for RRTC benchmarks."""

import numpy as np
from skrobot.model.primitives import Box, Cylinder

from plainmp.problem import Problem
from plainmp.psdf import GroundSDF, UnionSDF
from plainmp.robot_spec import FetchSpec, PandaSpec
from plainmp.utils import primitive_to_plainmp_sdf


def make_problem(scene, validator):
    if scene == "fetch":
        spec = FetchSpec()
        table = Box([1.0, 2.0, 0.05])
        table.translate([0.95, 0, 0.8])
        obstacles = [primitive_to_plainmp_sdf(table), GroundSDF(0.0)]
        start = np.array([0, 1.32, 1.40, -0.20, 1.72, 0, 1.66, 0])
        goal = np.array([0.386, 0.205, 1.41, 0.308, -1.82, 0.245, 0.417, 6.01])
    else:
        spec = PandaSpec()
        ground = Box([2, 2, 0.05])
        ground.translate([0, 0, -0.05])
        pole1, pole2 = Cylinder(0.05, 1.0), Cylinder(0.05, 1.0)
        pole1.translate([0.3, 0.3, 0.5])
        pole2.translate([-0.3, -0.3, 0.5])
        primitives = [ground, pole1, pole2]
        if scene == "panda_difficult":
            ceiling = Box([2, 2, 0.05])
            ceiling.translate([0, 0, 1])
            primitives.append(ceiling)
        obstacles = [primitive_to_plainmp_sdf(p) for p in primitives]
        start = np.array([-1.54, 1.54, 0, -0.1, 0, 1.5, 0.81])
        goal = np.array([1.54, 1.54, 0, -0.1, 0, 1.5, 0.81])
    constraint = spec.create_collision_const()
    constraint.set_sdf(UnionSDF(obstacles))
    lb, ub = spec.angle_bounds()
    resolution = np.full(len(start), 0.05) if validator == "box" else 0.05
    return Problem(start, lb, ub, goal, constraint, None, resolution, validator_type=validator)
