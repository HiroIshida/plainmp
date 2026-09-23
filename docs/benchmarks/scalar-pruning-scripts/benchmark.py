import os
"""Reproduce online collision-prediction experiments using identical query streams."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

ROOT = Path(os.environ["PLAINMP_SCALAR_STUDY"])


def load_variant(variant):
    # Avoid the machine's editable plainmp installation and its old native module.
    package = types.ModuleType("plainmp")
    package.__path__ = [str(ROOT / "source/src/plainmp")]
    sys.modules["plainmp"] = package
    shared = next((ROOT / "build" / variant).glob("_plainmp*.so"))
    spec = importlib.util.spec_from_file_location("plainmp._plainmp", shared)
    native = importlib.util.module_from_spec(spec)
    sys.modules["plainmp._plainmp"] = native
    spec.loader.exec_module(native)
    return native


def create_scene(name):
    import numpy as np
    from plainmp.robot_spec import PandaSpec, FetchSpec
    from plainmp.psdf import BoxSDF, CylinderSDF, SphereSDF, UnionSDF, Pose, CloudSDF
    from plainmp.problem import Problem

    def box(extents, xyz):
        return BoxSDF(np.array(extents), Pose(np.array(xyz), np.eye(3)))

    if name.startswith("panda"):
        robot = PandaSpec()
        start = np.array([-1.54, 1.54, 0, -0.1, 0, 1.5, 0.81])
        goal = np.array([1.54, 1.54, 0, -0.1, 0, 1.5, 0.81])
        obstacles = [box([2, 2, .05], [0, 0, -.05]),
                     CylinderSDF(.05, 1., Pose(np.array([.3, .3, .5]), np.eye(3))),
                     CylinderSDF(.05, 1., Pose(np.array([-.3, -.3, .5]), np.eye(3)))]
        if name == "panda_hard":
            obstacles.append(box([2, 2, .05], [0, 0, 1.]))
    else:
        robot = FetchSpec()
        if name.startswith("fetch_table"):
            start = np.array([0, 1.32, 1.40, -.20, 1.72, 0, 1.66, 0])
            goal = np.array([.386, .205, 1.41, .308, -1.82, .245, .417, 6.01])
            obstacles = [box([1, 2, .05], [.95, 0, .8])]
        else:
            start = np.array([0, 1.31999949, 1.40000015, -.20000077, 1.71999929, 0, 1.6600001, 0])
            goal = np.array([.386, .20565, 1.41370, .30791, -1.82230, .24521, .41718, 6.01064])
            obstacles = [box([1, 1, .05], [0, 0, -.1])]
            rng = np.random.RandomState(7)
            for _ in range(9 if name == "fetch_spheres9" else 4):
                xyz = [rng.uniform(.2, 1) + .2, -rng.uniform(-.6, .6), rng.uniform(.3, 1.5)]
                obstacles.append(SphereSDF(.15, Pose(np.array(xyz), np.eye(3))))
    if name.startswith("fetch_table_cloud"):
        # Same table volume sampling as example/fetch_plan.py --pcloud,
        # with fixed geometry seed; keep the earlier planning parameters.
        count = 100000 if name.endswith("100k") else 10000
        points = np.random.RandomState(424242).rand(count, 3) * [1., 2., .05]
        points += np.array([.95, 0, .8]) - np.array([.5, 1., .025])
        obstacles = [CloudSDF(points, .002)]
    if name in ["fetch_table_tilted", "panda_boxes"]:
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_euler("xyz", [.12, .08, .4]).as_matrix()
        if name == "fetch_table_tilted":
            obstacles = [BoxSDF(np.array([1., 2., .05]), Pose(np.array([.95, 0, .74]), rot))]
        else:
            obstacles = [obstacles[0], BoxSDF(np.array([.1, .1, 1.]), Pose(np.array([.3, .3, .5]), rot)),
                         BoxSDF(np.array([.1, .1, 1.]), Pose(np.array([-.3, -.3, .5]), rot.T))]
    cst = robot.create_collision_const(use_cache=False)
    cst.set_sdf(UnionSDF(obstacles))
    lb, ub = robot.angle_bounds()
    problem = Problem(start, lb, ub, goal, cst, None, 1 / 32, "euclidean")
    return robot, cst, problem
