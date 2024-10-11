import argparse
import time

import numpy as np
from skrobot.model.primitives import Box, PointCloudLink
from skrobot.models.fetch import Fetch
from skrobot.viewers import PyrenderViewer

from plainmp.ik import solve_ik
from plainmp.psdf import CloudSDF
from plainmp.robot_spec import FetchSpec
from plainmp.utils import set_robot_state, sksdf_to_cppsdf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="visualize the result")
    parser.add_argument("--pcloud", action="store_true", help="use point cloud for collision check")
    args = parser.parse_args()

    # create table sdf
    if args.pcloud:
        table_points = np.random.rand(1000, 3) * np.array([1.0, 2.0, 0.05]) - np.array(
            [0.5, 1.0, 0.025]
        )
        table_points += np.array([1.0, 0.0, 0.8])
        table = PointCloudLink(table_points)
        sdf = CloudSDF(table_points, 0.002)
    else:
        table = Box([1.0, 2.0, 0.05], with_sdf=True)
        table.translate([1.0, 0.0, 0.8])
        sdf = sksdf_to_cppsdf(table.sdf, False)

    # create problem
    fs = FetchSpec()
    eq_cst = fs.create_gripper_pose_const([0.7, +0.2, 0.95, 0, 0, 0])
    ineq_cst = fs.create_collision_const()
    ineq_cst.set_sdf(sdf)
    lb, ub = fs.angle_bounds()

    # solve it
    ts = time.time()
    ret = solve_ik(eq_cst, ineq_cst, lb, ub, q_seed=None, max_trial=10)
    print(f"after {ret.n_trial} trials, elapsed time: {(time.time() - ts) * 1000:.2f} msec")
    assert ret.success

    if args.visualize:
        fetch = Fetch()
        set_robot_state(fetch, fs.control_joint_names, ret.q)
        v = PyrenderViewer()
        v.add(fetch)
        v.add(table)
        v.show()
        time.sleep(1000)
