import argparse
import time

import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis, Box
from skrobot.viewers import PyrenderViewer

from plainmp.ik import solve_ik
from plainmp.ompl_solver import OMPLSolver, OMPLSolverConfig
from plainmp.problem import Problem
from plainmp.robot_spec import OpenArmV10DualSpec
from plainmp.utils import primitive_to_plainmp_sdf


def solve_ik_2pi_adjust(cst, ineq_cst, lb: np.ndarray, ub: np.ndarray, **kwargs):
    # NOTE: hack to fight against 2pi periodicity issue
    ub_tmp = ub.copy()
    ub_tmp[7] += 2 * np.pi
    if "max_trial" in kwargs:
        max_trial = kwargs.pop("max_trial")
        kwargs["max_trial"] = 1
    else:
        max_trial = 100  # default

    for _trial in range(max_trial):
        result = solve_ik(cst, ineq_cst, lb, ub_tmp, **kwargs)
        if not result.success:
            continue
        if result.q[7] > ub[7]:
            adjusted = result.q[7] - 2 * np.pi
            if lb[7] <= adjusted <= ub[7]:
                result.q[7] = adjusted
                return result
            else:
                result.success = False
                continue
        else:
            return result

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="visualize the result or not")
    args = parser.parse_args()

    s = OpenArmV10DualSpec()

    target_right = Coordinates(pos=[0.3, +0.2, 0.72])
    target_right.rotate(np.pi * 0.5, "z")
    target_left = Coordinates(pos=[0.3, -0.1, 0.52])
    target_left.rotate(-np.pi * 0.5, "z")
    cst = s.create_tcp_pose_const(target_right, target_left)
    ineq_cst = s.create_collision_const(self_collision=True)
    obstacle = Box(extents=[0.05, 1.0, 1.0], face_colors=[1.0, 0.0, 0.0, 0.5])
    obstacle.translate([0.45, 0.0, 0.5])
    ineq_cst.set_sdf(primitive_to_plainmp_sdf(obstacle))

    lb, ub = s.angle_bounds()
    result = solve_ik_2pi_adjust(cst, ineq_cst, lb, ub, max_trial=100, config=None)
    assert result.success
    print(f"collision aware ik solved in {result.elapsed_time} sec")

    # solve motion plan
    resolution = np.ones(14) * 0.05
    problem = Problem(np.zeros(14), lb, ub, result.q, ineq_cst, None, resolution)
    solver = OMPLSolver(OMPLSolverConfig(shortcut=True))
    result = solver.solve(problem)
    print(f"motion plan solved in {result.time_elapsed} sec")

    if args.visualize:
        print("loading robot model to visualize...")
        model = s.get_robot_model(with_mesh=True)
        v = PyrenderViewer()
        v.add(Axis.from_coords(target_right))
        v.add(Axis.from_coords(target_left))
        v.add(model)
        v.add(obstacle)
        v.show()

        input("Press Enter to start animation...")

        for q in result.traj.resample(20):
            s.set_skrobot_model_state(model, q)
            time.sleep(0.5)
            v.redraw()
        time.sleep(1000)
