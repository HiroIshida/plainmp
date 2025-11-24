import argparse
import time

import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis
from skrobot.viewers import PyrenderViewer

from plainmp.ik import IKConfig, solve_ik
from plainmp.robot_spec import OpenArmV10LarmSpec


def solve_ik_2pi_adjust(cst, ineq_cst, lb: np.ndarray, ub: np.ndarray, **kwargs):
    # NOTE: hack to fight against 2pi periodicity issue
    ub_tmp = ub.copy()
    ub_tmp[0] += 2 * np.pi
    result = solve_ik(cst, ineq_cst, lb, ub_tmp, **kwargs)
    if result.success:
        if result.q[0] > ub[0]:
            adjusted = result.q[0] - 2 * np.pi
            if lb[0] <= adjusted <= ub[0]:
                result.q[0] = adjusted
            else:
                result.success = False
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="visualize the result or not")
    args = parser.parse_args()
    np.random.seed(0)

    s = OpenArmV10LarmSpec()

    target = Coordinates(pos=[0.5, +0.2, 0.62])
    cst = s.create_tcp_pose_const(target)
    ineq_cst = s.create_collision_const()  # self collision
    lb, ub = s.angle_bounds()
    result = solve_ik_2pi_adjust(cst, None, lb, ub, max_trial=100, config=IKConfig(timeout=1000))
    assert result.success
    assert np.all(result.q >= lb) and np.all(result.q <= ub)
    print(f"collision aware ik solved in {result.elapsed_time} sec")

    if args.visualize:
        print("loading robot model to visualize...")
        model = s.get_robot_model(with_mesh=True)
        s.set_skrobot_model_state(model, result.q)
        v = PyrenderViewer()
        v.add(Axis.from_coords(target))
        v.add(model)
        v.show()
        time.sleep(1000)
