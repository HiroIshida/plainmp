import time

import numpy as np
from skmp.robot.jaxon import Jaxon
from skmp.robot.utils import set_robot_state
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis, Box
from skrobot.viewers import PyrenderViewer

import tinyfk
from plainmp.ik import solve_ik
from plainmp.robot_spec import JaxonSpec, RotType

jspec = JaxonSpec()
com_const = jspec.create_default_com_const()
efnames = ["rleg_end_coords", "lleg_end_coords", "rarm_end_coords", "larm_end_coords"]
start_coords_list = [
    Coordinates([0.0, -0.2, 0]),
    Coordinates([0.0, +0.2, 0]),
    Coordinates([0.6, -0.25, 0.25]).rotate(+np.pi * 0.5, "z"),
    Coordinates([0.6, +0.25, 0.25]).rotate(+np.pi * 0.5, "z"),
]
eq_cst = stand_pose_const = jspec.crate_pose_const_from_coords(
    efnames, start_coords_list, [RotType.XYZW] * 4
)

lb, ub = jspec.angle_bounds()
ret = solve_ik(eq_cst, com_const, lb, ub, q_seed=None, max_trial=100)
print(ret)
assert ret.success

# visualize
v = PyrenderViewer()
robot = Jaxon()
set_robot_state(robot, jspec.control_joint_names, ret.q, tinyfk.BaseType.FLOATING)
ground = Box([2, 2, 0.01])
for co in start_coords_list:
    ax = Axis.from_coords(co)
    v.add(ax)

ax = Axis.from_coords(robot.rleg_end_coords)
v.add(ax)
v.add(robot)
v.add(ground)
v.show()
time.sleep(10)
