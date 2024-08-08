import time

import numpy as np
from skmp.robot.jaxon import Jaxon
from skmp.robot.utils import set_robot_state
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis, Box
from skrobot.viewers import PyrenderViewer

import tinyfk
from plainmp.constraint import IneqCompositeCst
from plainmp.ik import solve_ik
from plainmp.psdf import UnionSDF
from plainmp.robot_spec import JaxonSpec, RotType
from plainmp.utils import sksdf_to_cppsdf

jspec = JaxonSpec(gripper_collision=False)
com_const = jspec.create_default_com_const(total_force_on_arm=10)
com_const_no = jspec.create_default_com_const()

# collision constraint against box
box = Box([0.4, 0.4, 0.4], with_sdf=True, face_colors=[0, 0, 255, 230])
box.translate([0.6, 0.0, 0.2])

ground = Box([2.0, 2.0, 0.03], with_sdf=True)
ground.translate([0.0, 0.0, -0.015])

table = Box([0.6, 1.0, 0.8], with_sdf=True)
table.rotate(np.pi * 0.5, "z")
table.translate([0.7, 0.0, 0.4])

sksdfs = [box.sdf, ground.sdf, table.sdf]
sdf = UnionSDF([sksdf_to_cppsdf(sdf) for sdf in sksdfs], False)

coll_cst = jspec.create_collision_const(False)
coll_cst.set_sdf(sdf)
ineq_cst = IneqCompositeCst([com_const, com_const_no, coll_cst])

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
ts = time.time()
for _ in range(100):
    ret = solve_ik(eq_cst, ineq_cst, lb, ub, q_seed=None, max_trial=100)
print(time.time() - ts)
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
v.add(table)
v.add(box)
v.show()
time.sleep(10)
