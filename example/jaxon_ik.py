import time

import numpy as np
from skmp.robot.jaxon import Jaxon
from skmp.robot.utils import set_robot_state
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Box
from skrobot.viewers import PyrenderViewer

import tinyfk
from plainmp.constraint import EqCompositeCst, IneqCompositeCst
from plainmp.ik import solve_ik
from plainmp.manifold_rrt.manifold_rrt_solver import ManiRRTConfig, ManiRRTConnectSolver
from plainmp.parallel import ParallelSolver
from plainmp.problem import Problem
from plainmp.psdf import UnionSDF
from plainmp.robot_spec import JaxonSpec, RotType
from plainmp.utils import sksdf_to_cppsdf

jspec = JaxonSpec(gripper_collision=False)
com_const = jspec.create_default_com_const(total_force_on_arm=10)
com_const_no = jspec.create_default_com_const()

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
ret1 = solve_ik(eq_cst, ineq_cst, lb, ub, q_seed=None, max_trial=100)
assert ret1.success

# solve goal IK
rarm_target = table.copy_worldcoords()
rarm_target.translate([0.0, -0.25, 0.66]).rotate(np.pi * 0.5, "z")
larm_target = table.copy_worldcoords()
larm_target.translate([0.0, +0.25, 0.66]).rotate(np.pi * 0.5, "z")
goal_coords_list = [
    Coordinates([0.0, -0.2, 0]),
    Coordinates([0.0, +0.2, 0]),
    rarm_target,
    larm_target,
]

goal_pose_cst = jspec.crate_pose_const_from_coords(efnames, goal_coords_list, [RotType.XYZW] * 4)
ret2 = solve_ik(goal_pose_cst, ineq_cst, lb, ub, q_seed=ret1.q, max_trial=100)
assert ret2.success

# solve RRT
stance_cst = jspec.crate_pose_const_from_coords(
    efnames[:2], start_coords_list[:2], [RotType.XYZW] * 2
)
relative_pose_cst = jspec.create_relative_pose_const(
    "rarm_end_coords", "larm_end_coords", np.array([0.5, 0.0, 0.0])
)
fixed_zaxis_cst = jspec.create_fixed_zaxis_const("rarm_end_coords")
eq_global_cst = EqCompositeCst([stance_cst, relative_pose_cst, fixed_zaxis_cst])

box_coll_cst = jspec.create_attached_box_collision_const(
    box, "rarm_end_coords", np.array([0.25, 0.0, -0.04])
)
table_sdf = sksdf_to_cppsdf(table.sdf)
box_coll_cst.set_sdf(table_sdf)


ineq_cst = IneqCompositeCst([com_const, coll_cst, box_coll_cst])

problem = Problem(ret1.q, lb, ub, ret2.q, ineq_cst, eq_global_cst, np.array([0.1] * 37))
solver = ManiRRTConnectSolver(ManiRRTConfig(10000))
psolver = ParallelSolver(solver, 8)

ts = time.time()
result = psolver.solve(problem)
print("Time: ", time.time() - ts)
print(result)

# visualize
v = PyrenderViewer()
ground = Box([2, 2, 0.01])
robot = Jaxon()
set_robot_state(robot, jspec.control_joint_names, ret1.q, tinyfk.BaseType.FLOATING)
robot.rarm_end_coords.assoc(box)

v.add(ground)
v.add(robot)
v.add(table)
v.add(box)
v.show()
time.sleep(2)

for q in result.traj:
    set_robot_state(robot, jspec.control_joint_names, q, tinyfk.BaseType.FLOATING)
    v.redraw()
    time.sleep(0.1)
time.sleep(10)
