import copy
import time

import numpy as np
from skrobot.model.primitives import Box
from skrobot.models.panda import Panda
from skrobot.viewers import PyrenderViewer

from plainmp.ompl_solver import OMPLSolver
from plainmp.problem import Problem
from plainmp.psdf import UnionSDF
from plainmp.robot_spec import PandaSpec
from plainmp.utils import primitive_to_plainmp_sdf, set_robot_state

ps = PandaSpec()
cst = ps.create_collision_const()

panda = Panda()
q0 = copy.deepcopy(panda.angle_vector())
q0[0:5] = 0.0
q0[0] = -1.54
q0[1] = 1.54
q0[3] = -0.1
q0[5] = 1.5

q1 = copy.deepcopy(q0)
q1[0] = 1.54
q1[1] = 1.54

panda.angle_vector(q1)

case = "d"
if case == "a":
    n_sample = 20
elif case == "b":
    n_sample = 50
elif case == "c":
    n_sample = 100
elif case == "d":
    n_sample = 300

low = np.array([-1.2, -1.2, 0.0])
high = np.array([1.2, 1.2, 1.2])
density = 0.001 * n_sample / np.prod(high - low)
print(f"density: {density}")

box_lits = []
ground = Box([0.8, 0.8, 0.05], with_sdf=True)
ground.translate([0, 0, -0.025])
box_lits.append(ground)
np.random.seed(0)
while len(box_lits) < n_sample:
    box = Box([0.1, 0.1, 0.1], with_sdf=True)
    pos = np.random.uniform(low, high)
    if abs(pos[0]) < 0.2 and pos[2] < 0.5:
        continue
    box.translate(pos)
    box_lits.append(box)

boxes_in_reach = []
for box in box_lits:
    if np.linalg.norm(box.worldpos()) < 1.0:
        boxes_in_reach.append(box)

ts = time.time()
dists = [np.linalg.norm(box.worldpos()) for box in boxes_in_reach]
sorted_idx = np.argsort(dists)
boxes_in_reach = [boxes_in_reach[idx] for idx in sorted_idx]
print(f"time elapsed to sort: {time.time() - ts}")

sdf = UnionSDF([primitive_to_plainmp_sdf(b) for b in boxes_in_reach])
cst.set_sdf(sdf)

lb, ub = ps.angle_bounds()
msbox = np.array([0.1, 0.12, 0.15, 0.18, 0.3, 0.3, 0.3])
problem = Problem(q0[:7], lb, ub, q1[:7], cst, None, msbox)
solver = OMPLSolver()
ret = solver.solve(problem)

v = PyrenderViewer()
v.add(panda)
for box in box_lits:
    v.add(box)
v.show()

for q in ret.traj.resample(20):
    set_robot_state(panda, ps.control_joint_names, q)
    time.sleep(0.2)

time.sleep(1000)
