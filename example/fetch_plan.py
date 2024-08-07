import time

import numpy as np
from skmp.robot.fetch import FetchConfig
from skmp.robot.utils import set_robot_state
from skmp.visualization.collision_visualizer import CollisionSphereVisualizationManager
from skrobot.model.primitives import Box
from skrobot.models import Fetch
from skrobot.viewers import PyrenderViewer

from plainmp.ompl_solver import OMPLSolver
from plainmp.problem import Problem
from plainmp.robot_spec import FetchSpec
from plainmp.utils import sksdf_to_cppsdf

fetch = FetchSpec()
cst = fetch.create_collision_const()

table = Box([1.0, 2.0, 0.05], with_sdf=True)
table.translate([1.0, 0.0, 0.8])
ground = Box([2.0, 2.0, 0.05], with_sdf=True)
sdfs = [sksdf_to_cppsdf(table.sdf), sksdf_to_cppsdf(ground.sdf)]
cst.set_sdfs(sdfs)
lb, ub = fetch.angle_bounds()
start = np.array([0.0, 1.31999949, 1.40000015, -0.20000077, 1.71999929, 0.0, 1.6600001, 0.0])
goal = np.array([0.386, 0.20565, 1.41370, 0.30791, -1.82230, 0.24521, 0.41718, 6.01064])
msbox = np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.2, 0.2])
problem = Problem(start, lb, ub, goal, cst, None, msbox)
solver = OMPLSolver()

times = []
for _ in range(100):
    ts = time.time()
    ret = solver.solve(problem)
    print(f"planning time {1000 * (time.time() - ts)} [ms]")
    times.append(time.time() - ts)
print(f"average planning time {1000 * np.mean(times)} [ms]")

# visualize
conf = FetchConfig()
fetch = Fetch()
set_robot_state(fetch, conf.get_control_joint_names(), goal)
v = PyrenderViewer()
colkin = conf.get_collision_kin()
colvis = CollisionSphereVisualizationManager(colkin, v)
colvis.update(fetch)
v.add(fetch)
v.add(table)
v.add(ground)
v.show()

time.sleep(1.0)
for q in ret.traj.resample(20):
    set_robot_state(fetch, conf.get_control_joint_names(), q)
    colvis.update(fetch)
    v.redraw()
    time.sleep(0.5)

time.sleep(1000)
