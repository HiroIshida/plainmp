import time

import numpy as np
from skrobot.model.primitives import Sphere
from skrobot.models.urdf import RobotModelFromURDF
from skrobot.viewers import PyrenderViewer

from plainmp.robot_spec import PandaSpec
from plainmp.utils import set_robot_state

s = PandaSpec()
cst = s.create_collision_const()
s.get_kin()

lb, ub = s.angle_bounds()
q = np.random.uniform(lb, ub)
cst.update_kintree(q, True)
model = RobotModelFromURDF(urdf_file=str(s.urdf_path))
set_robot_state(model, s.control_joint_names, q)

sk_all_spheres = []
for center, r in cst.get_all_spheres():
    sk_sphere = Sphere(r, pos=center, color=[0, 255, 0, 100])
    sk_all_spheres.append(sk_sphere)

v = PyrenderViewer()
v.add(model)
for sk_sphere in sk_all_spheres:
    v.add(sk_sphere)
v.show()

time.sleep(1000)
