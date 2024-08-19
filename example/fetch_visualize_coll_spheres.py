import time

import numpy as np
from skrobot.model.primitives import Sphere
from skrobot.models.fetch import Fetch
from skrobot.viewers import PyrenderViewer

from plainmp.robot_spec import FetchSpec
from plainmp.utils import set_robot_state

fs = FetchSpec()
cst = fs.create_collision_const()
q = np.zeros(8)
cst.update_kintree(q)


sk_group_spheres = []
for center, r in cst.get_group_spheres():
    sk_sphere = Sphere(r, pos=center, color=[255, 0, 0, 100])
    sk_group_spheres.append(sk_sphere)

sk_all_spheres = []
for center, r in cst.get_all_spheres():
    sk_sphere = Sphere(r, pos=center, color=[0, 255, 0, 100])
    sk_all_spheres.append(sk_sphere)

fetch = Fetch()
set_robot_state(fetch, fs.control_joint_names, q)
v = PyrenderViewer()
v.add(fetch)
for s in sk_group_spheres:
    v.add(s)
for s in sk_all_spheres:
    v.add(s)
v.show()
time.sleep(1000)
