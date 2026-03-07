import numpy as np
import tqdm
from skrobot.viewers import PyrenderViewer

from plainmp.robot_spec import OpenArmV10RarmSpec

if __name__ == "__main__":
    v = PyrenderViewer()
    spec = OpenArmV10RarmSpec()
    kin = spec.get_kin()

    joint_names = spec.control_joint_names
    joint_ids = kin.get_joint_ids(joint_names)

    n_bench = 10000

    q = np.array([1.54, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in tqdm.tqdm(range(n_bench)):
        tau = kin.get_gravity_term2(joint_ids, q)
