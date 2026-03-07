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

    # panda
    q = np.array([1.54, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in tqdm.tqdm(range(n_bench)):
        kin.set_joint_positions(joint_ids, q)
        tau = kin.get_gravity_term(joint_ids)
    print(tau)

    # skrobot
    model = spec.get_robot_model()
    joint_ids = np.array([model.joint_names.index(jn) for jn in joint_names])
    av = model.angle_vector()
    av[joint_ids] = q
    for _ in tqdm.tqdm(range(n_bench)):
        model.angle_vector(av)
        tau_all = model.inverse_dynamics(gravity=np.array([0, 0, -9.81]))
        tau = tau_all[joint_ids]
    print(tau)
