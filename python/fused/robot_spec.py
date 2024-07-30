import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np
import yaml
from fused.constraint import FusedSpheresCollisionChecker, SphereAttachentSpec
from fused.utils import sksdf_to_cppsdf
from skrobot.model.primitives import Box, Cylinder, Sphere
from skrobot.model.robot_model import RobotModel
from skrobot.utils.urdf import URDF, no_mesh_load_mode

_loaded_urdf_models: Dict[str, URDF] = {}


def load_urdf_model_using_cache(file_path: Path, deepcopy: bool = True):
    file_path = file_path.expanduser()
    assert file_path.exists()
    key = str(file_path)
    if key not in _loaded_urdf_models:
        with no_mesh_load_mode():
            urdf = URDF.load(str(file_path))
        _loaded_urdf_models[key] = urdf
    if deepcopy:
        return copy.deepcopy(_loaded_urdf_models[key])
    else:
        return _loaded_urdf_models[key]


class RobotSpec(ABC):
    def __init__(self, conf_file: Path):
        with open(conf_file, "r") as f:
            self.conf_dict = yaml.safe_load(f)

    @property
    def control_joint_names(self) -> List[str]:
        return self.conf_dict["control_joint_names"]

    @property
    def robot_model(self) -> RobotModel:
        return load_urdf_model_using_cache(self.urdf_path)

    @property
    def urdf_path(self) -> Path:
        return Path(self.conf_dict["urdf_path"]).expanduser()

    @abstractmethod
    def self_body_collision_primitives(self) -> Sequence[Union[Box, Sphere, Cylinder]]:
        pass

    def create_collision_const(self) -> FusedSpheresCollisionChecker:
        d = self.conf_dict["collision_spheres"]

        sphere_specs = []
        for link_name, vals in d.items():
            ignore_collision = vals["ignore_collision"]
            spheres_d = vals["spheres"]
            for spec in spheres_d:
                vals = np.array(spec)
                center, r = vals[:3], vals[3]
                sphere_specs.append(SphereAttachentSpec(link_name, center, r, ignore_collision))
        self_collision_pairs = self.conf_dict["self_collision_pairs"]
        sdfs = [sksdf_to_cppsdf(sk.sdf) for sk in self.self_body_collision_primitives()]
        cst = FusedSpheresCollisionChecker(
            str(self.urdf_path), self.control_joint_names, sphere_specs, self_collision_pairs, sdfs
        )
        return cst


class FetchSpec(RobotSpec):
    def __init__(self):
        p = Path(__file__).parent / "conf" / "fetch_conf.yaml"
        super().__init__(p)

    def self_body_collision_primitives(self) -> Sequence[Union[Box, Sphere, Cylinder]]:
        base = Cylinder(0.29, 0.32, face_colors=[255, 255, 255, 200], with_sdf=True)
        base.translate([0.005, 0.0, 0.2])
        torso = Box([0.16, 0.16, 1.0], face_colors=[255, 255, 255, 200], with_sdf=True)
        torso.translate([-0.12, 0.0, 0.5])

        neck_lower = Box([0.1, 0.18, 0.08], face_colors=[255, 255, 255, 200], with_sdf=True)
        neck_lower.translate([0.0, 0.0, 0.97])
        neck_upper = Box([0.05, 0.17, 0.15], face_colors=[255, 255, 255, 200], with_sdf=True)
        neck_upper.translate([-0.035, 0.0, 0.92])

        torso_left = Cylinder(0.1, 1.5, face_colors=[255, 255, 255, 200], with_sdf=True)
        torso_left.translate([-0.143, 0.09, 0.75])
        torso_right = Cylinder(0.1, 1.5, face_colors=[255, 255, 255, 200], with_sdf=True)
        torso_right.translate([-0.143, -0.09, 0.75])

        head = Cylinder(0.235, 0.12, face_colors=[255, 255, 255, 200], with_sdf=True)
        head.translate([0.0, 0.0, 1.04])
        self_body_obstacles = [base, torso, torso_left, torso_right]
        return self_body_obstacles
