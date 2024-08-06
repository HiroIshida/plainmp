import copy
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import yaml
from skrobot.coordinates import CascadedCoords, Coordinates
from skrobot.coordinates.math import (
    matrix2quaternion,
    rotation_matrix,
    rpy_angle,
    wxyz2xyzw,
)
from skrobot.model.primitives import Box, Cylinder, Sphere
from skrobot.model.robot_model import RobotModel
from skrobot.models.urdf import RobotModelFromURDF
from skrobot.utils.urdf import URDF, no_mesh_load_mode

from plainmp.constraint import (
    ComInPolytopeCst,
    LinkPoseCst,
    SphereAttachentSpec,
    SphereCollisionCst,
)
from plainmp.psdf import BoxSDF, Pose
from plainmp.tinyfk import KinematicModel
from plainmp.utils import sksdf_to_cppsdf

_loaded_urdf_models: Dict[str, URDF] = {}
_loaded_kin: Dict[str, KinematicModel] = {}


def load_urdf_model_using_cache(file_path: Path, deepcopy: bool = True):
    file_path = file_path.expanduser()
    assert file_path.exists()
    key = str(file_path)
    if key not in _loaded_urdf_models:
        with no_mesh_load_mode():
            model = RobotModelFromURDF(urdf_file=str(file_path))
        _loaded_urdf_models[key] = model
    if deepcopy:
        return copy.deepcopy(_loaded_urdf_models[key])
    else:
        return _loaded_urdf_models[key]


class RotType(Enum):
    IGNORE = 0
    RPY = 1
    XYZW = 2


class RobotSpec(ABC):
    def __init__(self, conf_file: Path, with_base: bool):
        with open(conf_file, "r") as f:
            self.conf_dict = yaml.safe_load(f)
        self.with_base = with_base

    def get_kin(self) -> KinematicModel:
        if str(self.urdf_path) not in _loaded_kin:
            with open(self.urdf_path, "r") as f:
                urdf_str = f.read()
            kin = KinematicModel(urdf_str)
            _loaded_kin[str(self.urdf_path)] = kin
        return _loaded_kin[str(self.urdf_path)]

    @abstractmethod
    def get_robot_model(self) -> RobotModel:
        ...

    @property
    def urdf_path(self) -> Path:
        return Path(self.conf_dict["urdf_path"]).expanduser()

    @property
    @abstractmethod
    def control_joint_names(self) -> List[str]:
        ...

    @abstractmethod
    def self_body_collision_primitives(self) -> Sequence[Union[Box, Sphere, Cylinder]]:
        pass

    @abstractmethod
    def angle_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def get_sphere_specs(self) -> List[SphereAttachentSpec]:
        # the below reads the all the sphere specs from the yaml file
        # but if you want to use the sphere specs for the specific links
        # you can override this method
        d = self.conf_dict["collision_spheres"]
        sphere_specs = []
        for link_name, vals in d.items():
            ignore_collision = vals["ignore_collision"]
            spheres_d = vals["spheres"]
            for spec in spheres_d:
                vals = np.array(spec)
                center, r = vals[:3], vals[3]
                sphere_specs.append(SphereAttachentSpec(link_name, center, r, ignore_collision))
        return sphere_specs

    def create_collision_const(self, self_collision: bool = True) -> SphereCollisionCst:
        sphere_specs = self.get_sphere_specs()
        if self_collision:
            if "self_collision_pairs" not in self.conf_dict:
                raise ValueError("self_collision_pairs is not defined in the yaml file")
            self_collision_pairs = self.conf_dict["self_collision_pairs"]
            sdfs = [sksdf_to_cppsdf(sk.sdf) for sk in self.self_body_collision_primitives()]
        else:
            self_collision_pairs = []
            sdfs = []
        with open(self.urdf_path, "r") as f:
            urdf_str = f.read()
        kin = KinematicModel(urdf_str)
        cst = SphereCollisionCst(
            kin, self.control_joint_names, self.with_base, sphere_specs, self_collision_pairs, sdfs
        )
        return cst

    def crate_pose_const_from_coords(
        self, link_names: List[str], link_poses: List[Coordinates], rot_types: List[RotType]
    ) -> LinkPoseCst:
        pose_list = []
        for co, rt in zip(link_poses, rot_types):
            pos = co.worldpos()
            if rt == RotType.RPY:
                ypr = rpy_angle(co.rotation)[0]
                rpy = [ypr[2], ypr[1], ypr[0]]
                pose = np.hstack([pos, rpy])
            elif rt == RotType.XYZW:
                quat_wxyz = matrix2quaternion(co.rotation)
                pose = np.hstack([pos, wxyz2xyzw(quat_wxyz)])
            else:
                pose = pos
            pose_list.append(pose)
        print(pose_list)
        return self.create_pose_const(link_names, pose_list)

    def create_pose_const(self, link_names: List[str], link_poses: List[np.ndarray]) -> LinkPoseCst:
        return LinkPoseCst(
            self.get_kin(), self.control_joint_names, self.with_base, link_names, link_poses
        )


class FetchSpec(RobotSpec):
    def __init__(self, with_base: bool = False):
        # set with_base = True only in testing
        p = Path(__file__).parent / "conf" / "fetch.yaml"
        super().__init__(p, with_base)

    def get_robot_model(self) -> RobotModel:
        return load_urdf_model_using_cache(self.urdf_path)

    @property
    def control_joint_names(self) -> List[str]:
        return self.conf_dict["control_joint_names"]

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

    def create_gripper_pose_const(self, link_pose: np.ndarray) -> LinkPoseCst:
        return self.create_pose_const(["gripper_link"], [link_pose])

    @staticmethod
    def angle_bounds() -> Tuple[np.ndarray, np.ndarray]:
        # it takes time to parse the urdf file so we do it here...
        min_angles = np.array(
            [0.0, -1.6056, -1.221, -np.pi * 2, -2.251, -np.pi * 2, -2.16, -np.pi * 2]
        )
        max_angles = np.array(
            [0.38615, 1.6056, 1.518, np.pi * 2, 2.251, np.pi * 2, 2.16, np.pi * 2]
        )
        return min_angles, max_angles

    @staticmethod
    def q_reset_pose() -> np.ndarray:
        return np.array([0.0, 1.31999949, 1.40000015, -0.20000077, 1.71999929, 0.0, 1.6600001, 0.0])

    @staticmethod
    def get_reachable_box() -> Tuple[np.ndarray, np.ndarray]:
        lb_reachable = np.array([-0.60046263, -1.08329689, -0.18025853])
        ub_reachable = np.array([1.10785484, 1.08329689, 2.12170273])
        return lb_reachable, ub_reachable


class JaxonSpec(RobotSpec):
    def __init__(self):
        p = Path(__file__).parent / "conf" / "jaxon.yaml"
        super().__init__(p, with_base=True)  # jaxon is free-floating, so with_base=True

    def get_kin(self):
        kin = super().get_kin()
        # the below is a workaround.
        try:
            # this raise error is those links are not attached.
            kin.get_link_ids(
                ["rarm_end_coords", "larm_end_coords", "rleg_end_coords", "lleg_end_coords"]
            )
        except ValueError:
            # so in the only first call of get_kin() the following code is executed.
            matrix = rotation_matrix(np.pi * 0.5, [0, 0, 1.0])
            rpy = np.flip(rpy_angle(matrix)[0])
            kin.add_new_link("rarm_end_coords", "LARM_LINK7", np.array([0, 0, -0.220]), rpy)
            kin.add_new_link("larm_end_coords", "LARM_LINK7", np.array([0, 0, -0.220]), rpy)
            kin.add_new_link("rleg_end_coords", "RLEG_LINK5", np.array([0, 0, -0.1]), np.zeros(3))
            kin.add_new_link("lleg_end_coords", "LLEG_LINK5", np.array([0, 0, -0.1]), np.zeros(3))
        return kin

    def get_robot_model(self) -> RobotModel:
        matrix = rotation_matrix(np.pi * 0.5, [0, 0, 1.0])
        model = load_urdf_model_using_cache(self.urdf_path)

        model.rarm_end_coords = CascadedCoords(model.RARM_LINK7, name="rarm_end_coords")
        model.rarm_end_coords.translate([0, 0, -0.220])
        model.rarm_end_coords.rotate_with_matrix(matrix, wrt="local")

        model.rarm_tip_coords = CascadedCoords(model.RARM_LINK7, name="rarm_end_coords")
        model.rarm_tip_coords.translate([0, 0, -0.3])
        model.rarm_tip_coords.rotate_with_matrix(matrix, wrt="local")

        model.larm_end_coords = CascadedCoords(model.LARM_LINK7, name="larm_end_coords")
        model.larm_end_coords.translate([0, 0, -0.220])
        model.larm_end_coords.rotate_with_matrix(matrix, wrt="local")

        model.rleg_end_coords = CascadedCoords(model.RLEG_LINK5, name="rleg_end_coords")
        model.rleg_end_coords.translate([0, 0, -0.1])

        model.lleg_end_coords = CascadedCoords(model.LLEG_LINK5, name="lleg_end_coords")
        model.lleg_end_coords.translate([0, 0, -0.1])
        return model

    @property
    def control_joint_names(self) -> List[str]:
        return self.conf_dict["control_joint_names"]

    def get_sphere_specs(self) -> List[SphereAttachentSpec]:
        # because legs are on the ground, we don't need to consider the spheres on the legs
        specs = super().get_sphere_specs()
        filtered = []
        for spec in specs:
            if spec.parent_link_name in ("RLEG_LINK5", "LLEG_LINK5"):
                continue
            filtered.append(spec)
        return filtered

    def self_body_collision_primitives(self) -> Sequence[Union[Box, Sphere, Cylinder]]:
        return []

    def create_default_stand_pose_const(self) -> LinkPoseCst:
        robot_model = self.get_robot_model()
        # set reset manip pose
        for jn, angle in zip(self.control_joint_names, self.reset_manip_pose_q):
            robot_model.__dict__[jn].joint_angle(angle)

        def skcoords_to_xyzrpy(co):
            pos = co.worldpos()
            ypr = rpy_angle(co.rotation)[0]
            rpy = [ypr[2], ypr[1], ypr[0]]
            return np.hstack([pos, rpy])

        rleg = robot_model.rleg_end_coords.copy_worldcoords()
        lleg = robot_model.lleg_end_coords.copy_worldcoords()
        return self.create_pose_const(
            ["rleg_end_coords", "lleg_end_coords"],
            [skcoords_to_xyzrpy(rleg), skcoords_to_xyzrpy(lleg)],
        )

    def create_default_com_const(self) -> ComInPolytopeCst:
        com_box = BoxSDF([0.25, 0.5, 0.0], Pose(np.array([0, 0, 0]), np.eye(3)))
        return ComInPolytopeCst(
            self.get_kin(), self.control_joint_names, self.with_base, com_box, []
        )

    @property
    def reset_manip_pose_q(self) -> np.ndarray:
        angle_table = {
            "RLEG": [0.0, 0.0, -0.349066, 0.698132, -0.349066, 0.0],
            "LLEG": [0.0, 0.0, -0.349066, 0.698132, -0.349066, 0.0],
            "CHEST": [0.0, 0.0, 0.0],
            "RARM": [0.0, 0.959931, -0.349066, -0.261799, -1.74533, -0.436332, 0.0, -0.785398],
            "LARM": [0.0, 0.959931, 0.349066, 0.261799, -1.74533, 0.436332, 0.0, -0.785398],
        }
        d = {}
        for key, values in angle_table.items():
            for i, angle in enumerate(values):
                d["{}_JOINT{}".format(key, i)] = angle
        return np.array([d[joint] for joint in self.control_joint_names])

    def angle_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.array(
            [
                -0.144751,
                -1.4208,
                -3.14159,
                -3.14159,
                -3.14159,
                0.2762,
                -3.14159,
                -3.14159,
                -2.19006,
                -2.19006,
                -3.14159,
                -3.14159,
                -1.5708,
                -1.54494,
                -1.41372,
                -1.41372,
                -1.02662,
                -1.0972,
                -0.725029,
                -0.523599,
                -2.11843,
                -2.11843,
                0.0,
                0.0,
                -1.38564,
                -1.38564,
                -1.0472,
                -1.0472,
                -0.198551,
                -0.0349066,
                -1.0566,
                -1.0,
                -1.0,
                0.0,
                -1.0,
                -1.0,
                -1.0,
            ]
        )
        ub = np.array(
            [
                1.4208,
                0.144751,
                3.14159,
                3.14159,
                -0.2762,
                3.14159,
                3.14159,
                3.14159,
                1.0472,
                1.0472,
                3.14159,
                3.14159,
                1.54494,
                1.5708,
                1.0472,
                1.0472,
                1.0972,
                1.02662,
                0.523599,
                0.725029,
                0.785398,
                0.785398,
                2.77419,
                2.77419,
                1.47343,
                1.47343,
                1.0472,
                1.0472,
                0.198551,
                0.610865,
                1.0566,
                2.0,
                1.0,
                3.0,
                1.0,
                1.0,
                1.0,
            ]
        )
        return lb, ub
