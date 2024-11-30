import copy
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml
from skrobot.coordinates import CascadedCoords, Coordinates
from skrobot.coordinates.math import matrix2quaternion, rpy_angle, wxyz2xyzw
from skrobot.model.primitives import Box, Cylinder, Sphere
from skrobot.model.robot_model import RobotModel
from skrobot.models.urdf import RobotModelFromURDF
from skrobot.sdf import UnionSDF
from skrobot.utils.urdf import URDF, no_mesh_load_mode

from plainmp.constraint import (
    AppliedForceSpec,
    ComInPolytopeCst,
    ConfigPointCst,
    FixedZAxisCst,
    LinkPoseCst,
    RelativePoseCst,
    SphereAttachmentSpec,
    SphereCollisionCst,
)
from plainmp.kinematics import KinematicModel
from plainmp.psdf import BoxSDF, Pose
from plainmp.utils import sksdf_to_cppsdf

_loaded_urdf_models: Dict[str, URDF] = {}
_loaded_yamls: Dict[str, Dict] = {}  # loading yaml is quite slow
N_MAX_CACHE = 200
_loaded_kin: "OrderedDict[str, KinematicModel]" = OrderedDict()


def load_urdf_model_using_cache(file_path: Path, deepcopy: bool = True, with_mesh: bool = False):
    file_path = file_path.expanduser()
    assert file_path.exists()
    key = str(file_path)
    if key not in _loaded_urdf_models:
        if with_mesh:
            model = RobotModelFromURDF(urdf_file=str(file_path))
        else:
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
        if str(conf_file) not in _loaded_yamls:
            with open(conf_file, "r") as f:
                self.conf_dict = yaml.safe_load(f)
            _loaded_yamls[str(conf_file)] = self.conf_dict
        else:
            self.conf_dict = _loaded_yamls[str(conf_file)]
        self.with_base = with_base
        self.uuid = str(uuid.uuid4())

    def get_kin(self) -> KinematicModel:
        # The kinematic chain is shared among the same robot spec.
        # This sharing mechanism is important to instantiate an composite
        # constraint, albeit its usage complexity.
        self_id = self.uuid
        if self_id not in _loaded_kin:
            with open(self.urdf_path, "r") as f:
                urdf_str = f.read()
            kin = KinematicModel(urdf_str)
            if len(_loaded_kin) > (N_MAX_CACHE - 1):
                _loaded_kin.popitem(last=False)
            _loaded_kin[self_id] = kin

        # For each of the end effectors defined in the conf file
        # Add it as a link to the kinematic chain if it is not present in the chain.
        for name in self.conf_dict["end_effectors"].keys():
            try:
                # Pass if end effector link is already in the kinematic chain.
                _loaded_kin[self_id].get_link_ids([name])
            except ValueError:
                # Add the unfound end effector link to the kinematic chain.
                consider_rotation = True
                _loaded_kin[self_id].add_new_link(
                    name,
                    self.conf_dict["end_effectors"][name]["parent_link"],
                    np.array(self.conf_dict["end_effectors"][name]["position"]),
                    np.array(self.conf_dict["end_effectors"][name]["rpy"]),
                    consider_rotation,
                )
        return _loaded_kin[self_id]

    def get_robot_model(self) -> RobotModel:
        model = load_urdf_model_using_cache(self.urdf_path)
        # Add end effectors defined in conf to the robot model as CascadedCoords
        for name in self.conf_dict["end_effectors"].keys():
            parent_link = self.conf_dict["end_effectors"][name]["parent_link"]
            pos = np.array(self.conf_dict["end_effectors"][name]["position"])
            rpy = np.array(self.conf_dict["end_effectors"][name]["rpy"])
            setattr(model, name, CascadedCoords(getattr(model, parent_link), name=name))
            link = getattr(model, name)
            link.rotate(rpy[0], "x")
            link.rotate(rpy[1], "y")
            link.rotate(rpy[2], "z")
            link.translate(pos)
        return model

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

    def angle_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        kin = self.get_kin()
        joint_ids = kin.get_joint_ids(self.control_joint_names)
        limits = kin.get_joint_position_limits(joint_ids)
        lb = np.array([l[0] for l in limits])
        ub = np.array([l[1] for l in limits])
        lb[lb == -np.inf] = -np.pi * 2
        ub[ub == np.inf] = np.pi * 2
        return lb, ub

    def get_sphere_specs(self) -> List[SphereAttachmentSpec]:
        # the below reads the all the sphere specs from the yaml file
        # but if you want to use the sphere specs for the specific links
        # you can override this method
        d = self.conf_dict["collision_spheres"]
        sphere_specs = []
        for parent_link_name, vals in d.items():
            ignore_collision = vals["ignore_collision"]
            spheres_d = vals["spheres"]
            radii = []
            positions = []
            for info in spheres_d:
                vals = np.array(info)
                center, r = vals[:3], vals[3]
                radii.append(r)
                positions.append(center)
            radii = np.array(radii)
            positions = np.array(positions).transpose()
            spec = SphereAttachmentSpec(
                "yaml-colsphere", parent_link_name, positions, radii, ignore_collision
            )
            sphere_specs.append(spec)
        return sphere_specs

    def create_fixed_zaxis_const(self, link_name: str) -> FixedZAxisCst:
        return FixedZAxisCst(self.get_kin(), self.control_joint_names, self.with_base, link_name)

    def create_collision_const(self, self_collision: bool = True) -> SphereCollisionCst:
        sphere_specs = self.get_sphere_specs()

        if ("self_collision_pairs" not in self.conf_dict) and len(
            self.self_body_collision_primitives()
        ) == 0:
            self_collision = False

        if self_collision:
            self_collision_pairs = self.conf_dict["self_collision_pairs"]
            sksdf = UnionSDF([p.sdf for p in self.self_body_collision_primitives()])
            cppsdf = sksdf_to_cppsdf(sksdf, create_bvh=False)
        else:
            self_collision_pairs = []
            cppsdf = None
        with open(self.urdf_path, "r") as f:
            f.read()
        kin = self.get_kin()
        cst = SphereCollisionCst(
            kin,
            self.control_joint_names,
            self.with_base,
            sphere_specs,
            self_collision_pairs,
            cppsdf,
        )
        return cst

    def create_config_point_const(self, q: np.ndarray) -> ConfigPointCst:
        return ConfigPointCst(self.get_kin(), self.control_joint_names, self.with_base, q)

    def crate_pose_const_from_coords(
        self, link_names: List[str], link_poses: List[Coordinates], rot_types: List[RotType]
    ) -> LinkPoseCst:
        print(
            "\033[93m[Warning] crate_pose_const_from_coords is deprecated, please use create_pose_const_from_coords.\033[0m"
        )
        return self.create_pose_const_from_coords(link_names, link_poses, rot_types)

    def create_pose_const_from_coords(
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
        return self.create_pose_const(link_names, pose_list)

    def create_pose_const(self, link_names: List[str], link_poses: List[np.ndarray]) -> LinkPoseCst:
        return LinkPoseCst(
            self.get_kin(), self.control_joint_names, self.with_base, link_names, link_poses
        )

    def create_relative_pose_const(
        self, link_name1: str, link_name2: str, relative_position: np.ndarray
    ) -> RelativePoseCst:
        return RelativePoseCst(
            self.get_kin(),
            self.control_joint_names,
            self.with_base,
            link_name1,
            link_name2,
            relative_position,
        )

    def create_attached_box_collision_const(
        self, box: Box, parent_link_name: str, relative_position: np.ndarray, n_grid: int = 6
    ) -> SphereCollisionCst:
        extent = box._extents
        grid = np.meshgrid(
            np.linspace(-0.5 * extent[0], 0.5 * extent[0], n_grid),
            np.linspace(-0.5 * extent[1], 0.5 * extent[1], n_grid),
            np.linspace(-0.5 * extent[2], 0.5 * extent[2], n_grid),
        )
        grid_points = np.stack([g.flatten() for g in grid], axis=1)
        grid_points = box.transform_vector(grid_points)
        grid_points = grid_points[box.sdf(grid_points) > -1e-2]

        points_from_center = grid_points - box.worldpos()
        points_from_link = points_from_center + relative_position
        radii = np.zeros(len(points_from_link))
        spec = SphereAttachmentSpec(
            "grid-box", parent_link_name, points_from_link.transpose(), radii, False
        )

        cst = SphereCollisionCst(
            self.get_kin(),
            self.control_joint_names,
            self.with_base,
            [spec],
            [],
            None,
        )
        return cst


class FetchSpec(RobotSpec):
    def __init__(self, with_base: bool = False):
        # set with_base = True only in testing
        p = Path(__file__).parent / "conf" / "fetch.yaml"
        super().__init__(p, with_base)
        if not self.urdf_path.exists():
            from skrobot.models.fetch import Fetch  # noqa

            Fetch()

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
    def q_reset_pose() -> np.ndarray:
        return np.array([0.0, 1.31999949, 1.40000015, -0.20000077, 1.71999929, 0.0, 1.6600001, 0.0])

    @staticmethod
    def get_reachable_box() -> Tuple[np.ndarray, np.ndarray]:
        lb_reachable = np.array([-0.60046263, -1.08329689, -0.18025853])
        ub_reachable = np.array([1.10785484, 1.08329689, 2.12170273])
        return lb_reachable, ub_reachable


class PandaSpec(RobotSpec):
    def __init__(self):
        p = Path(__file__).parent / "conf" / "panda.yaml"
        super().__init__(p, with_base=False)
        if not self.urdf_path.exists():
            from skrobot.models.panda import Panda  # noqa

            Panda()

    @property
    def control_joint_names(self) -> List[str]:
        return self.conf_dict["control_joint_names"]

    def get_robot_model(self) -> RobotModel:
        return load_urdf_model_using_cache(self.urdf_path)

    def self_body_collision_primitives(self) -> Sequence[Union[Box, Sphere, Cylinder]]:
        return []


class JaxonSpec(RobotSpec):
    gripper_collision: bool

    def __init__(self, gripper_collision: bool = True):
        p = Path(__file__).parent / "conf" / "jaxon.yaml"
        super().__init__(p, with_base=True)  # jaxon is free-floating, so with_base=True
        self.gripper_collision = gripper_collision

        if not self.urdf_path.exists():
            from robot_descriptions.jaxon_description import URDF_PATH  # noqa

    @property
    def control_joint_names(self) -> List[str]:
        return self.conf_dict["control_joint_names"]

    def get_sphere_specs(self) -> List[SphereAttachmentSpec]:
        # because legs are on the ground, we don't need to consider the spheres on the legs
        specs = super().get_sphere_specs()
        filtered = []

        ignore_list = ["RLEG_LINK5", "LLEG_LINK5"]
        if not self.gripper_collision:
            ignore_list.extend(
                [
                    "RARM_FINGER0",
                    "RARM_FINGER1",
                    "RARM_LINK7",
                    "LARM_FINGER0",
                    "LARM_FINGER1",
                    "LARM_LINK7",
                ]
            )

        for spec in specs:
            if spec.parent_link_name in ignore_list:
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

    def create_default_com_const(
        self, total_force_on_arm: Optional[float] = None
    ) -> ComInPolytopeCst:
        com_box = BoxSDF([0.25, 0.5, 0.0], Pose(np.array([0, 0, 0]), np.eye(3)))

        specs = []
        if total_force_on_arm is not None:
            specs.append(AppliedForceSpec("RARM_LINK7", 0.5 * total_force_on_arm))
            specs.append(AppliedForceSpec("LARM_LINK7", 0.5 * total_force_on_arm))

        return ComInPolytopeCst(
            self.get_kin(), self.control_joint_names, self.with_base, com_box, specs
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
        q_reset = np.array([d[joint] for joint in self.control_joint_names])
        base_pose = np.array([0, 0, 1.0, 0, 0, 0])
        return np.hstack([q_reset, base_pose])

    def angle_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        joint_lb, joint_ub = super().angle_bounds()
        base_lb = np.array([-1.0, -1.0, 0.0, -1.0, -1.0, -1.0])
        base_ub = np.array([2.0, 1.0, 3.0, 1.0, 1.0, 1.0])
        return np.hstack([joint_lb, base_lb]), np.hstack([joint_ub, base_ub])
