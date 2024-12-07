import copy
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml
from skrobot.coordinates import CascadedCoords, Coordinates, Transform
from skrobot.coordinates.math import (
    matrix2quaternion,
    quaternion2matrix,
    rpy_angle,
    rpy_matrix,
    wxyz2xyzw,
)
from skrobot.model.primitives import Box, Cylinder, Sphere
from skrobot.model.robot_model import RobotModel
from skrobot.models.urdf import RobotModelFromURDF
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
from plainmp.psdf import BoxSDF, Pose, UnionSDF
from plainmp.utils import primitive_to_plainmp_sdf

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
            if "include" in self.conf_dict:
                for relative_path in self.conf_dict["include"]:
                    depend_file = conf_file.parent / relative_path
                    with open(depend_file, "r") as f:
                        depend_dict = yaml.safe_load(f)
                    assert (
                        len(set(self.conf_dict.keys()) & set(depend_dict.keys())) == 0
                    ), "conflict"
                    self.conf_dict.update(depend_dict)
                    assert "include" not in depend_dict, "recursive include is not supported yet"

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

        # For each of the custom link defined in the conf file
        # add it as a link to the kinematic chain if it is not present in the chain.
        if "custom_links" in self.conf_dict:
            for name in self.conf_dict["custom_links"].keys():
                try:
                    # Pass if end effector link is already in the kinematic chain.
                    _loaded_kin[self_id].get_link_ids([name])
                except ValueError:
                    # Add the unfound end effector link to the kinematic chain.
                    consider_rotation = True
                    _loaded_kin[self_id].add_new_link(
                        name,
                        self.conf_dict["custom_links"][name]["parent_link"],
                        np.array(self.conf_dict["custom_links"][name]["position"]),
                        np.array(self.conf_dict["custom_links"][name]["rpy"]),
                        consider_rotation,
                    )
        return _loaded_kin[self_id]

    def reflect_joint_positions(self, table: Dict[str, float]) -> None:
        # Some robot (e.g. PR2) has many joints but only a few joints are controlled.
        # for example, say we want to control only the arm joints and other joints
        # (e.g. torso or larm) are fixed.
        # In this case, the user is responsible to reflect the joint positions
        # other than the controlled joints to the kinematic model.
        # This is because the kinematic model only updates the controlled joints
        # in the planning process, while assuming the other joints are fixed.
        # Note that kin model is shared among all the constraints.
        kin = self.get_kin()
        ids = kin.get_joint_ids(list(table.keys()))
        kin.set_joint_positions(ids, list(table.values()))

    def reflect_skrobot_model_to_kin(self, robot_model: RobotModel) -> None:
        # same comment as reflect_joint_positions method applies here
        # this is the wrapper for the skrobot model. reflect the skrobot's joint angles to
        # the plainmp's kinematic model
        table = {}
        for jn in robot_model.joint_names:
            angle = robot_model.__dict__[jn].joint_angle()
            table[jn] = angle
        self.reflect_joint_positions(table)

    def reflect_kin_to_skrobot_model(self, robot_model: RobotModel) -> None:
        # inverse of reflect_skrobot_model_to_kin
        joint_names = robot_model.joint_names
        kin = self.get_kin()
        joint_ids = kin.get_joint_ids(joint_names)
        angles = kin.get_joint_positions(joint_ids)
        robot_model.angle_vector(angles)

    def get_robot_model(self, with_mesh: bool = False) -> RobotModel:
        model = load_urdf_model_using_cache(self.urdf_path, with_mesh=with_mesh)
        # Add custom links defined in conf to the robot model as CascadedCoords
        if "custom_links" in self.conf_dict:
            for name in self.conf_dict["custom_links"].keys():
                parent_link = self.conf_dict["custom_links"][name]["parent_link"]
                pos = np.array(self.conf_dict["custom_links"][name]["position"])
                rpy = np.array(self.conf_dict["custom_links"][name]["rpy"])
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
    def control_joint_names(self) -> List[str]:
        return self.conf_dict["control_joint_names"]

    def self_body_collision_primitives(self) -> Sequence[Union[Box, Sphere, Cylinder]]:
        # Self body collision primitives are the primitive shapes that are attached to the
        # robot's base. This is useful when you know that certain parts of the robot
        # are fixed and well approximated by primitive shapes rather than spheres.
        # Typical use case if for base or torso of the robot.
        # Because these primitives are fixed, they are used only for self collision
        self_collision_primitives = []
        if "self_body_collision_primitives" in self.conf_dict:
            for prim_dict in self.conf_dict["self_body_collision_primitives"]:
                if prim_dict["type"] == "box":
                    extents = prim_dict["extents"]
                    obj = Box(extents, face_colors=[255, 255, 255, 200])
                elif prim_dict["type"] == "sphere":
                    # TODO: not tested well
                    radius = prim_dict["radius"]
                    obj = Sphere(radius, color=[255, 255, 255, 200])
                elif prim_dict["type"] == "cylinder":
                    radius = prim_dict["radius"]
                    height = prim_dict["height"]
                    obj = Cylinder(radius, height, face_colors=[255, 255, 255, 200])
                else:
                    raise ValueError("Invalid primitive type")
                position = np.array(prim_dict["position"])
                roll, pitch, yaw = np.array(prim_dict["rotation"])
                rotmat = rpy_matrix(yaw, pitch, roll)
                co = Coordinates(position, rotmat)
                obj.newcoords(co)
                self_collision_primitives.append(obj)
        return self_collision_primitives

    def angle_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        kin = self.get_kin()
        joint_ids = kin.get_joint_ids(self.control_joint_names)
        limits = kin.get_joint_position_limits(joint_ids)
        lb = np.array([l[0] for l in limits])
        ub = np.array([l[1] for l in limits])
        lb[lb == -np.inf] = -np.pi * 2
        ub[ub == np.inf] = np.pi * 2
        return lb, ub

    def transform_points_wrt_link(
        self, points: np.ndarray, link_name: str, q: np.ndarray
    ) -> np.ndarray:
        """Transform points from the world frame to the link frame
        The computation is done by the kinematic model
        Args:
            points: 3D points in the world frame
            link_name: link name
            q: joint angles
        """
        kin = self.get_kin()
        # TODO: this is not efficient. If performance is critical, we need to do it in c++ side
        kin.set_joint_positions(kin.get_joint_ids(self.control_joint_names), q)
        x, y, z, qx, qy, qz, qw = kin.debug_get_link_pose(link_name)
        mat = quaternion2matrix([qw, qx, qy, qz])
        tf_link_to_world = Transform(np.array([x, y, z]), mat)
        return tf_link_to_world.inverse_transformation().transform_vector(points)

    def parse_sphere_specs(self) -> List[SphereAttachmentSpec]:
        # the below reads the all the sphere specs from the yaml file
        # but if you want to use the sphere specs for the specific links
        # you can override this method

        if "only_self_collision_links" in self.conf_dict:
            # spheres of these links are only used for self collision
            # not for the collision with the environment
            only_self_colliision_links = set(self.conf_dict["only_self_collision_links"])
        else:
            only_self_colliision_links = set()

        d = self.conf_dict["collision_spheres"]
        sphere_specs = []
        for parent_link_name, vals in d.items():
            only_self_collision = parent_link_name in only_self_colliision_links
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
            spec = SphereAttachmentSpec(parent_link_name, positions, radii, only_self_collision)
            sphere_specs.append(spec)

        parent_link_name_set = set([spec.parent_link_name for spec in sphere_specs])
        assert len(parent_link_name_set) == len(sphere_specs), "duplicated parent link name"
        return sphere_specs

    def create_fixed_zaxis_const(self, link_name: str) -> FixedZAxisCst:
        return FixedZAxisCst(self.get_kin(), self.control_joint_names, self.with_base, link_name)

    def create_collision_const(
        self, self_collision: bool = True, attachements: Sequence[SphereAttachmentSpec] = tuple()
    ) -> SphereCollisionCst:
        """Create a collision constraint from the conf file
        Args:
            self_collision: If True, self collision is considered
            attachement: Extra attachment specs to be considered
        """
        sphere_specs = self.parse_sphere_specs()

        if len(attachements) > 0:  # merge the attachements
            for att in attachements:
                parent_link = att.parent_link_name
                merged = False
                for spec in sphere_specs:
                    if spec.parent_link_name == parent_link:
                        spec.relative_positions = np.hstack(
                            [spec.relative_positions, att.relative_positions]
                        )
                        spec.radii = np.hstack([spec.radii, att.radii])
                        merged = True
                        break
                if not merged:
                    sphere_specs.append(att)

        self_collision_pairs = []
        robot_anchor_sdf = None
        if self_collision:
            # Add self_collision_pairs only if exist in the conf file
            if "self_collision_pairs" in self.conf_dict:
                self_collision_pairs = self.conf_dict["self_collision_pairs"]

            # Add self_body_collision_primitives only if exist
            if len(self.self_body_collision_primitives()) > 0:
                sdfs = [primitive_to_plainmp_sdf(p) for p in self.self_body_collision_primitives()]
                robot_anchor_sdf = UnionSDF(sdfs)

        with open(self.urdf_path, "r") as f:
            f.read()
        kin = self.get_kin()
        cst = SphereCollisionCst(
            kin,
            self.control_joint_names,
            self.with_base,
            sphere_specs,
            self_collision_pairs,
            robot_anchor_sdf,
        )
        return cst

    def create_config_point_const(self, q: np.ndarray) -> ConfigPointCst:
        return ConfigPointCst(self.get_kin(), self.control_joint_names, self.with_base, q)

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
        """This constraint allows you to specify target poses for multiple links in a kinematic chain.
        Each pose can be specified in one of three formats:
        - 3D position only (x, y, z)
        - 6D pose with position and RPY angles (x, y, z, roll, pitch, yaw)
        - 7D pose with position and quaternion (x, y, z, qx, qy, qz, qw)
        """
        assert len(link_names) == len(link_poses)
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
        # Deprecated. Use create_collision_const with attachements instead.
        extent = box._extents
        grid = np.meshgrid(
            np.linspace(-0.5 * extent[0], 0.5 * extent[0], n_grid),
            np.linspace(-0.5 * extent[1], 0.5 * extent[1], n_grid),
            np.linspace(-0.5 * extent[2], 0.5 * extent[2], n_grid),
        )
        grid_points = np.stack([g.flatten() for g in grid], axis=1)
        grid_points = box.transform_vector(grid_points)
        sdf = primitive_to_plainmp_sdf(box)
        grid_points = grid_points[sdf.evaluate_batch(grid_points.T) > -1e-2]

        points_from_center = grid_points - box.worldpos()
        points_from_link = points_from_center + relative_position
        radii = np.zeros(len(points_from_link))
        spec = SphereAttachmentSpec(parent_link_name, points_from_link.transpose(), radii, False)

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


class PR2SpecBase(RobotSpec):
    def __init__(self):
        p = Path(__file__).parent / "conf" / self.get_yaml_file_name()
        super().__init__(p, with_base=False)
        if not self.urdf_path.exists():
            from skrobot.models.pr2 import PR2  # noqa

            PR2()  # this downloads the PR2 urdf into the cache

    @property
    def default_joint_positions(self) -> Dict[str, float]:
        return copy.deepcopy(self.conf_dict["default_joint_positions"])

    @property
    def q_default(self) -> np.ndarray:
        return np.array([self.default_joint_positions[jn] for jn in self.control_joint_names])

    @abstractmethod
    def get_yaml_file_name(self) -> str:
        pass


class PR2RarmSpec(PR2SpecBase):
    def get_yaml_file_name(self) -> str:
        return "pr2_rarm.yaml"

    def create_gripper_pose_const(self, link_pose: np.ndarray) -> LinkPoseCst:
        return self.create_pose_const(["r_gripper_tool_frame"], [link_pose])


class PR2LarmSpec(PR2SpecBase):
    def get_yaml_file_name(self) -> str:
        return "pr2_larm.yaml"

    def create_gripper_pose_const(self, link_pose: np.ndarray) -> LinkPoseCst:
        return self.create_pose_const(["l_gripper_tool_frame"], [link_pose])


class PR2DualarmSpec(PR2SpecBase):
    def get_yaml_file_name(self) -> str:
        return "pr2_dualarm.yaml"

    def create_gripper_pose_const(self, link_poses: Tuple[np.ndarray, np.ndarray]) -> LinkPoseCst:
        return self.create_pose_const(
            ["r_gripper_tool_frame", "l_gripper_tool_frame"], list(link_poses)
        )


class PandaSpec(RobotSpec):
    def __init__(self):
        p = Path(__file__).parent / "conf" / "panda.yaml"
        super().__init__(p, with_base=False)
        if not self.urdf_path.exists():
            from skrobot.models.panda import Panda  # noqa

            Panda()


class JaxonSpec(RobotSpec):
    gripper_collision: bool

    def __init__(self, gripper_collision: bool = True):
        p = Path(__file__).parent / "conf" / "jaxon.yaml"
        super().__init__(p, with_base=True)  # jaxon is free-floating, so with_base=True
        self.gripper_collision = gripper_collision

        if not self.urdf_path.exists():
            from robot_descriptions.jaxon_description import URDF_PATH  # noqa

    def parse_sphere_specs(self) -> List[SphereAttachmentSpec]:
        # because legs are on the ground, we don't need to consider the spheres on the legs
        specs = super().parse_sphere_specs()
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
