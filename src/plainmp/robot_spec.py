import copy
import subprocess
import time
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
    xyzw2wxyz,
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
    LinkPositionBoundCst,
    RelativePoseCst,
    SphereAttachmentSpec,
    SphereCollisionCst,
)
from plainmp.kinematics import BaseType, KinematicModel
from plainmp.psdf import BoxSDF, Pose, UnionSDF
from plainmp.utils import primitive_to_plainmp_sdf, set_robot_state

_loaded_urdf_models: Dict[str, URDF] = {}
_loaded_yamls: Dict[str, Dict] = {}  # loading yaml is quite slow
N_MAX_CACHE = 200
_loaded_kin: "OrderedDict[str, KinematicModel]" = OrderedDict()
_created_collision_csts: Dict[Tuple[str, bool], SphereCollisionCst] = {}


def load_urdf_model_using_cache(file_path: Path, with_mesh: bool = False, deepcopy: bool = True):
    file_path = file_path.expanduser()
    assert file_path.exists()
    key = f"{file_path}_{with_mesh}"
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
    """Base class for robot specifications.

    This class provides a unified interface for robot kinematics, constraints,
    and configuration management. It handles URDF loading, kinematic model creation,
    and constraint generation for motion planning and inverse kinematics.

    Parameters
    ----------
    conf_file : Path
        Path to the robot configuration YAML file.
    base_type : BaseType, default=BaseType.FIXED
        Type of robot base (FIXED, PLANAR, or FLOATING).
    use_fixed_spec_id : bool, default=False
        Controls the generation of the instance's identifier.
        If True, the class name is used as a fixed identifier, allowing resources
        to be shared across multiple instances. If False, a unique identifier is generated.
    spec_id : str, optional
        Custom identifier for this robot specification instance.
    """

    def __init__(
        self,
        conf_file: Path,
        base_type: BaseType = BaseType.FIXED,
        use_fixed_spec_id: bool = False,
        spec_id: Optional[str] = None,
    ):
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
        self.base_type = base_type

        if spec_id is not None:
            self._spec_id = self.__class__.__name__ + str(spec_id)
        else:
            if use_fixed_spec_id:
                self._spec_id = self.__class__.__name__ + str(base_type)
            else:
                self._spec_id = str(uuid.uuid4())

    def urdf_path_override(self) -> Optional[Path]:
        # Override this method to specify a custom URDF path.
        # This is useful for testing or when the URDF is not in the default location.
        return None

    def get_kin(self) -> KinematicModel:
        # The kinematic chain is shared among the same robot spec.
        # This sharing mechanism is important to instantiate an composite
        # constraint, albeit its usage complexity.
        self_id = self._spec_id
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
        pose_vec = self.coordinates_to_pose_vec(robot_model.worldcoords(), RotType.XYZW)
        self.get_kin().set_base_pose(pose_vec)

    def reflect_kin_to_skrobot_model(self, robot_model: RobotModel) -> None:
        # inverse of reflect_skrobot_model_to_kin
        joint_names = robot_model.joint_names
        kin = self.get_kin()
        joint_ids = kin.get_joint_ids(joint_names)
        angles = kin.get_joint_positions(joint_ids)
        robot_model.angle_vector(angles)

        pose_vec = kin.get_base_pose()
        position, quat = pose_vec[:3], pose_vec[3:]
        co = Coordinates(position, xyzw2wxyz(quat))
        robot_model.newcoords(co)

    def set_skrobot_model_state(self, robot_model: RobotModel, q: np.ndarray) -> None:
        set_robot_state(robot_model, self.control_joint_names, q, base_type=self.base_type)

    def extract_skrobot_model_q(
        self, robot_model: RobotModel, base_type: BaseType = BaseType.FIXED
    ) -> np.ndarray:
        angles = []
        for joint in self.control_joint_names:
            angle = robot_model.__dict__[joint].joint_angle()
            angles.append(angle)

        if base_type == BaseType.FLOATING:
            # TODO: should we return in rpz or quaternion?
            assert False, "Not implemented yet"

        if base_type == BaseType.PLANAR:
            x, y, _ = robot_model.worldpos()
            yaw, _, _ = rpy_angle(robot_model.worldrot())[0]
            angles.extend([x, y, yaw])

        return np.array(angles)

    def get_robot_model(self, with_mesh: bool = False, deepcopy: bool = True) -> RobotModel:
        # deepcopy is recommended to avoid the side effect of the skrobot model's internal
        # but it is not efficient. If performance is critical, you may want to set deepcopy=False
        model = load_urdf_model_using_cache(self.urdf_path, with_mesh=with_mesh, deepcopy=deepcopy)
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
        path = self.urdf_path_override()  # if you want to override the conf path
        if path is None:
            path = Path(self.conf_dict["urdf_path"]).expanduser()  # from conf file
        if path.suffix == ".xacro":
            xacro_installed = subprocess.run("which xacro", shell=True, check=False)
            if xacro_installed.returncode != 0:
                raise RuntimeError("xacro is not found.")
            urdf_path = path.with_suffix(".urdf")
            if not urdf_path.exists():
                subprocess.run(f"xacro {path} -o {urdf_path}", shell=True, check=True)
            return urdf_path
        else:
            return path

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

        if self.base_type != BaseType.FIXED and len(self_collision_primitives) > 0:
            err_msg = "self_body_collision_primitives are only used for the fixed base robot.\n"
            err_msg += "Please define collision spheres instead of primitives for\n"
            err_msg += "the robot with the floating/planar base."
            # TODO: maybe auto-generate spheres from the primitives??
            raise ValueError(err_msg)

        return self_collision_primitives

    def angle_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get joint angle bounds for the robot.

        Returns the lower and upper bounds for all controlled joints.
        Infinite bounds are clipped to [-2π, 2π] for numerical stability.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Lower bounds and upper bounds for joint angles.

        Examples
        --------
        >>> spec = FetchSpec()
        >>> lb, ub = spec.angle_bounds()
        >>> print(f"Joint bounds: [{lb[0]:.2f}, {ub[0]:.2f}] rad")
        """
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
            if self.base_type == BaseType.FIXED:
                only_self_collision = parent_link_name in only_self_colliision_links
            else:
                # If base is moving, we need to consider all the spheres for collision
                # against the environment.
                only_self_collision = False

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
        return FixedZAxisCst(self.get_kin(), self.control_joint_names, self.base_type, link_name)

    def get_self_collision_pairs(self) -> List[List[str]]:
        pairs = []
        if "self_collision_pairs" in self.conf_dict:  # for backward compatibility
            pairs.extend(self.conf_dict["self_collision_pairs"])

        if "self_collision_group_pairs" in self.conf_dict:
            for group_pair in self.conf_dict["self_collision_group_pairs"]:
                group1 = group_pair["group1"]
                group2 = group_pair["group2"]
                for link1 in group1:
                    for link2 in group2:
                        pairs.append([link1, link2])

        # NOTE: We intentionally do not remove duplicate collision pairs automatically.
        # Automatically reordering (e.g., sorting) the pairs could alter the user-specified
        # check order, which is critical for optimal performance due to significant variations
        # in collision check speed. Therefore, we require users to manually define
        # the desired order and raise an error if duplicates are detected.
        pair_set = set()
        for pair in pairs:
            key = tuple(sorted(pair))
            if key in pair_set:
                raise ValueError(f"Duplicated self collision pair: {key}")
            pair_set.add(key)
        return pairs

    def create_collision_const(
        self,
        self_collision: bool = True,
        attachments: Sequence[SphereAttachmentSpec] = tuple(),
        use_cache: bool = True,
        reorder_spheres: bool = True,
    ) -> SphereCollisionCst:
        """Create a collision constraint for motion planning and IK.

        This method creates a sphere-based collision constraint using the collision
        spheres defined in the robot configuration file. It supports both self-collision
        checking and environment collision checking.

        Parameters
        ----------
        self_collision : bool, default=True
            Whether to include self-collision constraints.
        attachments : Sequence[SphereAttachmentSpec], default=()
            Additional sphere attachments (e.g., for carried objects).
        use_cache : bool, default=True
            Whether to load cached collision constraints for performance.
        reorder_spheres : bool, default=True
            Whether to reorder spheres for optimal collision checking performance.

        Returns
        -------
        SphereCollisionCst
            Collision constraint object that can be used with solvers.

        Examples
        --------
        >>> spec = FetchSpec()
        >>> collision_cst = spec.create_collision_const(self_collision=True)
        >>> collision_cst.set_sdf(table_sdf)  # Add environment obstacles
        """

        key = (self._spec_id, self_collision)
        if len(attachments) == 0:  # caching is supported only if attachments are not given
            if use_cache and key in _created_collision_csts:
                return _created_collision_csts[key]

        sphere_specs = self.parse_sphere_specs()
        if len(attachments) > 0:  # merge the attachments
            for att in attachments:
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
            self_collision_pairs = self.get_self_collision_pairs()

            # Add self_body_collision_primitives only if exist
            if len(self.self_body_collision_primitives()) > 0:
                sdfs = [primitive_to_plainmp_sdf(p) for p in self.self_body_collision_primitives()]
                robot_anchor_sdf = UnionSDF(sdfs)

        kin = self.get_kin()
        cst = SphereCollisionCst(
            kin,
            self.control_joint_names,
            self.base_type,
            sphere_specs,
            self_collision_pairs,
            robot_anchor_sdf,
            reorder_spheres,
        )
        if use_cache and len(attachments) == 0:
            _created_collision_csts[key] = cst
        return cst

    def create_config_point_const(self, q: np.ndarray) -> ConfigPointCst:
        return ConfigPointCst(self.get_kin(), self.control_joint_names, self.base_type, q)

    def create_pose_const_from_coords(
        self,
        link_names: List[str],
        link_poses: List[Coordinates],
        rot_types: Optional[List[RotType]] = None,
    ) -> LinkPoseCst:
        if rot_types is None:
            rot_types = [RotType.XYZW] * len(link_poses)
        pose_list = []
        for co, rt in zip(link_poses, rot_types):
            pose = self.coordinates_to_pose_vec(co, rt)
            pose_list.append(pose)
        return self.create_pose_const(link_names, pose_list)

    def create_pose_const(self, link_names: List[str], link_poses: List[np.ndarray]) -> LinkPoseCst:
        """Create pose constraints for specified robot links.

        This constraint allows you to specify target poses for multiple links in a kinematic chain.
        Each pose can be specified in one of three formats:
        - 3D position only (x, y, z)
        - 6D pose with position and RPY angles (x, y, z, roll, pitch, yaw)
        - 7D pose with position and quaternion (x, y, z, qx, qy, qz, qw)

        Parameters
        ----------
        link_names : List[str]
            Names of the robot links to constrain.
        link_poses : List[np.ndarray]
            Target poses for each link. Format depends on array length (3, 6, or 7 elements).

        Returns
        -------
        LinkPoseCst
            Pose constraint object for use in IK or motion planning.

        Examples
        --------
        >>> spec = FetchSpec()
        >>> pose_cst = spec.create_pose_const(["gripper_link"], [[0.7, 0.2, 0.95, 0, 0, 0]])
        """
        assert len(link_names) == len(link_poses)
        return LinkPoseCst(
            self.get_kin(), self.control_joint_names, self.base_type, link_names, link_poses
        )

    def create_relative_pose_const(
        self, link_name1: str, link_name2: str, relative_position: np.ndarray
    ) -> RelativePoseCst:
        return RelativePoseCst(
            self.get_kin(),
            self.control_joint_names,
            self.base_type,
            link_name1,
            link_name2,
            relative_position,
        )

    def create_position_bound_const(
        self, link_name: str, axis: int, lb: Optional[float], ub: Optional[float]
    ) -> LinkPositionBoundCst:
        return LinkPositionBoundCst(
            self.get_kin(), self.control_joint_names, self.base_type, link_name, axis, lb, ub
        )

    def create_attached_box_collision_const(
        self,
        box: Box,
        parent_link_name: str,
        relative_position: np.ndarray,
        n_grid: int = 6,
        reorder_spheres: bool = True,
    ) -> SphereCollisionCst:
        # Deprecated. Use create_collision_const with attachments instead.
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
            self.base_type,
            [spec],
            [],
            None,
            reorder_spheres,
        )
        return cst

    @staticmethod
    def coordinates_to_pose_vec(co: Coordinates, rot_type: RotType) -> np.ndarray:
        """convert skrobot coordinates to the pose vector
        Args:
            co: skrobot coordinates
        Returns:
            pose vector: [x, y, z, orientation...]
        where orientation depends on the rot_type
        """
        pos = co.worldpos()
        if rot_type == RotType.RPY:
            ypr = rpy_angle(co.rotation)[0]
            rpy = [ypr[2], ypr[1], ypr[0]]
            return np.hstack([pos, rpy])
        elif rot_type == RotType.XYZW:
            quat_wxyz = matrix2quaternion(co.rotation)
            return np.hstack([pos, wxyz2xyzw(quat_wxyz)])
        else:
            return pos

    def debug_visualize_collision_spheres(self, skmodel: RobotModel) -> None:
        from skrobot.viewers import PyrenderViewer

        cst = self.create_collision_const(self_collision=False)
        lb, ub = self.angle_bounds()
        q = self.extract_skrobot_model_q(skmodel)
        cst.update_kintree(q, True)
        self.set_skrobot_model_state(skmodel, q)
        self.reflect_skrobot_model_to_kin(skmodel)

        v = PyrenderViewer()
        v.add(skmodel)
        for center, r in cst.get_all_spheres():
            sk_sphere = Sphere(r, pos=center, color=[0, 255, 0, 100])
            v.add(sk_sphere)
        v.show()
        print("==> Press [q] to close window")
        while not v.has_exit:
            time.sleep(0.1)
            v.redraw()


class FetchSpec(RobotSpec):
    """Robot specification for the Fetch mobile manipulator.

    Provides pre-configured settings and convenience methods for the Fetch robot,
    including gripper pose constraints and workspace bounds.

    Parameters
    ----------
    base_type : BaseType, default=BaseType.FIXED
        Type of robot base. Fetch typically uses FIXED base for manipulation tasks.
    use_fixed_spec_id : bool, default=False
        Whether to use a fixed identifier for resource sharing.
    spec_id : str, optional
        Custom identifier for this robot specification instance.

    Examples
    --------
    >>> fs = FetchSpec()
    >>> gripper_cst = fs.create_gripper_pose_const([0.7, 0.2, 0.95, 0, 0, 0])
    >>> collision_cst = fs.create_collision_const()
    >>> lb, ub = fs.angle_bounds()
    """

    def __init__(
        self,
        base_type: BaseType = BaseType.FIXED,
        use_fixed_spec_id: bool = False,
        spec_id: Optional[str] = None,
    ):
        p = Path(__file__).parent / "conf" / "fetch.yaml"
        super().__init__(
            p, base_type=base_type, use_fixed_spec_id=use_fixed_spec_id, spec_id=spec_id
        )
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
    def __init__(
        self,
        base_type: BaseType = BaseType.FIXED,
        use_fixed_spec_id: bool = False,
        spec_id: Optional[str] = None,
    ):
        p = Path(__file__).parent / "conf" / self.get_yaml_file_name()
        super().__init__(
            p, base_type=base_type, use_fixed_spec_id=use_fixed_spec_id, spec_id=spec_id
        )
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


class PR2BaseOnlySpec(PR2SpecBase):
    def __init__(self, use_fixed_spec_id: bool = False, spec_id: Optional[str] = None):
        super().__init__(
            base_type=BaseType.PLANAR, use_fixed_spec_id=use_fixed_spec_id, spec_id=spec_id
        )

    def get_yaml_file_name(self) -> str:
        return "pr2_base_only.yaml"

    def create_base_pose_const(self, pose3d: np.ndarray) -> LinkPoseCst:
        x, y, yaw = pose3d
        target = np.array([x, y, 0.0, 0.0, 0.0, yaw])
        return self.create_pose_const(["base_footprint"], [target])


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
    def __init__(self, use_fixed_spec_id: bool = False, spec_id: Optional[str] = None):
        p = Path(__file__).parent / "conf" / "panda.yaml"
        super().__init__(
            p, base_type=BaseType.FIXED, use_fixed_spec_id=use_fixed_spec_id, spec_id=spec_id
        )
        if not self.urdf_path.exists():
            from skrobot.models.panda import Panda  # noqa

            Panda()


class JaxonSpec(RobotSpec):
    gripper_collision: bool

    def __init__(
        self,
        gripper_collision: bool = True,
        use_fixed_spec_id: bool = False,
        spec_id: Optional[str] = None,
    ):
        p = Path(__file__).parent / "conf" / "jaxon.yaml"
        super().__init__(
            p, base_type=BaseType.FLOATING, use_fixed_spec_id=use_fixed_spec_id, spec_id=spec_id
        )
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
        rleg = robot_model.rleg_end_coords.copy_worldcoords()
        lleg = robot_model.lleg_end_coords.copy_worldcoords()
        return self.create_pose_const(
            ["rleg_end_coords", "lleg_end_coords"],
            [
                self.coordinates_to_pose_vec(rleg, RotType.RPY),
                self.coordinates_to_pose_vec(lleg, RotType.RPY),
            ],
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
            self.get_kin(), self.control_joint_names, self.base_type, com_box, specs
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
