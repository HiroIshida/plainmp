from typing import List, Sequence, Union

import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import rpy_matrix
from skrobot.model.primitives import Box, Cylinder, Sphere
from skrobot.model.robot_model import RobotModel

import plainmp.psdf as psdf
from plainmp.kinematics import BaseType


def box_to_grid_poitns(box: Box, N_points: Union[int, Sequence[int]]) -> np.ndarray:
    if isinstance(N_points, int):
        N_points = [N_points, N_points, N_points]
    x, y, z = np.array(box.extents) * 0.5
    xlin = np.linspace(-x, x, N_points[0])
    ylin = np.linspace(-y, y, N_points[1])
    zlin = np.linspace(-z, z, N_points[2])
    grid_points_local = np.array(np.meshgrid(xlin, ylin, zlin)).T.reshape(-1, 3)
    grid_points_world = box.transform_vector(grid_points_local)
    return grid_points_world


def primitive_to_plainmp_sdf(shape: Union[Sphere, Box, Cylinder]) -> psdf.SDFBase:
    if isinstance(shape, Sphere):
        pose = psdf.Pose(shape.worldpos(), shape.worldrot())
        sdf = psdf.SphereSDF(shape.radius, pose)
    elif isinstance(shape, Box):
        pose = psdf.Pose(shape.worldpos(), shape.worldrot())
        sdf = psdf.BoxSDF(shape.extents, pose)
    elif isinstance(shape, Cylinder):
        pose = psdf.Pose(shape.worldpos(), shape.worldrot())
        sdf = psdf.CylinderSDF(shape.radius, shape.height, pose)
    else:
        raise ValueError(f"Unsupported shape type {type(shape)}")
    return sdf


def set_robot_state(
    robot_model: RobotModel,
    joint_names: List[str],
    angles: np.ndarray,
    base_type: BaseType = BaseType.FIXED,
) -> None:
    if base_type == BaseType.FIXED:
        assert len(joint_names) == len(angles)
        av_joint = angles
    elif base_type == BaseType.FLOATING:
        assert len(joint_names) + 6 == len(angles)
        av_joint, av_base = angles[:-6], angles[-6:]
        xyz, rpy = av_base[:3], av_base[3:]
        co = Coordinates(pos=xyz, rot=rpy_matrix(*np.flip(rpy)))
        robot_model.newcoords(co)
    elif base_type == BaseType.PLANAR:
        assert len(joint_names) + 3 == len(angles)
        av_joint, av_base = angles[:-3], angles[-3:]
        xyz = [av_base[0], av_base[1], 0.0]
        rpy = [0.0, 0.0, av_base[2]]
        co = Coordinates(pos=xyz, rot=rpy_matrix(*np.flip(rpy)))
        robot_model.newcoords(co)
    else:
        assert False

    for joint_name, angle in zip(joint_names, av_joint):
        robot_model.__dict__[joint_name].joint_angle(angle)
