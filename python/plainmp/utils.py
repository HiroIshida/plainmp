from typing import List

import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import rpy_matrix
from skrobot.model.robot_model import RobotModel
from skrobot.sdf import BoxSDF, CylinderSDF, SphereSDF, UnionSDF

import plainmp.psdf as psdf


def sksdf_to_cppsdf(sksdf, create_bvh: bool = False) -> psdf.SDFBase:
    if isinstance(sksdf, BoxSDF):
        pose = psdf.Pose(sksdf.worldpos(), sksdf.worldrot())
        sdf = psdf.BoxSDF(sksdf._width, pose)
    elif isinstance(sksdf, CylinderSDF):
        pose = psdf.Pose(sksdf.worldpos(), sksdf.worldrot())
        sdf = psdf.CylinderSDF(sksdf._radius, sksdf._height, pose)
    elif isinstance(sksdf, SphereSDF):
        pose = psdf.Pose(sksdf.worldpos(), sksdf.worldrot())
        sdf = psdf.SphereSDF(sksdf._radius, pose)
    elif isinstance(sksdf, UnionSDF):
        for s in sksdf.sdf_list:
            if not isinstance(s, (BoxSDF, CylinderSDF)):
                raise ValueError("Unsupported SDF type")
        cpp_sdf_list = [sksdf_to_cppsdf(s, create_bvh) for s in sksdf.sdf_list]
        sdf = psdf.UnionSDF(cpp_sdf_list, create_bvh)
    else:
        raise ValueError(f"Unsupported SDF type {type(sksdf)}")
    return sdf


def set_robot_state(
    robot_model: RobotModel,
    joint_names: List[str],
    angles: np.ndarray,
    floating_base: bool = False,
) -> None:
    if floating_base:
        assert len(joint_names) + 6 == len(angles)
        av_joint, av_base = angles[:-6], angles[-6:]
        xyz, rpy = av_base[:3], av_base[3:]
        co = Coordinates(pos=xyz, rot=rpy_matrix(*np.flip(rpy)))
        robot_model.newcoords(co)
    else:
        assert len(joint_names) == len(angles)
        av_joint = angles

    for joint_name, angle in zip(joint_names, av_joint):
        robot_model.__dict__[joint_name].joint_angle(angle)
