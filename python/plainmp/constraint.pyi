from typing import Sequence, Tuple

import numpy as np
from scipy.sparse import csc_matrix

from plainmp.psdf import SDFBase

class ConstraintBase:
    def update_kintree(self, q: np.ndarray, high_accuracy: bool) -> None:
        """Update kinematic tree with given joint values
        Args:
            q: joint values
            high_accuracy: if False, sin/cos computation in FK is approximated
                via Taylor expansion
        """
        ...
    def evaluate(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...

class EqConstraintBase(ConstraintBase): ...

class IneqConstraintBase(ConstraintBase):
    def is_valid(self, q: np.ndarray) -> bool: ...

class ConfigPointCst(EqConstraintBase):
    def __init__(self, position: np.ndarray) -> None: ...

class LinkPoseCst(EqConstraintBase): ...
class RelativePoseCst(EqConstraintBase): ...
class FixedZAxisCst(EqConstraintBase): ...

class SphereAttachmentSpec:
    postfix: str
    parent_link_name: str
    relative_positions: np.ndarray
    radii: np.ndarray
    ignore_collision: bool

class SphereCollisionCst(IneqConstraintBase):
    def set_sdf(self, sdf: SDFBase) -> None: ...
    def get_sdf(self) -> SDFBase: ...
    def get_all_spheres(self) -> List[Tuple[np.ndarray, float]]: ...
    def get_group_spheres(self) -> List[Tuple[np.ndarray, float]]: ...

class ComInPolytopeCst(IneqConstraintBase): ...

class AppliedForceSpec:
    link_name: str
    force: float

    def __init__(self, link_name: str, force: np.ndarray) -> None: ...

# NOTE: actually EqCompositeCst is not a subclass of EqConstraintBase but has same interface
class EqCompositeCst(EqConstraintBase):
    def __init___(self, csts: Sequence[EqConstraintBase]) -> None: ...
    @property
    def constraints(self) -> Sequence[EqConstraintBase]: ...

# NOTE: actually IneqCompositeCst is not a subclass of IneqConstraintBase but has same interface
class IneqCompositeCst(IneqConstraintBase):
    def __init___(self, csts: Sequence[IneqConstraintBase]) -> None: ...
    @property
    def constraints(self) -> Sequence[EqConstraintBase]: ...

class SequentialCst:
    """A class to handle sequential constraints
    This class is intended to be used for optimization (or NLP) based motion planning
    where jacobian wrt trajectory value is needed
    """

    def __init__(self, T: int) -> None: ...
    def add_globally(self, cst: ConstraintBase) -> None:
        """Add a constraint to be evaluated globally t \in {0 .. T-1}"""
    def add_at(self, cst: ConstraintBase, t: int) -> None:
        """Add a constraint to be evaluated at time t"""
    def add_fixed_point_at(self, q_fix: np.ndarray, t: int) -> None: ...
    def add_motion_step_box_constraint(self, msbox: np.ndarray) -> None: ...
    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, csc_matrix]: ...
    def x_dim(self) -> int: ...
    def cst_dim(self) -> int: ...
    def finalize(self) -> None:
        """must be called before evaluate"""
