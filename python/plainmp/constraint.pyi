from typing import Sequence, Tuple

import numpy as np
from fused.psdf import SDFBase
from scipy.sparse import csc_matrix

class ConstraintBase:
    def update_kintree(self, q: np.ndarray) -> None: ...
    def evaluate(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...

class EqConstraintBase(ConstraintBase): ...

class IneqConstraintBase(ConstraintBase):
    def is_valid(self, q: np.ndarray) -> bool: ...

class ConfigPointCst(EqConstraintBase): ...
class LinkPoseCst(EqConstraintBase): ...
class RelativePoseCst(EqConstraintBase): ...

class SphereAttachmentSpec:
    name: str
    parent_link_name: str
    relative_position: np.ndarray
    radius: float
    ignore_collision: bool

class SphereCollisionCst(IneqConstraintBase):
    def set_sdf(self, sdf: SDFBase) -> None: ...

class ComInPolytopeCst(IneqConstraintBase): ...

class AppliedForceSpec:
    link_name: str
    force: float

    def __init__(self, link_name: str, force: np.ndarray) -> None: ...

# NOTE: actually EqCompositeCst is not a subclass of EqConstraintBase but has same interface
class EqCompositeCst(EqConstraintBase):
    def __init___(self, csts: Sequence[EqConstraintBase]) -> None: ...

# NOTE: actually IneqCompositeCst is not a subclass of IneqConstraintBase but has same interface
class IneqCompositeCst(IneqConstraintBase):
    def __init___(self, csts: Sequence[IneqConstraintBase]) -> None: ...

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
    def add_motion_step_box_constraint(self, q: np.ndarray) -> None: ...
    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, csc_matrix]: ...
    def x_dim(self) -> int: ...
    def c_dim(self) -> int: ...
    def determine_sparsity_pattern(self) -> None:
        """must be called before evaluate"""
