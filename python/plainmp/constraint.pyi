from typing import Sequence, Tuple

import numpy as np
from fused.psdf import SDFBase

class ConstraintBase:
    def update_kintree(self, q: np.ndarray) -> None: ...
    def evaluate(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...

class EqConstraintBase(ConstraintBase): ...

class IneqConstraintBase(ConstraintBase):
    def is_valid(self, q: np.ndarray) -> bool: ...

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
