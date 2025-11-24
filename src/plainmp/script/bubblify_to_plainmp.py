from dataclasses import dataclass
from typing import Any

import numpy as np
import tyro
import yaml


class FlowSeq(list):
    pass


def represent_flowseq(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


yaml.SafeDumper.add_representer(FlowSeq, represent_flowseq)


@dataclass(frozen=True)
class CollisionSphere:
    center: np.ndarray
    radius: float

    @staticmethod
    def from_dict(d: Any) -> "CollisionSphere":
        c = np.asarray(d["center"], dtype=float)
        if c.shape != (3,):
            raise ValueError(f"center must be length-3, got {c}")
        return CollisionSphere(center=c, radius=float(d["radius"]))


def main(path: str):
    # load bubblify style collision spheres
    with open(path, "r", encoding="utf-8") as f:
        root = yaml.safe_load(f)
    cs_raw = root.get("collision_spheres", {})
    link_to_spheres_table = {
        link: ([] if spheres is None else [CollisionSphere.from_dict(s) for s in spheres])
        for link, spheres in cs_raw.items()
    }

    # print out plainmp style collision spheres
    d = {"collision_spheres": {}}
    for link, spheres in link_to_spheres_table.items():
        d["collision_spheres"][link] = {}
        d["collision_spheres"][link]["spheres"] = []
        for sphere in spheres:
            flattened = np.concatenate([sphere.center, [sphere.radius]])
            d["collision_spheres"][link]["spheres"].append(FlowSeq(flattened.tolist()))

    yaml_str = yaml.safe_dump(d, sort_keys=False)
    print(yaml_str)


if __name__ == "__main__":
    tyro.cli(main)
