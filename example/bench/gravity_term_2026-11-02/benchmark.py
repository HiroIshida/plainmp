"""Measure the q-only gravity APIs against Pinocchio, with changing q.

Run with .venv/bin/python. Use --extension-dir for a saved baseline binary,
and --pinocchio-site if Pinocchio is installed outside the virtualenv.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import sysconfig
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def timed(function, arguments, passes):
    start = time.perf_counter_ns()
    for _ in range(passes):
        for args in arguments:
            function(*args)
    return time.perf_counter_ns() - start


def measure(pin, spec, options, rng):
    kin = spec.get_kin()
    names = spec.control_joint_names
    ids = kin.get_joint_ids(names)
    qs = [q.copy() for q in rng.uniform(*spec.angle_bounds(), (256, len(ids)))]
    xml = spec.urdf_path.read_text()
    full = pin.buildModelFromXML(xml)
    locked = [i for i in range(1, full.njoints) if full.names[i] not in names]
    reduced = pin.buildReducedModel(full, locked, pin.neutral(full))
    assert reduced.nv == len(ids)
    cases = [("plainmp_gravity_q", kin.get_gravity_term2, [(ids, q) for q in qs])]
    errors = {}
    for label, model in [("reduced", reduced), ("full", full)]:
        model.gravity.linear = np.array([0.0, 0.0, -9.80665])
        data = model.createData()
        pin_qs = []
        indices = [model.joints[model.getJointId(name)].idx_v for name in names]
        for q in qs:
            pq = pin.neutral(model)
            for name, value in zip(names, q):
                joint = model.joints[model.getJointId(name)]
                if joint.nq == 1:
                    pq[joint.idx_q] = value
                else:
                    assert joint.nq == 2
                    pq[joint.idx_q : joint.idx_q + 2] = [np.cos(value), np.sin(value)]
            pin_qs.append(pq)
        cases.append(
            (
                "pinocchio_gravity_" + label,
                pin.computeGeneralizedGravity,
                [(model, data, q) for q in pin_qs],
            )
        )
        error = 0.0
        for q, pq in zip(qs, pin_qs):
            actual = kin.get_gravity_term2(ids, q)
            expected = -pin.computeGeneralizedGravity(model, data, pq)[indices]
            np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
            error = max(error, float(np.max(np.abs(actual - expected))))
        errors[label] = error

    # Measure calls after mass changes, excluding the setter. Depending on the
    # implementation this may rebuild gravity preparation. Restore the mass
    # before all steady-state measurements.
    mass_name = next(
        link.attrib["name"]
        for link in ET.fromstring(xml).findall("link")
        if link.find("inertial") is not None
    )
    mass_id = kin.get_link_ids([mass_name])[0]
    original_mass = kin.get_link_mass(mass_id)
    rebuild_ns = []
    for i in range(128):
        kin.set_link_mass(mass_id, original_mass + (0.1 if i % 2 == 0 else 0.0))
        start = time.perf_counter_ns()
        kin.get_gravity_term2(ids, qs[0])
        rebuild_ns.append(time.perf_counter_ns() - start)
    kin.set_link_mass(mass_id, original_mass)

    native_input = options.output.with_name(type(spec).__name__ + ".txt")
    with native_input.open("w") as stream:
        stream.write(str(spec.urdf_path) + "\n")
        stream.write(f"{len(ids)} {len(qs)}\n" + " ".join(names) + "\n")
        for q in qs:
            stream.write(" ".join(map(str, q)) + "\n")
    for _, function, args in cases:
        timed(function, args, 16)
    raw = {name: [] for name, _, _ in cases}
    orders = []
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(options.rounds):
            order = rng.permutation(len(cases)).tolist()
            orders.append([cases[i][0] for i in order])
            for i in order:
                name, function, arguments = cases[i]
                raw[name].append(timed(function, arguments, options.passes))
    finally:
        if enabled:
            gc.enable()
    result = {}
    for name, samples in raw.items():
        us = np.asarray(samples) / (len(qs) * options.passes) / 1000
        result[name] = dict(
            median_us=float(np.median(us)),
            q25_us=float(np.quantile(us, 0.25)),
            q75_us=float(np.quantile(us, 0.75)),
            batch_ns=samples,
        )
        print(type(spec).__name__, name, f"{np.median(us):.3f} us/call", flush=True)
    return dict(
        robot=type(spec).__name__,
        joint_names=names,
        urdf_path=str(spec.urdf_path),
        urdf_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        qs=[q.tolist() for q in qs],
        pin_full_nv=full.nv,
        pin_reduced_nv=reduced.nv,
        correctness_error=errors,
        after_mass_update_call_median_us=float(np.median(rebuild_ns) / 1000),
        after_mass_update_call_ns=rebuild_ns,
        calls_per_batch=len(qs) * options.passes,
        round_orders=orders,
        results=result,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pinocchio-site", type=Path)
    parser.add_argument("--extension-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--passes", type=int, default=256)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument(
        "--robots", nargs="+", choices=["fetch", "openarm", "pr2"], default=["fetch", "openarm"]
    )
    options = parser.parse_args()
    if options.pinocchio_site:
        sys.path.append(str(options.pinocchio_site))
    import pinocchio as pin

    import plainmp

    if options.extension_dir:
        binary = options.extension_dir / ("_plainmp" + sysconfig.get_config_var("EXT_SUFFIX"))
        spec = importlib.util.spec_from_file_location("plainmp._plainmp", binary)
        module = importlib.util.module_from_spec(spec)
        sys.modules["plainmp._plainmp"] = module
        spec.loader.exec_module(module)
        plainmp._plainmp = module
    import plainmp._plainmp
    from plainmp.robot_spec import FetchSpec, OpenArmV10RarmSpec, PR2RarmSpec

    root = Path(__file__).resolve().parents[3]
    extension = Path(plainmp._plainmp.__file__).resolve()
    expected = options.extension_dir.resolve() if options.extension_dir else root / "build/native"
    assert extension.parent == expected
    assert Path(plainmp.__file__).resolve() == root / "src/plainmp/__init__.py"
    os.sched_setaffinity(0, {options.cpu})
    metadata = dict(
        timestamp=datetime.now(timezone.utc).isoformat(),
        python=sys.version,
        numpy=np.__version__,
        pinocchio=pin.__version__,
        cpu=command("lscpu"),
        affinity=sorted(os.sched_getaffinity(0)),
        head=command("git", "rev-parse", "HEAD"),
        branch=command("git", "branch", "--show-current"),
        extension=str(extension),
        extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
        source_diff=None if options.extension_dir else command("git", "diff", "--", "cpp/plainmp"),
        build_flags=(root / "build/native/CMakeFiles/_plainmp.dir/flags.make").read_text(),
        pinocchio_build="Existing 2.7.0 wheel; compiler flags not verified",
        options={k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()},
        method="Median of 15 batch means by default; Python calls and returned array costs included. "
        "256 changing configurations, accurate=True. Model setup, Pinocchio q conversion and output "
        "reordering excluded. Same interpreter, pinned CPU, sequential shuffled batches, GC disabled. "
        "The Pinocchio reduced model is built before timing. "
        "Pinocchio gravity sign is reversed only in correctness checks, outside timing.",
    )
    rng = np.random.default_rng(20261102)
    specs = {"fetch": FetchSpec, "openarm": OpenArmV10RarmSpec, "pr2": PR2RarmSpec}
    robots = [measure(pin, specs[name](), options, rng) for name in options.robots]
    options.output.write_text(json.dumps(dict(metadata=metadata, robots=robots), indent=2) + "\n")
    print("Saved", options.output, flush=True)


if __name__ == "__main__":
    main()
