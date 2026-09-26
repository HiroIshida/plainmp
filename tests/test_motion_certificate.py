"""Integration tests use an inline URDF and need no downloaded robot assets."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import plainmp.ompl_solver as ompl_solver
from plainmp.constraint import SphereAttachmentSpec, SphereCollisionCst
from plainmp.kinematics import BaseType, KinematicModel
from plainmp.problem import Problem
from plainmp.psdf import Pose, SphereSDF

ROBOT = """
<robot name="xy">
  <link name="base"/><link name="x_link"/><link name="tip"/>
  <joint name="x" type="prismatic">
    <parent link="base"/><child link="x_link"/><axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="y" type="prismatic">
    <parent link="x_link"/><child link="tip"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def create_problem(validator_type):
    model = KinematicModel(ROBOT)
    spheres = SphereAttachmentSpec("tip", np.zeros((3, 1)), np.array([0.05]), False)
    cst = SphereCollisionCst(model, ["x", "y"], BaseType.FIXED, [spheres], [], None, False)
    cst.set_sdf(SphereSDF(0.2, Pose(np.zeros(3), np.eye(3))))
    resolution = 0.04 if validator_type == "euclidean" else np.array([0.04, 0.04])
    return model, Problem(
        np.array([-0.8, 0.0]),
        -np.ones(2),
        np.ones(2),
        np.array([0.8, 0.0]),
        cst,
        None,
        resolution,
        validator_type,
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_runtime_flag_reaches_planner_and_simplifier(monkeypatch, enabled):
    captured = []
    original = ompl_solver.ValidatorConfig

    def make_config():
        config = original()
        captured.append(config)
        return config

    monkeypatch.setattr(ompl_solver, "ValidatorConfig", make_config)
    _, problem = create_problem("euclidean")
    config = ompl_solver.OMPLSolverConfig(enable_interval_pruning=enabled)
    result = ompl_solver.OMPLSolver(config).solve(problem)
    assert result.success
    ompl_solver.simplify_path(
        result.traj,
        problem.lb,
        problem.ub,
        problem.global_ineq_const,
        problem.resolution,
        problem.validator_type,
        refine_seq=[ompl_solver.RefineType.SHORTCUT],
        enable_interval_pruning=enabled,
    )
    assert len(captured) == 2
    assert all(c.enable_interval_pruning == enabled for c in captured)


def plan_records(enabled):
    ompl_solver.set_log_level_none()
    ompl_solver.set_random_seed(67867967)
    records = []
    for validator_type in ("euclidean", "box"):
        for budget in (3, 10000):
            for refine in ([], [ompl_solver.RefineType.SHORTCUT]):
                for _ in range(10):
                    model, problem = create_problem(validator_type)
                    config = ompl_solver.OMPLSolverConfig(
                        n_max_call=budget, refine_seq=refine, enable_interval_pruning=enabled
                    )
                    result = ompl_solver.OMPLSolver(config).solve(problem)
                    final = model.get_joint_positions(model.get_joint_ids(["x", "y"]))
                    records.append(
                        {
                            "success": result.success,
                            "calls": result.n_call,
                            "path": result.traj.numpy().tolist() if result.success else None,
                            "final_joints": final,
                        }
                    )
    return records


def test_runtime_on_off_preserves_plans():
    # Separate processes give OMPL identical initial RNG state. This compares
    # settings of the same extension, not separately compiled ON/OFF binaries.
    results = []
    for enabled in (False, True):
        output = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), str(int(enabled))],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        results.append(json.loads(output.stdout))
    assert any(record["success"] for record in results[0])
    assert any(not record["success"] for record in results[0])
    assert results[0] == results[1]


if __name__ == "__main__":
    print(json.dumps(plan_records(bool(int(sys.argv[1])))))
