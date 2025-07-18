# plainmp  [![build & test](https://github.com/HiroIshida/plainmp/actions/workflows/build_and_test.yaml/badge.svg)](https://github.com/HiroIshida/plainmp/actions/workflows/build_and_test.yaml) [![format](https://github.com/HiroIshida/plainmp/actions/workflows/check_format.yaml/badge.svg)](https://github.com/HiroIshida/plainmp/actions/workflows/check_format.yaml) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14271046.svg)](https://doi.org/10.5281/zenodo.14271046) [![PyPI - Version](https://img.shields.io/pypi/v/plainmp)](https://pypi.org/project/plainmp)

![result](https://github.com/user-attachments/assets/fde49de1-f583-4f1d-933b-5853cea6bccc)

The project is licensed under the BSD 3 License (see [LICENSE](./LICENSE-BSD3)), except for the code in `cpp` directory which is licensed under the MPL2.0 (see [cpp/LICENSE-MPL2](cpp/LICENSE-MPL2)).

plainmp provides:
- Fast sampling-based motion planning (e.g., **less than 1ms** for moderate problems using RRTConnect)
- Collision-aware inverse kinematics (IK) solver
- Motion planning/IK for various models (e.g. movable base, dual-arm, object attachment)
- Flexible framework for defining various robot model and motion planning problems
- Collision checking for primitives (sphere/box/cylinder...) and/or point cloud vs. robot
- (Beta) Sampling-based constrained motion planning solver (e.g., whole-body humanoid)
- (Beta) SQP-based constrained motion planning (will be used as smoother for sampling-based planner)

Note that plainmp currently heavily relies on approximations of robot body by spheres.

The TODO list is
- Speed up IK by reimplementing current Python/C++ mix into pure C++
- Auto-generate collision spheres from URDF instead of manual sphere definitions

Related/depeding projects:
- plainmp is a rewrite of my previous projects [scikit-motionplan](https://github.com/HiroIshida/scikit-motionplan) and [tinyfk](https://github.com/HiroIshida/tinyfk) to achieve 100x speedup
- plainmp depends on [OMPL](https://github.com/ompl/ompl) (with unmerged PR of [ERTConnect](https://github.com/ompl/ompl/pull/783)) for SBMP algorithms
- plainmp deponds on [scikit-robt](https://github.com/iory/scikit-robot) framework for visualization and planning problem specifications
- [benchmark](https://github.com/HiroIshida/bench_plainmp_and_vamp) with [VAMP](https://github.com/KavrakiLab/vamp) (the world fastest motion planner as of 2024 to my knowledge) shows that
    - AMD Ryzen 7 7840HS (256-bit AVX) VAMP is faster (1.3x, 1.1x, 4.8x, 12.5x)
    - ARM Neoverse-N1 (128-bit NEON) both seems to be comparable (0.53x, 0.41x, 2.2x, 4.8x)
    - x-s are time ratio plainmp/VAMP for 4 different settings with resolution of 1/32

## Performance example
panda dual bars: median 0.17 ms | panda ceiled dual bars: median 0.65 ms | fetch table: median 0.62 ms


<img src="https://github.com/user-attachments/assets/9bcb776c-3e60-4715-9371-e54403b06abe" width="260" /> <img src="https://github.com/user-attachments/assets/b9ef3966-f638-46d6-8355-b1b40f536310" width="260" /> <img src="https://github.com/user-attachments/assets/d6bd4e28-70a8-45d3-8a75-f704b3734a36" width="260" />

<img src="https://github.com/user-attachments/assets/a9a0e2b0-85d6-4178-9fbf-0a57a16ebeae" width="260" /> <img src="https://github.com/user-attachments/assets/f8b61603-84bd-4e72-a348-1ab93ecb3b65" width="260" /> <img src="https://github.com/user-attachments/assets/bf529b32-74fa-4819-92d6-33d187f38870" width="260" />

\* resolution is isotropically set to 0.05 rad or 0.05 m with [box motion validator](./cpp/ompl/motion_validator.hpp)

\* On my laptop (AMD Ryzen 7 7840HS)

The plots are generated by the following commands:
```bash
python3 example/bench/panda_plan.py  # panda dual bars
python3 example/bench/panda_plan.py --difficult  # panda ceiled dual bars
python3 example/bench/fetch_plan.py  # fetch table
```

## installation and usage (Ubuntu/macOS)
On Ubuntu:
```bash
sudo apt install libeigen3-dev libboost-all-dev libompl-dev libspatialindex-dev freeglut3-dev libsuitesparse-dev libblas-dev liblapack-dev
pip install plainmp  # or build from source after git submodules update --init
```
On macOS:
```bash
brew install eigen boost ompl spatialindex suite-sparse openblas lapack
pip install plainmp  # or build from source after git submodules update --init
```

Then try examples in [example directory](./example) with `--visualize` option. Note that you may need to install the following for visualization:
```bash
pip uninstall -y pyrender && pip install git+https://github.com/mmatl/pyrender.git --no-cache-dir
```

## Troubleshooting
- Segmentation faults and other C++ runtime errors may occur when multiple OMPL versions are present - typically when installed via both from Ubuntu and ROS. To temporarily resolve this, disable ROS workspace sourcing in your shell or remove either OMPL installation.

## How to add a new robot model
**\* Feel free to open an issue and include your (public) URDF file/link! I might be able to create a custom sphere model for that.**
- (step 1) Prepare a URDF file. Note that [robot_description](https://github.com/robot-descriptions/robot_descriptions.py) package might be useful.
- (step 2) Implement a new class inheriting `RobotSpec` class in [src/plainmp/robot_spec.py](./src/plainmp/robot_spec.py).
- (step 3) Write yaml file defining urdf location/collision information/control joints/end effector in (see [example yaml files](./src/plainmp/conf/)).
- NOTE: In step 3, you need to manually define the collision spheres for the robot (This is actually tedious and takes an hour or so). For this purpose, a visualizer script like [this](./example/misc/panda_visualize_coll_spheres.py) might be helpful to check the collision spheres defined in the yaml file. The output of the this visualizer looks like figure below.
<img src="https://github.com/user-attachments/assets/e7f36c3a-5fc8-45ee-8583-f1c5f38bf561" width="400" />

## Note on motion validator of motion planning
We provides two types of motion validator type `box` and `euclidean`.
- The `box` type split the motion segment into waypoints by clipping the segment with the collision box. You need to specify box widths for each joint.
    - e.g. `bow_width = np.ones(7) / 20` for panda robot
- The `euclidean` type is instead split the segment by euclidean distance of `resolution` parameter.
    - e.g. `resolution = 1/20` for panda robot
- see [problem.py](./src/plainmp/problem.py) for how to specify the motion validator.
