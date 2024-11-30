# plainmp  [![build & test](https://github.com/HiroIshida/plainmp/actions/workflows/build_and_test.yaml/badge.svg)](https://github.com/HiroIshida/plainmp/actions/workflows/build_and_test.yaml) [![format](https://github.com/HiroIshida/plainmp/actions/workflows/check_format.yaml/badge.svg)](https://github.com/HiroIshida/plainmp/actions/workflows/check_format.yaml)
plainmp provides:
- Very fast motion planning (**less than 1ms** for moderate problems)
- Collision-aware inverse kinematics (IK) solver
- Collision checking for primitives (sphere/box/cylinder...) and/or point cloud vs. robot
- (Beta) Sampling-based constrained motion planning solver (e.g., whole-body humanoid)
- (Beta) SQP-based constrained motion planning (will be used as smoother for sampling-based planner)

The implementation
- is written in C++ (see [cpp directory](./cpp)) and wrapped by pybind11 (see [python directory](./python)).
- is a rewrite of my previous projects [scikit-motionplan](https://github.com/HiroIshida/scikit-motionplan) and [tinyfk](https://github.com/HiroIshida/tinyfk)
- relies on spheres approximations of robot body
- depends on [OMPL](https://github.com/ompl/ompl) (with unmerged PR of [ERTConnect](https://github.com/ompl/ompl/pull/783)) for SBMP algorithms
- deponds on [scikit-robt](https://github.com/iory/scikit-robot) framework for visualization and planning problem specifications

## Performance example
panda dual bars: median 0.16 ms | panda ceiled dual bars: median 0.61 ms | fetch table: median 0.57 ms
* resolution is isotropically set to 0.05 rad or 0.05 m with [box motion validator](./cpp/ompl/motion_validator.hpp)

<img src="https://github.com/user-attachments/assets/9bcb776c-3e60-4715-9371-e54403b06abe" width="260" /> <img src="https://github.com/user-attachments/assets/b9ef3966-f638-46d6-8355-b1b40f536310" width="260" /> <img src="https://github.com/user-attachments/assets/d6bd4e28-70a8-45d3-8a75-f704b3734a36" width="260" />

<img src="https://github.com/user-attachments/assets/a9a0e2b0-85d6-4178-9fbf-0a57a16ebeae" width="260" /> <img src="https://github.com/user-attachments/assets/f8b61603-84bd-4e72-a348-1ab93ecb3b65" width="260" /> <img src="https://github.com/user-attachments/assets/bf529b32-74fa-4819-92d6-33d187f38870" width="260" />

The plots are generated by the following commands:
```bash
python3 example/bench/panda_plan.py  # panda dual bars
python3 example/bench/panda_plan.py --difficult  # panda ceiled dual bars
python3 example/bench/fetch_plan.py  # fetch table
```

## installation and usage
```bash
sudo apt-get install libspatialindex-dev freeglut3-dev libsuitesparse-dev libblas-dev liblapack-dev  # for scikit-robot
sudo apt install libeigen3-dev libboost-all-dev  # plainmp dependencies
pip install scikit-build
pip install plainmp
```
Then try examples in [example directory](./example) with `--visualize` option. Note that you may need to install the following for visualization:
```bash
pip uninstall -y pyrender && pip install git+https://github.com/mmatl/pyrender.git --no-cache-dir
```

## How to add a new robot model
- (step 1) Prepare a URDF file. Note that [robot_description](https://github.com/robot-descriptions/robot_descriptions.py) package might be useful.
- (step 2) Implement a new class inheriting `RobotSpec` class in [python/plainmp/robot_spec.py](./python/plainmp/robot_spec.py).
- (step 3) Write yaml file defining urdf location/collision information/control joints/end effector in (see [example yaml files](./python/plainmp/conf/)).
- NOTE: In step 3, you need to manually define the collision spheres for the robot (This is actually tedious and takes an hour or so). For this purpose, a visualizer script like [this](./example/misc/panda_visualize_coll_spheres.py) might be helpful to check the collision spheres defined in the yaml file. The output of the this visualizer looks like figure below.
- TODO: Adding automatic collision sphere generation from urdf (tried but not satisfactory yet. Manually defined spheres are more accurate)
<img src="https://github.com/user-attachments/assets/e7f36c3a-5fc8-45ee-8583-f1c5f38bf561" width="400" />

## Note on motion validator of motion planning
We provides two types of motion validator type `box` and `euclidean`.
- The `box` type split the motion segment into waypoints by clipping the segment with the collision box. You need to specify box widths for each joint.
    - e.g. `resolution = np.ones(7) / 20` for panda robot
- The `euclidean` type is instead split the segment by euclidean distance of `resolution` parameter.
    - e.g. `resolution = 1/20` for panda robot
- see [problem.py](./python/plainmp/problem.py) for how to specify the motion validator.
