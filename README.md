# plainmp  [![build & test](https://github.com/HiroIshida/plainmp/actions/workflows/build_and_test.yaml/badge.svg)](https://github.com/HiroIshida/plainmp/actions/workflows/build_and_test.yaml) [![format](https://github.com/HiroIshida/plainmp/actions/workflows/check_format.yaml/badge.svg)](https://github.com/HiroIshida/plainmp/actions/workflows/check_format.yaml)

plainmp provides:
- Very fast motion planning (**less than 1ms** for moderate problems)
- Collision-aware inverse kinematics 
- (Beta) Sampling-based constrained motion planning solver (e.g., whole-body humanoid)
- (Beta) SQP-based constrained motion planning (will be used as smoother for sampling-based planner)

plainmp
- is written in C++ (see [cpp directory](./cpp)) and wrapped by pybind11 (see [python directory](./python)).
- is a rewrite of my previous projects [scikit-motionplan](https://github.com/HiroIshida/scikit-motionplan) and [tinyfk](https://github.com/HiroIshida/tinyfk)
- relies on spheres approximations of robot body
- depends on [OMPL](https://github.com/ompl/ompl) (with unmerged PR of [ERTConnect](https://github.com/ompl/ompl/pull/783)) for SBMP algorithms
- deponds on [scikit-robt](https://github.com/iory/scikit-robot) framework for visualization and planning problem specifications

## Performance example

## installation
```bash
sudo apt-get install libspatialindex-dev freeglut3-dev libsuitesparse-dev libblas-dev liblapack-dev  # for scikit-robot
sudo apt install libeigen3-dev libboost-all-dev  # plainmp dependencies
pip install scikit-build
pip install plainmp
```
