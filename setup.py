import sys
try:
    from skbuild import setup
    from setuptools import find_packages
except ImportError:
    raise Exception

is_editable_install = '--editable' in sys.argv or 'develop' in sys.argv
if is_editable_install:  # I don't know why, but this is necessary for editable install
    packages = ["plainmp"]
else:
    packages = find_packages(where="python")

setup(
    name="plainmp",
    version="0.2.3",
    description="Very fast motion planning for articulated robot, through a bit of premature-optimization (C++ core with Python bindings) *less than 1ms for moderate problems",
    author="Hirokazu Ishida",
    install_requires=["numpy", "scipy", "scikit-robot>=0.0.44", "pyyaml", "robot_descriptions", "osqp<1.0.0"],
    packages=packages,
    package_dir={"": "python"},
    package_data={"plainmp": ["*.pyi", "conf/*.yaml", "conf/pr2_common/*.yaml"]},
    license="MPL2.0 for C++ core and BSD3 for Python bindings",
    license_files=["cpp/LICENSE-MPL2", "LICENSE-BSD3"],
    cmake_install_dir="python/plainmp/",
)
