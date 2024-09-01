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
    version="0.0.12",
    description="experimental",
    author="Hirokazu Ishida",
    license="PolyForm NonCommercial 1.0.0",
    install_requires=["numpy", "scipy", "scikit-robot", "pyyaml", "ompl-thin", "robot_descriptions", "osqp"],
    packages=packages,
    package_dir={"": "python"},
    package_data={"plainmp": ["*.pyi", "conf/*.yaml"]},
    cmake_install_dir="python/plainmp/",
)
