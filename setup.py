try:
    from skbuild import setup
    from setuptools import find_packages
except ImportError:
    raise Exception

setup(
    name="plainmp",
    version="0.0.3",
    description="experimental",
    author="Hirokazu Ishida",
    license="PolyForm NonCommercial 1.0.0",
    install_requires=["numpy", "scipy", "scikit-robot", "pyyaml", "ompl-thin", "robot_descriptions", "osqp"],
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    package_data={"plainmp": ["*.pyi", "conf/*.yaml"]},
    cmake_install_dir="python/plainmp/",
)
