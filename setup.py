try:
    from skbuild import setup
except ImportError:
    raise Exception

setup(
    name="plainmp",
    version="0.0.2",
    description="GPL release version of (legacy) plainmp",
    author="Hirokazu Ishida",
    license="GPLv3",
    install_requires=["numpy", "scipy", "scikit-robot", "pyyaml", "ompl-thin"],
    packages=["plainmp"],
    package_dir={"": "python"},
    package_data={"plainmp": ["*.pyi", "conf/*.yaml"]},
    cmake_install_dir="python/plainmp/",
)
