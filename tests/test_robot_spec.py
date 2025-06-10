from typing import Type

import pytest

from plainmp.robot_spec import (
    FetchSpec,
    JaxonSpec,
    PandaSpec,
    PR2BaseOnlySpec,
    PR2DualarmSpec,
    PR2RarmSpec,
    RobotSpec,
)

spec_classes = [
    FetchSpec,
    PandaSpec,
    PR2RarmSpec,
    PR2BaseOnlySpec,
    PR2DualarmSpec,
    JaxonSpec,
]


@pytest.mark.parametrize("spec_t", spec_classes)
def test_robot_specs_fixed_id(spec_t: Type[RobotSpec]):
    spec = spec_t(use_fixed_spec_id=True)
    kin = spec.get_kin()

    spec2 = spec_t(use_fixed_spec_id=True)
    kin2 = spec2.get_kin()
    assert id(kin) == id(kin2)


@pytest.mark.parametrize("spec_t", spec_classes)
def test_robot_specs_rand_id(spec_t: Type[RobotSpec]):
    spec = spec_t(use_fixed_spec_id=False)
    kin = spec.get_kin()

    spec2 = spec_t(use_fixed_spec_id=False)
    kin2 = spec2.get_kin()
    assert id(kin) != id(kin2)


@pytest.mark.parametrize("spec_t", spec_classes)
def test_robot_specs_custom_id(spec_t: Type[RobotSpec]):
    spec = spec_t(spec_id="hoge")
    kin = spec.get_kin()

    spec2 = spec_t(spec_id="hoge")
    kin2 = spec2.get_kin()

    spec3 = spec_t(spec_id="fuga")
    kin3 = spec3.get_kin()

    assert id(kin) == id(kin2)
    assert id(kin) != id(kin3)
