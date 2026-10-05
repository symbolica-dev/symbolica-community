"""Showcase physics definitions through the installed HEPKit owners."""
import importlib.util
from pathlib import Path
import sys

import pytest
from symbolica import E, S
from symbolica.community import hepkit as hep

pytestmark = pytest.mark.skipif(
    not hasattr(hep, "fastsecdec"), reason="requires experimental-fastsecdec owner setup"
)


PATH = Path(__file__).parents[1] / "examples/hep/fastsecdec_inputs.py"
SPEC = importlib.util.spec_from_file_location("fastsecdec_inputs", PATH)
inputs = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inputs
SPEC.loader.exec_module(inputs)


@pytest.mark.parametrize("builder,loops,propagators", [
    (inputs.massive_triangle, 1, 3),
    (inputs.massless_box, 1, 4),
    (inputs.rank_two_box, 1, 4),
    (inputs.coupled_sunset, 2, 3),
])
def test_native_owners_and_integral_conventions(builder, loops, propagators):
    value = builder()
    arguments = value.integral_arguments()
    assert arguments["diagram"] is value.diagram
    assert arguments["kinematics"] is value.kinematics
    assert isinstance(value.model, hep.Model)
    assert value.diagram.loop_count == loops
    assert len(value.diagram.internal_edges) == propagators
    assert arguments["measure_multiplier"] == E("1")
    assert value.dimension == 4 - 2 * value.regulator
    assert value.scalar_numerator() != E("0")
    assert value.diagram.overall_factor_expression() == E("1")


def test_massive_triangle_preserves_native_mass_and_spacelike_scale():
    value = inputs.massive_triangle(mass=2, s=-3)
    p = hep.Kinematics.external_momentum()
    assert value.kinematics.scalar_product(p(1), p(2)) == E("-3/2")
    assert list(value.scalar_values.values()) == [E("2")]
    assert value.scalar_numerator() == E("1")
    assert value.model.parameter("mt").value == 2 + 0j


def test_box_numerator_toggle_preserves_the_kinematics():
    scalar = inputs.massless_box(s12=-2, s23=-3)
    tensor = inputs.rank_two_box(s12=-2, s23=-3)
    k, p = hep.Kinematics.loop_momentum(), hep.Kinematics.external_momentum()
    kin = tensor.kinematics
    expected = kin.scalar_product(k(0), k(0)) + 3 * kin.scalar_product(k(0), p(0)) * kin.scalar_product(k(0), p(1))
    assert (tensor.scalar_numerator() - expected).expand() == E("0")
    assert scalar.scalar_numerator() == E("1")
    assert kin.scalar_product(p(0), p(1)) == E("-1")
    assert kin.scalar_product(p(1), p(2)) == E("-3/2")
    assert kin.scalar_product(p(0), p(2)) == E("5/2")


def test_noninteger_form_values_are_exact_binary_rationals_for_native_families():
    value = inputs.massive_triangle(mass=0.1, s=-0.2)
    mass = next(iter(value.scalar_values))
    assert value.scalar_values[mass] == E("3602879701896397/36028797018963968")
    p = hep.Kinematics.external_momentum()
    assert value.kinematics.scalar_product(p(1), p(2)) == -value.scalar_values[mass]
    assert value.model.parameter("mt").value == complex(0.1)
    original = value.diagram.propagator_family(kinematics=value.kinematics)
    bound = hep.IntegralFamily(
        original.loop_momenta,
        original.external_momenta,
        [term.replace(mass, value.scalar_values[mass]) for term in original.denominators],
        kinematics=value.kinematics,
    )
    u, f = bound.symanzik(list(S("showcase_input_test::x1", "showcase_input_test::x2", "showcase_input_test::x3")))
    assert u != E("0") and f != E("0")


def test_sunset_preserves_the_mixed_loop_numerator_and_coupled_propagator():
    value = inputs.coupled_sunset(s=-3)
    k, p = hep.Kinematics.loop_momentum(), hep.Kinematics.external_momentum()
    kin = value.kinematics
    expected = kin.scalar_product(k(0), k(1)) + 2 * kin.scalar_product(k(0), p(0))
    assert (value.scalar_numerator() - expected).expand() == E("0")
    family = value.diagram.propagator_family(kinematics=kin)
    third = k(0) + k(1) + p(0)
    mass = next(iter(value.scalar_values))
    denominators = [term.replace(mass, E("0")) for term in family.denominators]
    assert (denominators[2] - kin.scalar_product(third, third)).expand() == E("0")
    assert value.max_order == 1


@pytest.mark.parametrize("builder,arguments", [
    (inputs.massive_triangle, {"mass": 0}),
    (inputs.massless_box, {"s12": 1}),
    (inputs.rank_two_box, {"s23": 0}),
    (inputs.coupled_sunset, {"s": float("nan")}),
])
def test_showcase_controls_reject_outside_the_declared_euclidean_inputs(builder, arguments):
    with pytest.raises(ValueError):
        builder(**arguments)
