"""One-loop reduction and master evaluation in the host's Symbolica kernel."""

import importlib
import math
import signal
import subprocess
import sys

import pytest

from symbolica import E, Expression, Replacement, S
from symbolica.community import hep
from symbolica.community.hep import oneloop


def evaluate_coefficients(reduction, parameters, values, mu_squared=None):
    coefficients = oneloop.reduction_coefficients(reduction, mu_squared=mu_squared)
    assert len(coefficients) == 3
    assert all(isinstance(coefficient, Expression) for coefficient in coefficients)
    constants = dict(zip(parameters, values, strict=True))
    return [complex(coefficient.evaluate(constants)) for coefficient in coefficients]


def scalar_family(masses, invariants, dimension=None):
    """Build ordinary Feynkit denominators with independently named offsets."""
    dimension = S("shared_family_test::D") if dimension is None else dimension
    loop = S("shared_family_test::ell")
    external = [S(f"shared_family_test::offset_{i}") for i in range(1, len(masses))]
    kinematics = hep.Kinematics(dimension, momenta=[loop, *external])
    pairs = dict(zip(
        ((i, j) for i in range(len(masses)) for j in range(i + 1, len(masses))),
        invariants, strict=True,
    ))
    for i, left in enumerate(external, 1):
        kinematics = kinematics.with_scalar_product(left, left, pairs[0, i])
        for j, right in enumerate(external[i:], i + 1):
            kinematics = kinematics.with_scalar_product(
                left, right, (pairs[0, i] + pairs[0, j] - pairs[i, j]) / 2,
            )
    return hep.IntegralFamily(
        [loop], external,
        [kinematics.scalar_product(loop + offset, loop + offset) - mass
         for offset, mass in zip([E("0"), *external], masses, strict=True)],
        kinematics=kinematics,
    )


def test_namespace_exposes_reduction_without_duplicate_family_classes():
    assert importlib.import_module("symbolica.community.hep.oneloop") is oneloop
    for name in ("Reduction", "MasterIntegral"):
        assert getattr(oneloop, name).__module__ == "symbolica.community.hep.oneloop"
    assert not hasattr(oneloop, "Propagator")
    assert not hasattr(oneloop, "IntegralFamily")
    assert callable(oneloop.reduce)
    assert oneloop.EXPRESSION_INTEROP


@pytest.mark.parametrize("name", ["A0", "B0", "dB0", "C0", "D0"])
def test_exported_master_primitives_accept_symbolic_arguments_and_carry_native_hooks(name):
    primitive = getattr(oneloop, name)
    assert callable(primitive)
    assert not oneloop.is_initialized()
    mass, invariant, scale = S("exports_test::m2", "exports_test::s", "exports_test::mu2")
    arguments = {
        "A0": [mass + 1, scale],
        "B0": [invariant, mass + 1, mass + 1, scale],
        "dB0": [invariant, mass + 1, mass + 1, scale],
        "C0": [invariant, 2 * invariant, 3 * invariant, *([mass + 1] * 3), scale],
        "D0": [*(i * invariant for i in range(1, 7)), *([mass + 1] * 4), scale],
    }[name]
    master = primitive(*arguments)
    assert isinstance(master, Expression)
    # Parser identity is intentional here: the constructed call must use the
    # registered master head, with the same namespace and callbacks.
    assert master == E(f"oneloopmaster::{name}")(*arguments)
    coefficients = oneloop.master_coefficients(master)
    assert coefficients == [primitive(tag, *arguments) for tag in (0, -1, -2)]
    point = {mass: 1, invariant: -1, scale: 5}
    values = [coefficient.evaluate(point) for coefficient in coefficients]
    numeric_arguments = [argument.evaluate(point) for argument in arguments]
    expected = getattr(oneloop, name.lower())(*numeric_arguments, backend="native")
    assert values == pytest.approx(expected, rel=1e-11, abs=1e-12)
    assert not oneloop.is_initialized()


def test_reduction_requires_the_shared_family_type():
    from types import SimpleNamespace

    family = scalar_family([E("2")], [])
    imitation = SimpleNamespace(
        loop_momenta=family.loop_momenta,
        external_momenta=family.external_momenta,
        denominators=family.denominators,
        scalar_products=family.scalar_products,
        kinematics=family.kinematics,
        is_complete=family.is_complete,
        is_independent=family.is_independent,
    )
    with pytest.raises(TypeError):
        oneloop.reduce(imitation, [1])


@pytest.mark.parametrize("name", ["d", "xll", "xq1", "den1"])
def test_scalar_parameters_are_not_captured_by_internal_coordinates(name):
    parameter = S(f"oneloopmaster::{name}")
    family = scalar_family([E("2")], [])
    result = oneloop.reduce(family, [1], numerator=parameter).to_expression()
    assert (result - parameter * oneloop.A0(2, 1)).together() == E("0")


def test_scalar_parameter_survives_internal_onshell_regularization():
    parameter = S("oneloopmaster::reg_delta")
    family = scalar_family([E("2")] * 3, [E("0"), E("-3"), E("-2")])
    result = oneloop.reduce(family, [1, 1, 1], numerator=parameter).to_expression()
    expected = parameter * oneloop.C0(0, -2, -3, 2, 2, 2, 1)
    assert (result - expected).together() == E("0")


@pytest.mark.parametrize("legacy_input", ["numerator", "mass", "invariant"])
def test_legacy_reducer_symbols_are_rejected(legacy_input):
    mass = S("oneloopreduce::legacy_mass") if legacy_input == "mass" else E("2")
    invariant = S("oneloopreduce::legacy_s") if legacy_input == "invariant" else E("-3")
    numerator = (
        E("oneloopreduce::dot(oneloopreduce::k,oneloopreduce::q1)")
        if legacy_input == "numerator" else E("1")
    )
    # An old loop dot product must not silently pass through as a scalar
    # coefficient and produce an apparently successful, incomplete reduction.
    with pytest.raises(ValueError, match="oneloopreduce::"):
        oneloop.reduce(scalar_family([mass] * 2, [invariant]), [1, 1], numerator=numerator)


@pytest.mark.parametrize("composite", [False, True])
@pytest.mark.parametrize("entrypoint", ["master", "reduction", "coefficients"])
def test_legacy_scale_is_rejected_at_every_expression_entrypoint(composite, entrypoint):
    scale = S("oneloopreduce::legacy_scale")
    if composite:
        scale += E("1")
    reduction = oneloop.reduce(scalar_family([E("2")], []), [1])
    ((_, master),) = reduction.terms
    with pytest.raises(ValueError, match="oneloopreduce::"):
        if entrypoint == "master":
            master.to_expression(mu_squared=scale)
        elif entrypoint == "reduction":
            reduction.to_expression(mu_squared=scale)
        else:
            oneloop.reduction_coefficients(reduction, mu_squared=scale)


def test_symbolic_reduction_keeps_numerical_initialization_lazy():
    script = """
from symbolica import E, S
from symbolica.community import hep
from symbolica.community.hep import oneloop
assert not oneloop.is_initialized()
D, ell, external, s = S("lazy::D", "lazy::ell", "lazy::external", "s")
kin = hep.Kinematics(D, momenta=[ell, external]).with_scalar_product(external, external, s)
family = hep.IntegralFamily([ell], [external], [
    kin.scalar_product(ell, ell), kin.scalar_product(ell + external, ell + external),
], kinematics=kin)
reduction = oneloop.reduce(family, [1, 1]).simplify()
assert reduction.to_expression() == oneloop.B0(s, 0, 0, 1)
assert not oneloop.is_initialized()
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30,
    )
    output = result.stdout + result.stderr
    if result.returncode == -signal.SIGABRT and (
        "Cannot start another restricted Symbolica thread while this user's thread allowance is in use."
        in output
    ):
        pytest.skip("The parent pytest process holds the restricted Symbolica thread allowance")
    assert result.returncode == 0, output


@pytest.mark.parametrize(
    "kind,head,masses,invariants,arguments",
    [
        ("tadpole", "A0", ["m0"], [], ["m0"]),
        ("bubble", "B0", ["m0", "m1"], ["p01"], ["p01", "m0", "m1"]),
        (
            "triangle", "C0", ["m0", "m1", "m2"], ["p01", "p02", "p12"],
            ["p01", "p12", "p02", "m0", "m1", "m2"],
        ),
        (
            "box", "D0", ["m0", "m1", "m2", "m3"],
            ["p01", "p02", "p03", "p12", "p13", "p23"],
            ["p01", "p12", "p23", "p03", "p02", "p13", "m0", "m1", "m2", "m3"],
        ),
    ],
)
def test_scalar_master_mapping(kind, head, masses, invariants, arguments):
    family = scalar_family(list(map(E, masses)), list(map(E, invariants)))
    reduction = oneloop.reduce(family, [1] * len(masses)).simplify()
    ((coefficient, master),) = reduction.terms
    assert isinstance(reduction, oneloop.Reduction)
    assert isinstance(master, oneloop.MasterIntegral)
    assert coefficient == E("1")
    assert (master.kind, master.head) == (kind, head)
    assert master.arguments == list(map(E, arguments))
    assert isinstance(family, hep.IntegralFamily)
    assert reduction.dimension == family.kinematics.dimension

    for scale in (None, E("renormalization_scale_squared")):
        scale_argument = E("1") if scale is None else scale
        expected = getattr(oneloop, head)(
            *map(E, arguments), scale_argument,
        )
        assert master.to_expression(mu_squared=scale) == expected
        assert reduction.to_expression(mu_squared=scale) == expected
        coefficients = oneloop.reduction_coefficients(reduction, mu_squared=scale)
        assert coefficients == [
            getattr(oneloop, head)(tag, *map(E, arguments), scale_argument)
            for tag in (0, -1, -2)
        ]
        assert all(
            "oneloopreduce::" not in expression.to_canonical_string()
            for expression in [expected, *coefficients]
        )


@pytest.mark.parametrize(
    "masses,invariants,expected_poles",
    [
        ([2], [], [2, 0]),
        ([2, 2], [-1], [1, 0]),
        ([2, 2, 2], [-1, -3, -2], [0, 0]),
        ([2, 2, 2, 2], [-1, -5, -4, -2, -6, -3], [0, 0]),
    ],
)
def test_all_scalar_master_symbols_have_native_evaluation_hooks(masses, invariants, expected_poles):
    family = scalar_family([E(str(mass)) for mass in masses], [E(str(s)) for s in invariants])
    reduction = oneloop.reduce(family, [1] * len(masses)).simplify()
    # No function definitions, manual evaluator, or compiled expression is
    # supplied: tagged master symbols must carry their native numerical hooks.
    values = evaluate_coefficients(reduction, [], [])
    assert all(math.isfinite(value.real) and math.isfinite(value.imag) for value in values)
    assert abs(values[0]) > 0
    assert values[1:] == pytest.approx(expected_poles, abs=1e-12)


@pytest.mark.parametrize("sign", [-1, 1])
def test_selected_exact_triangle_branch_retains_parameters_and_matches_native_hook(sign):
    invariant = S("oneloop_test::inspection_s", is_real=True)
    mass = S("oneloop_test::inspection_mass_squared", is_positive=True)
    triangle = oneloop.C0
    exact = oneloop.get_expression(triangle(0, 0, invariant, 0, mass, 0, 1))
    selected = oneloop.select_branch(
        exact, [Replacement(invariant, E(str(2 * sign))), Replacement(mass, E("1"))],
    )
    assert {invariant, mass} <= set(selected[0].get_all_symbols(False))
    assert "if(" not in selected[0].to_canonical_string()
    # Evaluate away from the probe to catch accidental substitution of the
    # branch-selection kinematics into the returned symbolic expression.
    point = {invariant: 3 * sign, mass: 1}
    values = [coefficient.evaluate(point) for coefficient in selected]
    native = [
        triangle(tag, 0, 0, invariant, 0, mass, 0, 1).evaluate(point)
        for tag in (0, -1, -2)
    ]
    assert values == pytest.approx(native, rel=1e-11, abs=1e-12)


def test_rank_one_triangle_matches_the_scalar_master_combination():
    mass, p1, p2, s, scale = S(
        "oneloop_test::mass", "oneloop_test::p1", "oneloop_test::p2",
        "oneloop_test::s", "oneloop_test::scale",
    )
    family = scalar_family([mass] * 3, [p1, s, p2])
    numerator = family.kinematics.scalar_product(family.loop_momenta[0], family.external_momenta[0])
    reduction = oneloop.reduce(family, [1, 1, 1], numerator=numerator).simplify()
    assert sorted(master.kind for _, master in reduction.terms) == ["bubble", "bubble", "triangle"]
    bubble, triangle = (oneloop.B0, oneloop.C0)
    expected = (
        bubble(s, mass, mass, 1) - bubble(p2, mass, mass, 1)
        - p1 * triangle(p1, p2, s, mass, mass, mass, 1)
    ) / 2
    assert (reduction.to_expression() - expected).expand() == E("0")
    assert "oneloopreduce::" not in reduction.to_expression().to_canonical_string()

    actual = evaluate_coefficients(
        reduction, [mass, p1, p2, s, scale], [2 + 0j, -1 + 0j, -2 + 0j, -3 + 0j, 5 + 0j],
        mu_squared=scale,
    )
    bubble_s = [bubble(tag, -3, 2, 2, 5).evaluate({}) for tag in (0, -1, -2)]
    bubble_p2 = [bubble(tag, -2, 2, 2, 5).evaluate({}) for tag in (0, -1, -2)]
    scalar_triangle = [
        triangle(tag, -1, -2, -3, 2, 2, 2, 5).evaluate({})
        for tag in (0, -1, -2)
    ]
    expected_values = [(a - b + c) / 2 for a, b, c in zip(bubble_s, bubble_p2, scalar_triangle)]
    assert actual == pytest.approx(expected_values, rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("scale", [None, E("5")])
def test_raised_tadpole_retains_the_dimension_dependent_finite_part(scale):
    mass = S("oneloop_test::tadpole_mass_squared")
    family = scalar_family([mass], [])
    reduction = oneloop.reduce(family, [2]).simplify()
    ((coefficient, master),) = reduction.terms
    dimension = family.kinematics.dimension
    assert reduction.dimension == dimension
    assert (coefficient - (dimension - 2) / (2 * mass)).expand() == E("0")
    assert master.head == "A0"
    a0 = oneloop.A0
    scale_argument = E("1") if scale is None else scale
    expected_coefficients = [
        (a0(0, mass, scale_argument) - a0(-1, mass, scale_argument)) / mass,
        (a0(-1, mass, scale_argument) - a0(-2, mass, scale_argument)) / mass,
        a0(-2, mass, scale_argument) / mass,
    ]
    coefficients = oneloop.reduction_coefficients(reduction, mu_squared=scale)
    assert all(
        (actual - expected).expand() == E("0")
        for actual, expected in zip(coefficients, expected_coefficients, strict=True)
    )
    actual = evaluate_coefficients(reduction, [mass], [2 + 0j], mu_squared=scale)
    scale_value = 1 if scale is None else 5
    # Differentiating A0 with respect to mass squared removes its finite +1.
    assert actual == pytest.approx([-math.log(2 / scale_value), 1, 0], rel=1e-12, abs=1e-12)


def test_raised_massless_bubble_mixes_pole_into_the_finite_part():
    s = S("oneloop_test::bubble_s")
    family = scalar_family([E("0")] * 2, [s])
    reduction = oneloop.reduce(family, [2, 1]).simplify()
    ((coefficient, master),) = reduction.terms
    dimension = family.kinematics.dimension
    assert (coefficient + (dimension - 3) / s).expand() == E("0")
    assert master.head == "B0"
    actual = evaluate_coefficients(reduction, [s], [-3 + 0j], mu_squared=E("2"))
    finite, pole, double_pole = [
        oneloop.B0(tag, -3, 0, 0, 2).evaluate({})
        for tag in (0, -1, -2)
    ]
    expected = [(-finite + 2 * pole) / -3, (-pole + 2 * double_pole) / -3, -double_pole / -3]
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert actual == pytest.approx([-math.log(3 / 2) / 3, 1 / 3, 0], rel=1e-12, abs=1e-12)


def test_quadratic_dimension_coefficient_mixes_double_pole_into_finite_part():
    dimension, s = S("other_dimension::D", "oneloop_test::triangle_s")
    family = scalar_family([E("0")] * 3, [E("0"), s, E("0")], dimension)
    reduction = oneloop.reduce(family, [1, 1, 1], numerator=(dimension - 4) ** 2).simplify()
    assert reduction.dimension == dimension
    coefficients = oneloop.reduction_coefficients(reduction)
    assert coefficients[1:] == [E("0"), E("0")]
    actual = evaluate_coefficients(reduction, [s], [-3 + 0j])
    double_pole = oneloop.C0(-2, 0, 0, -3, 0, 0, 0, 1).evaluate({})
    assert abs(double_pole) > 0
    # (d-4)^2 = 4*eps^2, so only the master's double pole contributes.
    assert actual == pytest.approx([4 * double_pole, 0, 0], rel=1e-12, abs=1e-12)


def test_coefficient_pole_requires_unavailable_higher_order_masters():
    family = scalar_family([E("2")], [])
    dimension = family.kinematics.dimension
    reduction = oneloop.reduce(family, [1], numerator=1 / (dimension - 4)).simplify()
    # Exact symbolic reduction remains useful even when finite/pole evaluation
    # would require the master's positive powers of epsilon.
    assert reduction.to_expression() == oneloop.A0(2, 1) / (dimension - 4)
    with pytest.raises(ValueError, match="pole at d=4"):
        oneloop.reduction_coefficients(reduction)


def test_fractional_dimension_coefficient_is_rejected():
    family = scalar_family([E("2")], [])
    dimension = family.kinematics.dimension
    reduction = oneloop.reduce(family, [1], numerator=(dimension - 4) ** E("1/2"))
    with pytest.raises(ValueError, match="integer-power Taylor expansion"):
        oneloop.reduction_coefficients(reduction)


@pytest.mark.parametrize("dependent_argument", ["scale", "mass", "invariant"])
def test_dimension_dependent_master_arguments_are_rejected(dependent_argument):
    dimension = S("shared_family_test::D")
    mass = dimension if dependent_argument == "mass" else E("2")
    invariant = dimension if dependent_argument == "invariant" else E("-3")
    scale = dimension if dependent_argument == "scale" else None
    family = scalar_family([mass] * 2, [invariant], dimension)
    reduction = oneloop.reduce(family, [1, 1]).simplify()
    assert isinstance(reduction.to_expression(mu_squared=scale), Expression)
    with pytest.raises(ValueError, match="independent of"):
        oneloop.reduction_coefficients(reduction, mu_squared=scale)


@pytest.mark.parametrize("zero_kind", ["scaleless", "zero_numerator"])
def test_zero_reduction_has_three_zero_expression_coefficients(zero_kind):
    mass = E("0") if zero_kind == "scaleless" else E("2")
    numerator = E("0") if zero_kind == "zero_numerator" else E("1")
    reduction = oneloop.reduce(scalar_family([mass], []), [1], numerator=numerator).simplify()
    assert reduction.to_expression() == E("0")
    coefficients = oneloop.reduction_coefficients(reduction)
    assert all(isinstance(value, Expression) for value in coefficients)
    assert coefficients == [E("0"), E("0"), E("0")]


def test_reduction_coefficients_are_shared_with_feynkit():
    mass = S("oneloop_test::shared_mass_squared")
    reduction = oneloop.reduce(scalar_family([mass], []), [1])
    coefficients = oneloop.reduction_coefficients(reduction)
    assert all(isinstance(value, Expression) for value in coefficients)
    expression = hep.TensorReducer.feynkit(E("4")).reduce(coefficients[0] + coefficients[1])
    value = complex(expression.evaluate({mass: 2 + 0j}))
    assert value == pytest.approx(2 * (2 - math.log(2)), rel=1e-12, abs=1e-12)


def test_the_same_family_is_usable_by_one_loop_and_rustred():
    invariant = S("shared_backends::s")
    family = scalar_family([E("0")] * 2, [invariant])
    ibp = hep.IBPFamily(family, name="shared_bubble")
    assert ibp.denominator_count == len(family.denominators) == 2
    one_loop = oneloop.reduce(family, [2, 1]).to_expression()
    solution = ibp.reduce_laporta([[2, 1]], max_depth=2)
    integral = S("shared_backends::I")
    ibp_result = solution.reduce([2, 1], integral=integral)
    ibp_result = ibp_result.replace(integral(1, 1), oneloop.B0(invariant, 0, 0, 1))
    assert (one_loop - ibp_result).together() == E("0")


def test_diagram_generated_family_is_reduced_directly():
    from pathlib import Path

    model = hep.Model(Path(__file__).parents[1] / "examples/hep/scalar_phi3.json")
    diagram = model.process(["scalar_0"], ["scalar_0"]).generate_diagrams(
        loops=1, max_vertices=2,
        allow_self_loops=False,
    ).diagrams[0]
    family = diagram.integral_family()
    external = family.external_momenta[0]
    invariant = S("diagram_oneloop_test::s")
    family = diagram.integral_family(
        kinematics=family.kinematics.with_scalar_product(external, external, invariant),
    )
    assert len(family.denominators) == len(diagram.internal_edges) == 2
    reduction = oneloop.reduce(family, [1, 1])
    ((coefficient, master),) = reduction.terms
    assert coefficient == E("1")
    assert master.head == "B0"
    assert master.arguments[0] == invariant
    assert reduction.dimension == family.kinematics.dimension


def test_shifted_normalized_denominators_and_numerator_preserve_routing():
    dimension, loop, external, mass, invariant = S(
        "shifted_test::Dimension", "shifted_test::integration_momentum",
        "shifted_test::external_leg", "shifted_test::mass2", "shifted_test::s",
    )
    kin = hep.Kinematics(dimension, momenta=[loop, external]).with_scalar_product(
        external, external, invariant,
    )
    # The first propagator defines l=k+2p; the second is (l+p)^2-m².
    # Denominator normalization contributes (1/2)^-1 * (-3)^-1 = -2/3.
    shifted = hep.IntegralFamily([loop], [external], [
        (kin.scalar_product(loop + 2 * external, loop + 2 * external) - mass) / 2,
        -3 * (kin.scalar_product(loop + 3 * external, loop + 3 * external) - mass),
    ], kinematics=kin)
    result = oneloop.reduce(shifted, [1, 1], numerator=kin.scalar_product(loop, external))
    # Equal-mass bubble symmetry gives integral(l.p) = -s/2 * B0;
    # hence integral(k.p) = -5s/2 * B0 before the normalization factor.
    expected = 5 * invariant / 3 * oneloop.B0(invariant, mass, mass, 1)
    assert (result.to_expression() - expected).together() == E("0")
    assert result.dimension == dimension


def test_independent_numerator_direction_uses_the_full_kinematics():
    dimension, loop, external, transverse, invariant, transverse_sq, mass = S(
        "transverse_test::D", "transverse_test::ell", "transverse_test::p",
        "transverse_test::q", "transverse_test::s", "transverse_test::t", "transverse_test::m2",
    )
    kin = hep.Kinematics(dimension, momenta=[loop, external, transverse])
    kin = kin.with_scalar_product(external, external, invariant)
    kin = kin.with_scalar_product(transverse, transverse, transverse_sq)
    kin = kin.with_scalar_product(external, transverse, E("0"))
    family = hep.IntegralFamily([loop], [external, transverse], [
        kin.scalar_product(loop, loop) - mass,
        kin.scalar_product(loop + external, loop + external) - mass,
    ], kinematics=kin)
    numerator = kin.scalar_product(loop, transverse) ** 2
    result = oneloop.reduce(family, [1, 1], numerator=numerator).to_expression()
    projected = transverse_sq / (dimension - 1) * (
        kin.scalar_product(loop, loop) - kin.scalar_product(loop, external) ** 2 / invariant
    )
    reference = oneloop.reduce(family, [1, 1], numerator=projected).to_expression()
    assert (result - reference).together() == E("0")
    assert result != E("0")
    completed = family.complete(candidates=[kin.scalar_product(loop, transverse)])
    assert completed.denominators == [*family.denominators, kin.scalar_product(loop, transverse)]
    assert (
        oneloop.reduce(completed, [1, 1, -2]).to_expression() - result
    ).together() == E("0")
    assert (
        oneloop.reduce(completed, [1, 1, 0]).to_expression()
        - oneloop.reduce(family, [1, 1]).to_expression()
    ).together() == E("0")


@pytest.mark.parametrize("powers", [[], [1], [1, 1, 1], [40, 1], [-(2**31), 1], [2**31 - 1, 1]])
def test_invalid_powers_raise_value_error(powers):
    family = scalar_family([E("0")] * 2, [S("powers_test::s")])
    with pytest.raises(ValueError):
        oneloop.reduce(family, powers)


def test_nonpositive_propagator_powers_are_numerator_factors():
    mass, invariant = S("signed_powers_test::m2", "signed_powers_test::s")
    family = scalar_family([mass, mass], [invariant])
    numerator_result = oneloop.reduce(family, [-1, 1]).to_expression()
    assert (numerator_result - invariant * oneloop.A0(mass, 1)).together() == E("0")
    assert oneloop.reduce(family, [0, 0]).to_expression() == E("0")


def test_powers_are_explicit():
    with pytest.raises(TypeError):
        oneloop.reduce(scalar_family([E("2")], []))


@pytest.mark.parametrize("dimension", [E("4"), E("6")])
def test_integer_dimension_is_rejected_before_epsilon_terms_are_lost(dimension):
    family = scalar_family([E("2")], [], dimension)
    with pytest.raises(ValueError, match="symbolic.*dimension|dimension.*symbol"):
        oneloop.reduce(family, [2])


def test_multiple_loops_are_rejected():
    dimension, k, ell = S("two_loop_test::D", "two_loop_test::k", "two_loop_test::ell")
    kin = hep.Kinematics(dimension, momenta=[k, ell])
    family = hep.IntegralFamily([k, ell], [], [
        kin.scalar_product(k, k) - 2, kin.scalar_product(ell, ell) - 3,
    ], kinematics=kin)
    with pytest.raises(ValueError, match="one loop|one-loop|single loop|single-loop|exactly one"):
        oneloop.reduce(family, [1, 1])


def test_positive_power_eikonal_denominator_is_rejected():
    dimension, k, p = S("eikonal_test::D", "eikonal_test::k", "eikonal_test::p")
    kin = hep.Kinematics(dimension, momenta=[k, p]).with_scalar_product(p, p, E("-3"))
    family = hep.IntegralFamily([k], [p], [
        kin.scalar_product(k, k) - 2, kin.scalar_product(k, p),
    ], kinematics=kin)
    with pytest.raises(ValueError, match="quadratic|eikonal"):
        oneloop.reduce(family, [1, 1])


@pytest.mark.parametrize("numerator_kind", ["bare_momentum", "unknown_function", "inverse_product"])
def test_unsupported_loop_numerators_are_rejected(numerator_kind):
    family = scalar_family([E("2")] * 2, [E("-3")])
    loop, external = family.loop_momenta[0], family.external_momenta[0]
    numerator = {
        "bare_momentum": loop,
        "unknown_function": S("unsupported_test::opaque")(loop),
        "inverse_product": 1 / family.kinematics.scalar_product(loop, external),
    }[numerator_kind]
    with pytest.raises(ValueError):
        oneloop.reduce(family, [1, 1], numerator=numerator)
