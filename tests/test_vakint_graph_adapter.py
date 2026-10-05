"""Real graph/family ingress tests; the numerical gate needs the RustRed host."""

from pathlib import Path

import pytest
from symbolica import N, Float, Replacement, S
from symbolica.community import hepkit as hep
from symbolica.community.hepkit import vakint


def h_input():
    model = hep.Model.phi_3_4()
    source = (Path(__file__).parents[1] / "examples/hep/data/rustred_four_loop/h.dot").read_text()
    diagram = hep.FeynmanDiagram.from_dot(model, source)
    dimension, mass_squared = S("d", "vakint::muvsq")
    original = diagram.integral_family(kinematics=hep.Kinematics(dimension))
    substitutions = {model.particle("phi").mass: mass_squared ** (N(1) / 2)}
    replacements = [Replacement(left, right) for left, right in substitutions.items()]
    family = hep.IntegralFamily(original.loop_momenta, [],
        [den.replace_multiple(replacements) for den in original.denominators],
        kinematics=original.kinematics)
    p1, p2 = S("reference_p1", "reference_p2")
    kin = hep.Kinematics(dimension, momenta=[*family.loop_momenta, p1, p2])
    k1, k2, k3, _ = family.loop_momenta
    sp = kin.scalar_product
    numerator = (sp(k1, k2) ** 2 + sp(p1, k3) * sp(k3, p2)
                 + sp(p1, p2) * sp(k2 + k1, k2))
    return diagram, family, numerator, substitutions, (p1, p2)


def converted(data, **options):
    diagram, family, numerator, substitutions, spectators = data
    return vakint.integral_from_diagram(diagram, family, numerator,
        parameter_substitutions=substitutions, external_momenta=spectators, **options)


def test_import_initializes_namespaced_imaginary_placeholder():
    # Module import above executes the real native initializer in the accepted
    # single installed host. The placeholder remains a symbol until evaluation.
    placeholder = S("vakint::𝑖")
    assert placeholder**2 != N(-1)
    engine = vakint.Vakint(evaluation_order=[])
    result, error = engine.numerical_evaluation(placeholder, params={})
    assert error is None
    # These exact 0/1 components need no high-precision-to-f64 comparison.
    assert result.to_list() == [(0, (0.0, 1.0))]


def test_native_indexed_tensor_projection_without_form():
    # Same raw-index rank-two identity as Vakint's existing native tensor test.
    # It exercises both registered loop/external heads before the full H gate.
    k, p, dot, topo, epsilon = S("vakint::k", "vakint::p", "vakint::dot",
                                "vakint::topo", "vakint::ε")
    topology = topo(S("vakint_test::native_tensor_case"))
    source = k(1, 101) * k(1, 102) * p(1, 101) * p(1, 102) * topology
    expected = dot(k(1), k(1)) * dot(p(1), p(1)) * topology / (4 - 2*epsilon)
    engine = vakint.Vakint(evaluation_order=[], allow_unknown_integrals=True,
        tensor_reduction_method="feynkit", use_dot_product_notation=True,
        form_exe_path="/this/path/must/not/be/invoked/by-tensor-tests")
    reduced = engine.tensor_reduce(source)
    assert (reduced - expected).expand().together() == N(0)


def test_h_ingress_retains_graph_routing_and_native_rich_expression():
    data = h_input()
    integral = converted(data)
    assert isinstance(integral, vakint.VakintExpression)
    expression = integral.to_expression()
    k, p, dot = S("vakint::k", "vakint::p", "vakint::dot")
    expected_numerator = dot(k(1), k(2))**2 + dot(p(1), k(3))*dot(k(3), p(2)) + dot(p(1), p(2))*(dot(k(2), k(2)) + dot(k(1), k(2)))
    # No string ingress; exact numerator mapping is independently reconstructed.
    scalar = vakint.integral_from_diagram(data[0], data[1], N(1),
        parameter_substitutions=data[3]).to_expression()
    assert (expression - expected_numerator * scalar).expand() == N(0)
    assert integral._repr_html_()
    assert integral.formatted(max_terms=10)._repr_html_()
    assert vakint.VakintExpression(expression).to_expression() == expression


def test_native_rich_numerical_precision_and_wrapper_ingress_in_single_host():
    # The acceptance runner imports one installed Symbolica DSO in a fresh
    # process. These Python printer/extraction checks must not run embedded in
    # a Rust unit-test binary with another copy of Symbolica's global state.
    for cls in (vakint.Vakint, vakint.VakintExpression,
                vakint.VakintNumericalResult, vakint.VakintEvaluationMethod):
        assert cls.__module__ == "symbolica.community.hepkit.vakint"
    method = vakint.VakintEvaluationMethod.new_rustred_method(substitute_masters=False)
    engine = vakint.Vakint(run_time_decimal_precision=48, evaluation_order=[method],
        tensor_reduction_method="feynkit",
        form_exe_path="/this/path/must/not/be/invoked/by-rich-view-tests")
    epsilon = S("vakint::ε")
    decimal = "1.2345678901234567890123456789012345"
    source = N(Float(decimal, decimal_digits=48)) * epsilon**-2
    result = engine.numerical_result_from_expression(source)
    expression = result.to_expression()
    assert expression == engine.numerical_result_to_expression(result)
    assert expression != N(float(decimal)) * epsilon**-2
    roundtrip = engine.numerical_result_from_expression(expression)
    matches, detail = result.compare_to(roundtrip, relative_threshold=1e-40)
    assert matches, detail
    rounded = engine.numerical_result_from_expression(N(float(decimal)) * epsilon**-2)
    assert result.compare_to(rounded, relative_threshold=1e-30)[0] is False
    custom_epsilon = S("vakint_test::custom_epsilon")
    assert result.to_expression(custom_epsilon) == expression.replace_multiple([
        Replacement(epsilon, custom_epsilon)])
    with pytest.raises(ValueError, match="Symbolica variable"):
        result.to_expression(N(1))
    assert result._repr_html_()
    assert result.formatted(precision=48)._repr_html_()


@pytest.mark.parametrize("invalid", [True, 1.5, "1"])
def test_boolean_noninteger_and_string_powers_refuse(invalid):
    data = h_input()
    powers = [1] * 9 + [0]
    powers[0] = invalid
    with pytest.raises(ValueError, match="integer"):
        converted(data, powers=powers)
    powers = [1] * 9 + [invalid]
    with pytest.raises(ValueError, match="integer"):
        converted(data, powers=powers)


def test_power_and_family_ambiguities_refuse():
    data = h_input()
    with pytest.raises(ValueError, match="physical powers"):
        converted(data, powers=[0] + [1] * 8 + [0])
    with pytest.raises(ValueError, match="auxiliary powers"):
        converted(data, powers=[1] * 10)
    diagram, family, numerator, substitutions, spectators = data
    wrong = hep.IntegralFamily(family.loop_momenta, [],
        [-family.denominators[0], *family.denominators[1:]], kinematics=family.kinematics)
    with pytest.raises(ValueError, match="order/sign/masses"):
        vakint.integral_from_diagram(diagram, wrong, numerator,
            parameter_substitutions=substitutions, external_momenta=spectators)
    with pytest.raises(ValueError, match="order/sign/masses"):
        vakint.integral_from_diagram(diagram, family, numerator, external_momenta=spectators)
    with pytest.raises(ValueError, match="unconverted"):
        vakint.integral_from_diagram(diagram, family, numerator, parameter_substitutions=substitutions)


def test_dot_and_auxiliary_powers_are_native_not_rewritten_text():
    data = h_input()
    powers = [2] + [1] * 8 + [-1]
    wrapped = converted(data, powers=powers)
    assert isinstance(wrapped.to_expression(), type(N(0)))
    assert wrapped._repr_html_()


def test_integrated_momentum_assumptions_cannot_be_dropped_by_ingress():
    diagram, family, _, substitutions, _ = h_input()
    k = family.loop_momenta[0]
    constrained = family.kinematics.with_scalar_product(k, k, N(7))
    # Current HEPKit rejects this already at its native family boundary. If
    # that boundary ever accepts such a descriptor, the adapter must still
    # reject its modified physical denominators rather than drop the premise.
    with pytest.raises((hep.DiagramError, ValueError), match="on-shell assumptions|order/sign/masses"):
        physical = diagram.propagator_family(kinematics=constrained)
        replacements = [Replacement(left, right) for left, right in substitutions.items()]
        wrong = hep.IntegralFamily(physical.loop_momenta, [],
            [den.replace_multiple(replacements) for den in physical.denominators],
            kinematics=constrained)
        vakint.integral_from_diagram(diagram, wrong, N(1), parameter_substitutions=substitutions)


def test_h_rank_four_original_32_digit_reference_with_invalid_form():
    # Original Vakint test_integrate_4l_h_rank_4 reference, unchanged scales,
    # f64 external-input boundary and original 30-digit comparison threshold.
    integral = converted(h_input())
    engine = vakint.Vakint(run_time_decimal_precision=32,
        number_of_terms_in_epsilon_expansion=5, integral_normalization_factor="MSbar",
        mu_r_sq_symbol=S("vakint::mursq"), tensor_reduction_method="feynkit",
        evaluation_order=[vakint.VakintEvaluationMethod.new_rustred_method()],
        form_exe_path="/this/path/must/not/be/invoked/by-rustred-acceptance")
    evaluated = engine.evaluate(integral.to_expression())
    vectors = {i: (0.17 * (i + 1), 0.4 * (i + 2), 0.3 * (i + 3), 0.12 * (i + 4)) for i in (1, 2)}
    result, error = engine.numerical_evaluation(evaluated,
        params={"muvsq": 3.0, "mursq": 5.0}, externals=vectors)
    values = (
        (-4, "1.809145974886785501452557650622e-9"),
        (-3, "1.862208677723707446525921998109e-8"),
        (-2, "8.865577059648962609058045604434e-8"),
        (-1, "4.064224364096375531168559391726e-7"),
        (0, "7.705260630861442737917312763495e-6"),
    )
    epsilon = S("vakint::ε")
    expected = sum((N(Float(value, decimal_digits=32)) * epsilon**power for power, value in values), N(0))
    reference = engine.numerical_result_from_expression(expected)
    matches, detail = result.compare_to(reference, relative_threshold=1e-30, error=error, max_pull=1.0)
    assert matches, detail
    assert result.to_expression() == engine.numerical_result_to_expression(result)
    assert result.formatted(precision=32)._repr_html_()
