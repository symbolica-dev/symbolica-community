"""Numerical integration uses the installed host's native objects and precision."""

from symbolica import ComplexFloat, E, Float, S
from symbolica.community import hepkit as hep
from symbolica.community.hep import integration as numerical
import pytest


def precise(value):
    return ComplexFloat(value, decimal_digits=90)


def assert_close(actual, expected, digits=20):
    assert isinstance(actual, ComplexFloat)
    assert abs(actual - expected) < Float(f"1e-{digits}", decimal_digits=90)


def tadpole():
    dimension, loop, epsilon = S(
        "integration_test::D", "integration_test::k", "integration_test::eps"
    )
    kinematics = hep.Kinematics(dimension, momenta=[loop])
    family = hep.IntegralFamily(
        [loop], [], [kinematics.scalar_product(loop, loop) - E("1")],
        kinematics=kinematics,
    )
    return family, epsilon


def test_existing_family_and_precise_laurent_values():
    family, epsilon = tadpole()
    evaluator = numerical.IntegralEvaluator()
    result, = evaluator.evaluate(family, [[1]], {}, epsilon, last=0)
    assert result.verified_digits == 20
    assert_close(result.coefficients[-1], precise("1"))
    assert isinstance(result.comparison_errors[-1], Float)
    assert result.working_bits > 64


def test_exact_finite_epsilon_samples_and_tadpole_recurrence():
    family, epsilon = tadpole()
    evaluator = numerical.IntegralEvaluator(options=numerical.EvaluationOptions(workers=2))
    samples = [E("1/10"), E("1/20")]
    rows = evaluator.evaluate_samples(family, [[1], [2]], {}, epsilon, samples)
    for eps, (first, second) in zip(("0.1", "0.05"), rows):
        # Gamma(eps) = (eps-1)*Gamma(eps-1), including the Minkowski sign.
        assert_close(second, (1 - precise(eps)) * first, digits=40)
    with pytest.raises(numerical.InvalidInputError):
        evaluator.evaluate_samples(family, [[1]], {}, epsilon, [])
    with pytest.raises(numerical.InvalidInputError):
        evaluator.evaluate_samples(family, [[1]], {}, epsilon, [E("0")])


def test_regular_differential_system_keeps_evidence_distinct():
    x = S("integration_test::x")
    system = numerical.DifferentialSystem(x, [[E("1")]])
    boundary = numerical.BoundaryData(
        precise("0"), [precise("1")], verified_digits=60, provenance="exact unit"
    )
    result = system.transport(boundary, [precise("1")])
    assert_close(result.coefficients[0][0], precise("1").exp())
    assert result.verified_digits is None
    assert result.input_verified_digits == 60
    assert result.provenance == "exact unit"


def test_native_boundary_generation_then_physical_transport():
    dimension, loop, epsilon, mass_squared = S(
        "integration_seed::D", "integration_seed::k", "integration_seed::eps",
        "integration_seed::mass_squared",
    )
    kinematics = hep.Kinematics(dimension, momenta=[loop])
    family = hep.IntegralFamily(
        [loop], [], [kinematics.scalar_product(loop, loop) - mass_squared],
        kinematics=kinematics,
    )
    prepared = numerical.IntegralEvaluator().prepare(
        family, [[1]], [mass_squared], epsilon,
        branch_domain="positive real mass squared",
    )
    cache = numerical.BoundaryCache()
    seed = prepared.generate_boundary(cache, {mass_squared: E("1")})
    assert seed.verified_digits >= 30 and not seed.cache_hit
    leading, last = prepared.required_master_range({mass_squared: E("2")}, -1, 0)
    transported = prepared.transport(
        cache, {mass_squared: E("2")}, leading, last, admit_straight_path=True
    )
    result, = prepared.project_targets(transported, -1, 0)
    assert result.evidence_kind == "propagated_boundary"
    assert result.verified_digits >= 20
    assert_close(result.coefficients[-1], precise("2"))
    assert isinstance(result.absolute_errors[-1], Float)


def test_physical_cache_restart_and_intermediate_reuse(tmp_path):
    x, epsilon, master = S(
        "integration_cache::x", "integration_cache::eps", "integration_cache::I"
    )
    flow = numerical.KinematicTransport(
        epsilon, {x: [[epsilon / (1 + x)]]}, [master], E("1"),
        branch_domain="real x >= 0; positive 1+x",
    )
    cache = numerical.BoundaryCache()
    seed_provenance = "exact (1+x)^epsilon at x=0"
    flow.add_boundary(
        cache, {x: E("0")}, [[precise("1")], [precise("0")], [precise("0")]], 0,
        verified_digits=60,
        comparison_errors=[[Float("0", decimal_digits=90)] for _ in range(3)],
        provenance=seed_provenance,
    )
    first = flow.evaluate(cache, {x: E("1")}, 0, 2, admit_straight_path=True)
    assert first.verified_digits >= 20 and not first.cache_hit
    assert seed_provenance in first.provenance
    logarithm = precise("2").log()
    assert_close(first.coefficients[1][0], logarithm)
    assert_close(first.coefficients[2][0], logarithm * logarithm / 2)
    assert len(cache) > 1
    cache.save(tmp_path / "boundaries")
    restored = numerical.BoundaryCache.load(tmp_path / "boundaries")
    repeat = flow.evaluate(restored, {x: E("1")}, 0, 2)
    assert repeat.cache_hit and repeat.steps == 0
    assert repeat.coefficients == first.coefficients
    def matches_selected_source(entry, result):
        offset = result.leading_power - entry.leading_power
        return (
            entry.identity == result.identity
            and entry.coordinates == result.starting_coordinates
            and entry.provenance in result.provenance
            and entry.input_verified_digits == result.input_verified_digits
            and entry.verified_digits == result.verified_digits
            and entry.working_bits == result.working_bits
            and offset >= 0
            and entry.coefficients[offset:offset + len(result.coefficients)] == result.coefficients
            and entry.comparison_errors[offset:offset + len(result.comparison_errors)] == result.comparison_errors
        )
    assert any(matches_selected_source(entry, repeat) for entry in restored.entries())
    nearby = flow.evaluate(restored, {x: E("11/10")}, 0, 2, admit_straight_path=True)
    assert nearby.starting_coordinates[x] != E("0")
    assert len(restored) >= len(cache)
    assert_close(nearby.coefficients[1][0], precise("2.1").log())
    assert seed_provenance in nearby.provenance
    # Loading seed data into an existing calculation must preserve intermediate
    # points accumulated during previous transport. Self-merges must not deadlock.
    retained = len(restored)
    evidence = {
        (entry.identity, entry.coordinates[x], entry.leading_power,
         len(entry.coefficients), entry.input_verified_digits,
         entry.working_bits, entry.provenance): (
            entry.provenance, entry.verified_digits, entry.input_verified_digits,
            entry.working_bits, entry.coefficients, entry.comparison_errors,
        )
        for entry in restored.entries()
    }
    restored.extend(cache)
    restored.extend(restored)
    assert len(restored) == retained
    merged = numerical.BoundaryCache()
    merged.extend(restored)
    merged.extend(cache)
    assert len(merged) == retained
    merged.save(tmp_path / "merged")
    reloaded = numerical.BoundaryCache.load(tmp_path / "merged")
    assert {
        (entry.identity, entry.coordinates[x], entry.leading_power,
         len(entry.coefficients), entry.input_verified_digits,
         entry.working_bits, entry.provenance): (
            entry.provenance, entry.verified_digits, entry.input_verified_digits,
            entry.working_bits, entry.coefficients, entry.comparison_errors,
        )
        for entry in reloaded.entries()
    } == evidence
    nearby_repeat = flow.evaluate(reloaded, {x: E("11/10")}, 0, 2)
    assert nearby_repeat.cache_hit and nearby_repeat.steps == 0
    assert nearby_repeat.coefficients == nearby.coefficients
    assert any(matches_selected_source(entry, nearby_repeat) for entry in reloaded.entries())
    assert seed_provenance in nearby_repeat.provenance


def test_cancelled_evaluation_and_descriptive_types():
    family, epsilon = tadpole()
    control = numerical.ComputationControl()
    control.cancel()
    with pytest.raises(numerical.CalculationCancelled):
        numerical.IntegralEvaluator().evaluate(
            family, [[1]], {}, epsilon, control=control
        )
    for name in numerical.__all__:
        value = getattr(numerical, name)
        if isinstance(value, type):
            assert value.__module__ == "symbolica.community.hep.integration"
            assert not any(brand in name for brand in ("Rust", "Symbolica", "AMFlow", "DiffExp"))


def test_graph_substitutions_are_simultaneous_and_applied_once():
    model = hep.Model.phi3()
    dimension, numerator, epsilon = S(
        "feynkit_graph::D", "integration_graph::a", "integration_graph::eps"
    )
    momentum = E("gammalooprs::P(1)")
    kinematics = hep.Kinematics(dimension, momenta=[momentum]).with_scalar_product(
        momentum, momentum, E("-1")
    )
    diagram = hep.FeynmanDiagram.from_dot(model, '''
        digraph bubble {
            num="integration_graph::a";
            incoming [style=invis]; outgoing [style=invis];
            incoming -> a [id=0, particle="phi"];
            b -> outgoing [id=1, particle="phi"];
            a -> b [id=2, particle="phi", lmb_id=0];
            b -> a [id=3, particle="phi"];
        }
    ''')
    # a -> eps survives this simultaneous substitution. A second application
    # would turn it into 3 and incorrectly restore the bubble's 1/eps pole.
    result = numerical.IntegralEvaluator().evaluate_diagram(
        diagram, kinematics,
        {numerator: epsilon, epsilon: E("3"), S("UFO::mass"): E("0"),
         S("UFO::g"): E("1")},
        epsilon,
    )
    assert_close(result.coefficients[-1], precise("0"))
    assert abs(abs(result.coefficients[0]) - Float("1", decimal_digits=90)) < Float(
        "1e-20", decimal_digits=90
    )
