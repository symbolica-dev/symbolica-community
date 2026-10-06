"""Supplied reductions retain native family domains and reusable cache identities."""

import pytest

from symbolica import ComplexFloat, E, Float, S
from symbolica.community import hepkit as hep
from symbolica.community.hep import integration as numerical


def precise(value):
    return ComplexFloat(value, decimal_digits=90)


def tadpole(mass):
    dimension, loop, epsilon = S(
        "supplied_python::D", "supplied_python::k", "supplied_python::eps"
    )
    kinematics = hep.Kinematics(dimension, momenta=[loop])
    family = hep.IntegralFamily(
        [loop], [], [kinematics.scalar_product(loop, loop) - mass],
        kinematics=kinematics,
    )
    return family, epsilon


def test_supplied_reductions_preserve_original_and_explicit_nonzero_conditions():
    mass = S("supplied_python::mass_squared")
    family, epsilon = tadpole(mass)
    # The rational identity is the ordinary tadpole recurrence. Its original
    # expression additionally excludes mass_squared = +/-1 before cancellation.
    coefficient = (1 - epsilon) * (mass * mass - 1) / (mass * (mass - 1) * (mass + 1))
    tables = numerical.ReductionTables().with_family(
        family, epsilon, [([2], [([1], coefficient)])], [[1]],
        nonzero_conditions=[mass - 7],
    )
    prepared = numerical.IntegralEvaluator(reductions=tables).prepare(
        family, [[1], [2]], [mass], epsilon,
        branch_domain="positive mass squared within the original reduction domain",
    )
    assert prepared.basis == [[1]]
    assert prepared.target_reductions[0] == [([1], E("1"))]
    powers, retained = prepared.target_reductions[1][0]
    assert powers == [1]
    assert (retained - (1 - epsilon) / mass).together().cancel() == E("0")
    assert {mass, mass - 1, mass + 1, mass - 7} <= set(prepared.nonzero_conditions)
    prepared.required_master_range({mass: E("2")}, -1, 0)
    for excluded in (-1, 0, 1, 7):
        with pytest.raises(numerical.InvalidInputError, match="nonzero reduction condition"):
            prepared.required_master_range({mass: E(str(excluded))}, -1, 0)


def test_supplied_table_admission_is_immutable_and_rejects_uncovered_leaves():
    mass = S("supplied_python::mass_squared")
    family, epsilon = tadpole(mass)
    rules = [([2], [([1], (1 - epsilon) / mass)])]
    empty = numerical.ReductionTables()
    empty_identity = numerical.IntegralEvaluator(reductions=empty).backend_identity
    tables = empty.with_family(family, epsilon, rules, [[1]])
    evaluator = numerical.IntegralEvaluator(reductions=tables)
    identity = evaluator.backend_identity
    assert identity != empty_identity
    assert numerical.IntegralEvaluator(reductions=empty).backend_identity == empty_identity
    with pytest.raises(numerical.InvalidInputError, match="already exists"):
        tables.with_family(family, epsilon, rules, [[1]])
    with pytest.raises(numerical.IncompleteReductionError, match="uncovered integral"):
        empty.with_family(family, epsilon, rules, [])
    with pytest.raises(numerical.InvalidInputError, match="zero reduction condition"):
        empty.with_family(family, epsilon, rules, [[1]], nonzero_conditions=[E("0")])
    assert evaluator.backend_identity == identity
    assert evaluator.prepare(
        family, [[1]], [mass], epsilon, branch_domain="positive mass squared"
    ).basis == [[1]]


@pytest.mark.parametrize(
    "changed", ["denominator_order", "routing", "mass", "epsilon", "dimension",
                "numerator_role", "specialization"],
)
def test_supplied_tables_do_not_cross_native_family_scopes(changed):
    dimension, loop, external, epsilon, mass, other_mass, invariant = S(
        "supplied_scope::D", "supplied_scope::k", "supplied_scope::p",
        "supplied_scope::eps", "supplied_scope::m2", "supplied_scope::n2",
        "supplied_scope::s",
    )
    kinematics = hep.Kinematics(dimension, momenta=[loop, external]).with_scalar_product(
        external, external, invariant
    )
    square = kinematics.scalar_product(loop, loop)
    mixed = kinematics.scalar_product(loop, external)
    denominators = [square - mass, square + 2 * mixed + invariant - other_mass]

    def family(denominators):
        return hep.IntegralFamily([loop], [external], denominators, kinematics=kinematics)

    original = family(denominators)
    tables = numerical.ReductionTables().with_family(
        original, epsilon, [([2, 0], [([1, 0], (1 - epsilon) / mass)])], [[1, 0]],
    )
    evaluator = numerical.IntegralEvaluator(reductions=tables)
    variables = [mass, other_mass, invariant]
    assert evaluator.prepare(
        original, [[1, 0]], variables, epsilon, branch_domain="fixed physical sheet"
    ).basis == [[1, 0]]
    options = {}
    if changed == "denominator_order":
        denominators.reverse()
    elif changed == "routing":
        denominators[1] -= 4 * mixed
    elif changed == "mass":
        denominators[0] -= 1
    elif changed == "epsilon":
        epsilon = S("supplied_scope::other_epsilon")
    elif changed == "dimension":
        evaluator = numerical.IntegralEvaluator(
            reductions=tables, options=numerical.EvaluationOptions(dimension=6)
        )
    elif changed == "numerator_role":
        options["physical_propagators"] = 1
    else:
        variables = [other_mass, invariant]
        options["fixed_parameters"] = {mass: E("2")}
    with pytest.raises(numerical.IncompleteReductionError, match="no supplied reduction table"):
        evaluator.prepare(
            family(denominators), [[1, 0]], variables, epsilon,
            branch_domain="fixed physical sheet", **options,
        )


def constant_flow(basis, condition, normalization):
    x, epsilon = S("supplied_cache::x", "supplied_cache::eps")
    return numerical.KinematicTransport(
        epsilon, {x: [[E("0"), E("0")], [E("0"), E("0")]]},
        basis, normalization, branch_domain="declared real domain",
        nonzero_conditions=[condition],
    )


def seed(flow, cache, point, values):
    return flow.add_boundary(
        cache, point, [[precise(value) for value in values]], 0,
        verified_digits=60, comparison_errors=[[Float("0", decimal_digits=90)] * 2],
        provenance="exact constant vector in this ordered basis and normalization",
    )


@pytest.mark.parametrize("changed", ["basis_order", "conditions", "normalization"])
def test_cache_reload_separates_basis_conditions_and_normalization(tmp_path, changed):
    x, first, second = S("supplied_cache::x", "supplied_cache::first", "supplied_cache::second")
    basis, condition, normalization = [first, second], x - 1, E("1")
    original = constant_flow(basis, condition, normalization)
    cache = numerical.BoundaryCache()
    original_seed = seed(original, cache, {x: E("0")}, ["2", "3"])
    cache.save(tmp_path / "boundary-cache")
    restored = numerical.BoundaryCache.load(tmp_path / "boundary-cache")
    values = ["2", "3"]
    if changed == "basis_order":
        basis, values = basis[::-1], values[::-1]
    elif changed == "conditions":
        condition = x - 2
    else:
        normalization, values = E("2"), ["4", "6"]
    other = constant_flow(basis, condition, normalization)
    assert other.identity != original.identity
    retained = len(restored)
    with pytest.raises(numerical.IncompleteReductionError, match="no compatible cached"):
        other.evaluate(restored, {x: E("0")}, 0, 0)
    assert len(restored) == retained
    other_seed = seed(other, restored, {x: E("0")}, values)
    assert len(restored) == retained + 1
    for flow, expected in [(original, original_seed), (other, other_seed)]:
        result = flow.evaluate(restored, {x: E("0")}, 0, 0)
        assert result.cache_hit and result.steps == 0
        assert result.identity == flow.identity
        assert result.coefficients == expected.coefficients


def test_original_nonzero_condition_blocks_regular_kernel_path_and_endpoint():
    x, first, second = S("supplied_cache::x", "supplied_cache::first", "supplied_cache::second")
    flow = constant_flow([first, second], x - 1, E("1"))
    cache = numerical.BoundaryCache()
    seed(flow, cache, {x: E("0")}, ["2", "3"])
    retained = len(cache)
    # The differential matrix is zero everywhere, but its derivation excludes
    # x=1. Endpoint checks and whole-path admission must retain that restriction.
    with pytest.raises(numerical.InvalidInputError, match="nonzero reduction condition"):
        seed(flow, cache, {x: E("1")}, ["2", "3"])
    with pytest.raises(numerical.IncompleteReductionError, match="no compatible cached"):
        flow.evaluate(cache, {x: E("2")}, 0, 0, admit_straight_path=True)
    assert len(cache) == retained


def test_native_auxiliary_family_table_evaluates_finite_epsilon_samples():
    original, epsilon = tadpole(E("2"))
    eta = S("symbolica_amflow::eta")
    auxiliary, _ = tadpole(2 + eta)
    tables = numerical.ReductionTables().with_family(
        auxiliary, epsilon, [([2], [([1], (1 - epsilon) / (2 + eta))])], [[1]],
    )
    samples = [E("1/97")]
    supplied, = numerical.IntegralEvaluator(reductions=tables).evaluate_samples(
        original, [[1], [2]], {}, epsilon, samples
    )
    independent, = numerical.IntegralEvaluator().evaluate_samples(
        original, [[1], [2]], {}, epsilon, samples
    )
    for actual, expected in zip(supplied, independent):
        assert isinstance(actual, ComplexFloat)
        assert abs(actual - expected) < Float("1e-20", decimal_digits=90)
    expected_ratio = (1 - precise("1") / precise("97")) / 2
    assert abs(supplied[1] - expected_ratio * supplied[0]) < Float("1e-40", decimal_digits=90)
