"""Actual-Pyodide automatic evaluation with shared HEPKit families and RustRed.

No reference boundary is supplied to the evaluator. Decimal references below
are independent gamma-function evaluations (mpmath, 90 decimal digits):
-Gamma(-9/10) and Gamma(1/10)*Gamma(9/10)^2/Gamma(9/5).
They are comparison data only.
"""

import tempfile

from symbolica import ComplexFloat, E, Expression, Float, S
from symbolica.community import hepkit as hep
from symbolica.community.hep import integration


def _precise(value):
    return ComplexFloat(value, decimal_digits=90)


def _close(actual, expected, digits=20):
    assert isinstance(actual, ComplexFloat)
    assert abs(actual - _precise(expected)) < Float(f"1e-{digits}", decimal_digits=90), (
        actual, expected,
    )


assert integration.automatic_boundary_generation_available
try:
    integration.EvaluationOptions(workers=2)
except integration.UnsupportedInputError as error:
    assert "one worker" in str(error)
else:
    raise AssertionError("Browser options silently accepted multiple workers")
dimension, loop, external, epsilon, mass = S(
    "wasm_automatic::D", "wasm_automatic::k", "wasm_automatic::p",
    "wasm_automatic::epsilon", "wasm_automatic::mass_squared",
)
kinematics = hep.Kinematics(dimension, momenta=[loop])
family = hep.IntegralFamily(
    [loop], [], [kinematics.scalar_product(loop, loop) - mass], kinematics=kinematics,
)
options = integration.EvaluationOptions(digits=20, workers=1)
evaluator = integration.IntegralEvaluator(options=options)
control = integration.ComputationControl()
samples = evaluator.evaluate_samples(
    family, [[1], [2]], {mass: E("1")}, epsilon, [E("1/10")], control=control,
)
first, second = samples[0]
_close(first, "10.5705641096319242625472079747393357695006429178759748256111119671491837592475789017079598")
assert abs(second - _precise("0.9") * first) < Float("1e-35", decimal_digits=90)
assert control.poll(), "Completed synchronous work should retain progress events"
assert not control.poll(), "Polling must drain retained events"
print("WASM integral evaluation: finite-epsilon tadpole and progress passed")

prepared = evaluator.prepare(
    family, [[1], [2]], [mass], epsilon, branch_domain="positive real mass squared",
)
assert type(prepared) is integration.PreparedIntegralFamily
assert prepared.basis == [[1]]
assert prepared.target_reductions[0] == [([1], E("1"))]
powers, coefficient = prepared.target_reductions[1][0]
assert powers == [1] and isinstance(coefficient, Expression)
assert (coefficient - (1 - epsilon) / mass).together().cancel() == E("0")
assert mass in prepared.nonzero_conditions
tables = integration.ReductionTables().with_family(
    family, epsilon, [([2], [([1], coefficient)])], [[1]],
)
supplied = integration.IntegralEvaluator(reductions=tables).prepare(
    family, [[1], [2]], [mass], epsilon, branch_domain="positive real mass squared",
)
assert supplied.basis == prepared.basis
assert supplied.target_reductions == prepared.target_reductions
print("WASM integral evaluation: exact and supplied tadpole reductions passed")

result, = evaluator.evaluate(family, [[1]], {mass: E("1")}, epsilon, last=0)
assert result.verified_digits >= 20 and result.working_bits > 64
_close(result.coefficients[-1], "1")
_close(result.coefficients[0], "0.4227843350984671393934879099175975689578406640600764011942327651151322732223353290630529")
print("WASM integral evaluation: tadpole Laurent expansion verified to 20 digits")

bubble_kinematics = hep.Kinematics(dimension, momenta=[loop, external]).with_scalar_product(
    external, external, E("-1"),
)
square = bubble_kinematics.scalar_product(loop, loop)
mixed = bubble_kinematics.scalar_product(loop, external)
bubble = hep.IntegralFamily(
    [loop], [external], [square, square + 2 * mixed - 1], kinematics=bubble_kinematics,
)
bubble_rows = evaluator.evaluate_samples(bubble, [[1, 1]], {}, epsilon, [E("1/10")])
_close(bubble_rows[0][0], "11.6644879019310665161138835314585845565958250798934088592437943219726711523653995234827552")
print("WASM integral evaluation: finite-epsilon massless bubble passed")

cache = integration.BoundaryCache()
seed = prepared.generate_boundary(cache, {mass: E("1")})
assert not seed.cache_hit and seed.verified_digits >= 30
print("WASM integral evaluation: fresh boundary verified to at least 30 digits")
leading, last = prepared.required_master_range({mass: E("2")}, -1, 0)
transported = prepared.transport(cache, {mass: E("2")}, leading, last, admit_straight_path=True)
projected = prepared.project_targets(transported, -1, 0)
assert all(value.evidence_kind == "propagated_boundary" for value in projected)
assert all(value.verified_digits >= 20 for value in projected)
_close(projected[0].coefficients[-1], "2")
_close(projected[1].coefficients[-1], "1")
with tempfile.TemporaryDirectory() as directory:
    cache.save(directory)
    restored = integration.BoundaryCache.load(directory)
    repeated = prepared.transport(restored, {mass: E("2")}, leading, last)
    assert repeated.cache_hit and repeated.steps == 0
    assert repeated.coefficients == transported.coefficients
    assert repeated.comparison_errors == transported.comparison_errors

# Keep many wrappers alive to exercise CPython's eight-byte heap alignment.
retained_boundaries = [integration.BoundaryData(_precise("0"), [_precise("1")]) for _ in range(64)]
assert all(item.values == [_precise("1")] for item in retained_boundaries)
del retained_boundaries

cancelled = integration.ComputationControl()
cancelled.cancel()
try:
    evaluator.evaluate(family, [[1]], {mass: E("1")}, epsilon, control=cancelled)
except integration.CalculationCancelled:
    pass
else:
    raise AssertionError("Pre-cancelled automatic evaluation ran")

automatic_integral_validation = {
    "automatic_boundary_generation": True,
    "exact_tadpole_ibp": True,
    "scoped_supplied_reductions": True,
    "finite_epsilon_tadpole_and_bubble": True,
    "tadpole_laurent_verified_digits": result.verified_digits,
    "fresh_boundary_verified_digits": seed.verified_digits,
    "generated_boundary_transport_and_restart": True,
    "precancelled_evaluation": True,
    "retained_progress": True,
    "multiple_workers_rejected": True,
    "execution": "synchronous single worker",
}
