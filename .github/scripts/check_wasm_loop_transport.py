"""Small actual-Pyodide gate for supplied numerical transport and exact restart."""

import tempfile

from symbolica import ComplexFloat, E, Expression, Float, S, get_citations
from symbolica.community.hep import integration
from symbolica.community.hepkit import Model, Process

_numerical_reference_ids = {
    "arXiv:2607.08477", "arXiv:2006.05510", "arXiv:2201.11669", "arXiv:2112.07578",
}
assert not _numerical_reference_ids.intersection(c.id for c in get_citations())


def _number(value):
    return value.as_integer_ratio(), value.precision


def _evidence(result):
    return (
        [[(_number(z.real), _number(z.imag)) for z in row]
         for row in result.coefficients],
        [[_number(error) for error in row] for row in result.comparison_errors],
        result.verified_digits,
        result.input_verified_digits,
        result.working_bits,
        result.provenance,
    )


def check_higgs_standard_model():
    model = Model.standard_model()
    original = model.to_json()
    extended = integration.HiggsJetAmplitude.with_form_factor_vertices(model)
    assert type(extended) is Model
    assert model.to_json() == original
    assert extended.particle_by_pdg(21).name == "g"
    assert extended.particle_by_pdg(25).name == "H"
    assert isinstance(extended.parameter("GGGHEWWW_ForFac1_RE").symbol, Expression)
    assert isinstance(extended.coupling("GGGH_HEFT_C1").expression, Expression)
    vertices = ["GGGHEWWW", "GGGHEWZZ", "GGGHHEFT"]
    for name in vertices:
        assert extended.vertex_rule(name).particles == ["g", "g", "g", "H"]
    assert isinstance(extended.process(["g", "g"], ["g", "H"], vertex_allow=vertices), Process)
    before_collision = extended.to_json()
    try:
        integration.HiggsJetAmplitude.with_form_factor_vertices(extended)
    except integration.InvalidInputError as error:
        assert "already exists" in str(error)
    else:
        raise AssertionError("Conflicting Higgs-jet declarations were overwritten")
    assert extended.to_json() == before_collision and model.to_json() == original
    assert Model.from_json(before_collision).to_json() == before_collision
    citations = {c.id: c for c in get_citations()}
    assert _numerical_reference_ids <= citations.keys()
    for identifier in _numerical_reference_ids:
        assert citations[identifier].reasons
        assert identifier.split(":")[1] in citations[identifier].to_bibtex()
    assert citations.keys() == {c.id for c in get_citations()}


def check_supplied_loop_transport():
    assert integration.automatic_boundary_generation_available
    for name in ("IntegralEvaluator", "PreparedIntegralFamily", "ReductionTables"):
        assert hasattr(integration, name)
        assert name in integration.__all__
    for kind in ("planar", "nonplanar"):
        system = integration.HiggsJetIntegralSystem(kind)
        assert system.automatic_boundary_generation_available
        assert hasattr(system, "generate_boundary")
        assert len(system.configurations()) == 8

    x, epsilon, master = S("wasm_transport::x", "wasm_transport::epsilon", "wasm_transport::I")
    options = integration.EvaluationOptions(digits=20, guard_digits=30, series_order=32)
    flow = integration.KinematicTransport(
        epsilon, {x: [[epsilon / (1 + x)]]}, [master], E("1"),
        branch_domain="real x >= 0; positive 1+x", options=options,
    )
    # An exactly supplied nonzero dyadic value with 415 declared bits also
    # exercises the owner binary codec, rather than a trivial zero-only cache.
    seed = ComplexFloat(Float.from_ratio(1, 3, precision=415))
    zero = ComplexFloat(Float.from_ratio(0, 1, precision=415))
    cache = integration.BoundaryCache()
    provenance = "exact supplied rounded dyadic constant times (1+x)^epsilon"
    flow.add_boundary(
        cache, {x: E("0")}, [[seed], [zero], [zero]], 0,
        verified_digits=60, comparison_errors=[[zero.real] for _ in range(3)],
        provenance=provenance,
    )
    before = flow.evaluate(cache, {x: E("0")}, 0, 2)
    assert before.cache_hit and before.steps == 0
    with tempfile.TemporaryDirectory() as directory:
        cache.save(directory)
        restored = integration.BoundaryCache.load(directory)
        assert _evidence(flow.evaluate(restored, {x: E("0")}, 0, 2)) == _evidence(before)
        result = flow.evaluate(restored, {x: E("1")}, 0, 2, admit_straight_path=True)
        assert not result.cache_hit and result.steps > 0
        assert 20 <= result.verified_digits <= result.input_verified_digits == 60
        logarithm = ComplexFloat("2", decimal_digits=100).log()
        tolerance = Float("1e-20", decimal_digits=100)
        assert abs(result.coefficients[1][0] - seed * logarithm) < tolerance
        assert abs(result.coefficients[2][0] - seed * logarithm * logarithm / 2) < tolerance
        assert len(restored) > 1
        stored_evidence = [
            (entry.identity, entry.coordinates, entry.leading_power, _evidence(entry))
            for entry in restored.entries()
        ]
        restored.save(directory)
        reloaded = integration.BoundaryCache.load(directory)
        assert [
            (entry.identity, entry.coordinates, entry.leading_power, _evidence(entry))
            for entry in reloaded.entries()
        ] == stored_evidence
        hit = flow.evaluate(reloaded, {x: E("1")}, 0, 2)
        assert hit.cache_hit and hit.steps == 0
        assert _evidence(hit)[:3] == _evidence(result)[:3]
        assert hit.working_bits == result.working_bits
        # Reusing a derived boundary inherits its achieved accuracy, rather
        # than relabelling it with the original seed's stronger input cap.
        assert hit.input_verified_digits == result.verified_digits
        assert any(
            entry.identity == hit.identity
            and entry.coordinates == hit.starting_coordinates
            and _evidence(entry)[:-1] == _evidence(hit)[:-1]
            and entry.provenance in hit.provenance
            for entry in reloaded.entries()
        )
        assert provenance in result.provenance and provenance in hit.provenance
        nearby = flow.evaluate(reloaded, {x: E("11/10")}, 0, 2, admit_straight_path=True)
        assert nearby.starting_coordinates[x] != E("0")
        assert abs(nearby.coefficients[1][0] - seed * ComplexFloat("2.1", decimal_digits=100).log()) < tolerance
        count = len(reloaded)
        control = integration.ComputationControl()
        control.cancel()
        try:
            flow.evaluate(reloaded, {x: E("2")}, 0, 2, admit_straight_path=True, control=control)
        except integration.CalculationCancelled:
            pass
        else:
            raise AssertionError("Cancelled browser transport ran")
        assert len(reloaded) == count


check_supplied_loop_transport()
check_higgs_standard_model()
