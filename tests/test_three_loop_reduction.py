"""Real native closure/reduction gates for the explicit three-loop notebook."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 and 3.10
    import tomli as tomllib

import pytest

pytest.importorskip("symbolica")
from symbolica import E, S
from symbolica.community import hepkit as hep


SPEC = importlib.util.spec_from_file_location(
    "three_loop_reduction_support",
    Path(__file__).parents[1] / "examples/hep/three_loop_reduction_support.py",
)
support = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(support)


def test_generation_requires_explicit_start_and_cannot_restart():
    class Native:
        def __init__(self):
            self.calls = []

        def start_family_candidates(self, source, **options):
            self.calls.append((source, options))
            return object()

    native = Native()
    run = support.ThreeLoopRun(native, "test-only-source")
    assert run.poll()["state"] == "ready"
    assert native.calls == []
    assert run.result is run.candidate is run.closing is run.inspection is None
    assert run.reductions == {}
    assert run.start(n_cores=1)
    assert not run.start(n_cores=1)
    assert native.calls == [("test-only-source", {
        "input_format": "toml", "n_cores": 1,
    })]


def test_graph_routing_matches_the_certified_source():
    dot, source = support.graph_inputs()
    model = hep.Model.phi_3_4()
    diagram = hep.FeynmanDiagram.from_dot(model, dot)
    routed = diagram.propagator_family(kinematics=hep.Kinematics(S("d")))
    mass = model.particle("phi").mass
    family = hep.IntegralFamily(
        routed.loop_momenta, routed.external_momenta,
        [denominator.replace(mass, E("1")) for denominator in routed.denominators],
        kinematics=routed.kinematics,
    )
    support.assert_source_matches_family(source, family)
    assert hep.IBPFamily(family, name="K6_graph_test").denominator_count == 6
    with pytest.raises(AssertionError):
        support.assert_source_matches_family(source.replace("k1^2-1", "k1^2+1"), family)


@pytest.fixture(scope="module")
def closed_family(tmp_path_factory):
    _, source = support.graph_inputs()
    run = support.ThreeLoopRun(hep.rustred, source)
    assert run.start(n_cores=1)
    assert run.session.wait(timeout=120), "small K6 candidate generation timed out"
    assert run.poll()["state"] == "generated"
    candidate = run.result
    assert candidate.status == "uncertified-candidates"
    assert run.candidate.metadata()["closure_claim"] is False
    generated = tomllib.loads(candidate.to_toml())
    assert generated["solved_sectors"] == 38
    assert generated["zero_sectors"] == 26
    assert generated["generated_rules"] == 623

    certificate = hep.rustred.certify_candidates(candidate.bundle)
    assert certificate.status == "generated-durable"
    artifact_path = tmp_path_factory.mktemp("three-loop-native") / "k6.rr"
    artifact_path.write_bytes(certificate.artifact)
    artifact = artifact_path.read_bytes()
    inspected = hep.rustred.inspect_closing_artifact(artifact)
    assert inspected.status == "inspected"
    inspection = tomllib.loads(inspected.to_toml())
    metadata = inspection["artifact"]
    assert metadata["arity"] == 6
    assert metadata["in_scope_zero_sectors"] == 26
    assert metadata["root_power_lower"] == [-(2**63)] * 6
    assert metadata["root_power_upper"] == [2**63 - 1] * 6
    masters = {tuple(master["powers"]) for master in metadata["masters"]}
    assert len(masters) == 38
    validation = inspection["validation"]
    assert validation["source_rows"] == 9
    assert validation["replayed_source_rows"] > 0
    assert validation["guarded_rules"] > 0

    cache = {}

    def reduce(powers):
        key = tuple(powers)
        if key not in cache:
            reduction = hep.rustred.reduce_with_closing_artifact(artifact, list(key))
            assert reduction.status == "reduced"
            assert reduction.target_powers == list(key)
            for term in reduction.terms:
                assert tuple(term.master_powers) in masters
                assert term.common_mass_squared_power == sum(term.master_powers) - sum(key)
            cache[key] = reduction
        return cache[key]

    return artifact_path, masters, reduce


def as_expression(reduction):
    integral = S("three_loop_test_I")
    return sum((E(term.unit_mass_coefficient) * integral(*term.master_powers)
                for term in reduction.terms), E("0"))


def test_all_single_dots_reduce_to_masters_and_obey_exact_homogeneity(closed_family):
    _, _, reduce = closed_family
    scalar = [1] * 6
    dotted = []
    for index in range(6):
        target = scalar.copy()
        target[index] += 1
        result = reduce(target)
        assert result.terms
        dotted.append(as_expression(result))
    assert len(reduce([2, 1, 1, 1, 1, 1]).terms) == 30
    dimension = E("rustred::{}::d")
    # With D_i = q_i^2 - 1, overall loop-momentum scaling fixes this sign.
    residual = sum(dotted, E("0")) - (3 * dimension / 2 - 6) * as_expression(reduce(scalar))
    assert residual.together() == E("0")


@pytest.mark.parametrize("target, coefficient", [
    ([2, 1, 1, 0, 0, 0], "(rustred::d-2)/2"),
    ([2, 2, 1, 0, 0, 0], "(rustred::d-2)^2/4"),
    ([3, 1, 1, 0, 0, 0], "(rustred::d-2)*(rustred::d-4)/8"),
    ([1, 1, 1, -1, 0, 0], "1"),
    ([1, 1, 1, -2, 0, 0], "1+4/rustred::d"),
])
def test_factorized_pinches_and_numerators(closed_family, target, coefficient):
    _, _, reduce = closed_family
    tadpoles = as_expression(reduce([1, 1, 1, 0, 0, 0]))
    residual = as_expression(reduce(target)) - E(coefficient) * tadpoles
    assert residual.together() == E("0")


def test_scaleless_sector_reduces_to_zero(closed_family):
    _, _, reduce = closed_family
    assert reduce([1, 1, 0, 0, 0, 0]).terms == []
    assert reduce([0, 0, 0, 0, 0, 0]).terms == []


def test_fresh_process_loads_only_the_written_closing_artifact(closed_family):
    path, masters, reduce = closed_family
    target = [2, 1, 1, 1, 1, 1]
    script = """
import json
from pathlib import Path
import sys
from symbolica.community import hepkit as hep
artifact = Path(sys.argv[1]).read_bytes()
result = hep.rustred.reduce_with_closing_artifact(artifact, [2, 1, 1, 1, 1, 1])
print(json.dumps({"status": result.status, "terms": [
    [term.master_powers, term.unit_mass_coefficient, term.common_mass_squared_power]
    for term in result.terms
]}))
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        check=True, text=True, capture_output=True, timeout=60,
    )
    cold = json.loads(child.stdout)
    assert cold["status"] == "reduced"
    cold_terms = {
        tuple(powers): (E(coefficient), mass_power)
        for powers, coefficient, mass_power in cold["terms"]
    }
    warm_terms = {
        tuple(term.master_powers): (E(term.unit_mass_coefficient), term.common_mass_squared_power)
        for term in reduce(target).terms
    }
    assert cold_terms.keys() == warm_terms.keys()
    assert cold_terms.keys() <= masters
    for powers, (coefficient, mass_power) in cold_terms.items():
        expected, expected_power = warm_terms[powers]
        assert (coefficient - expected).together() == E("0")
        assert mass_power == expected_power
