"""Real native closure/reduction gates for the explicit three-loop notebook."""

import importlib.util
import json
from collections import Counter
from pathlib import Path
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 and 3.10
    import tomli as tomllib

import pytest

pytest.importorskip("symbolica")
pytest.importorskip("marimo")
from symbolica import E, S
from symbolica.community import hepkit as hep


SPEC = importlib.util.spec_from_file_location(
    "three_loop_reduction_notebook",
    Path(__file__).parents[1] / "examples/hep/three_loop_reduction.py",
)
support = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(support)


def test_generation_requires_explicit_start_and_cannot_restart():
    class Native:
        def __init__(self):
            self.calls = []

        def execution_capabilities(self):
            return hep.rustred.execution_capabilities()

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


def graph_family_and_source():
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
    return family, source


def test_graph_routing_matches_the_certified_source():
    family, source = graph_family_and_source()
    support.assert_source_matches_family(source, family)
    assert hep.IBPFamily(family, name="K6_graph_test").denominator_count == 6
    with pytest.raises(AssertionError):
        support.assert_source_matches_family(source.replace("k1^2-1", "k1^2+1"), family)


def test_native_terminal_normalization_identifies_five_integral_types():
    family, source = graph_family_and_source()
    support.assert_source_matches_family(source, family)
    ibp = hep.IBPFamily(family, name="K6_terminal_normalization_test")
    session = ibp.start_generation(n_cores=1)
    assert session.wait(timeout=30), "small K6 candidate generation timed out"
    candidate = session.result().artifact()

    # Use the candidate from this exact graph family: the source-input closing
    # fixture has a different family fingerprint despite matching denominators.
    def raw_terminals():
        sectors = candidate.sectors(start=0, limit=100)
        assert len(sectors["items"]) == sectors["total"]
        result = set()
        for sector in sectors["items"]:
            page = candidate.terminals(sector["ordinal"], start=0, limit=100)
            assert len(page["items"]) == page["total"]
            result.update(tuple(powers) for powers in page["items"])
        return result

    raw = raw_terminals()
    assert len(raw) == 38
    assert all(set(powers) <= {0, 1} for powers in raw)
    normalized = ibp.normalize_candidate_terminals(candidate)
    metadata = normalized.metadata()
    assert metadata["unique_raw_terminals"] == 38
    assert metadata["unit_aliases"] == 33
    assert metadata["canonical_terminals"] == 5
    assert metadata["skipped"] == []
    assert metadata["exact_within_family"] is True
    assert metadata["closure_claim"] is False
    assert metadata["master_minimality_claim"] is False

    # Native canonical keys in the example's checked denominator order. These
    # are exact integral types, not an additional proof of master minimality.
    expected = {
        (0, 0, 1, 0, 1, 1): 16,  # T3,1: three one-loop tadpoles
        (0, 0, 1, 1, 1, 1): 12,  # T4,1: sunset times one-loop tadpole
        (0, 1, 1, 1, 1, 0): 3,   # T4,2: three-loop basketball
        (0, 1, 1, 1, 1, 1): 6,   # T5,1: five-line vacuum
        (1, 1, 1, 1, 1, 1): 1,   # T6,1: tetrahedron / Mercedes
    }
    terminals = normalized.terminals(start=0, limit=100)
    assert terminals["total"] == len(terminals["items"]) == 5
    assert {tuple(powers) for powers in terminals["items"]} == set(expected)
    relations = normalized.relations(start=0, limit=100)
    assert relations["total"] == len(relations["items"]) == 38
    seen, multiplicities = set(), Counter()
    for row in relations["items"]:
        relation = normalized.relation(row["ordinal"])
        integral = tuple(relation["integral"])
        assert integral == tuple(row["integral"])
        assert integral not in seen
        seen.add(integral)
        assert len(relation["rhs"]) == 1
        term = relation["rhs"][0]
        representative = tuple(term["integral"])
        assert representative in expected
        coefficient = normalized.coefficient(term["coefficient_id"])
        assert coefficient["numerator"] == coefficient["denominator"] == "1"
        multiplicities[representative] += 1
    assert seen == raw
    assert multiplicities == expected
    assert raw_terminals() == raw  # Normalization leaves the candidate unchanged.


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
