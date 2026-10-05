"""Small real-native smoke gates; the four-loop notebook is a separate run."""

import importlib

import pytest
from symbolica import S
from symbolica.community import hepkit as hep


def tadpole(*, cut=False):
    d, k, m2 = S("stream_test_d", "stream_test_k", "stream_test_m2")
    kin = hep.Kinematics(d, momenta=[k])
    family = hep.IntegralFamily(
        [k], [], [kin.scalar_product(k, k) - m2], kinematics=kin
    )
    return hep.IBPFamily(family, name="stream_test", cut=[cut])


def test_embedded_namespace_keeps_native_types_and_exceptions():
    native = hep.rustred
    package = importlib.import_module("symbolica.community.hepkit.rustred")
    for name in [
        "CandidateGenerationSession", "CandidateArtifact", "RustRedError",
        "TerminalNormalization",
        "RustRedInputError", "RustRedExecutionError", "RustRedLimitError",
        "RustRedCoordinatorPoisonedError",
    ]:
        assert getattr(package, name) is getattr(native, name)
    assert package.family_candidates is native.family_candidates


def test_native_family_generation_and_lazy_artifact():
    family = tadpole()
    session = family.start_generation(n_cores=1, event_capacity=4, numerical_depth=1)
    assert session.wait(timeout=120), "tiny native smoke did not finish"
    assert session.snapshot()["state"] == "completed"
    batch = session.poll_events(max_events=4)
    assert batch["schema"] == "rustred.candidate-generation-events.v1"
    assert batch["snapshot"]["done"]
    result = session.result()
    artifact = result.artifact()
    metadata = artifact.metadata()
    assert metadata["arity"] == 1
    assert metadata["total_rules"] > 0
    assert metadata["decoded_coefficients"] == 0
    assert metadata["closure_claim"] is False
    assert "closure certificate" in artifact._repr_html_()
    sectors = artifact.sectors()["items"]
    sector = next(s["ordinal"] for s in sectors if s["total_rules"])
    artifact.rules(sector)
    artifact.terminals(sector)
    rule = artifact.rule(sector, 0)
    assert artifact.metadata()["decoded_coefficients"] == 0
    coefficient = rule["rhs"][0]["coefficient_id"]
    detail = artifact.coefficient(coefficient)
    # These are bounded display strings, not an algebra round-trip format.
    assert detail["denominator"]
    assert detail["denominator_terms"] > 0
    assert artifact.metadata()["decoded_coefficients"] == 1
    with pytest.raises(hep.rustred.RustRedLimitError):
        artifact.coefficient(coefficient, max_output_bytes=1)
    reopened = hep.rustred.CandidateArtifact.open(result.bundle)
    assert reopened.metadata()["decoded_coefficients"] == 0
    assert reopened.metadata()["total_rules"] == metadata["total_rules"]


def test_generation_rejects_cuts_without_changing_cut_aware_bridge():
    family = tadpole(cut=True)
    with pytest.raises(ValueError, match="cut families"):
        family.start_generation(n_cores=1)
    assert family.reduce_laporta([[0]], max_depth=0).reduce([0]) == []


def test_build_capabilities_and_original_parameter_legend():
    family = tadpole()
    arities = family.compiled_runtime_arities()
    assert arities == sorted(set(arities)) and 1 in arities
    bindings = family.parameter_bindings
    assert bindings
    assert all(internal != original for internal, original in bindings)
    assert {original for _, original in bindings} == set(
        S("stream_test_d", "stream_test_m2")
    )


def test_explicit_native_terminal_normalization_and_bounded_views():
    family = tadpole()
    session = family.start_generation(n_cores=1, numerical_depth=1,
                                      bundle_max_entries=10000000)
    assert session.wait(timeout=120)
    result = session.result()
    artifact = result.artifact()
    before = artifact.metadata()
    normalization = family.normalize_candidate_terminals(artifact)
    metadata = normalization.metadata()
    assert metadata["exact_within_family"] is True
    assert metadata["closure_claim"] is False
    assert metadata["master_minimality_claim"] is False
    assert metadata["original_ibp_source_replay_claim"] is False
    assert metadata["raw_terminal_records"] == before["total_terminals"]
    assert metadata["unique_raw_terminals"] > 0
    assert len(normalization.terminals(start=0, limit=1)["items"]) <= 1
    rows = normalization.relations(start=0, limit=1)["items"]
    assert len(rows) == 1
    relation = normalization.relation(rows[0]["ordinal"], max_output_bytes=8192)
    assert relation["integral"] == rows[0]["integral"]
    if relation["rhs"]:
        cid = relation["rhs"][0]["coefficient_id"]
        assert normalization.coefficient(cid, max_output_bytes=8192)["denominator"]
    assert normalization.sidecar()
    # Normalization has a separate coefficient table and does not eagerly
    # decode the candidate recurrence coefficients.
    assert artifact.metadata() == before
    reopened = hep.rustred.CandidateArtifact.open(result.bundle, bundle_max_entries=10000000)
    assert reopened.metadata()["decoded_coefficients"] == 0
