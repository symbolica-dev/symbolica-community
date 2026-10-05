"""Euclidean acceptance orchestration and comparison; no two-loop computation."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from symbolica import ComplexFloat, Float, S
from symbolica.community.hep import integration as numerical


EXAMPLES = Path(__file__).parents[1] / "examples" / "hep"


@pytest.fixture
def anchors(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    spec = importlib.util.spec_from_file_location("gg_hg_anchor_test", EXAMPLES / "gg_hg_anchor_acceptance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_result(anchors, topology):
    """Comparison-only test fixture; never a production boundary provider."""
    specification = anchors.ANCHORS[topology]
    system = SimpleNamespace(
        coordinates=S(*(f"anchor_comparison::{topology}_{name}" for name in ("s", "t", "b"))),
        roots={S(f"anchor_comparison::{topology}_root{i}"): None
               for i in range(specification["roots"])},
        dimension=specification["dimension"], provenance="synthetic test system",
    )
    reference = json.loads((anchors.DATA / f"{topology}-regenerated-anchor.json").read_text())
    values = reference.get("coefficients", reference.get("coefficients_by_epsilon"))
    result = SimpleNamespace(
        leading_power=0, coordinates=anchors.native_point(system, topology)[0],
        root_sheets=anchors.native_point(system, topology)[1],
        verified_digits=20, working_bits=333, identity=f"test-only-{topology}", cache_hit=False,
        provenance="Native automatic auxiliary-mass boundary; no numerical reference seed; synthetic TEST fixture",
        coefficients=[[ComplexFloat(value["real"], value["imaginary"], decimal_digits=100)
                       for value in row] for row in values],
        comparison_errors=[[Float("1e-50", decimal_digits=100) for _ in row] for row in values],
    )
    return system, result, reference


@pytest.mark.parametrize("topology,count", [("planar", 240), ("nonplanar", 305)])
def test_anchor_comparison_preserves_recorded_allowances_including_zeros(anchors, topology, count):
    system, result, reference = synthetic_result(anchors, topology)
    errors = reference.get("absolute_errors", reference.get("source_absolute_errors_by_epsilon"))
    allowance = ComplexFloat(errors[0][0], decimal_digits=100)
    result.coefficients[0][0] += allowance / 2
    comparison = anchors.compare_anchor(result, system, topology, digits=20)
    assert comparison["passed"] and comparison["checked_coefficients"] == count
    assert comparison["reference_accuracy_cap_digits"] == 40
    assert comparison["reference"]["report_sha256"] == reference["report_sha256"]
    assert comparison["reference"]["system_source"] == reference["system_source"]
    result.coefficients[0][0] += allowance * 2
    comparison = anchors.compare_anchor(result, system, topology, digits=20)
    assert not comparison["passed"]
    assert comparison["failures"][0]["canonical_index"] == 1


def test_anchor_comparison_rejects_inflated_native_uncertainty(anchors):
    system, result, _ = synthetic_result(anchors, "planar")
    result.comparison_errors[0][0] = Float("1e-10", decimal_digits=100)
    with pytest.raises(ValueError, match="mixed accuracy budget"):
        anchors.compare_anchor(result, system, "planar", digits=20)


def arguments(directory):
    return SimpleNamespace(directory=directory, digits=20, guard_digits=40, order=80,
                           workers=1, max_steps=1000, max_precision_attempts=3,
                           case_batch=16, force=False, resume=False, cancel_file=None)


def fake_generators(anchors, monkeypatch, *, fail_second=False):
    calls = []
    systems = {}
    for topology in anchors.ANCHORS:
        system, result, _ = synthetic_result(anchors, topology)

        def generate(evaluator, cache, point, sheets, *, last, recompute, control,
                     topology=topology, result=result):
            assert point == result.coordinates and sheets == result.root_sheets
            assert last == 4 and not recompute
            assert evaluator.options.digits == 20
            calls.append(topology)
            if fail_second and topology == "nonplanar":
                raise numerical.IncompleteReductionError("synthetic second-family interruption")
            return result

        system.generate_boundary = generate
        systems[topology] = system
    monkeypatch.setattr(numerical, "HiggsJetIntegralSystem", lambda topology: systems[topology])
    return calls


def test_anchor_runner_reads_references_only_after_both_native_calls(anchors, tmp_path, monkeypatch):
    calls = fake_generators(anchors, monkeypatch)
    read_bytes = Path.read_bytes
    reference_reads = []

    def guarded_read(path):
        if path.parent == anchors.DATA:
            assert calls == ["planar", "nonplanar"]
            reference_reads.append(path.name)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    report = anchors.run(arguments(tmp_path))
    assert report["status"] == "passed" and report["checked_coefficients"] == 545
    assert len(reference_reads) == 2
    assert len(list((tmp_path / "runs").glob("*.json"))) == 1
    assert all((tmp_path / name / "boundaries" / "physical-boundaries.bin").exists()
               for name in anchors.ANCHORS)
    from gg_hg_acceptance import verify_anchor_report

    proof = verify_anchor_report(tmp_path / "anchor-acceptance.json", report["runtime_attestation"])
    assert proof["checked_coefficients"] == 545 and proof["minimum_verified_digits"] == 20
    assert len(reference_reads) == 2, "Proof verification must not open numerical reference fixtures."


def test_failed_second_native_call_never_reads_reference_and_keeps_first_bank(anchors, tmp_path, monkeypatch):
    fake_generators(anchors, monkeypatch, fail_second=True)
    read_bytes = Path.read_bytes

    def guarded_read(path):
        assert path.parent != anchors.DATA, "Numerical reference read before both native successes"
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    with pytest.raises(numerical.IncompleteReductionError, match="second-family"):
        anchors.run(arguments(tmp_path))
    report = json.loads((tmp_path / "anchor-acceptance.json").read_text())
    assert report["status"] == "incomplete_or_failed"
    assert report["references_loaded"] is False and report["comparisons"] == {}
    assert set(report["native"]) == {"planar"}
    assert (tmp_path / "planar" / "boundaries" / "physical-boundaries.bin").exists()


def test_forced_anchor_interruption_invalidates_both_banks_but_keeps_exact_work(anchors, tmp_path, monkeypatch):
    systems = {name: synthetic_result(anchors, name)[0] for name in anchors.ANCHORS}
    options = []
    original_options = numerical.EvaluationOptions

    def record_options(**kwargs):
        options.append(kwargs)
        return original_options(**kwargs)

    def interrupt(*args, **kwargs):
        assert kwargs["recompute"] is True
        raise numerical.CalculationCancelled("synthetic forced interruption")

    for name, system in systems.items():
        system.generate_boundary = interrupt
        bank = tmp_path / name / "boundaries" / "physical-boundaries.bin"
        bank.parent.mkdir(parents=True)
        bank.write_bytes(b"old incompatible bank, bypassed by explicit force")
        exact = tmp_path / name / "exact-reductions" / "retained"
        exact.parent.mkdir()
        exact.write_bytes(b"exact work")
        sample = tmp_path / name / "completed-samples" / "sample-old.bin"
        sample.parent.mkdir()
        sample.write_bytes(b"old completed numerical sample")
    monkeypatch.setattr(numerical, "HiggsJetIntegralSystem", lambda topology: systems[topology])
    monkeypatch.setattr(numerical, "EvaluationOptions", record_options)
    args = arguments(tmp_path)
    args.force = True
    with pytest.raises(numerical.CalculationCancelled):
        anchors.run(args)
    assert options[0]["reuse_samples"] is False
    report = json.loads((tmp_path / "anchor-acceptance.json").read_text())
    for name in systems:
        assert len(numerical.BoundaryCache.load(tmp_path / name / "boundaries")) == 0
        assert (tmp_path / name / "exact-reductions" / "retained").read_bytes() == b"exact work"
        assert not (tmp_path / name / "completed-samples" / "sample-old.bin").exists()
        archived = Path(report["archived_numerical_directories"][f"{name}/completed-samples"])
        assert (archived / "sample-old.bin").read_bytes() == b"old completed numerical sample"


def test_incomplete_forced_directory_preparation_refuses_resume(anchors, tmp_path, monkeypatch):
    systems = {name: synthetic_result(anchors, name)[0] for name in anchors.ANCHORS}
    for name in systems:
        directory = tmp_path / name / "completed-samples"
        directory.mkdir(parents=True)
        (directory / "sample-old.bin").write_bytes(b"prior sample")
    monkeypatch.setattr(numerical, "HiggsJetIntegralSystem", lambda topology: systems[topology])
    original_rename = Path.rename

    def fail_second_directory(path, destination):
        if path == tmp_path / "nonplanar" / "completed-samples":
            raise OSError("synthetic interrupted force preparation")
        return original_rename(path, destination)

    monkeypatch.setattr(Path, "rename", fail_second_directory)
    args = arguments(tmp_path)
    args.force = True
    with pytest.raises(OSError, match="force preparation"):
        anchors.run(args)
    assert (tmp_path / "force-preparation-pending.json").exists()
    assert not (tmp_path / "planar" / "completed-samples").exists()
    assert (tmp_path / "nonplanar" / "completed-samples" / "sample-old.bin").exists()
    args.force, args.resume = False, True
    with pytest.raises(RuntimeError, match="rerun with --force before --resume"):
        anchors.run(args)


@pytest.mark.parametrize("changed,message", [
    ("native_identity", "native source identity"),
    ("extension", "installed extension"),
    ("steering", "steering sources"),
    ("anchor_runner", "anchor runner"),
    ("incomplete", "Euclidean proof"),
    ("accuracy", "anchor evidence"),
    ("shape", "canonical dimension"),
    ("inflated_error", "mixed accuracy budget"),
])
def test_anchor_proof_rejects_other_sources_or_incomplete_accuracy(
    anchors, tmp_path, monkeypatch, changed, message,
):
    fake_generators(anchors, monkeypatch)
    report = anchors.run(arguments(tmp_path))
    current = copy.deepcopy(report["runtime_attestation"])
    if changed == "native_identity":
        report["runtime_attestation"]["native_identity_witness"]["identity"] = "other-native-graph"
    elif changed == "extension":
        report["runtime_attestation"]["loaded_extension"]["sha256"] = "different-extension"
    elif changed == "steering":
        report["runtime_attestation"]["execution_sources"]["controller"]["sha256"] = "different-controller"
    elif changed == "anchor_runner":
        report["anchor_runner"]["sha256"] = "different-runner"
    elif changed == "incomplete":
        report["checked_coefficients"] = 240
    elif changed == "accuracy":
        report["native"]["nonplanar"]["verified_digits"] = 19
    elif changed == "shape":
        report["native"]["nonplanar"]["coefficients"][0].pop()
    else:
        report["native"]["planar"]["comparison_errors"][0][0] = "1e-10"
    path = tmp_path / "modified-proof.json"
    path.write_text(json.dumps(report))
    from gg_hg_acceptance import verify_anchor_report

    with pytest.raises(ValueError, match=message):
        verify_anchor_report(path, current)
