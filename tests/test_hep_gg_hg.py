"""Lightweight stage safety checks; these never compute two-loop boundaries."""

import importlib.util
import hashlib
from pathlib import Path
from time import monotonic, sleep
from types import SimpleNamespace

import pytest

from symbolica import ComplexFloat, E, Float, S
from symbolica.community.hep import integration as numerical


EXAMPLES = Path(__file__).parents[1] / "examples" / "hep"
SPEC = importlib.util.spec_from_file_location("gg_hg_controller_test", EXAMPLES / "gg_hg_support.py")
SUPPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUPPORT)


@pytest.fixture
def session(tmp_path):
    session = SUPPORT.CalculationSession(
        EXAMPLES / "data" / "gg_hg" / "native-model.json", tmp_path,
    )
    try:
        yield session
    finally:
        session.close()


def populate_banks(session):
    """Use actual native persistence with a small, exactly constant system."""
    x, epsilon, master = S("gg_hg_control::x", "gg_hg_control::eps", "gg_hg_control::I")
    flow = numerical.KinematicTransport(
        epsilon, {x: [[E("0")]]}, [master], E("1"), branch_domain="real x",
    )

    def add(cache, point, provenance):
        return flow.add_boundary(
            cache, {x: E(point)}, [[ComplexFloat("1", decimal_digits=90)]], 0,
            verified_digits=60, comparison_errors=[[Float("0", decimal_digits=90)]],
            provenance=provenance,
        )

    seed = add(session.seeds, "0", "exact constant seed")
    session.cache.extend(session.seeds)
    add(session.cache, "1", "accepted intermediate; source: exact constant seed")
    session.seeds.save(session.directory / "seeds")
    session.cache.save(session.directory / "transport")
    return seed


def cancelled(*args, **kwargs):
    raise numerical.CalculationCancelled("controlled stage interruption")


def test_native_inputs_do_not_read_numerical_references(tmp_path, monkeypatch):
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert path.name not in {"coherent-reference.json", "amplitude-validation.json"}
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    session = SUPPORT.CalculationSession(
        EXAMPLES / "data" / "gg_hg" / "native-model.json", tmp_path,
    )
    try:
        assert len(session.configurations) == 16
        assert len(session.seeds) == len(session.cache) == 0
        for name, configuration in session.configurations:
            assert session._destination(session.systems[name], configuration) == configuration.destination
        assert session.results == {} and session.observables is None
    finally:
        session.close()


def test_interrupted_nearby_transport_cannot_mix_physical_points(session):
    old = SimpleNamespace(coefficients=[["old point"]])
    session.results = {configuration.label: old for _, configuration in session.configurations}
    session.observables = SimpleNamespace(verified_relative_digits={"old": 50})
    session.form_factors = {"W": old, "Z": old}
    new = SimpleNamespace(coefficients=[["new point"]])
    calls = 0

    def evaluate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            cancelled()
        return new

    for name, system in list(session.systems.items()):
        session.systems[name] = SimpleNamespace(coordinates=system.coordinates, evaluate=evaluate)
    previous_point = session.point[:]
    session.submit("transport", nearby=True)
    with pytest.raises(numerical.CalculationCancelled):
        session.wait()
    assert session.point[0] == previous_point[0] + E("1/100000")
    assert list(session.results.values()) == [new]
    assert session.observables is None and session.form_factors == {}
    assert "CalculationCancelled" in session.snapshot()["status"]
    with pytest.raises(RuntimeError, match="sixteen configurations"):
        session.assemble()


def test_interrupted_forced_boundaries_cannot_restore_old_numerical_banks(session, monkeypatch):
    populate_banks(session)
    session.results = {"old": object()}
    session.observables = object()
    exact = session.directory / "exact-reductions" / "existing-exact-entry"
    exact.parent.mkdir()
    exact.write_text("retained exact work")
    sample = session.directory / "completed-samples" / "existing-sample"
    sample.parent.mkdir()
    sample.write_text("retained checkpoint, ignored during forced recomputation")
    options = []
    native_options = numerical.EvaluationOptions

    def record_options(**kwargs):
        options.append(kwargs)
        return native_options(**kwargs)

    monkeypatch.setattr(numerical, "EvaluationOptions", record_options)
    session.systems = {name: SimpleNamespace(generate_boundary=cancelled) for name in session.systems}
    session.submit("boundaries", recompute=True)
    with pytest.raises(numerical.CalculationCancelled):
        session.wait()
    assert len(numerical.BoundaryCache.load(session.directory / "seeds")) == 0
    assert len(numerical.BoundaryCache.load(session.directory / "transport")) == 0
    assert session.results == {} and session.observables is None
    assert options[0]["reuse_samples"] is False
    assert options[0]["cache_directory"] == exact.parent
    assert exact.read_text() == "retained exact work"
    assert sample.exists()


def test_interrupted_forced_transport_persists_only_saved_seeds(session):
    populate_banks(session)
    seed_evidence = SUPPORT.boundary_evidence(session.seeds)
    for name, system in list(session.systems.items()):
        session.systems[name] = SimpleNamespace(coordinates=system.coordinates, evaluate=cancelled)
    session.submit("transport", recompute=True)
    with pytest.raises(numerical.CalculationCancelled):
        session.wait()
    restored = numerical.BoundaryCache.load(session.directory / "transport")
    assert SUPPORT.boundary_evidence(restored) == seed_evidence


def test_loading_boundaries_preserves_accumulated_points_and_provenance(session):
    seed = populate_banks(session)
    before = SUPPORT.boundary_evidence(session.cache)
    session.systems = {
        name: SimpleNamespace(generate_boundary=lambda *args, **kwargs: seed)
        for name in session.systems
    }
    session.submit("boundaries")
    assert len(session.wait()) == 16
    restored = numerical.BoundaryCache.load(session.directory / "transport")
    assert SUPPORT.boundary_evidence(restored) == before


def test_repeated_hit_check_requires_completed_transport(session):
    with pytest.raises(RuntimeError, match="Complete transport"):
        session.restart_and_repeat()


@pytest.fixture
def acceptance(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    spec = importlib.util.spec_from_file_location("gg_hg_acceptance_test", EXAMPLES / "gg_hg_acceptance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_forced_refinement_interruption_retains_only_new_precision_restart_state(session, acceptance):
    populate_banks(session)
    session.seed_digits = 40
    directory = session.directory / "completed-samples"
    directory.mkdir()
    (directory / "sample-cold-30.bin").write_bytes(b"previous cold work")

    def samples():
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in directory.glob("sample-*.bin")}

    def new_sample_then_cancel(evaluator, cache, *args, **kwargs):
        assert kwargs["recompute"] is True
        assert len(cache) == 0
        (directory / "sample-forced-40.bin").write_bytes(b"new complete refined sample")
        deadline = monotonic() + 5
        while not kwargs["control"].cancelled:
            if monotonic() >= deadline:
                raise AssertionError("Interruption monitor did not request cancellation.")
            sleep(0.001)
        cancelled()

    session.systems = {
        name: SimpleNamespace(generate_boundary=new_sample_then_cancel) for name in session.systems
    }
    record, persisted = acceptance.interrupt_forced_boundaries(
        session, samples, 1, poll_interval=0.001,
    )
    assert record["phase"] == "forced_refinement" and record["seed_digits"] == 40
    assert record["baseline_sample_files"] == record["new_complete_samples"] == 1
    assert record["retained_sample_files"] == 2
    assert record["elapsed_ns"] > record["cancellation_latency_ns"] > 0
    assert record["verified_configurations_before_cancel"] == 0
    assert persisted == samples()
    assert len(numerical.BoundaryCache.load(session.directory / "seeds")) == 0
    assert len(numerical.BoundaryCache.load(session.directory / "transport")) == 0


def test_interruption_monitor_preserves_native_failures(session, acceptance):
    def reduction_failure(*args, **kwargs):
        raise numerical.IncompleteReductionError("no complete sample exists")

    session.systems = {
        name: SimpleNamespace(generate_boundary=reduction_failure) for name in session.systems
    }
    with pytest.raises(numerical.IncompleteReductionError, match="no complete sample"):
        acceptance.interrupt_forced_boundaries(session, lambda: {}, 1, poll_interval=0.001)
