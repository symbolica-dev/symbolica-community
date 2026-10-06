"""Lightweight stage safety checks; these never compute two-loop boundaries."""

import importlib.util
import asyncio
import hashlib
import json
import runpy
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from threading import Event, Lock
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


def constant_boundary_adder():
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

    return flow, x, add


def populate_banks(session):
    """Use actual native persistence with a small, exactly constant system."""
    _, _, add = constant_boundary_adder()
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


def test_cooperative_transport_checkpoints_yields_and_cancels(tmp_path, monkeypatch):
    def no_threads(*args, **kwargs):
        raise AssertionError("Cooperative execution must not create worker threads")

    monkeypatch.setattr(SUPPORT, "ThreadPoolExecutor", no_threads)
    session = SUPPORT.CalculationSession(
        EXAMPLES / "data" / "gg_hg" / "native-model.json", tmp_path, cooperative=True,
    )
    flow, coordinate, add = constant_boundary_adder()
    add(session.cache, "0", "exact constant seed")
    session.configurations = [
        ("constant", SimpleNamespace(label=f"point-{i}", root_sheets={}))
        for i in range(3)
    ]
    session.systems = {"constant": SimpleNamespace(
        evaluate=lambda cache, destination, sheets, **kw: flow.evaluate(
            cache, destination, 0, 0, admit_straight_path=True, control=kw["control"],
        ),
    )}
    session._destination = lambda system, configuration: {
        coordinate: E(str(int(configuration.label[-1]) + 1)),
    }

    async def run():
        session.submit("transport")
        with pytest.raises(RuntimeError, match="wait_async"):
            session.wait()
        with pytest.raises(RuntimeError, match="already running"):
            session.submit("transport")
        # A browser event gets a turn between synchronous native calls.
        while not session.results and not session._future.done():
            await asyncio.sleep(0)
        if session._future.done():
            await session.wait_async()
        assert list(session.results) == ["point-0"]
        saved = numerical.BoundaryCache.load(tmp_path / "transport")
        assert SUPPORT.boundary_evidence(saved) == SUPPORT.boundary_evidence(session.cache)
        session.cancel()
        with pytest.raises(numerical.CalculationCancelled):
            await session.wait_async()
        assert list(session.results) == ["point-0"]
        assert "CalculationCancelled" in session.snapshot()["status"]
        session.submit("transport")
        result = await session.wait_async()
        assert len(result) == 3 and result["point-0"].cache_hit
        assert session.wait() == result
        session.submit("restart")
        assert all(r.cache_hit for r in (await session.wait_async()).values())

    try:
        asyncio.run(run())
    finally:
        session.close()


def test_cooperative_wait_timeout_retains_work(tmp_path):
    session = SUPPORT.CalculationSession(
        EXAMPLES / "data" / "gg_hg" / "native-model.json", tmp_path, cooperative=True,
    )

    async def run():
        ready = asyncio.Event()
        session._future = asyncio.create_task(ready.wait())
        with pytest.raises(TimeoutError):
            await session.wait_async(timeout=0)
        assert not session._future.cancelled()
        ready.set()
        assert await session.wait_async() is True

    try:
        asyncio.run(run())
    finally:
        session.close()


def test_supplied_only_build_rejects_native_recomputation_before_reset(session, monkeypatch):
    populate_banks(session)
    before = SUPPORT.boundary_evidence(session.cache)
    session.automatic_boundary_generation_available = False
    with pytest.raises(RuntimeError, match="requires a native build"):
        session.submit("boundaries", recompute=True)
    with pytest.raises(RuntimeError, match="requires a native build"):
        session.generate_boundaries(recompute=True)
    assert SUPPORT.boundary_evidence(session.cache) == before
    assert not (session.directory / "numerical-history").exists()


def test_insufficient_supplied_accuracy_preserves_banks_without_native_refinement(session, monkeypatch):
    populate_banks(session)
    before = SUPPORT.boundary_evidence(session.cache)
    session.automatic_boundary_generation_available = False
    session.results = {configuration.label: object() for _, configuration in session.configurations}
    session.amplitude = SimpleNamespace(evaluate=lambda *a, **kw: SimpleNamespace(
        verified_relative_digits={"ew": 19, "heft": 40, "interference": 21},
    ))
    monkeypatch.setattr(numerical, "HiggsJetFormFactorProjector", lambda: SimpleNamespace(
        evaluate=lambda *a: SimpleNamespace(values=[], absolute_errors=[], provenance="supplied input"),
    ))

    def must_not_regenerate(**kwargs):
        raise AssertionError("A supplied-only build cannot launch native boundary generation")

    session.generate_boundaries = must_not_regenerate
    with pytest.raises(numerical.AccuracyError, match="Import more accurate boundaries"):
        session.assemble(recompute=True)
    assert SUPPORT.boundary_evidence(session.cache) == before
    assert not (session.directory / "numerical-history").exists()
    assert session.observables is None and session.form_factors == {}


def test_boundary_step_budget_preserves_precision_and_physical_transport(session, monkeypatch):
    native_options = numerical.EvaluationOptions
    requested = []

    def record_options(**kwargs):
        requested.append(kwargs)
        return native_options(**kwargs)

    monkeypatch.setattr(numerical, "EvaluationOptions", record_options)
    session.seed_digits = 40
    physical = session._options()
    boundary = session._options(seeds=True)
    assert requested[0]["max_steps"] == 1000
    assert requested[1]["max_steps"] == 2000
    assert physical.digits == session.digits
    assert boundary.digits == 40
    for options, guard, order in ((physical, 20, 16), (boundary, 60, 96)):
        assert options.guard_digits == guard
        assert options.series_order == order
        # Exercise the native options handoff without evaluating an integral.
        prepared = numerical.IntegralEvaluator(options=options).options
        assert prepared.digits == options.digits
        assert prepared.guard_digits == options.guard_digits
        assert prepared.series_order == options.series_order


def test_interrupted_nearby_transport_cannot_mix_physical_points(session):
    old = SimpleNamespace(coefficients=[["old point"]])
    session.results = {configuration.label: old for _, configuration in session.configurations}
    session.observables = SimpleNamespace(verified_relative_digits={"old": 50})
    session.form_factors = {"W": old, "Z": old}
    new = SimpleNamespace(coefficients=[["new point"]], cache_hit=False, inserted_points=0)
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
    assert not sample.exists()
    generation = session.numerical_generation()
    archived = session.directory / generation["archive_directory"]
    assert (archived / "completed-samples" / sample.name).read_text() == (
        "retained checkpoint, ignored during forced recomputation"
    )
    assert len(numerical.BoundaryCache.load(archived / "seeds")) == 1
    assert len(numerical.BoundaryCache.load(archived / "transport")) == 2


@pytest.mark.parametrize("complete_configuration", [False, True])
def test_same_precision_forced_restart_excludes_untouched_old_samples(
    session, monkeypatch, complete_configuration,
):
    populate_banks(session)
    session.configurations = [
        session.configurations[0],
        next(item for item in session.configurations if item[0] == "nonplanar"),
    ]
    raw = session.directory / "completed-samples"
    raw.mkdir()
    (raw / "sample-planar.bin").write_bytes(b"deterministic planar payload")
    (raw / "sample-nonplanar.bin").write_bytes(b"old untouched nonplanar payload")
    exact = session.directory / "exact-reductions" / "entry"
    exact.parent.mkdir()
    exact.write_bytes(b"retained exact reduction")
    _, _, add = constant_boundary_adder()
    options = []
    native_options = numerical.EvaluationOptions

    def capture(**kwargs):
        options.append(kwargs)
        return native_options(**kwargs)

    monkeypatch.setattr(numerical, "EvaluationOptions", capture)

    def planar(evaluator, cache, *args, **kwargs):
        assert not (raw / "sample-nonplanar.bin").exists()
        # Recomputed same-precision values may have identical keys AND bytes.
        (raw / "sample-planar.bin").write_bytes(b"deterministic planar payload")
        if not complete_configuration:
            cancelled()
        add(cache, "2", "new planar boundary")
        return SimpleNamespace(cache_hit=False)

    session.systems = {
        "planar": SimpleNamespace(generate_boundary=planar),
        "nonplanar": SimpleNamespace(generate_boundary=cancelled),
    }
    session.submit("boundaries", recompute=True)
    with pytest.raises(numerical.CalculationCancelled):
        session.wait()
    generation = session.numerical_generation()
    assert all(option["digits"] == 30 and not option["reuse_samples"] for option in options)
    assert len(session.seeds) == int(complete_configuration)
    session.close()
    resumed = SUPPORT.CalculationSession(
        EXAMPLES / "data" / "gg_hg" / "native-model.json", session.directory,
        seed_digits=30,
    )
    try:
        resumed.configurations = session.configurations
        assert resumed.numerical_generation() == generation
        assert len(resumed.seeds) == int(complete_configuration)
        resumed_options = len(options)

        def reuse_planar(*args, **kwargs):
            assert (raw / "sample-planar.bin").read_bytes() == b"deterministic planar payload"
            assert not (raw / "sample-nonplanar.bin").exists()
            return SimpleNamespace(cache_hit=False)

        def compute_nonplanar(*args, **kwargs):
            assert not (raw / "sample-nonplanar.bin").exists()
            (raw / "sample-nonplanar.bin").write_bytes(b"new nonplanar work")
            return SimpleNamespace(cache_hit=False)

        resumed.systems = {
            "planar": SimpleNamespace(generate_boundary=reuse_planar),
            "nonplanar": SimpleNamespace(generate_boundary=compute_nonplanar),
        }
        resumed.submit("boundaries")
        assert len(resumed.wait()) == 2
        assert all(option["reuse_samples"] and option["digits"] == 30
                   for option in options[resumed_options:])
        assert resumed.numerical_generation() == generation
        archived = session.directory / generation["archive_directory"] / "completed-samples"
        assert (archived / "sample-nonplanar.bin").read_bytes() == b"old untouched nonplanar payload"
        assert exact.read_bytes() == b"retained exact reduction"
    finally:
        resumed.close()


def test_force_preparation_failure_requires_explicit_recovery_in_fresh_session(session, monkeypatch):
    populate_banks(session)
    raw = session.directory / "completed-samples"
    raw.mkdir()
    (raw / "sample-old.bin").write_bytes(b"pre-force sample")
    real_rename = Path.rename
    failed = False

    def interrupted_rename(path, target):
        nonlocal failed
        if path == session.directory / "transport" and not failed:
            failed = True
            raise OSError("controlled failure between numerical-bank renames")
        return real_rename(path, target)

    monkeypatch.setattr(Path, "rename", interrupted_rename)
    session.submit("boundaries", recompute=True)
    with pytest.raises(OSError, match="between numerical-bank renames"):
        session.wait()
    assert (session.directory / "force-preparation-pending.json").exists()
    assert (session.directory / "transport" / "physical-boundaries.bin").exists()
    session.close()
    recovery = SUPPORT.CalculationSession(
        EXAMPLES / "data" / "gg_hg" / "native-model.json", session.directory,
    )
    try:
        assert len(recovery.seeds) == len(recovery.cache) == 0
        for stage in ("boundaries", "transport", "amplitude", "restart"):
            recovery.submit(stage)
            with pytest.raises(RuntimeError, match="Recompute boundaries"):
                recovery.wait()
        recovery.systems = {name: SimpleNamespace(generate_boundary=cancelled)
                            for name in recovery.systems}
        recovery.submit("boundaries", recompute=True)
        with pytest.raises(numerical.CalculationCancelled):
            recovery.wait()
        assert not (session.directory / "force-preparation-pending.json").exists()
        assert recovery.numerical_generation()["generation"] != "initial"
        assert not list(raw.glob("sample-*.bin"))
        history = session.directory / "numerical-history"
        assert [p.read_bytes() for p in history.rglob("sample-old.bin")] == [b"pre-force sample"]
        assert len(list(history.rglob("previous-preparation.json"))) == 1
    finally:
        recovery.close()


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


def test_exact_transport_hits_persist_unsaved_points_once(session, monkeypatch):
    populate_banks(session)
    flow, x, add = constant_boundary_adder()
    add(session.cache, "2", "valid in-memory point after a failed prior save")
    before = SUPPORT.boundary_evidence(session.cache)
    real_save, saves = numerical.BoundaryCache.save, []

    def save(cache, directory):
        saves.append(directory)
        return real_save(cache, directory)

    monkeypatch.setattr(numerical.BoundaryCache, "save", save)
    for name, system in list(session.systems.items()):
        session.systems[name] = SimpleNamespace(
            coordinates=system.coordinates,
            evaluate=lambda cache, *args, **kwargs: flow.evaluate(cache, {x: E("2")}, 0, 0),
        )
    results = session.transport()
    assert len(results) == 16
    assert all(result.cache_hit and result.steps == 0 for result in results.values())
    assert saves == [session.directory / "transport"]
    restored = numerical.BoundaryCache.load(session.directory / "transport")
    assert SUPPORT.boundary_evidence(restored) == before


def test_transport_save_failure_retries_without_losing_new_points(session, monkeypatch):
    populate_banks(session)
    flow, x, _ = constant_boundary_adder()
    real_save, saves = numerical.BoundaryCache.save, []

    def save(cache, directory):
        saves.append(directory)
        if len(saves) == 2:
            raise OSError("controlled checkpoint failure after successful transport")
        return real_save(cache, directory)

    monkeypatch.setattr(numerical.BoundaryCache, "save", save)
    for name, system in list(session.systems.items()):
        session.systems[name] = SimpleNamespace(
            coordinates=system.coordinates,
            evaluate=lambda cache, *args, **kwargs: flow.evaluate(
                cache, {x: E("2")}, 0, 0, admit_straight_path=True,
            ),
        )
    with pytest.raises(OSError, match="checkpoint failure"):
        session.transport()
    before = SUPPORT.boundary_evidence(session.cache)
    assert len(numerical.BoundaryCache.load(session.directory / "transport")) < len(session.cache)
    results = session.transport()
    assert all(result.cache_hit and result.steps == 0 for result in results.values())
    assert len(saves) == 3  # initial save, failed new-point save, durable retry
    restored = numerical.BoundaryCache.load(session.directory / "transport")
    assert SUPPORT.boundary_evidence(restored) == before


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


@pytest.mark.parametrize("failure_stage", ["admission", "second_projection", "observable_evaluation"])
def test_forced_amplitude_failure_cannot_restore_old_derived_values(session, monkeypatch, failure_stage):
    old = SimpleNamespace(verified_relative_digits={"ew": 60})
    session.observables, session.form_factors = old, {"old": object()}
    if failure_stage != "admission":
        session.results = {configuration.label: object() for _, configuration in session.configurations}
    calls = []

    def project(*args):
        calls.append(args)
        if len(calls) == 2 and failure_stage == "second_projection":
            cancelled()
        return SimpleNamespace(values=[], absolute_errors=[], provenance="new projection")

    exact_kernel = SimpleNamespace(evaluate=cancelled)
    session.amplitude = exact_kernel
    projector = SimpleNamespace(evaluate=project)
    monkeypatch.setattr(numerical, "HiggsJetFormFactorProjector", lambda: projector)
    session.submit("amplitude", recompute=True)
    with pytest.raises(RuntimeError if failure_stage == "admission" else numerical.CalculationCancelled):
        session.wait()
    assert session.observables is None and session.form_factors == {}
    assert session.amplitude is exact_kernel
    if failure_stage != "admission":
        assert len(session.results) == 16

        def restarted_projection(*args):
            raise RuntimeError("load-or-assemble must retry projection")

        monkeypatch.setattr(projector, "evaluate", restarted_projection)
        session.submit("amplitude")
        with pytest.raises(RuntimeError, match="must retry projection"):
            session.wait()


def test_forced_amplitude_uses_current_exact_kinematics_and_all_transport_blocks(session, monkeypatch):
    old = SimpleNamespace(verified_relative_digits={"ew": 60})
    new = SimpleNamespace(verified_relative_digits={"ew": 60, "heft": 60, "interference": 60})
    session.observables, session.form_factors = old, {"old": object()}
    session.results = {configuration.label: object() for _, configuration in session.configurations}
    session.point = [E("71"), E("-29/3"), E("1")]
    projection_calls, evaluation_calls, constructions = [], [], []
    projected = []

    def project(*args):
        projection_calls.append(args)
        mass = ("W", "Z")[(len(projection_calls) - 1) % 2]
        assert list(args[:3]) == session.point and args[3] == session.masses[mass]
        for topology, actual in zip(("planar", "nonplanar"), args[4:]):
            ordered = sorted((c for name, c in session.configurations
                              if name == topology and c.mass == mass), key=lambda c: c.permutation)
            assert actual == [session.results[c.label] for c in ordered]
        result = SimpleNamespace(values=[mass], absolute_errors=[mass + " error"], provenance="current " + mass)
        projected.append(result)
        return result

    def evaluate(*args, **kwargs):
        evaluation_calls.append((args, kwargs))
        assert list(args[:3]) == session.point
        assert list(args[3:]) == [projected[-2].values, projected[-1].values,
                                 projected[-2].absolute_errors, projected[-1].absolute_errors]
        assert "current W" in kwargs["provenance"] and "current Z" in kwargs["provenance"]
        return new

    exact_kernel = SimpleNamespace(evaluate=evaluate)
    session.amplitude = exact_kernel

    def construct():
        constructions.append(True)
        return SimpleNamespace(evaluate=project)

    monkeypatch.setattr(numerical, "HiggsJetFormFactorProjector", construct)
    assert session.assemble() is old
    session.submit("amplitude", recompute=True)
    assert session.wait() is new
    assert session.amplitude is exact_kernel
    assert len(projection_calls) == 2 and len(evaluation_calls) == 1
    assert session.form_factors == dict(zip(("W", "Z"), projected))
    assert len(constructions) == 1
    # A retained exact projector must still use new momenta, masses and masters.
    session.point = [E("72"), E("-31/3"), E("1")]
    session.masses = {"W": E("2/5"), "Z": E("3/5")}
    session.results = {configuration.label: object() for _, configuration in session.configurations}
    session.submit("amplitude", recompute=True)
    assert session.wait() is new
    assert len(projection_calls) == 4 and len(evaluation_calls) == 2
    assert session.form_factors == dict(zip(("W", "Z"), projected[-2:]))
    assert len(constructions) == 1


@pytest.fixture
def acceptance(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    spec = importlib.util.spec_from_file_location("gg_hg_acceptance_test", EXAMPLES / "gg_hg_acceptance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("seed_digits", [30, 40])
def test_forced_interruption_retains_only_new_generation_restart_state(session, acceptance, seed_digits):
    populate_banks(session)
    session.seed_digits = seed_digits
    directory = session.directory / "completed-samples"
    directory.mkdir()
    (directory / "sample-cold-30.bin").write_bytes(b"previous cold work")

    def samples():
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in directory.glob("sample-*.bin")}

    def new_sample_then_cancel(evaluator, cache, *args, **kwargs):
        assert kwargs["recompute"] is True
        assert len(cache) == 0
        if seed_digits == 30:
            # Neither filename nor deterministic payload distinguishes fresh
            # work; the durable generation must drive the interruption gate.
            name, payload = "sample-cold-30.bin", b"previous cold work"
        else:
            name, payload = "sample-forced-40.bin", b"new complete refined sample"
        # Match the native checkpoint contract: only a complete file becomes
        # visible to the monitor. Direct write_bytes can expose an empty file
        # between opening and writing, making its observed hash nondeterministic.
        temporary = directory / f".{name}.tmp"
        with temporary.open("wb") as output:
            output.write(payload[:1])
            output.flush()
            sleep(0.01)  # The polling monitor must ignore this partial temporary.
            assert samples() == {}
            output.write(payload[1:])
        temporary.replace(directory / name)
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
    assert record["phase"] == "forced_refinement" and record["seed_digits"] == seed_digits
    assert record["baseline_sample_files"] == record["new_complete_samples"] == 1
    assert record["retained_sample_files"] == record["archived_sample_files"] == 1
    assert record["archived_sample_payloads_preserved"]
    assert record["numerical_generation"] == session.numerical_generation()
    assert record["previous_generation"]["generation"] == "initial"
    assert all(timing["numerical_generation"] == record["numerical_generation"]
               for timing in session.snapshot()["timings"])
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


def configuration_indices(session):
    def key(point):
        return tuple(sorted((str(k), str(v)) for k, v in point.items()))

    positions = {key(configuration.start): index
                 for index, (_, configuration) in enumerate(session.configurations)}
    return lambda point: positions[key(point)]


def wait_for_saved_seeds(session, count):
    deadline = monotonic() + 10
    while monotonic() < deadline:
        if len(numerical.BoundaryCache.load(session.directory / "seeds")) >= count:
            return
        sleep(0.001)
    raise AssertionError("A completed configuration was not progressively persisted.")


@pytest.mark.parametrize("workers,boundary_workers,per_configuration", [(5, 2, 2), (2, 4, 1)])
def test_concurrent_boundaries_respect_budget_persist_progress_and_return_input_order(
    session, monkeypatch, workers, boundary_workers, per_configuration,
):
    populate_banks(session)
    session.configurations = session.configurations[:4]
    session.workers, session.boundary_workers = workers, boundary_workers
    position = configuration_indices(session)
    _, _, add = constant_boundary_adder()
    first_started, release_first = Event(), Event()
    lock = Lock()
    active = peak = 0
    private_caches, initial_lengths, allocations = [], [], []
    monkeypatch.setattr(numerical, "IntegralEvaluator", lambda *, options: SimpleNamespace(options=options))

    def generate(evaluator, cache, point, *args, **kwargs):
        nonlocal active, peak
        index = position(point)
        with lock:
            active += 1
            peak = max(active, peak)
            private_caches.append(cache)
            initial_lengths.append(len(cache))
            allocations.append(evaluator.options.workers)
        try:
            if index == 0:
                first_started.set()
                assert release_first.wait(10)
            else:
                assert first_started.wait(10)
            add(cache, str(index + 2), f"independent configuration {index}")
            return SimpleNamespace(cache_hit=False)
        finally:
            with lock:
                active -= 1

    session.systems = {name: SimpleNamespace(generate_boundary=generate) for name in session.systems}
    session.submit("boundaries")
    try:
        wait_for_saved_seeds(session, 2)
        assert not session.snapshot()["done"]
        assert peak == 2
    finally:
        release_first.set()
    results = session.wait()
    assert [label for label, _ in results] == [c.label for _, c in session.configurations]
    assert allocations == [per_configuration] * 4
    assert peak * per_configuration <= workers
    assert initial_lengths == [1] * 4
    assert len({id(cache) for cache in private_caches}) == 4
    assert len(numerical.BoundaryCache.load(session.directory / "seeds")) == 5
    assert len(numerical.BoundaryCache.load(session.directory / "transport")) == 6
    timings = [entry for entry in session.snapshot()["timings"] if entry["stage"] == "boundary_configuration"]
    assert len(timings) == 4
    assert all(entry["sample_workers"] == per_configuration and entry["elapsed_ns"] > 0
               for entry in timings)


class NativePanicSurrogate(BaseException):
    """PyO3 PanicException has the same direct BaseException ancestry."""


@pytest.mark.parametrize("failure_type", [numerical.IncompleteReductionError, NativePanicSurrogate])
def test_concurrent_boundary_failure_preserves_completed_sibling_and_exception(session, failure_type):
    populate_banks(session)
    session.configurations = session.configurations[:3]
    session.workers = session.boundary_workers = 2
    position = configuration_indices(session)
    _, _, add = constant_boundary_adder()
    fail, sibling_started, sibling_cancelled = Event(), Event(), Event()
    original_failure = failure_type("independent configuration failed")

    def generate(evaluator, cache, point, *args, **kwargs):
        if position(point) == 0:
            add(cache, "2", "completed before sibling reduction failure")
            return SimpleNamespace(cache_hit=False)
        if position(point) == 1:
            assert fail.wait(10) and sibling_started.wait(10)
            raise original_failure
        sibling_started.set()
        deadline = monotonic() + 10
        while not kwargs["control"].cancelled:
            assert monotonic() < deadline
            sleep(0.001)
        sibling_cancelled.set()
        cancelled()

    session.systems = {name: SimpleNamespace(generate_boundary=generate) for name in session.systems}
    session.submit("boundaries")
    try:
        wait_for_saved_seeds(session, 2)
    finally:
        fail.set()
    with pytest.raises(failure_type, match="independent configuration") as raised:
        session.wait()
    assert raised.value is original_failure
    assert session._control.cancelled
    assert sibling_cancelled.is_set()
    state = session.snapshot()
    assert failure_type.__name__ in state["status"]
    failed_label = session.configurations[1][1].label
    assert any(entry.get("configuration") == failed_label
               and entry["outcome"].startswith(failure_type.__name__)
               for entry in state["timings"])
    assert len(numerical.BoundaryCache.load(session.directory / "seeds")) == 2
    assert len(numerical.BoundaryCache.load(session.directory / "transport")) == 3


def test_concurrent_cancellation_keeps_completed_configurations(session):
    populate_banks(session)
    session.configurations = session.configurations[:3]
    session.workers = session.boundary_workers = 2
    position = configuration_indices(session)
    _, _, add = constant_boundary_adder()

    def generate(evaluator, cache, point, *args, **kwargs):
        if position(point) == 0:
            add(cache, "2", "completed before user cancellation")
            return SimpleNamespace(cache_hit=False)
        deadline = monotonic() + 10
        while not kwargs["control"].cancelled:
            assert monotonic() < deadline
            sleep(0.001)
        cancelled()

    session.systems = {name: SimpleNamespace(generate_boundary=generate) for name in session.systems}
    session.submit("boundaries")
    wait_for_saved_seeds(session, 2)
    session.cancel()
    with pytest.raises(numerical.CalculationCancelled):
        session.wait()
    assert len(numerical.BoundaryCache.load(session.directory / "seeds")) == 2
    assert len(numerical.BoundaryCache.load(session.directory / "transport")) == 3


def test_concurrent_exact_cache_hits_preserve_both_banks(session):
    populate_banks(session)
    session.configurations = session.configurations[:4]
    session.workers, session.boundary_workers = 8, 4
    before = (SUPPORT.boundary_evidence(session.seeds), SUPPORT.boundary_evidence(session.cache))
    flow, x, _ = constant_boundary_adder()

    def generate(evaluator, cache, *args, **kwargs):
        return flow.evaluate(cache, {x: E("0")}, 0, 0)

    session.systems = {name: SimpleNamespace(generate_boundary=generate) for name in session.systems}
    session.submit("boundaries")
    assert all(result.cache_hit and result.steps == 0 for _, result in session.wait())
    assert (SUPPORT.boundary_evidence(session.seeds), SUPPORT.boundary_evidence(session.cache)) == before
    assert SUPPORT.boundary_evidence(numerical.BoundaryCache.load(session.directory / "seeds")) == before[0]
    assert SUPPORT.boundary_evidence(numerical.BoundaryCache.load(session.directory / "transport")) == before[1]


def test_headless_progress_is_persisted_while_a_configuration_is_running(session, acceptance):
    populate_banks(session)
    session.configurations = session.configurations[:2]
    session.workers = session.boundary_workers = 2
    position = configuration_indices(session)
    _, _, add = constant_boundary_adder()
    release, progress_saved = Event(), Event()
    report = {"status": "running"}
    previous_timing = {"stage": "previous_session", "elapsed_ns": 123}

    def generate(evaluator, cache, point, *args, **kwargs):
        index = position(point)
        if index == 1:
            assert release.wait(10)
        add(cache, str(index + 2), f"configuration {index} with durable progress")
        return SimpleNamespace(cache_hit=False)

    def publish(state):
        acceptance.persist_progress(
            session.directory, report, state, stage="boundaries", mode="cold",
            elapsed_ns=456, seed_digits=30, archived_timings=[previous_timing],
        )
        if not state["done"] and any(entry["outcome"] == "completed" for entry in state["timings"]):
            progress_saved.set()

    session.systems = {name: SimpleNamespace(generate_boundary=generate) for name in session.systems}
    session.submit("boundaries")
    with ThreadPoolExecutor(max_workers=1) as monitor:
        future = monitor.submit(acceptance.wait_for_stage, session, publish, poll_interval=0.01)
        try:
            assert progress_saved.wait(10)
            wait_for_saved_seeds(session, 2)
            partial = json.loads((session.directory / "acceptance.json").read_text())
            assert partial["progress"]["done"] is False
            assert partial["progress"]["stage"] == "boundaries"
            assert partial["progress"]["seed_digits"] == 30
            assert partial["timings"][0] == previous_timing
            assert partial["progress"]["recent_events"]
            assert len(session.seeds) == 2 and len(session.cache) == 3
            assert session.results == {} and session.observables is None
        finally:
            release.set()
        assert len(future.result(timeout=10)) == 2
    final = json.loads((session.directory / "acceptance.json").read_text())
    assert final["progress"]["done"] is True
    assert final["timings"][0] == previous_timing
    assert not (session.directory / "acceptance.json.tmp").exists()


def test_headless_warm_completion_does_not_wait_for_poll_interval(session, acceptance, monkeypatch):
    value = object()
    session._future = Future()
    session._future.set_result(value)
    updates = []

    def unexpected_sleep(*args):
        raise AssertionError("Completed work must not wait for a polling sleep.")

    monkeypatch.setattr(acceptance, "sleep", unexpected_sleep)
    started = monotonic()
    assert acceptance.wait_for_stage(session, updates.append, poll_interval=60) is value
    assert monotonic() - started < 1
    assert updates[-1]["done"]


def test_headless_progress_wait_continues_after_a_pending_future_timeout(session, acceptance):
    value = object()
    session._future = Future()
    updates = []

    def publish(state):
        updates.append(state)
        if len(updates) == 2:
            # Only finish after Future.result has timed out and polling resumes.
            assert not state["done"]
            session._future.set_result(value)

    assert acceptance.wait_for_stage(session, publish, poll_interval=0.001) is value
    assert [state["done"] for state in updates] == [False, False, True]


# Python 3.11+ aliases FutureTimeoutError to the built-in exception.
@pytest.mark.parametrize("failure_type", list(dict.fromkeys([
    TimeoutError, FutureTimeoutError, NativePanicSurrogate,
])))
@pytest.mark.parametrize("already_done", [False, True])
def test_headless_progress_wait_preserves_original_failures(
    session, acceptance, failure_type, already_done, monkeypatch,
):
    original = failure_type("original computation failure")
    session._future = Future()
    if already_done:
        session._future.set_exception(original)
    else:
        wait = session.wait

        def finish_on_wait(timeout=None):
            if not session._future.done():
                session._future.set_exception(original)
            return wait(timeout=timeout)

        monkeypatch.setattr(session, "wait", finish_on_wait)
    updates = []
    with pytest.raises(failure_type) as raised:
        acceptance.wait_for_stage(session, updates.append, poll_interval=0.01)
    assert raised.value is original
    assert [state["done"] for state in updates] == [already_done, True]


def test_acceptance_attests_loaded_extension_and_native_source_identity(acceptance):
    import symbolica.core as core

    evidence = acceptance.runtime_attestation()
    extension = evidence["loaded_extension"]
    assert Path(extension["path"]) == Path(core.__file__).resolve()
    assert extension["size_bytes"] == Path(core.__file__).stat().st_size
    assert len(extension["sha256"]) == 64
    x, epsilon, master = S(
        "integration_acceptance_attestation_v1::x",
        "integration_acceptance_attestation_v1::epsilon",
        "integration_acceptance_attestation_v1::master",
    )
    native = numerical.KinematicTransport(
        epsilon, {x: [[E("0")]]}, [master], E("1"),
        branch_domain="native source attestation v1",
    )
    assert evidence["native_identity_witness"]["identity"] == native.identity
    assert evidence["execution_sources"]["controller"]["path"] == str(EXAMPLES / "gg_hg_support.py")
    acceptance.verify_runtime_attestation(evidence)


@pytest.mark.parametrize("changed", ["execution_source", "extension"])
def test_acceptance_rejects_changed_execution_input(acceptance, tmp_path, monkeypatch, changed):
    import symbolica.core as core

    source = tmp_path / "steering.py"
    source.write_text("original executed source\n")
    monkeypatch.setattr(acceptance if changed == "execution_source" else core, "__file__", str(source))
    evidence = acceptance.runtime_attestation()
    source.write_text("changed source after computation started\n")
    with pytest.raises(RuntimeError, match="changed during this run"):
        acceptance.verify_runtime_attestation(evidence)
    original = (evidence["execution_sources"]["acceptance"] if changed == "execution_source"
                else evidence["loaded_extension"])
    assert original["sha256"] == hashlib.sha256(
        b"original executed source\n"
    ).hexdigest()


def test_acceptance_rejects_optimized_python_before_starting_work(acceptance, tmp_path):
    destination = tmp_path / "must-not-start"
    process = subprocess.run(
        [sys.executable, "-O", acceptance.__file__, "--directory", str(destination)],
        capture_output=True, text=True, timeout=30,
    )
    assert process.returncode != 0
    assert "Acceptance requires enabled assertions" in process.stderr
    assert not destination.exists()


@pytest.mark.parametrize("change,error,message", [
    ("1e-22", "1e-30", "coefficient uncertainty estimates"),
    ("1e-18", "1e-16", "mixed accuracy target"),
])
def test_refinement_rejects_underreported_errors_and_insufficient_accuracy(
    acceptance, change, error, message,
):
    value = ComplexFloat("1", decimal_digits=90)
    allowance = Float(error, decimal_digits=90)
    previous = SimpleNamespace(coefficients=[[value]], comparison_errors=[[allowance]])
    result = SimpleNamespace(
        coefficients=[[value + ComplexFloat(change, decimal_digits=90)]],
        comparison_errors=[[allowance]],
    )
    with pytest.raises(AssertionError, match=message):
        acceptance.check_coefficient_refinement(
            previous, result, label="controlled refinement",
            tolerance=Float("1e-20", decimal_digits=90), one=Float("1", decimal_digits=90),
        )


def test_refinement_accepts_difference_covered_by_combined_native_errors(acceptance):
    value = ComplexFloat("1", decimal_digits=90)
    previous = SimpleNamespace(
        coefficients=[[value]], comparison_errors=[[Float("6e-26", decimal_digits=90)]],
    )
    result = SimpleNamespace(
        coefficients=[[value + ComplexFloat("1e-25", decimal_digits=90)]],
        comparison_errors=[[Float("6e-26", decimal_digits=90)]],
    )
    assert acceptance.check_coefficient_refinement(
        previous, result, label="controlled refinement",
        tolerance=Float("1e-20", decimal_digits=90), one=Float("1", decimal_digits=90),
    ) == 1


@pytest.fixture
def transport_reference_coverage():
    reference = json.loads((EXAMPLES / "data" / "gg_hg" / "coherent-reference.json").read_text())
    labels = [case["label"] for case in reference["cases"]]
    native = {
        case["label"]: SimpleNamespace(coefficients=[list(row) for row in case["reference_values"]])
        for case in reference["cases"]
    }
    return reference, native, labels


def test_transport_reference_coverage_includes_all_sixteen_configurations(acceptance, transport_reference_coverage):
    reference, native, labels = transport_reference_coverage
    assert acceptance.check_transport_reference_coverage(native, reference, labels) == 4360


@pytest.mark.parametrize("mutation, message", [
    ("duplicate_reference", "unique configurations"),
    ("missing_reference", "unique configurations"),
    ("unknown_reference", "differs from native inputs"),
    ("missing_native", "Native configuration coverage"),
    ("reference_coefficient", "4,360 transport coefficients"),
    ("native_coefficient", "4,360 transport coefficients"),
])
def test_transport_reference_coverage_rejects_missing_or_duplicate_data(
    acceptance, transport_reference_coverage, mutation, message,
):
    reference, native, labels = transport_reference_coverage
    if mutation == "duplicate_reference":
        reference["cases"][1]["label"] = labels[0]
    elif mutation == "missing_reference":
        reference["cases"].pop()
    elif mutation == "unknown_reference":
        reference["cases"][0]["label"] = "unknown physical configuration"
    elif mutation == "missing_native":
        native.pop(labels[0])
    elif mutation == "reference_coefficient":
        reference["cases"][0]["reference_values"][0].pop()
    else:
        native[labels[0]].coefficients[0].pop()
    with pytest.raises(AssertionError, match=message):
        acceptance.check_transport_reference_coverage(native, reference, labels)


@pytest.mark.parametrize("filename", ["coherent-reference.json", "amplitude-validation.json"])
def test_comparison_reference_records_exact_payload_hash_and_provenance(acceptance, filename):
    path = EXAMPLES / "data" / "gg_hg" / filename
    payload = path.read_bytes()
    expected = json.loads(payload)
    reference, evidence = acceptance.load_comparison_reference(path)
    assert reference == expected
    assert evidence["path"] == str(path.resolve())
    assert evidence["size_bytes"] == len(payload)
    assert evidence["sha256"] == hashlib.sha256(payload).hexdigest()
    assert evidence["schema"] == expected["schema"]
    keys = (("provenance", "purpose") if filename == "coherent-reference.json" else
            ("source_input_sha256", "references", "scope", "reference_accuracy", "physical_s_t_MH_squared"))
    assert evidence["provenance"] == {name: expected[name] for name in keys}


def test_form_factor_comparison_uses_all_recorded_reference_allowances(acceptance):
    reference = json.loads((EXAMPLES / "data/gg_hg/amplitude-validation.json").read_text())
    native = {
        block["mass"]: SimpleNamespace(
            values=[ComplexFloat(row["real"], row["imaginary"], decimal_digits=100)
                    + ComplexFloat(row["absolute_error"], decimal_digits=100) / 2
                    for row in block["values"]],
            absolute_errors=[Float("1e-50", decimal_digits=100) for _ in block["values"]],
            verified_relative_digits=[40 for _ in block["values"]],
        )
        for block in reference["form_factors"]
    }
    report = acceptance.compare_form_factors(native, reference)
    assert report["components"] == 8
    for block in reference["form_factors"]:
        assert [row["reference_absolute_error"] for row in report["comparison"][block["mass"]]] == [
            row["absolute_error"] for row in block["values"]
        ]
    allowance = reference["form_factors"][0]["values"][0]["absolute_error"]
    native["W"].values[0] += ComplexFloat(allowance, decimal_digits=100) * 2
    with pytest.raises(AssertionError, match="form-factor uncertainty estimates"):
        acceptance.compare_form_factors(native, reference)


@pytest.mark.parametrize("nearby", [False, True])
def test_notebook_displays_eight_native_form_factors_at_the_current_point(monkeypatch, nearby):
    marimo = pytest.importorskip("marimo", minversion="0.24.0")
    reference = json.loads((EXAMPLES / "data/gg_hg/amplitude-validation.json").read_text())
    point = [E(value) for value in reference["physical_s_t_MH_squared"]]
    if nearby:
        point[0] += E("1/100000")
    form_factors = {
        block["mass"]: SimpleNamespace(
            values=[ComplexFloat(row["real"], row["imaginary"], decimal_digits=100)
                    for row in block["values"]],
            absolute_errors=[Float("1e-40", decimal_digits=100) for _ in block["values"]],
            verified_relative_digits=[30 for _ in block["values"]],
        )
        for block in reference["form_factors"]
    }
    native_values = {name: Float(value, decimal_digits=100)
                     for name, value in reference["expected_observables"].items()}
    display_session = SimpleNamespace(
        automatic_boundary_generation_available=True,
        point=point, masses={"W": E("5399/13074"), "Z": E("7775/14631")},
        results={}, amplitude=SimpleNamespace(diagrams=[]), form_factors=form_factors,
        observables=SimpleNamespace(
            values=native_values,
            absolute_errors={name: Float("1e-40", decimal_digits=100) for name in native_values},
            verified_relative_digits={name: 30 for name in native_values},
            provenance="Synthetic display fixture; no native computation requested",
        ),
        snapshot=lambda: {"done": True, "status": "ready", "timings": [], "events": []},
    )
    captured = []
    table = marimo.ui.table

    def record_table(data, *args, **kwargs):
        if kwargs.get("label") == "Native W/Z form factors and propagated uncertainties":
            captured.extend(data)
        return table(data, *args, **kwargs)

    monkeypatch.setattr(marimo.ui, "table", record_table)
    app = runpy.run_path(str(EXAMPLES / "gg_hg.py"), run_name="notebook_display_test")["app"]
    app.run(defs={"session": display_session})
    assert [row["form factor"] for row in captured] == [f"{mass}{i}" for mass in ("W", "Z") for i in range(1, 5)]
    for row in captured:
        mass, index = row["form factor"][0], int(row["form factor"][1]) - 1
        assert row["native result"] == str(form_factors[mass].values[index])
        assert row["propagated absolute uncertainty"] == str(form_factors[mass].absolute_errors[index])
        assert row["achieved relative digits"] == 30
        if nearby:
            assert row["reference at recorded point"] == "different kinematics"
            assert row["reference absolute uncertainty"] == row["absolute difference"] == "—"
        else:
            assert row["reference at recorded point"] == row["native result"]
            assert row["reference absolute uncertainty"] != "—"
            assert Float(row["absolute difference"], decimal_digits=100) == 0
