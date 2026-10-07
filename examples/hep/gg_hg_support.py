"""Staged Higgs-plus-jet calculation used by the notebook and acceptance runner.

Comparison references never enter the evaluator. The model and optional supplied
starting values come from disk; equations, exact basis maps and coordinates come
from the extension. Native execution also supports fresh boundary generation.
"""

from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
import asyncio
import json
import os
from pathlib import Path
import sys
from threading import Lock
from time import perf_counter_ns
from uuid import uuid4


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x") as output:
            json.dump(value, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def boundary_evidence(cache):
    """Stable inspection of every numerical value and its retained evidence."""
    records = [
        (
            entry.identity,
            tuple(sorted((str(k), str(v)) for k, v in entry.coordinates.items())),
            tuple(sorted((str(k), v) for k, v in entry.root_sheets.items())),
            entry.leading_power, entry.verified_digits, entry.input_verified_digits,
            entry.working_bits, entry.provenance,
            tuple(tuple(row) for row in entry.coefficients),
            tuple(tuple(row) for row in entry.comparison_errors),
        )
        for entry in cache.entries()
    ]
    # Sort through a printable key, but compare native arbitrary-precision
    # objects themselves so reload checks do not lose digits to formatting.
    return sorted(records, key=lambda record: tuple(str(value) for value in record))


class CalculationSession:
    def __init__(self, model_path, directory, *, seed_digits=30, digits=20, workers=1,
                 boundary_workers=1, boundary_bundle=None, cooperative=None):
        from symbolica import E
        from symbolica.community import hepkit as hep
        from symbolica.community.hep import integration
        from symbolica.community.hep.integration import (
            BoundaryCache,
            HiggsJetIntegralSystem,
        )

        if workers < 1 or boundary_workers < 1:
            raise ValueError("Worker budgets must be positive.")
        self.cooperative = sys.platform == "emscripten" if cooperative is None else cooperative
        if self.cooperative and (workers != 1 or boundary_workers != 1):
            raise ValueError("Cooperative notebook execution requires one worker.")
        self.automatic_boundary_generation_available = integration.automatic_boundary_generation_available
        self.directory = Path(directory)
        self.boundary_bundle = None if boundary_bundle is None else Path(boundary_bundle)
        self.model = hep.Model.from_json(Path(model_path).read_text())
        self.systems = {
            name: HiggsJetIntegralSystem(name)
            for name in ("planar", "nonplanar")
        }
        self.configurations = [
            (name, configuration)
            for name, system in self.systems.items()
            for configuration in system.configurations()
        ]
        self.point = [E(s) for s in (
            "7173070292440521/111284741846000",
            "-12058167788971/339319588980", "1",
        )]
        self.masses = {"W": E("5399/13074"), "Z": E("7775/14631")}
        self.seed_digits = seed_digits
        self.digits = digits
        self.workers = workers
        self.boundary_workers = boundary_workers
        self.seeds = self._load_cache("seeds", BoundaryCache)
        self.cache = self._load_cache("transport", BoundaryCache)
        self.results = {}
        self.amplitude = None
        self._form_factor_projector = None
        self.observables = None
        self.form_factors = {}
        self.timings = []
        self.events = []
        self.status = "ready"
        self._pool = None if self.cooperative else ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="higgs-jet",
        )
        self._lock = Lock()
        self._future = None
        self._control = None

    def _load_cache(self, name, cache_type):
        # Allow constructing a recovery session even if a previous force was
        # interrupted between directory moves. Only explicit force may proceed.
        if self._preparation_pending():
            return cache_type()
        path = self.directory / name
        if (path / "physical-boundaries.bin").exists():
            return cache_type.load(path)
        return cache_type()

    def _preparation_pending(self):
        return (self.directory / "force-preparation-pending.json").exists()

    def numerical_generation(self):
        """Return committed numerical-generation evidence, or None during reset."""
        with self._lock:
            return self._read_numerical_generation()

    def _read_numerical_generation(self):
        if self._preparation_pending():
            return None
        path = self.directory / "numerical-generation.json"
        if not path.exists():
            return {"schema": 1, "generation": "initial", "archive_directory": None}
        try:
            record = json.loads(path.read_text())
            generation = record["generation"]
            if (record["schema"] != 1 or not isinstance(generation, str)
                    or len(generation) != 32
                    or any(c not in "0123456789abcdef" for c in generation)
                    or record["archive_directory"] != f"numerical-history/{generation}"):
                raise ValueError("Invalid numerical-generation record")
            return record
        except (ValueError, KeyError, TypeError) as error:
            raise RuntimeError(
                "Invalid numerical generation; run Recompute boundaries to recover."
            ) from error

    def _require_numerical_generation(self):
        if self.numerical_generation() is None:
            raise RuntimeError(
                "Forced boundary preparation was interrupted; run Recompute boundaries "
                "(recompute=True) before loading or computing other stages."
            )

    def _generation_evidence(self):
        # Progress/finally reporting must preserve the original stage failure.
        try:
            return self._read_numerical_generation()
        except (OSError, RuntimeError) as error:
            return {"status": "unavailable", "error": str(error)}

    def _prepare_forced_boundaries(self, cache_type):
        """Archive numerical work before any new sample, keeping exact reductions."""
        generation = uuid4().hex
        archive = self.directory / "numerical-history" / generation
        archive.mkdir(parents=True)
        pending = self.directory / "force-preparation-pending.json"
        if pending.exists():
            # Preserve the prior interrupted preparation's history on recovery.
            (archive / "previous-preparation.json").write_bytes(pending.read_bytes())
        record = {
            "schema": 1, "generation": generation,
            "archive_directory": f"numerical-history/{generation}",
        }
        _atomic_json(pending, record)
        for name in ("seeds", "transport", "completed-samples", "numerical-generation.json"):
            old = self.directory / name
            if old.exists():
                old.rename(archive / name)
        self.seeds, self.cache = cache_type(), cache_type()
        self.seeds.save(self.directory / "seeds")
        self.cache.save(self.directory / "transport")
        (self.directory / "completed-samples").mkdir()
        _atomic_json(self.directory / "numerical-generation.json", record)
        pending.unlink()

    def _options(self, *, seeds=False, recompute=False, workers=None):
        """Budget extra step attempts for the long auxiliary-mass boundary path.

        Nonplanar seed propagation can exhaust 1,000 accepted/rejected trials
        while still advancing with valid local error checks. Allow 2,000 for
        seeds; physical transport keeps 1,000. Physical transport starts with
        guard 20/order 16, then retains the solver's adaptive accuracy checks.
        """
        from symbolica.community.hep.integration import EvaluationOptions

        return EvaluationOptions(
            digits=self.seed_digits if seeds else self.digits,
            guard_digits=60 if seeds else 20,
            series_order=96 if seeds else 16,
            max_steps=2000 if seeds else 1000,
            workers=self.workers if workers is None else workers,
            cache_directory=self.directory / "exact-reductions",
            sample_cache_directory=self.directory / "completed-samples",
            reuse_samples=not recompute,
        )

    def submit(self, stage, *, recompute=False, nearby=False):
        """Start a stage; browser transport yields between configurations."""
        from symbolica.community.hep.integration import ComputationControl

        with self._lock:
            if self._future is not None and not self._future.done():
                raise RuntimeError("A stage is already running; cancel or wait for it.")
            if stage not in ("supplied_boundaries", "boundaries", "transport", "amplitude", "restart"):
                raise ValueError("Unknown calculation stage")
            if stage == "boundaries" and not self.automatic_boundary_generation_available:
                raise RuntimeError("This build uses supplied boundaries; automatic generation requires a native build.")
            loop = asyncio.get_running_loop() if self.cooperative else None
            self._control = ComputationControl()
            self.status = f"running {stage}"
            if self.cooperative:
                self._future = loop.create_task(self._run_cooperatively(stage, recompute, nearby))
                # Preserve failures for wait_async while avoiding unobserved-task
                # warnings when the notebook only reads the progress panel.
                self._future.add_done_callback(lambda task: None if task.cancelled() else task.exception())
            else:
                self._future = self._pool.submit(self._run, stage, recompute, nearby)
        return self._future

    def _finish_stage(self, stage, recompute, nearby, started, outcome):
        with self._lock:
            self.status = f"{stage}: {outcome}"
            self.timings.append({
                "stage": stage, "recompute": recompute, "nearby": nearby,
                "elapsed_ns": perf_counter_ns() - started, "outcome": outcome,
                "numerical_generation": self._generation_evidence(),
            })

    async def _run_cooperatively(self, stage, recompute, nearby):
        # Also prevents an eager asyncio task factory running under submit's lock.
        await asyncio.sleep(0)
        if stage != "transport":
            return self._run(stage, recompute, nearby)
        started = perf_counter_ns()
        outcome = "completed"
        try:
            for _ in self._transport_steps(recompute=recompute, nearby=nearby):
                # Each completed configuration is already checkpointed. Native
                # calls are synchronous: browser cancellation is observed at
                # the next yield, rather than during an individual ODE solve.
                await asyncio.sleep(0)
            return self.results
        except BaseException as exc:
            outcome = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._finish_stage(stage, recompute, nearby, started, outcome)

    def _run(self, stage, recompute, nearby):
        started = perf_counter_ns()
        outcome = "completed"
        try:
            if stage == "supplied_boundaries":
                result = self.load_supplied_boundaries()
            elif stage == "boundaries":
                result = self.generate_boundaries(recompute=recompute)
            elif stage == "transport":
                result = self.transport(recompute=recompute, nearby=nearby)
            elif stage == "amplitude":
                result = self.assemble(recompute=recompute)
            else:
                result = self.restart_and_repeat()
            return result
        except BaseException as exc:
            outcome = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._finish_stage(stage, recompute, nearby, started, outcome)

    def cancel(self):
        with self._lock:
            if self._control is not None:
                self._control.cancel()

    def snapshot(self):
        with self._lock:
            if self._control is not None:
                self.events.extend(self._control.poll())
                self.events = self.events[-100:]
            return {
                "status": self.status, "events": list(self.events),
                "timings": list(self.timings),
                "numerical_generation": self._generation_evidence(),
                "done": self._future is None or self._future.done(),
            }

    def wait(self, timeout=None):
        """Headless entry point: propagate typed native errors to the caller."""
        if self.cooperative and self._future is not None:
            if not self._future.done():
                raise RuntimeError("Use await session.wait_async() for cooperative execution.")
            return self._future.result()
        return None if self._future is None else self._future.result(timeout=timeout)

    async def wait_async(self, timeout=None):
        """Await a stage without blocking the browser event loop or cancelling it on timeout."""
        if self._future is None:
            return None
        future = self._future if self.cooperative else asyncio.wrap_future(self._future)
        return await asyncio.wait_for(asyncio.shield(future), timeout)

    def close(self):
        """Cancel current work; native execution also joins its worker."""
        self.cancel()
        if self._pool is not None:
            self._pool.shutdown(wait=True)

    def _invalidate_results(self):
        # A partial stage must never leave a mixture of old and new kinematics
        # that could pass the sixteen-configuration amplitude admission check.
        self.results = {}
        self.form_factors = {}
        self.observables = None

    def load_supplied_boundaries(self):
        """Import the shipped starting values as supplied evidence in this build."""
        import importlib.util

        self._require_numerical_generation()
        if self.boundary_bundle is None:
            raise ValueError("No supplied boundary bundle was configured")
        path = Path(__file__).with_name("gg_hg_boundaries.py")
        spec = importlib.util.spec_from_file_location("gg_hg_boundary_bundle", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pending, results = module.load_boundary_bundle(
            self.boundary_bundle, self.systems, control=self._control,
        )
        # Validation completes before either of the session's banks changes.
        self._invalidate_results()
        self.seeds.extend(pending)
        self.cache.extend(pending)
        self.seeds.save(self.directory / "seeds")
        self.cache.save(self.directory / "transport")
        with self._lock:
            self.events.append(
                "Loaded sixteen supplied AMF starting configurations; "
                "physical transport and amplitude remain to be computed."
            )
        return results

    def generate_boundaries(self, *, recompute=False):
        if not self.automatic_boundary_generation_available:
            raise RuntimeError("This build uses supplied boundaries; automatic generation requires a native build.")
        from symbolica.community.hep.integration import (
            BoundaryCache, CalculationCancelled, ComputationControl, IntegralEvaluator,
        )

        self._invalidate_results()
        if recompute:
            with self._lock:
                self._prepare_forced_boundaries(BoundaryCache)
        else:
            self._require_numerical_generation()
        if not self.configurations:
            return []
        concurrent = min(self.boundary_workers, self.workers, len(self.configurations))
        sample_workers = self.workers // concurrent
        snapshot = BoundaryCache()
        snapshot.extend(self.seeds)
        with self._lock:
            if self._control is None:
                self._control = ComputationControl()
            control = self._control
        completed = [None] * len(self.configurations)

        def evaluate(topology, configuration):
            started = perf_counter_ns()
            if control.cancelled:
                raise CalculationCancelled("Boundary stage cancelled before configuration start.")
            private_cache = BoundaryCache()
            private_cache.extend(snapshot)
            evaluator = IntegralEvaluator(options=self._options(
                seeds=True, recompute=recompute, workers=sample_workers,
            ))
            with self._lock:
                self.events.append(
                    f"Starting boundary {configuration.label} with {sample_workers} sample workers."
                )
            outcome, cache_hit = "completed", None
            try:
                result = self.systems[topology].generate_boundary(
                    evaluator, private_cache, configuration.start, configuration.root_sheets,
                    recompute=recompute, control=control,
                )
                cache_hit = result.cache_hit
            except BaseException as exc:
                outcome = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                with self._lock:
                    self.timings.append({
                        "stage": "boundary_configuration", "configuration": configuration.label,
                        "seed_digits": self.seed_digits, "sample_workers": sample_workers,
                        "elapsed_ns": perf_counter_ns() - started,
                        "outcome": outcome, "cache_hit": cache_hit,
                        "numerical_generation": self._generation_evidence(),
                    })
            return result, private_cache

        failure = None
        with ThreadPoolExecutor(max_workers=concurrent, thread_name_prefix="higgs-jet-boundary") as pool:
            futures = {
                pool.submit(evaluate, topology, configuration): (index, configuration.label)
                for index, (topology, configuration) in enumerate(self.configurations)
            }
            for future in as_completed(futures):
                index, label = futures[future]
                try:
                    result, private_cache = future.result()
                    # Only this coordinator writes the growing banks. Merge
                    # successful siblings even after cancellation or failure.
                    self.seeds.extend(private_cache)
                    self.seeds.save(self.directory / "seeds")
                    self.cache.extend(private_cache)
                    self.cache.save(self.directory / "transport")
                    completed[index] = (label, result)
                    with self._lock:
                        self.events.append(f"Saved boundary {label}; cache hit: {result.cache_hit}.")
                except CancelledError:
                    continue  # A queued sibling cancelled after the original failure.
                except BaseException as exc:
                    # PyO3 native panics derive directly from BaseException.
                    # Preserve their type while cooperatively stopping siblings.
                    if failure is None:
                        failure = exc
                    control.cancel()
                    for sibling in futures:
                        sibling.cancel()
            if failure is not None:
                raise failure
        return completed

    def _destination(self, system, configuration):
        s, t, h = self.point
        u = h - s - t
        a, b = [(s, u), (s, t), (u, t), (t, s)][configuration.permutation - 1]
        m = self.masses[configuration.mass]
        return dict(zip(system.coordinates, [a / m, b / m, h / m]))

    def transport(self, *, recompute=False, nearby=False):
        for _ in self._transport_steps(recompute=recompute, nearby=nearby):
            pass
        return self.results

    def _transport_steps(self, *, recompute=False, nearby=False):
        from symbolica import E
        from symbolica.community.hep.integration import BoundaryCache, CalculationCancelled

        self._require_numerical_generation()
        self._invalidate_results()
        if recompute:
            self.cache = BoundaryCache.load(self.directory / "seeds")
        # Persist once even on an exact repeat: a previous failed save can leave
        # valid computed points only in memory. New results still checkpoint
        # individually, so interruption never discards successful transport.
        self.cache.save(self.directory / "transport")
        if nearby:
            self.point = [self.point[0] + E("1/100000"), *self.point[1:]]
        for topology, configuration in self.configurations:
            if self._control is not None and self._control.cancelled:
                raise CalculationCancelled("Transport cancelled before the next configuration.")
            system = self.systems[topology]
            result = system.evaluate(
                self.cache, self._destination(system, configuration),
                configuration.root_sheets, options=self._options(), control=self._control,
            )
            self.results[configuration.label] = result
            if not result.cache_hit or result.inserted_points:
                self.cache.save(self.directory / "transport")
            yield configuration.label

    def assemble(self, *, recompute=False, _refinements=0):
        from symbolica import E
        from symbolica.community.hep.integration import (
            AccuracyError, HiggsJetAmplitude, HiggsJetFormFactorProjector,
        )

        if recompute:
            self.form_factors, self.observables = {}, None
        self._require_numerical_generation()
        if self.observables is not None and not recompute and all(
            value is not None and value >= self.digits
            for value in self.observables.verified_relative_digits.values()
        ):
            return self.observables
        if len(self.results) != 16:
            raise RuntimeError("Transport all sixteen configurations before amplitude assembly.")
        if self.amplitude is None:
            self.amplitude = HiggsJetAmplitude(self.model, control=self._control)
        s, t, higgs_mass_squared = self.point
        # The projector owns exact, kinematics-independent expressions. Keep
        # those expressions, but reevaluate all weights and source checks below.
        if self._form_factor_projector is None:
            self._form_factor_projector = HiggsJetFormFactorProjector()
        projector = self._form_factor_projector
        form_factors = {}
        for mass in ("W", "Z"):
            blocks = {}
            for topology in ("planar", "nonplanar"):
                configurations = sorted(
                    (c for name, c in self.configurations if name == topology and c.mass == mass),
                    key=lambda c: c.permutation,
                )
                blocks[topology] = [self.results[c.label] for c in configurations]
            form_factors[mass] = projector.evaluate(
                s, t, higgs_mass_squared, self.masses[mass], blocks["planar"], blocks["nonplanar"],
            )
        w, z = form_factors["W"], form_factors["Z"]
        mw2, mz2 = self.masses["W"], self.masses["Z"]
        parameters = {
            self.model.parameter("aEWM1").symbol: E("128"),
            self.model.parameter("aS").symbol: E("118/1000"),
            self.model.parameter("MZ").symbol: mz2.sqrt(),
            self.model.parameter("Gf").symbol:
                E("𝜋") * E("1/128") * mz2 / (E("2").sqrt() * mw2 * (mz2 - mw2)),
        }
        observables = self.amplitude.evaluate(
            s, t, higgs_mass_squared, w.values, z.values, w.absolute_errors, z.absolute_errors,
            parameters=parameters,
            provenance="Native canonical boundaries and physical transport; " + w.provenance + "; " + z.provenance,
            digits=self.digits, control=self._control,
        )
        achieved = observables.verified_relative_digits
        if any(value is None or value < self.digits for value in achieved.values()):
            if not self.automatic_boundary_generation_available:
                raise AccuracyError(
                    f"Supplied boundaries did not yield {self.digits} observable digits: {achieved}. "
                    "Import more accurate boundaries generated with the native build."
                )
            if _refinements >= 2:
                raise AccuracyError(
                    f"Observable accuracy did not reach {self.digits} digits after seed refinement: {achieved}"
                )
            self.seed_digits += 10
            with self._lock:
                self.events.append(f"Refining native seeds to {self.seed_digits} digits because propagated observable uncertainty exceeds the target.")
            self.generate_boundaries(recompute=True)
            self.transport()
            return self.assemble(recompute=True, _refinements=_refinements + 1)
        self.form_factors, self.observables = form_factors, observables
        return self.observables

    def restart_and_repeat(self):
        from symbolica.community.hep.integration import BoundaryCache

        self._require_numerical_generation()
        if len(self.results) != len(self.configurations):
            raise RuntimeError("Complete transport before checking an exact repeated hit.")
        before = {name: result.coefficients for name, result in self.results.items()}
        evidence = boundary_evidence(self.cache)
        self.cache.save(self.directory / "transport")
        self.cache = BoundaryCache.load(self.directory / "transport")
        assert boundary_evidence(self.cache) == evidence
        result = self.transport()
        assert all(value.cache_hit and value.steps == 0 for value in result.values())
        assert before == {name: value.coefficients for name, value in result.items()}
        return result
