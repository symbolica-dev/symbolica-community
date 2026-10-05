"""Staged native Higgs-plus-jet calculation used by the notebook and acceptance runner.

No reference values enter this module. Only the scoped native model is loaded
from disk; equations, exact basis maps and coordinates come from the extension.
"""

from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from time import perf_counter_ns


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
                 boundary_workers=1):
        from symbolica import E
        from symbolica.community import hepkit as hep
        from symbolica.community.hep.integration import (
            BoundaryCache,
            HiggsJetIntegralSystem,
        )

        if workers < 1 or boundary_workers < 1:
            raise ValueError("Worker budgets must be positive.")
        self.directory = Path(directory)
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
        self.observables = None
        self.form_factors = {}
        self.timings = []
        self.events = []
        self.status = "ready"
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="higgs-jet")
        self._lock = Lock()
        self._future = None
        self._control = None

    def _load_cache(self, name, cache_type):
        path = self.directory / name
        if (path / "physical-boundaries.bin").exists():
            return cache_type.load(path)
        return cache_type()

    def _options(self, *, seeds=False, recompute=False, workers=None):
        from symbolica.community.hep.integration import EvaluationOptions

        return EvaluationOptions(
            digits=self.seed_digits if seeds else self.digits,
            guard_digits=60,
            series_order=96,
            workers=self.workers if workers is None else workers,
            cache_directory=self.directory / "exact-reductions",
            sample_cache_directory=self.directory / "completed-samples",
            reuse_samples=not recompute,
        )

    def submit(self, stage, *, recompute=False, nearby=False):
        """Start one stage without blocking the notebook's progress/cancel UI."""
        from symbolica.community.hep.integration import ComputationControl

        with self._lock:
            if self._future is not None and not self._future.done():
                raise RuntimeError("A stage is already running; cancel or wait for it.")
            if stage not in ("boundaries", "transport", "amplitude", "restart"):
                raise ValueError("Unknown calculation stage")
            self._control = ComputationControl()
            self.status = f"running {stage}"
            self._future = self._pool.submit(self._run, stage, recompute, nearby)
        return self._future

    def _run(self, stage, recompute, nearby):
        started = perf_counter_ns()
        outcome = "completed"
        try:
            if stage == "boundaries":
                result = self.generate_boundaries(recompute=recompute)
            elif stage == "transport":
                result = self.transport(recompute=recompute, nearby=nearby)
            elif stage == "amplitude":
                result = self.assemble(recompute=recompute)
            else:
                result = self.restart_and_repeat()
            return result
        except Exception as exc:
            outcome = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            with self._lock:
                self.status = f"{stage}: {outcome}"
                self.timings.append({
                    "stage": stage, "recompute": recompute, "nearby": nearby,
                    "elapsed_ns": perf_counter_ns() - started, "outcome": outcome,
                })

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
                "done": self._future is None or self._future.done(),
            }

    def wait(self):
        """Headless entry point: propagate typed native errors to the caller."""
        return None if self._future is None else self._future.result()

    def close(self):
        """Join computation before releasing a headless or replaced session."""
        self.cancel()
        self._pool.shutdown(wait=True)

    def _invalidate_results(self):
        # A partial stage must never leave a mixture of old and new kinematics
        # that could pass the sixteen-configuration amplitude admission check.
        self.results = {}
        self.form_factors = {}
        self.observables = None

    def generate_boundaries(self, *, recompute=False):
        from symbolica.community.hep.integration import (
            BoundaryCache, CalculationCancelled, ComputationControl, IntegralEvaluator,
        )

        self._invalidate_results()
        if recompute:
            self.seeds = BoundaryCache()
            self.seeds.save(self.directory / "seeds")
            # Persist invalidation before expensive work. Cancellation must not
            # make the previous transport bank reappear when the session reloads.
            self.cache = BoundaryCache()
            self.cache.save(self.directory / "transport")
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
            except Exception as exc:
                outcome = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                with self._lock:
                    self.timings.append({
                        "stage": "boundary_configuration", "configuration": configuration.label,
                        "seed_digits": self.seed_digits, "sample_workers": sample_workers,
                        "elapsed_ns": perf_counter_ns() - started,
                        "outcome": outcome, "cache_hit": cache_hit,
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
                except Exception as exc:
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
        from symbolica import E
        from symbolica.community.hep.integration import BoundaryCache

        self._invalidate_results()
        if recompute:
            self.cache = BoundaryCache.load(self.directory / "seeds")
            self.cache.save(self.directory / "transport")
        if nearby:
            self.point = [self.point[0] + E("1/100000"), *self.point[1:]]
        for topology, configuration in self.configurations:
            system = self.systems[topology]
            result = system.evaluate(
                self.cache, self._destination(system, configuration),
                configuration.root_sheets, options=self._options(), control=self._control,
            )
            self.results[configuration.label] = result
            self.cache.save(self.directory / "transport")
        return self.results

    def assemble(self, *, recompute=False, _refinements=0):
        from symbolica import E
        from symbolica.community.hep.integration import (
            AccuracyError, HiggsJetAmplitude, HiggsJetFormFactorProjector,
        )

        if self.observables is not None and not recompute and all(
            value is not None and value >= self.digits
            for value in self.observables.verified_relative_digits.values()
        ):
            return self.observables
        if len(self.results) != 16:
            raise RuntimeError("Transport all sixteen configurations before amplitude assembly.")
        if self.amplitude is None:
            self.amplitude = HiggsJetAmplitude(self.model, control=self._control)
        projector = HiggsJetFormFactorProjector()
        for mass in ("W", "Z"):
            blocks = {}
            for topology in ("planar", "nonplanar"):
                configurations = sorted(
                    (c for name, c in self.configurations if name == topology and c.mass == mass),
                    key=lambda c: c.permutation,
                )
                blocks[topology] = [self.results[c.label] for c in configurations]
            self.form_factors[mass] = projector.evaluate(
                *self.point, self.masses[mass], blocks["planar"], blocks["nonplanar"],
            )
        w, z = self.form_factors["W"], self.form_factors["Z"]
        mw2, mz2 = self.masses["W"], self.masses["Z"]
        parameters = {
            self.model.parameter("aEWM1").symbol: E("128"),
            self.model.parameter("aS").symbol: E("118/1000"),
            self.model.parameter("MZ").symbol: mz2.sqrt(),
            self.model.parameter("Gf").symbol:
                E("𝜋") * E("1/128") * mz2 / (E("2").sqrt() * mw2 * (mz2 - mw2)),
        }
        self.observables = self.amplitude.evaluate(
            *self.point, w.values, z.values, w.absolute_errors, z.absolute_errors,
            parameters=parameters,
            provenance="Native canonical boundaries and physical transport; " + w.provenance + "; " + z.provenance,
            digits=self.digits, control=self._control,
        )
        achieved = self.observables.verified_relative_digits
        if any(value is None or value < self.digits for value in achieved.values()):
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
        return self.observables

    def restart_and_repeat(self):
        from symbolica.community.hep.integration import BoundaryCache

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
