"""Executed only inside actual Pyodide by gg_hg_pyodide.mjs.

Physics and scheduling stay in the frozen notebook controller. Existing native
acceptance helpers check coefficient evidence and refinement allowances. This
file records exact numbers and orchestrates staged and restart observations.
"""

import gzip
import json
import os
from pathlib import Path
import sys
from fractions import Fraction
from time import perf_counter_ns
from types import SimpleNamespace

assert sys.platform == "emscripten", "This runner has no native-Python fallback"
assert not os.environ.get("SYMBOLICA_LICENSE")
assert not os.environ.get("SYMBOLICA_LICENSE_KEY")
sys.path.insert(0, "/acceptance/inputs")

from symbolica import ComplexFloat, E, Float, is_licensed
from symbolica.community.hep import integration
import gg_hg_support as support
import gg_hg_boundaries as loader
import gg_hg_acceptance as acceptance
from gg_hg_anchor_acceptance import check_coefficient_evidence

assert is_licensed()  # Official wasm32 behavior, with no key installed.
assert not integration.automatic_boundary_generation_available
assert not hasattr(integration, "IntegralEvaluator")
INPUT = Path("/acceptance/inputs")
OUTPUT = Path("/acceptance/output")
DATA = INPUT / "data/gg_hg"
TOLERANCE = Float("1e-20", decimal_digits=100)
ONE = Float("1", decimal_digits=100)


def number(value):
    numerator, denominator = value.as_integer_ratio()
    return [str(numerator), str(denominator), value.precision]


def complex_number(value):
    return [number(value.real), number(value.imag)]


def decode(value):
    # Reuse the supplied-input decoder: no conversion through binary64/decimal.
    ratio = Fraction(value[0])
    return loader._number([str(ratio.numerator), str(ratio.denominator), value[1]])


def symbol_name(expression):
    return expression.get_name().rsplit("::", 1)[-1]


def coordinates(values):
    return {symbol_name(key): str(value) for key, value in values.items()}


def boundary(value):
    return {
        "identity": value.identity,
        "coordinates": coordinates(value.coordinates),
        "root_sheets": {symbol_name(key): sign for key, sign in value.root_sheets.items()},
        "leading_power": value.leading_power,
        "coefficients": [[complex_number(z) for z in row] for row in value.coefficients],
        "comparison_errors": [[number(e) for e in row] for row in value.comparison_errors],
        "verified_digits": value.verified_digits,
        "input_verified_digits": value.input_verified_digits,
        "working_bits": value.working_bits,
        "provenance": value.provenance,
    }


def bank(cache):
    return sorted((boundary(entry) for entry in cache.entries()),
                  key=lambda entry: json.dumps([entry["identity"], entry["coordinates"], entry["root_sheets"]], sort_keys=True))


def scientific_results(session):
    return {label: boundary(result) for label, result in session.results.items()}


def numerical_evidence(record):
    # A hit adds provenance and exposes the selected cached source's achieved
    # digits as its input cap. Numeric evidence remains exactly unchanged.
    return {key: value for key, value in record.items()
            if key not in ("provenance", "input_verified_digits")}


def assert_same_numerical_results(previous, current):
    assert set(previous) == set(current)
    for label in previous:
        assert numerical_evidence(previous[label]) == numerical_evidence(current[label]), label


def reference_result(record):
    return SimpleNamespace(
        coefficients=[[ComplexFloat(decode(z[0]), decode(z[1])) for z in row]
                      for row in record["coefficients"]],
        comparison_errors=[[decode(e) for e in row] for row in record["comparison_errors"]],
    )


class Benchmark:
    def __init__(self):
        self.previous = None
        if acceptance_resume:
            self.previous = json.loads((OUTPUT / "acceptance.json").read_text())
        self.report = {
            "schema": "gg-hg-pyodide-scientific-acceptance-v1", "status": "running",
            "stage": acceptance_stage, "resumed": bool(acceptance_resume),
            "settings": {"digits": 20, "guard_digits": 20, "series_order": 16, "workers": 1},
            "phases": [], "cases": [], "comparisons": {},
            "scope": "Actual Pyodide; unchanged supplied seed evidence, notebook controller and native owner algorithms. References enter comparisons only.",
        }
        self.reference = None
        self.session = None

    def finish_checkpoint(self):
        # A checkpoint failure must not replace the numerical exception that
        # caused this stage to unwind. The host also preserves the prior fully
        # published generation if a later generation could not be completed.
        active_error = sys.exception()
        try:
            self.save(checkpoint=True)
        except BaseException as error:
            if active_error is None:
                raise
            self.report.setdefault("checkpoint_failures", []).append(f"{type(error).__name__}: {error}")
            print(f"Checkpoint failed while preserving {type(active_error).__name__}: {error}", flush=True)

    def save(self, *, checkpoint=False):
        if checkpoint and self.session is not None:
            # Describe the durable bank copied by the host bridge, including on
            # a failure after memory admission but before the controller's save.
            path = OUTPUT / "cache/transport"
            persisted = integration.BoundaryCache.load(path) if (path / "physical-boundaries.bin").exists() else integration.BoundaryCache()
            self.report["checkpoint_bank"] = bank(persisted)
        acceptance.write_acceptance_report(OUTPUT, self.report)
        if checkpoint:
            publish_checkpoint()

    def native_reference(self):
        if self.reference is None:
            # First reached only after an actual computed result exists.
            self.reference = json.loads((INPUT / "native-reference.json").read_text())
            assert self.reference["settings"]["digits"] == 20
            assert self.reference["settings"]["workers"] == 1
            labels = [case["label"] for case in self.reference["cases"]]
            assert len(labels) == len(set(labels)) == 16
            assert self.reference["origin"] == json.loads((DATA / "boundaries-manifest.json").read_text())["origin"]
            self.report["native_reference_metadata"] = {
                key: self.reference[key] for key in ("scope", "source_digest", "dependency_digest", "settings", "origin", "reference_note")
            }
        return self.reference

    def check_result(self, label, result):
        topology, configuration = next((topology, c) for topology, c in self.configurations if c.label == label)
        system = self.session.systems[topology]
        assert result.coordinates == self.session._destination(system, configuration)
        assert result.root_sheets == configuration.root_sheets
        assert result.leading_power == 0
        assert 20 <= result.verified_digits <= result.input_verified_digits <= 40
        check_coefficient_evidence(result.coefficients, result.comparison_errors, system.dimension, 20)
        if result.cache_hit:
            actual = boundary(result)
            selected = [entry for entry in bank(self.session.cache)
                        if numerical_evidence(entry) == numerical_evidence(actual)]
            assert len(selected) == 1, (label, "exact cached numerical evidence")
            source = selected[0]
            assert result.input_verified_digits == source["verified_digits"]
            assert result.provenance == "compatible exact-coordinate cache hit; source: " + source["provenance"]
        record = next(case["boundary"] for case in self.native_reference()["cases"] if case["label"] == label)
        assert {name.rsplit("::", 1)[-1]: E(value) for name, value in record["coordinates"]} == {
            symbol_name(key): value for key, value in result.coordinates.items()
        }
        assert {name.rsplit("::", 1)[-1]: sign for name, sign in record["root_germ"]} == {
            symbol_name(key): sign for key, sign in result.root_sheets.items()
        }
        assert record["leading"] == 0 and record["last"] == 4
        assert 20 <= record["verified_digits"] <= record["input_verified_digits"] <= 40
        previous = reference_result(record)
        check_coefficient_evidence(previous.coefficients, previous.comparison_errors, system.dimension, 20)
        count = acceptance.check_coefficient_refinement(
            previous, result, label=label, tolerance=TOLERANCE, one=ONE,
        )
        return {"coefficients": count, "native_errors_and_difference_meet_requested20": True,
                "within_combined_native_and_pyodide_errors": True,
                "native_verified_digits": record["verified_digits"],
                "pyodide_verified_digits": result.verified_digits}

    async def stage(self, action, mode, **kwargs):
        started = perf_counter_ns()
        phase = {"action": action, "mode": mode, "status": "running"}
        self.report["phases"].append(phase)
        original_steps = self.session._transport_steps
        phase_cases = []

        def observed_steps(**options):
            steps = original_steps(**options)
            while True:
                before = perf_counter_ns()
                try:
                    label = next(steps)
                except StopIteration:
                    return
                elapsed = perf_counter_ns() - before
                result = self.session.results[label]
                # Timing closes before serialization and comparison. The native
                # controller has already saved the completed configuration.
                check_started = perf_counter_ns()
                comparison = self.check_result(label, result)
                record = {
                    "label": label, "mode": mode, "controller_step_ns": elapsed,
                    "owner_evaluation_ns": result.elapsed_nanoseconds,
                    "cache_hit": result.cache_hit, "steps": result.steps,
                    "rejected_steps": result.rejected_steps, "inserted_points": result.inserted_points,
                    "starting_coordinates": coordinates(result.starting_coordinates),
                    "attempted_starting_points": [coordinates(p) for p in result.attempted_starting_points],
                    "source_attempt_outcomes": result.source_attempt_outcomes,
                    "diagnostics": {name: getattr(result, name) for name in (
                        "predicate_evaluations", "rational_trials", "rational_steps",
                        "rational_fallbacks", "last_rational_fallback",
                        "fundamental_boundary_charts", "fundamental_boundary_fallback",
                        "fundamental_boundary_retry", "conditioning_digits",
                    )},
                    "boundary": boundary(result), "comparison": comparison,
                }
                record["validation_and_encoding_ns"] = perf_counter_ns() - check_started
                self.report["cases"].append(record)
                phase_cases.append(record)
                phase["last_completed_configuration"] = label
                self.save(checkpoint=True)
                print(f"{mode} {label}: {elapsed / 1e9:.6f}s, hit={result.cache_hit}, digits={result.verified_digits}", flush=True)
                yield label

        self.session._transport_steps = observed_steps
        try:
            self.session.submit(action, **kwargs)
            result = await self.session.wait_async()
            if action == "amplitude":
                phase["native_amplitude_evaluation_ns"] = result.elapsed_nanoseconds
            phase["status"] = "passed"
            return result
        except BaseException as error:
            phase["status"] = "failed"
            phase["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            self.session._transport_steps = original_steps
            phase["wall_elapsed_ns_including_validation_and_host_checkpoints"] = perf_counter_ns() - started
            phase["controller_transport_ns"] = sum(c["controller_step_ns"] for c in phase_cases)
            phase["owner_evaluation_ns"] = sum(c["owner_evaluation_ns"] for c in phase_cases)
            phase["cold_configurations"] = sum(not c["cache_hit"] for c in phase_cases)
            phase["cache_hits"] = sum(c["cache_hit"] for c in phase_cases)
            phase["controller_status"] = self.session.snapshot()
            self.finish_checkpoint()

    async def initialize(self):
        started = perf_counter_ns()
        self.session = support.CalculationSession(
            DATA / "native-model.json", OUTPUT / "cache", digits=20, workers=1,
            boundary_workers=1, boundary_bundle=DATA, cooperative=True,
        )
        assert self.session._pool is None
        assert not self.session.automatic_boundary_generation_available
        options = self.session._options()
        assert (options.digits, options.guard_digits, options.series_order, options.workers) == (20, 20, 16, 1)
        self.configurations = self.session.configurations[:]
        assert len(self.configurations) == 16
        self.report["session_construction_and_existing_cache_load_ns"] = perf_counter_ns() - started
        self.report["runtime_attestation"] = acceptance.runtime_attestation()
        self.report["initial_cache_entries"] = len(self.session.cache)
        if self.previous:
            assert self.previous["runtime_attestation"] == self.report["runtime_attestation"]
            assert bank(self.session.cache) == self.previous["checkpoint_bank"]
            self.report["resumed_bank_exact"] = True
            self.report["previous_run"] = {
                "status": self.previous["status"], "stage": self.previous["stage"],
                "passed_scope": self.previous.get("passed_scope"),
                "phases": self.previous["phases"],
            }
        self.save()
        if not acceptance_resume:
            seeds = await self.stage("supplied_boundaries", "cold_import")
        else:
            # Validate the original input independently without replacing the
            # resumed, growing bank with a seed-only bank.
            _, seeds = loader.load_boundary_bundle(DATA, self.session.systems)
        encoded = json.loads(gzip.decompress((DATA / "boundaries.json.gz").read_bytes()))["boundaries"]
        assert len(seeds) == len(encoded) == 16
        count = 0
        for record in encoded:
            result = seeds[record["label"]]
            actual = boundary(result)
            assert actual["coefficients"] == record["coefficients"]
            assert actual["comparison_errors"] == record["comparison_errors"]
            assert result.verified_digits == result.input_verified_digits == 40
            assert result.working_bits == record["working_bits"] == 415
            count += sum(map(len, result.coefficients))
        assert count == 4360
        before = bank(self.session.cache)
        for name in ("seeds", "transport"):
            cache = self.session.seeds if name == "seeds" else self.session.cache
            cache.save(OUTPUT / "cache" / name)
            restored = integration.BoundaryCache.load(OUTPUT / "cache" / name)
            assert bank(restored) == bank(cache)
        self.report["supplied_import"] = {
            "configurations": 16, "complex_coefficients": count,
            "all_values_errors_and_precisions_exact": True, "binary_restart_exact": True,
            "input_verified_digits": 40, "working_bits": 415,
        }
        assert before == bank(self.session.cache)
        self.report["checkpoint_bank"] = bank(self.session.cache)
        self.save(checkpoint=True)

    async def pilot(self):
        selected = [next((name, c) for name, c in self.configurations if name == topology)
                    for topology in ("planar", "nonplanar")]
        self.session.configurations = selected
        try:
            await self.stage("transport", "cold_pilot")
            assert len(self.session.results) == 2
            assert all(not r.cache_hit and r.steps > 0 for r in self.session.results.values())
            self.report["pilot_results"] = scientific_results(self.session)
            self.report["pilot_passed"] = True
        finally:
            self.session.configurations = self.configurations
        self.report["checkpoint_bank"] = bank(self.session.cache)
        self.save(checkpoint=True)

    def relative_comparison(self, value, error, expected, expected_error, label):
        # Comparison arithmetic uses the same owner numbers as the existing
        # acceptance helpers. No new evaluator or uncertainty estimator.
        difference = abs(value - expected)
        assert error.is_finite() and error >= 0
        assert expected_error.is_finite() and expected_error >= 0
        assert difference <= error + expected_error, (label, "combined errors")
        for sample, bound in ((value, error), (expected, expected_error)):
            assert abs(sample) > 0
            assert bound <= TOLERANCE * abs(sample), (label, "relative error")
            assert difference <= TOLERANCE * abs(sample), (label, "relative difference")
        return {"absolute_difference": number(difference), "combined_error": number(error + expected_error),
                "both_errors_and_difference_meet20_relative_digits": True}

    def amplitude_evidence(self):
        native = self.native_reference()
        factors = {}
        comparisons = {"form_factors": {}, "observables": {}}
        assert set(self.session.form_factors) == {"W", "Z"}
        for mass, result in self.session.form_factors.items():
            reference = next(r for r in native["form_factors"] if r["mass"] == mass)
            assert len(result.values) == len(reference["values"][0]) == 4
            assert all(d is not None and d >= 20 for d in result.verified_relative_digits)
            factors[mass] = {"values": [complex_number(v) for v in result.values],
                             "absolute_errors": [number(e) for e in result.absolute_errors],
                             "verified_relative_digits": result.verified_relative_digits,
                             "provenance": result.provenance}
            comparisons["form_factors"][mass] = [self.relative_comparison(
                value, error, ComplexFloat(decode(encoded[0]), decode(encoded[1])), decode(allowance), f"{mass}/{index}",
            ) for index, (value, error, encoded, allowance) in enumerate(zip(
                result.values, result.absolute_errors, reference["values"][0], reference["absolute_errors"], strict=True), 1)]
        observables = self.session.observables
        assert set(observables.values) == {"electroweak_squared", "interference", "effective_squared"}
        assert all(d is not None and d >= 20 for d in observables.verified_relative_digits.values())
        values = {}
        for label, value in observables.values.items():
            reference = next(r for r in native["observables"] if r["label"] == label)
            comparisons["observables"][label] = self.relative_comparison(
                value, observables.absolute_errors[label], decode(reference["value"]), decode(reference["absolute_error"]), label,
            )
            values[label] = {"value": number(value), "absolute_error": number(observables.absolute_errors[label]),
                             "arithmetic_change": number(observables.arithmetic_changes[label]),
                             "verified_relative_digits": observables.verified_relative_digits[label]}
        self.report["form_factors"] = factors
        self.report["observables"] = values
        self.report["amplitude_working_bits"] = observables.working_bits
        self.report["native_comparison"] = comparisons
        # Archived external allowances remain independent, including EW's
        # documented 19-digit cap; they are not upgraded by this fresh run.
        reference, metadata = acceptance.load_comparison_reference(DATA / "amplitude-validation.json")
        self.report["archived_reference"] = metadata
        self.report["archived_form_factor_comparison"] = acceptance.compare_form_factors(self.session.form_factors, reference)
        archived = {}
        for label, value in observables.values.items():
            limits = reference["reference_accuracy"][label]
            expected = Float(reference["expected_observables"][label], decimal_digits=100)
            allowance = Float(limits["input_absolute_error"], decimal_digits=100) + Float(limits["rounding_absolute_error"], decimal_digits=100)
            difference = abs(value - expected)
            assert difference <= allowance + observables.absolute_errors[label]
            archived[label] = {"difference": number(difference), "reference_allowance": number(allowance),
                               "reference_conditional_relative_digits": limits["conditional_relative_digits"]}
        self.report["archived_observable_comparison"] = archived

    async def full(self):
        mode = "resumed_transport" if acceptance_resume else "continuation_after_pilot"
        await self.stage("transport", mode)
        assert len(self.session.results) == 16
        reference, metadata = acceptance.load_comparison_reference(DATA / "coherent-reference.json")
        count = acceptance.check_transport_reference_coverage(
            self.session.results, reference, [c.label for _, c in self.configurations],
        )
        assert count == 4360
        self.report["complete_transport_results"] = scientific_results(self.session)
        self.report["transport_comparison"] = {"coefficients": count, "configurations": 16,
                                                 "native_comparison_checked_per_configuration": True}
        self.report["archived_transport_reference"] = metadata
        archived_count = 0
        maximum = Float("0", decimal_digits=100)
        for case in reference["cases"]:
            assert case["source_verified_digits_cap"] >= 20
            result = self.session.results[case["label"]]
            for values, expected_values in zip(result.coefficients, case["reference_values"], strict=True):
                for value, expected in zip(values, expected_values, strict=True):
                    comparison_value = ComplexFloat(expected["real"], expected["imaginary"], decimal_digits=100)
                    scaled_difference = abs(value - comparison_value) / max(ONE, abs(comparison_value))
                    assert scaled_difference <= TOLERANCE, case["label"]
                    maximum = max(maximum, scaled_difference)
                    archived_count += 1
        assert archived_count == 4360
        self.report["archived_transport_comparison"] = {
            "coefficients": archived_count, "mixed_digits_checked": 20,
            "maximum_scaled_difference": number(maximum),
            "source_verified_digits_caps": {case["label"]: case["source_verified_digits_cap"] for case in reference["cases"]},
            "endpoint_deltas": {case["label"]: case["endpoint_delta"] for case in reference["cases"]},
        }
        self.report["checkpoint_bank"] = bank(self.session.cache)
        self.save(checkpoint=True)
        await self.stage("amplitude", "first_use_amplitude")
        self.amplitude_evidence()
        before = scientific_results(self.session)
        before_bank = bank(self.session.cache)
        await self.stage("restart", "binary_reloaded_transport")
        assert_same_numerical_results(before, scientific_results(self.session))
        assert bank(self.session.cache) == before_bank
        assert all(r.cache_hit and r.steps == 0 and r.inserted_points == 0 for r in self.session.results.values())
        await self.stage("transport", "warm_transport")
        assert_same_numerical_results(before, scientific_results(self.session))
        assert bank(self.session.cache) == before_bank
        assert all(r.cache_hit and r.steps == 0 and r.inserted_points == 0 for r in self.session.results.values())
        first_values = self.report["observables"]
        first_factors = self.report["form_factors"]
        await self.stage("amplitude", "warm_amplitude", recompute=True)
        self.amplitude_evidence()
        assert self.report["observables"] == first_values
        for mass in ("W", "Z"):
            assert {key: value for key, value in self.report["form_factors"][mass].items() if key != "provenance"} == {
                key: value for key, value in first_factors[mass].items() if key != "provenance"
            }
        self.report["binary_restart_and_warm_results_exact"] = True
        self.report["all_observables_meet_requested20"] = True
        self.report["checkpoint_bank"] = bank(self.session.cache)


async def main():
    benchmark = Benchmark()
    try:
        await benchmark.initialize()
        if not acceptance_resume:
            await benchmark.pilot()
        if acceptance_stage == "full":
            await benchmark.full()
        benchmark.report["status"] = "passed"
        benchmark.report["passed_scope"] = "full16+8FF+3observables+restart+warm" if acceptance_stage == "full" else "exact16seed_import+restart+planar_nonplanar_pilot"
    except BaseException as error:
        benchmark.report["status"] = "failed"
        benchmark.report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if benchmark.session is not None:
            benchmark.report["checkpoint_bank"] = bank(benchmark.session.cache)
            benchmark.session.close()
        benchmark.finish_checkpoint()
