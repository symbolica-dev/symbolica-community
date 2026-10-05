"""Long native acceptance: python gg_hg_acceptance.py --directory PATH [--resume].

Run with a release-built community extension. Reference files are opened only
after native seed generation, transport and amplitude assembly have completed.
Ordinary notebook smoke tests do not invoke this program.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter_ns, sleep
import hashlib
import importlib
import sys
from typing import Any

from gg_hg_support import CalculationSession, boundary_evidence


def file_attestation(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"path": str(path), "size_bytes": size, "sha256": digest.hexdigest()}


def runtime_attestation():
    """Identify the loaded native owner and the Python code steering this run."""
    from symbolica import E, S
    from symbolica.community.hep.integration import KinematicTransport

    core = importlib.import_module("symbolica.core")
    x, epsilon, master = S(
        "integration_acceptance_attestation_v1::x",
        "integration_acceptance_attestation_v1::epsilon",
        "integration_acceptance_attestation_v1::master",
    )
    # This fixed, unevaluated connection asks the native owner for its existing
    # source-sensitive identity. Python does not duplicate the fingerprint rules.
    witness = KinematicTransport(
        epsilon, {x: [[E("0")]]}, [master], E("1"),
        branch_domain="native source attestation v1",
    )
    return {
        "schema": 1,
        "native_identity_witness": {
            "contract": "zero connection, integration_acceptance_attestation_v1 symbols, normalization 1",
            "identity": witness.identity,
        },
        "loaded_extension": file_attestation(core.__file__),
        "execution_sources": {
            "acceptance": file_attestation(__file__),
            "controller": file_attestation(CalculationSession.__init__.__code__.co_filename),
            "notebook": file_attestation(
                Path(CalculationSession.__init__.__code__.co_filename).with_name("gg_hg.py"),
            ),
        },
        "python": {"version": sys.version, "executable": sys.executable},
    }


def verify_runtime_attestation(recorded):
    if runtime_attestation() != recorded:
        raise RuntimeError("The native extension or acceptance sources changed during this run.")


def verify_anchor_report(path, current_runtime):
    """Require the separate Euclidean proof from this installed native graph.

    Only the proof is inspected here. Numerical comparison fixtures are not
    opened and no anchor coefficients are supplied to the physical evaluator.
    Files may be relocated; their contents and the native identity must agree.
    """
    from gg_hg_anchor_acceptance import ANCHORS, SCHEMA, check_coefficient_evidence
    from symbolica import ComplexFloat, Float

    payload = Path(path).read_bytes()
    report = json.loads(payload)
    try:
        if (report["schema"] != SCHEMA or report["status"] != "passed"
                or report["runtime_attestation_verified_at_completion"] is not True
                or report["references_loaded"] is not True or report["checked_coefficients"] != 545
                or report["options"]["digits"] < 20
                or set(report["native"]) != set(ANCHORS)
                or set(report["comparisons"]) != set(ANCHORS)):
            raise ValueError("Anchor report has no complete passed 20-digit Euclidean proof.")
        recorded = report["runtime_attestation"]
        if recorded["native_identity_witness"] != current_runtime["native_identity_witness"]:
            raise ValueError("Anchor report was computed with a different native source identity.")
        for key in ("sha256", "size_bytes"):
            if recorded["loaded_extension"][key] != current_runtime["loaded_extension"][key]:
                raise ValueError("Anchor report was computed with a different installed extension.")
        for name in ("acceptance", "controller", "notebook"):
            if recorded["execution_sources"][name]["sha256"] != current_runtime["execution_sources"][name]["sha256"]:
                raise ValueError("Anchor report used different acceptance steering sources.")
        runner = file_attestation(Path(__file__).with_name("gg_hg_anchor_acceptance.py"))
        if report["anchor_runner"]["sha256"] != runner["sha256"]:
            raise ValueError("Anchor report used a different anchor runner.")
        for topology, expected in ANCHORS.items():
            native, comparison = report["native"][topology], report["comparisons"][topology]
            if (not isinstance(native["identity"], str) or not native["identity"]
                    or native["coordinates"] != expected["coordinates"] or native["root_sheets"] != "principal"
                    or native["dimension"] != expected["dimension"] or native["leading_power"] != 0
                    or native["last_power"] != 4 or native["verified_digits"] < report["options"]["digits"]
                    or native["uncertainties_within_requested_budget"] is not True
                    or "no numerical reference seed" not in native["provenance"]
                    or comparison["passed"] is not True or comparison["failures"]
                    or comparison["checked_coefficients"] != 5 * expected["dimension"]
                    or comparison["reference_accuracy_cap_digits"] != 40
                    or comparison["reference_loaded_after_both_native_successes"] is not True
                    or comparison["reference"]["sha256"] != expected["reference_sha256"]):
                raise ValueError(f"Incomplete Euclidean anchor evidence for {topology}.")
            if (any(not isinstance(value[part], str) for row in native["coefficients"]
                    for value in row for part in ("real", "imaginary"))
                    or any(not isinstance(value, str) for row in native["comparison_errors"] for value in row)):
                raise ValueError("Anchor report numerical evidence must retain decimal strings.")
            precision = max(100, report["options"]["digits"] + 40)
            check_coefficient_evidence(
                [[ComplexFloat(value["real"], value["imaginary"], decimal_digits=precision)
                  for value in row] for row in native["coefficients"]],
                [[Float(value, decimal_digits=precision) for value in row]
                 for row in native["comparison_errors"]],
                expected["dimension"], report["options"]["digits"],
            )
    except (KeyError, TypeError) as error:
        raise ValueError("Malformed Euclidean anchor report.") from error
    return {
        "path": str(Path(path).resolve()), "sha256": hashlib.sha256(payload).hexdigest(),
        "checked_coefficients": 545, "minimum_verified_digits": min(
            result["verified_digits"] for result in report["native"].values()
        ),
        "native_source_identity_matched": True, "installed_extension_matched": True,
    }


def check_coefficient_refinement(previous, result, *, label, tolerance, one):
    """Check both requested accuracy and the native uncertainty estimates."""
    count = 0
    rows = zip(
        previous.coefficients, result.coefficients,
        previous.comparison_errors, result.comparison_errors, strict=True,
    )
    for old_row, new_row, old_errors, new_errors in rows:
        for old, new, old_error, new_error in zip(
            old_row, new_row, old_errors, new_errors, strict=True,
        ):
            assert old_error.is_finite() and old_error >= 0, label
            assert new_error.is_finite() and new_error >= 0, label
            difference = abs(old - new)
            assert difference <= tolerance * max(one, abs(new)), (label, "mixed accuracy target")
            assert difference <= old_error + new_error, (label, "coefficient uncertainty estimates")
            count += 1
    return count


def compare_form_factors(native, reference):
    """Compare all W/Z components using the archived absolute allowances."""
    from symbolica import ComplexFloat, Float

    assert set(native) == {"W", "Z"}
    references = {entry["mass"]: entry["values"] for entry in reference["form_factors"]}
    assert len(reference["form_factors"]) == len(references) == 2
    assert set(references) == set(native)
    comparison = {}
    count = 0
    for mass in ("W", "Z"):
        assert [entry["index"] for entry in references[mass]] == [1, 2, 3, 4]
        comparison[mass] = []
        rows = zip(
            native[mass].values, native[mass].absolute_errors,
            native[mass].verified_relative_digits, references[mass], strict=True,
        )
        for value, error, digits, expected in rows:
            expected_value = ComplexFloat(expected["real"], expected["imaginary"], decimal_digits=100)
            expected_error = Float(expected["absolute_error"], decimal_digits=100)
            assert error.is_finite() and error >= 0, mass
            assert expected_error.is_finite() and expected_error >= 0, mass
            difference = abs(value - expected_value)
            assert difference <= error + expected_error, (mass, expected["index"], "form-factor uncertainty estimates")
            comparison[mass].append({
                "index": expected["index"], "value": str(value),
                "absolute_difference": str(difference),
                "native_absolute_error": str(error),
                "native_verified_relative_digits": digits,
                "reference_absolute_error": expected["absolute_error"],
            })
            count += 1
    assert count == 8
    return {"components": count, "comparison": comparison}


def write_acceptance_report(directory, report):
    temporary = directory / "acceptance.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(directory / "acceptance.json")


def persist_progress(directory, report, state, *, stage, mode, elapsed_ns,
                     seed_digits, archived_timings=()):
    """Write observed progress without touching numerical caches or results."""
    report["progress"] = {
        "stage": stage, "mode": mode, "elapsed_ns": elapsed_ns,
        "seed_digits": seed_digits, "status": state["status"], "done": state["done"],
        "recent_events": state["events"],
        "numerical_generation": state.get("numerical_generation"),
        "completed_sample_files": sum(1 for _ in (directory / "completed-samples").glob("sample-*.bin")),
    }
    report["timings"] = list(archived_timings) + state["timings"]
    write_acceptance_report(directory, report)


def wait_for_stage(session, on_progress, *, poll_interval=5):
    """Poll long work, but return immediately when a warm stage completes."""
    if poll_interval <= 0:
        raise ValueError("The progress interval must be positive.")
    while True:
        on_progress(session.snapshot())
        try:
            value = session.wait(timeout=poll_interval)
        except TimeoutError:
            if not session.snapshot()["done"]:
                continue
            # A calculation can itself raise TimeoutError. Re-read a completed
            # future to preserve its original exception instead of polling forever.
            value = session.wait()
        except BaseException:
            on_progress(session.snapshot())
            raise
        on_progress(session.snapshot())
        return value


def interrupt_forced_boundaries(session, samples, after_samples, *, poll_interval=1,
                                on_progress=None):
    """Cancel new-generation work before its first verified configuration."""
    from symbolica.community.hep.integration import CalculationCancelled

    if after_samples <= 0:
        raise ValueError("The interruption sample count must be positive.")
    before = samples()
    previous_generation = session.numerical_generation()
    started = perf_counter_ns()
    session.submit("boundaries", recompute=True)
    while True:
        state = session.snapshot()
        if on_progress is not None:
            on_progress(state)
        if state["done"]:
            session.wait()  # Preserve any typed failure instead of hiding it.
            raise AssertionError("Boundary generation finished before the requested interruption.")
        generation = session.numerical_generation()
        # At equal precision even freshly computed payloads and filenames may
        # be identical. Only the committed generation proves they are new work.
        persisted = samples() if generation is not None and generation != previous_generation else {}
        if len(persisted) >= after_samples:
            break
        sleep(poll_interval)
    cancelled_at = perf_counter_ns()
    session.cancel()
    try:
        if on_progress is None:
            session.wait()
        else:
            wait_for_stage(session, on_progress, poll_interval=poll_interval)
    except CalculationCancelled:
        pass
    else:
        raise AssertionError("Expected typed cancellation after complete sample checkpoint.")
    finished = perf_counter_ns()
    after = samples()
    assert generation is not None
    assert session.numerical_generation() == generation
    assert all(after.get(name) == digest for name, digest in persisted.items())
    assert len(session.seeds) == 0, "Interruption occurred after a verified configuration was saved."
    archive = session.directory / generation["archive_directory"]
    assert all(
        hashlib.sha256((archive / "completed-samples" / name).read_bytes()).hexdigest() == digest
        for name, digest in before.items()
    ), "Pre-force samples must be retained only in the archived generation."
    return {
        "phase": "forced_refinement", "seed_digits": session.seed_digits,
        "baseline_sample_files": len(before), "retained_sample_files": len(after),
        "new_complete_samples": len(after), "archived_sample_files": len(before),
        "previous_generation": previous_generation, "numerical_generation": generation,
        "archived_sample_payloads_preserved": True,
        "elapsed_ns": finished - started,
        "cancellation_latency_ns": finished - cancelled_at,
        "sample_payloads_preserved": True, "verified_configurations_before_cancel": 0,
    }, after


def main():
    if not __debug__:
        raise RuntimeError("Acceptance requires enabled assertions; remove -O or PYTHONOPTIMIZE.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1,
                        help="Total sample-worker budget, partitioned across concurrent boundary configurations.")
    parser.add_argument("--boundary-workers", type=int, default=1,
                        help="Maximum simultaneous boundary configurations (bounded by --workers).")
    parser.add_argument("--seed-digits", type=int, default=30)
    parser.add_argument("--interrupt-after-samples", type=int, default=0,
                        help="During higher-precision forced refinement, cancel after this many new complete samples, then reload and resume.")
    parser.add_argument("--refine-digits", type=int, default=10,
                        help="Extra seed digits for forced independent recomputation (must be positive).")
    parser.add_argument("--anchor-report", type=Path,
                        help="Require a passed Euclidean-anchor report from this installed native graph.")
    args = parser.parse_args()
    if args.refine_digits <= 0 or args.interrupt_after_samples < 0 or min(args.workers, args.boundary_workers) < 1:
        parser.error("Refinement and worker budgets must be positive; interruption sample count must be nonnegative.")
    if args.directory.exists() and any(args.directory.iterdir()) and not args.resume:
        parser.error("Cold acceptance requires an empty directory; use --resume to reuse it.")
    args.directory.mkdir(parents=True, exist_ok=True)
    data = Path(__file__).parent / "data" / "gg_hg"
    attestation = runtime_attestation()
    anchor_evidence = verify_anchor_report(args.anchor_report, attestation) if args.anchor_report else None
    session = CalculationSession(
        data / "native-model.json", args.directory,
        seed_digits=args.seed_digits, workers=args.workers, boundary_workers=args.boundary_workers,
    )
    report: dict[str, Any] = {
        "status": "running", "mode": "resumed" if args.resume else "cold", "stages": [],
        "runtime_attestation": attestation,
        "euclidean_anchor_proof": anchor_evidence,
        "scope": "Physical seeds, transport, amplitude and restart; independent Euclidean anchor reports are a separate prerequisite.",
        "resources": {
            "sample_worker_budget": args.workers, "boundary_workers": args.boundary_workers,
            "effective_boundary_workers": min(args.boundary_workers, args.workers, len(session.configurations)),
            "sample_workers_per_configuration": args.workers // min(
                args.boundary_workers, args.workers, len(session.configurations),
            ),
        },
        "inputs": {
            "model_sha256": hashlib.sha256((data / "native-model.json").read_bytes()).hexdigest(),
            "point": [str(value) for value in session.point],
            "mass_squared": {name: str(value) for name, value in session.masses.items()},
            "systems": {name: system.provenance for name, system in session.systems.items()},
        },
    }
    archived_timings = []

    def write_report():
        write_acceptance_report(args.directory, report)

    def progress_callback(stage, mode, started):
        return lambda state: persist_progress(
            args.directory, report, state, stage=stage, mode=mode,
            elapsed_ns=perf_counter_ns() - started,
            seed_digits=session.seed_digits, archived_timings=archived_timings,
        )

    def run(stage, *, mode=None, **kwargs):
        initial_seed_digits = session.seed_digits
        started = perf_counter_ns()
        session.submit(stage, **kwargs)
        value = wait_for_stage(session, progress_callback(stage, mode, started))
        report["stages"].append({
            "stage": stage, "mode": mode,
            "seed_digits": initial_seed_digits, "final_seed_digits": session.seed_digits,
            "observable_digits": session.digits,
            "elapsed_ns": perf_counter_ns() - started, **kwargs,
        })
        write_report()
        return value

    def samples():
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (args.directory / "completed-samples").glob("sample-*.bin")}

    write_report()
    try:
        seeds = run("boundaries", mode="resumed" if args.resume else "cold")
        assert len(seeds) == 16
        assert all(
            result.verified_digits >= session.seed_digits
            and "no numerical reference seed" in result.provenance
            for _, result in seeds
        )
        report["native_seeds"] = {
            label: {
                "identity": result.identity, "working_bits": result.working_bits,
                "verified_digits": result.verified_digits, "provenance": result.provenance,
            }
            for label, result in seeds
        }
        run("transport", mode="resumed" if args.resume else "cold")
        observables = run("amplitude", mode="cold")
        assert all(d is not None and d >= 20 for d in observables.verified_relative_digits.values())
        assert all(r.verified_digits >= 20 for r in session.results.values())

        # Reference access is confined to comparison, after the result exists.
        from symbolica import ComplexFloat, Float

        reference = json.loads((data / "coherent-reference.json").read_text())
        count = 0
        max_scaled_difference = Float("0", decimal_digits=100)
        one = Float("1", decimal_digits=100)
        tolerance = Float("1e-20", decimal_digits=100)
        for case in reference["cases"]:
            result = session.results[case["label"]]
            assert case["source_verified_digits_cap"] >= 20
            assert result.leading_power == 0
            assert "Native automatic auxiliary-mass boundary;" in result.provenance
            for native_row, reference_row in zip(result.coefficients, case["reference_values"], strict=True):
                for value, expected in zip(native_row, reference_row, strict=True):
                    expected = ComplexFloat(expected["real"], expected["imaginary"], decimal_digits=100)
                    scale = max(one, abs(expected))
                    difference = abs(value - expected) / scale
                    assert difference <= tolerance, (case["label"], str(difference))
                    max_scaled_difference = max(max_scaled_difference, difference)
                    count += 1
        assert count == 4360
        report["reference_comparison"] = {
            "coefficients": count, "checked_mixed_digits": 20,
            "maximum_scaled_difference": str(max_scaled_difference),
            "reference_precision": "Preserved component source caps/precision and endpoint delta in coherent-reference.json",
            "source_verified_digits_caps": {
                case["label"]: case["source_verified_digits_cap"] for case in reference["cases"]
            },
            "endpoint_deltas": {case["label"]: case["endpoint_delta"] for case in reference["cases"]},
        }
        report["observables"] = {
            name: {"value": str(value), "absolute_error": str(observables.absolute_errors[name]),
                   "relative_digits": observables.verified_relative_digits[name]}
            for name, value in observables.values.items()
        }
        observable_reference = json.loads((data / "amplitude-validation.json").read_text())
        report["form_factor_comparison"] = compare_form_factors(
            session.form_factors, observable_reference,
        )
        report["observable_comparison"] = {}
        for name, value in observables.values.items():
            expected = Float(observable_reference["expected_observables"][name], decimal_digits=100)
            metadata = observable_reference["reference_accuracy"][name]
            reference_error = (Float(metadata["input_absolute_error"], decimal_digits=100)
                               + Float(metadata["rounding_absolute_error"], decimal_digits=100))
            difference = abs(value - expected)
            assert difference <= observables.absolute_errors[name] + reference_error, name
            report["observable_comparison"][name] = {
                "absolute_difference": str(difference),
                "reference_absolute_uncertainty": str(reference_error),
                "reference_conditional_relative_digits": metadata["conditional_relative_digits"],
            }
        evidence = boundary_evidence(session.cache)
        run("restart", mode="reloaded")
        assert boundary_evidence(session.cache) == evidence
        report["binary_restart"] = {
            "entries": len(evidence), "values_and_evidence_preserved": True,
            "exact_repeat_without_steps": True,
        }
        warm = run("transport", mode="warm")
        assert all(r.cache_hit and r.steps == 0 for r in warm.values())
        warm_observables = run("amplitude", mode="warm")
        assert all(d is not None and d >= 20
                   for d in warm_observables.verified_relative_digits.values())
        for name, value in warm_observables.values.items():
            assert abs(value - observables.values[name]) <= (
                warm_observables.absolute_errors[name] + observables.absolute_errors[name]
            ), name

        # Independent precision refinement forces new numerical work while
        # allowing the exact reduction cache to survive. Transport also starts
        # afresh from the newly verified seeds, not old endpoints.
        original = {name: result for name, result in session.results.items()}
        session.seed_digits += args.refine_digits
        if args.interrupt_after_samples:
            interruption, interrupted_samples = interrupt_forced_boundaries(
                session, samples, args.interrupt_after_samples,
                on_progress=progress_callback("boundaries", "forced_refinement_interrupted", perf_counter_ns()),
            )
            report["interruption"] = interruption
            report["stages"].append({
                "stage": "boundaries", "mode": "forced_refinement_interrupted",
                "seed_digits": session.seed_digits, "observable_digits": session.digits,
                "elapsed_ns": interruption["elapsed_ns"], "recompute": True,
                "outcome": "CalculationCancelled",
            })
            seed_digits = session.seed_digits
            archived_timings.extend(session.snapshot()["timings"])
            session.close()
            session = CalculationSession(
                data / "native-model.json", args.directory,
                seed_digits=seed_digits, workers=args.workers, boundary_workers=args.boundary_workers,
            )
            write_report()
            # Resume only this forced generation's completed samples, including
            # when filenames happen to equal those from archived generations.
            assert len(session.seeds) == len(session.cache) == 0
            assert session.numerical_generation() == interruption["numerical_generation"]
            refined_seeds = run("boundaries", mode="resumed_refinement")
            resumed_samples = samples()
            assert all(resumed_samples.get(name) == digest
                       for name, digest in interrupted_samples.items())
            interruption["sample_payloads_preserved_after_resume"] = True
        else:
            refined_seeds = run("boundaries", mode="forced_refinement", recompute=True)
        assert all(
            not result.cache_hit and result.verified_digits >= session.seed_digits
            and "no numerical reference seed" in result.provenance
            for _, result in refined_seeds
        )
        refined = run("transport", mode="forced_refinement", recompute=True)
        assert any(not r.cache_hit for r in refined.values())
        refined_count = 0
        for label, result in refined.items():
            refined_count += check_coefficient_refinement(
                original[label], result, label=label, tolerance=tolerance, one=one,
            )
        assert refined_count == 4360
        refined_observables = run("amplitude", mode="forced_refinement", recompute=True)
        refined_form_factors = compare_form_factors(session.form_factors, observable_reference)
        for name, value in refined_observables.values.items():
            assert refined_observables.verified_relative_digits[name] >= 20
            assert abs(value - observables.values[name]) <= (
                refined_observables.absolute_errors[name] + observables.absolute_errors[name]
            ), name
        report["forced_precision_refinement"] = {
            "seed_digits": session.seed_digits, "old_numerical_caches_bypassed": True,
            "new_precision_samples_resumed": bool(args.interrupt_after_samples),
            "exact_reductions_reusable": True, "checked_mixed_digits": 20,
            "coefficients_with_consistent_uncertainties": refined_count,
            "form_factor_comparison": refined_form_factors,
        }
        accumulated = session.cache.entries()
        seeds = session.seeds.entries()
        nearby = run("transport", mode="nearby", nearby=True)
        assert any(not r.cache_hit for r in nearby.values())
        reused = []
        for label, result in nearby.items():
            candidates = [entry for entry in accumulated
                          if entry.identity == result.identity
                          and entry.coordinates == result.starting_coordinates
                          and entry.provenance in result.provenance]
            assert candidates, (label, "selected source is absent from the retained bank")
            if not any(entry.identity == result.identity
                       and entry.coordinates == result.starting_coordinates for entry in seeds):
                reused.append(label)
            assert "Native automatic auxiliary-mass boundary;" in result.provenance
        assert reused, "Nearby transport did not reuse any accumulated physical point."
        run("amplitude", mode="nearby", recompute=True)
        report["nearby"] = {
            "point": [str(a) for a in session.point],
            "sources": {name: str(result.starting_coordinates) for name, result in nearby.items()},
            "inserted_points": sum(result.inserted_points for result in nearby.values()),
            "configurations_using_accumulated_points": reused,
            "source_provenance_preserved": True,
        }
        verify_runtime_attestation(report["runtime_attestation"])
        report["runtime_attestation_verified_at_completion"] = True
        report["status"] = "passed"
    except BaseException as exc:
        session.cancel()
        report["status"] = "incomplete_or_failed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        write_report()  # Record cancellation/failure before native workers drain.
        raise
    finally:
        session.close()
        report["timings"] = archived_timings + session.snapshot()["timings"]
        write_report()


if __name__ == "__main__":
    main()
