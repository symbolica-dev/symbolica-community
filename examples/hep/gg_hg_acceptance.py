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

from gg_hg_support import CalculationSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed-digits", type=int, default=30)
    parser.add_argument("--interrupt-after-samples", type=int, default=0,
                        help="Before acceptance, cancel after this many new complete samples and reload the session.")
    parser.add_argument("--refine-digits", type=int, default=10,
                        help="Extra seed digits for forced independent recomputation (must be positive).")
    args = parser.parse_args()
    if args.refine_digits <= 0 or args.interrupt_after_samples < 0:
        parser.error("Refinement must be positive; interruption sample count must be nonnegative.")
    if args.directory.exists() and any(args.directory.iterdir()) and not args.resume:
        parser.error("Cold acceptance requires an empty directory; use --resume to reuse it.")
    args.directory.mkdir(parents=True, exist_ok=True)
    data = Path(__file__).parent / "data" / "gg_hg"
    session = CalculationSession(
        data / "native-model.json", args.directory,
        seed_digits=args.seed_digits, workers=args.workers,
    )
    report = {"status": "running", "mode": "resumed" if args.resume else "cold", "stages": []}

    def write_report():
        temporary = args.directory / "acceptance.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.directory / "acceptance.json")

    def run(stage, **kwargs):
        started = perf_counter_ns()
        session.submit(stage, **kwargs)
        value = session.wait()
        report["stages"].append({"stage": stage, "elapsed_ns": perf_counter_ns() - started, **kwargs})
        write_report()
        return value

    def samples():
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (args.directory / "completed-samples").glob("sample-*.bin")}

    try:
        if args.interrupt_after_samples:
            from symbolica.community.hep.integration import CalculationCancelled

            before = samples()
            session.submit("boundaries")
            while not session.snapshot()["done"]:
                persisted = samples()
                if len(persisted.keys() - before.keys()) >= args.interrupt_after_samples:
                    break
                sleep(1)
            else:
                session.wait()  # Preserve any typed failure instead of hiding it.
                raise AssertionError("Boundary generation finished before the requested interruption.")
            cancelled_at = perf_counter_ns()
            session.cancel()
            try:
                session.wait()
            except CalculationCancelled:
                pass
            else:
                raise AssertionError("Expected typed cancellation after complete sample checkpoint.")
            after = samples()
            assert all(after.get(name) == digest for name, digest in persisted.items())
            report["interruption"] = {
                "retained_samples": len(after),
                "cancellation_latency_ns": perf_counter_ns() - cancelled_at,
                "sample_payloads_preserved": True,
            }
            session.close()
            session = CalculationSession(
                data / "native-model.json", args.directory,
                seed_digits=args.seed_digits, workers=args.workers,
            )
            write_report()
        run("boundaries")
        run("transport")
        observables = run("amplitude")
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
            assert result.leading_power == 0
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
        }
        report["observables"] = {
            name: {"value": str(value), "absolute_error": str(observables.absolute_errors[name]),
                   "relative_digits": observables.verified_relative_digits[name]}
            for name, value in observables.values.items()
        }
        observable_reference = json.loads((data / "amplitude-validation.json").read_text())
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
        run("restart")
        warm = run("transport")
        assert all(r.cache_hit and r.steps == 0 for r in warm.values())

        # Independent precision refinement forces new numerical work while
        # allowing the exact reduction cache to survive. Transport also starts
        # afresh from the newly verified seeds, not old endpoints.
        original = {name: result for name, result in session.results.items()}
        session.seed_digits += args.refine_digits
        refined_seeds = run("boundaries", recompute=True)
        assert all(not result.cache_hit for _, result in refined_seeds)
        refined = run("transport", recompute=True)
        assert any(not r.cache_hit for r in refined.values())
        for label, result in refined.items():
            previous = original[label]
            for old_row, new_row in zip(previous.coefficients, result.coefficients, strict=True):
                for old, new in zip(old_row, new_row, strict=True):
                    assert abs(old - new) <= tolerance * max(one, abs(new)), label
        refined_observables = run("amplitude", recompute=True)
        for name, value in refined_observables.values.items():
            assert refined_observables.verified_relative_digits[name] >= 20
            assert abs(value - observables.values[name]) <= (
                refined_observables.absolute_errors[name] + observables.absolute_errors[name]
            ), name
        report["forced_precision_refinement"] = {
            "seed_digits": session.seed_digits, "numerical_caches_bypassed": True,
            "exact_reductions_reusable": True, "checked_mixed_digits": 20,
        }
        nearby = run("transport", nearby=True)
        assert any(not r.cache_hit for r in nearby.values())
        run("amplitude", recompute=True)
        report["nearby"] = {
            "point": [str(a) for a in session.point],
            "sources": {name: str(result.starting_coordinates) for name, result in nearby.items()},
            "inserted_points": sum(result.inserted_points for result in nearby.values()),
        }
        report["status"] = "passed"
    except BaseException as exc:
        session.cancel()
        report["status"] = "incomplete_or_failed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        session.close()
        report["timings"] = session.snapshot()["timings"]
        write_report()


if __name__ == "__main__":
    main()
