"""Independently check serialized Pyodide outputs using exact rational arithmetic.

These inequalities follow the existing full-profile acceptance checker archived
in RustFlow reports/performance/2026-10-06-physical-transport-compact-profile.
This checks results only; it implements no transport or amplitude algorithm.
"""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def number(value):
    if len(value) == 3:
        numerator, denominator, bits = value
        assert isinstance(numerator, str) and isinstance(denominator, str)
        result = Fraction(int(numerator), int(denominator))
        assert result.denominator == int(denominator)
        # Computed wasm32 Astro values retain requested precision metadata;
        # astro-float-num rounds its mantissa precision upward to 32-bit words.
        # Compare the complete dyadic value, never round it back to `bits`.
        # This is separate from the unchanged stricter supplied-input decoder.
        storage_bits = ((bits + 31) // 32) * 32
    else:
        assert len(value) == 2
        result, bits = Fraction(value[0]), value[1]
        storage_bits = bits  # The native reference was computed with MPFR.
    assert type(bits) is int and bits >= 2
    assert result.denominator & (result.denominator - 1) == 0
    magnitude = abs(result.numerator)
    if magnitude:
        trailing_zeros = (magnitude & -magnitude).bit_length() - 1
        assert magnitude.bit_length() - trailing_zeros <= storage_bits
    return result


def compare(value, error, expected, expected_error, *, relative=False):
    real, imaginary = map(number, value)
    reference_real, reference_imaginary = map(number, expected)
    error, reference_error = number(error), number(expected_error)
    assert error >= 0 and reference_error >= 0
    difference_squared = (real - reference_real) ** 2 + (imaginary - reference_imaginary) ** 2
    bound_squared = (error + reference_error) ** 2
    assert difference_squared <= bound_squared, "combined admitted errors"
    for r, i, e in ((real, imaginary, error), (reference_real, reference_imaginary, reference_error)):
        norm = r * r + i * i
        scale = norm if relative else max(Fraction(1), norm)
        assert scale > 0
        assert e * e <= scale / 10**40, "individual requested20 error"
        assert difference_squared <= scale / 10**40, "requested20 difference"
    return {"difference_squared": str(difference_squared), "combined_error_squared": str(bound_squared)}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(run_directory, previous_checkpoint=None):
    run = json.loads((run_directory / "run.json").read_text())
    assert run["schema"] == "gg-hg-actual-pyodide-run-v1" and run["status"] == "passed"
    assert run["available_parallelism"] == 1 and run["license_environment_supplied"] is False
    assert run["validation"]["schema"] == "supplied-loop-transport-runtime-v1"
    assert run["validation"]["supplied_loop_transport"] is True
    assert run["validation"]["wheel_sha256"] == run["identity"]["wheel_sha256"]
    sources = run_directory / "sources"
    for name, record in run["identity"]["sources"].items():
        path = sources / name
        assert path.stat().st_size == record["bytes"] and file_hash(path) == record["sha256"]
    reference_path = sources / "native-reference.json"
    assert file_hash(reference_path) == run["identity"]["native_reference_sha256"]
    reference = json.loads(reference_path.read_text())
    report = run["acceptance"]
    current = json.loads((run_directory / "checkpoint/CURRENT").read_text())
    checkpoint = run_directory / "checkpoint" / current["generation"]
    checkpoint_manifest = json.loads((checkpoint / "checkpoint.json").read_text())
    assert checkpoint_manifest["identity"] == run["identity"]
    for name, record in checkpoint_manifest["members"].items():
        path = checkpoint / name
        assert path.stat().st_size == record["bytes"] and file_hash(path) == record["sha256"]
    assert json.loads((checkpoint / "acceptance.json").read_text()) == report
    assert report["status"] == "passed"
    assert report["settings"] == {"digits": 20, "guard_digits": 20, "series_order": 16, "workers": 1}
    assert report["supplied_import"]["complex_coefficients"] == 4360
    assert report["supplied_import"]["all_values_errors_and_precisions_exact"] is True
    assert report["supplied_import"]["binary_restart_exact"] is True
    full = run["stage"] == "full"
    results = report["complete_transport_results"] if full else report["pilot_results"]
    expected = {case["label"]: case["boundary"] for case in reference["cases"]}
    assert len(expected) == 16
    if full:
        assert set(results) == set(expected)
    else:
        assert len(results) == 2 and sum("Planar_EW1" in label for label in results) == 1
    counts = {}
    for label, result in results.items():
        archived = expected[label]
        assert result["root_sheets"] == {key.rsplit("::", 1)[-1]: sign for key, sign in archived["root_germ"]}
        assert {key: Fraction(value.replace(" ", "")) for key, value in result["coordinates"].items()} == {
            key.rsplit("::", 1)[-1]: Fraction(value) for key, value in archived["coordinates"]
        }
        assert result["leading_power"] == archived["leading"] == 0 and archived["last"] == 4
        for sample in (result, archived):
            assert 20 <= sample["verified_digits"] <= sample["input_verified_digits"] <= 40
            assert len(sample["coefficients"]) == len(sample["comparison_errors"]) == 5
        count = 0
        for values, errors, old_values, old_errors in zip(result["coefficients"], result["comparison_errors"],
                                                         archived["coefficients"], archived["comparison_errors"], strict=True):
            for value, error, old_value, old_error in zip(values, errors, old_values, old_errors, strict=True):
                compare(value, error, old_value, old_error)
                count += 1
        assert count == (240 if "Planar_EW1" in label else 305)
        counts[label] = count
    assert sum(counts.values()) == (4360 if full else 545)
    resumed_comparison = None
    if full and "resumed_from" in run:
        prior = run["resumed_from"]
        previous_directory = (previous_checkpoint or Path(prior["directory"])) / prior["generation"]
        previous_manifest = json.loads((previous_directory / "checkpoint.json").read_text())
        assert previous_manifest["identity"] == run["identity"]
        previous_path = previous_directory / "acceptance.json"
        assert file_hash(previous_path) == previous_manifest["members"]["acceptance.json"]["sha256"]
        previous = json.loads(previous_path.read_text())
        previous_results = {case["label"]: case["boundary"] for case in previous["cases"]}
        previous_results.update(previous.get("pilot_results", {}))
        resumed_count = 0
        for label, original in previous_results.items():
            for key, value in original.items():
                if key not in ("provenance", "input_verified_digits"):
                    assert results[label][key] == value, (label, key, "fresh-process exact reuse")
            resumed_count += sum(map(len, original["coefficients"]))
        resumed_comparison = {"previous_report_sha256": file_hash(previous_path),
                              "complex_coefficients_exact": resumed_count,
                              "values_errors_precisions_and_achieved_evidence_exact": True}
    comparisons = {"form_factors": {}, "observables": {}}
    if full:
        assert report["all_observables_meet_requested20"] and report["binary_restart_and_warm_results_exact"]
        assert set(report["form_factors"]) == {"W", "Z"}
        for mass, result in report["form_factors"].items():
            archived = next(record for record in reference["form_factors"] if record["mass"] == mass)
            assert len(result["values"]) == len(archived["values"][0]) == 4
            assert all(d >= 20 for d in result["verified_relative_digits"] + archived["verified_relative_digits"])
            comparisons["form_factors"][mass] = [compare(value, error, old_value, old_error, relative=True)
                for value, error, old_value, old_error in zip(result["values"], result["absolute_errors"],
                    archived["values"][0], archived["absolute_errors"], strict=True)]
        assert set(report["observables"]) == {record["label"] for record in reference["observables"]}
        assert len(report["observables"]) == 3
        zero = ["0", "1", 100]
        for label, result in report["observables"].items():
            archived = next(record for record in reference["observables"] if record["label"] == label)
            assert result["verified_relative_digits"] >= 20 and archived["verified_relative_digits"] >= 20
            comparisons["observables"][label] = compare([result["value"], zero], result["absolute_error"],
                                                        [archived["value"], zero], archived["absolute_error"], relative=True)
        assert report["archived_observable_comparison"]["electroweak_squared"]["reference_conditional_relative_digits"] == 19
    return {"schema": "gg-hg-exact-pyodide-comparison-v1", "status": "passed", "scope": report["passed_scope"],
            "checker_sha256": file_hash(Path(__file__)),
            "run_sha256": file_hash(run_directory / "run.json"), "identity": run["identity"],
            "coefficient_counts": counts, "complex_coefficients": sum(counts.values()),
            "fresh_process_resume": resumed_comparison,
            "all_differences_within_combined_errors_and_requested20": True,
            "arithmetic": "Exact Python integers and Fraction; no floating-point comparison rounding",
            "representation": "Computed wasm32 Astro mantissas may use ceil(requested_precision/32)*32 bits; requested precision metadata and complete dyadic values are preserved. MPFR references retain their exact bit limit. Supplied415-bit import remains unchanged.",
            **comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--previous-checkpoint", type=Path,
                        help="Relocated previous checkpoint directory; identity and report hashes must still match")
    args = parser.parse_args()
    report = verify(args.run_directory, args.previous_checkpoint)
    with args.output.open("x") as destination:
        json.dump(report, destination, indent=2)
        destination.write("\n")
    print(f"Exact comparison passed: {report['complex_coefficients']} complex coefficients, "
          f"{sum(map(len, report['form_factors'].values()))} form factors, {len(report['observables'])} observables")


if __name__ == "__main__":
    main()
