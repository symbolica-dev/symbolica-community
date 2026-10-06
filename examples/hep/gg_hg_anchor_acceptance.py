"""Independent Euclidean anchors through the installed native loop evaluator.

Run with a release-built community extension, for example:
  python gg_hg_anchor_acceptance.py --directory /path/to/new/anchors --workers 16
Resume with --resume; --force bypasses numerical reuse but keeps exact reductions.
Both native calculations finish before either comparison fixture is opened.
Use one process per cache directory. Completed samples and verified banks are separate.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter_ns, time_ns
from typing import Any, TypedDict

from gg_hg_acceptance import file_attestation, runtime_attestation, verify_runtime_attestation


SCHEMA = "native-gg-hg-euclidean-anchors-v1"


class AnchorSpecification(TypedDict):
    family: str
    coordinates: list[str]
    dimension: int
    roots: int
    reference_sha256: str


# Mathematical inputs are independent of all numerical comparison files.
ANCHORS: dict[str, AnchorSpecification] = {
    "planar": {"family": "Planar_EW1", "coordinates": ["-1/10", "-1/25", "-1/50"],
               "dimension": 48, "roots": 2,
               "reference_sha256": "c575eecd914b04052d4905ac929a227f716009eccc1bb5b72a4d2f9149c38a07"},
    "nonplanar": {"family": "NP_EW1", "coordinates": ["-1/10", "-1/5", "-1"],
                  "dimension": 61, "roots": 8,
                  "reference_sha256": "c5e62b3a72e5a2d09988c4bc85f476b48956731d819e1adaad7d784d1172c94a"},
}
DATA = Path(__file__).parent / "data" / "gg_hg" / "anchors"


def native_point(system, topology):
    from symbolica import E

    specification = ANCHORS[topology]
    if system.dimension != specification["dimension"] or len(system.roots) != specification["roots"]:
        raise ValueError("Native canonical basis does not match the Euclidean anchor contract.")
    return (
        dict(zip(system.coordinates, map(E, specification["coordinates"]), strict=True)),
        {root: 1 for root in system.roots},
    )


def check_native_evidence(result, system, topology, digits):
    point, sheets = native_point(system, topology)
    if (result.leading_power != 0 or result.coordinates != point or result.root_sheets != sheets
            or result.verified_digits is None or result.verified_digits < digits
            or "no numerical reference seed" not in (result.provenance or "")):
        raise ValueError("Native anchor lacks the required point, branch, or independent accuracy evidence.")
    check_coefficient_evidence(result.coefficients, result.comparison_errors, system.dimension, digits)


def check_coefficient_evidence(coefficients, comparison_errors, dimension, digits):
    """Apply the same native-number evidence checks to a result or its saved proof."""
    from symbolica import Float

    if len(coefficients) != 5 or len(comparison_errors) != 5:
        raise ValueError("Anchor evidence must cover epsilon powers zero through four.")
    one = Float("1", decimal_digits=max(100, digits + 40))
    tolerance = Float(f"1e-{digits}", decimal_digits=max(100, digits + 40))
    for values, errors in zip(coefficients, comparison_errors, strict=True):
        if len(values) != dimension or len(errors) != dimension:
            raise ValueError("Anchor evidence has an incorrect canonical dimension.")
        for value, error in zip(values, errors, strict=True):
            if (not value.is_finite() or not error.is_finite() or error < 0
                    or error > tolerance * max(one, abs(value))):
                raise ValueError("Native anchor uncertainty exceeds the requested mixed accuracy budget.")


def compare_anchor(result, system, topology, *, digits, reference_directory=DATA):
    """Comparison only: never called until both native generations have succeeded."""
    from symbolica import ComplexFloat, E, Float

    check_native_evidence(result, system, topology, digits)
    specification = ANCHORS[topology]
    path = reference_directory / f"{topology}-regenerated-anchor.json"
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != specification["reference_sha256"]:
        raise ValueError("Euclidean comparison fixture differs from its recorded source hash.")
    record = json.loads(payload)
    coordinates = record.get("coordinates", record.get("variables"))
    epsilon_range = record.get("epsilon_range", [record.get("leading_epsilon_power"),
                                                  record.get("last_epsilon_power")])
    reference_digits = record.get("verified_digits", record.get("evidence_digits"))
    values = record.get("coefficients", record.get("coefficients_by_epsilon"))
    errors = record.get("absolute_errors", record.get("source_absolute_errors_by_epsilon"))
    if (record["family"] != specification["family"] or epsilon_range != [0, 4]
            or reference_digits != 40 or record["basis_order"] != list(range(1, system.dimension + 1))
            or set(coordinates) != {"s", "t", "b"}
            or [E(coordinates[name]) for name in ("s", "t", "b")]
            != [E(value) for value in specification["coordinates"]]
            or record["root_germs"] != {f"root{i + 1}": "principal" for i in range(specification["roots"])}
            or "no additional gamma" not in record["normalization"]
            or len(values) != 5 or len(errors) != 5):
        raise ValueError("Incompatible Euclidean comparison convention.")
    precision = max(100, digits + 40)
    maximum = Float("0", decimal_digits=precision)
    failures = []
    checked = 0
    for power, (native, native_errors, expected, expected_errors) in enumerate(zip(
        result.coefficients, result.comparison_errors, values, errors, strict=True,
    )):
        if len(expected) != system.dimension or len(expected_errors) != system.dimension:
            raise ValueError("Incomplete Euclidean comparison coefficients.")
        for index, (value, error, item, allowance) in enumerate(zip(
            native, native_errors, expected, expected_errors, strict=True,
        ), start=1):
            reference_value = ComplexFloat(item["real"], item["imaginary"], decimal_digits=precision)
            reference_error = Float(allowance, decimal_digits=precision)
            if not reference_value.is_finite() or not reference_error.is_finite() or reference_error < 0:
                raise ValueError("Invalid Euclidean comparison value or allowance.")
            difference = abs(value - reference_value)
            maximum = max(maximum, difference)
            if difference > error + reference_error:
                failures.append({"epsilon_power": power, "canonical_index": index,
                                 "absolute_difference": str(difference),
                                 "combined_allowance": str(error + reference_error)})
            checked += 1
    return {
        "passed": not failures, "checked_coefficients": checked,
        "reference_accuracy_cap_digits": reference_digits,
        "reference_loaded_after_both_native_successes": True,
        "maximum_absolute_difference": str(maximum), "failures": failures,
        "metric": "absolute complex difference <= native error + recorded reference error",
        "reference": {"file": path.name, "sha256": digest,
                      "provenance": record.get("provenance", record.get("evidence")),
                      "normalization": record["normalization"],
                      **{key: record[key] for key in ("report_path", "report_sha256", "system_source",
                                                     "source_fixture_sha256", "system_path",
                                                     "system_sha256", "working_digits", "metric")
                         if key in record}},
    }


def write_report(directory, run_path, report):
    text = json.dumps(report, indent=2) + "\n"
    for path in (run_path, directory / "anchor-acceptance.json"):
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(text)
        temporary.replace(path)


def run(args):
    from symbolica.community.hep.integration import (
        BoundaryCache, ComputationControl, EvaluationOptions,
        HiggsJetIntegralSystem, IntegralEvaluator,
    )

    started = perf_counter_ns()
    args.directory.mkdir(parents=True, exist_ok=True)
    run_path = args.directory / "runs" / f"{time_ns()}-{os.getpid()}.json"
    run_path.parent.mkdir(exist_ok=True)
    systems = {name: HiggsJetIntegralSystem(name) for name in ANCHORS}
    report: dict[str, Any] = {
        "schema": SCHEMA, "status": "running", "native": {}, "comparisons": {},
        "runtime_attestation": runtime_attestation(), "anchor_runner": file_attestation(__file__),
        "options": {name: getattr(args, name) for name in (
            "digits", "guard_digits", "order", "workers", "max_steps", "max_precision_attempts",
            "case_batch", "force", "resume",
        )},
        "references_loaded": False,
        "accuracy_metric": "mixed max(1,abs(value)); native empirical precision/sample refinement, not interval bounds",
    }
    control = ComputationControl()
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="euclidean-anchors")
    results = {}

    def save():
        report["elapsed_ns"] = perf_counter_ns() - started
        write_report(args.directory, run_path, report)

    save()
    try:
        pending = args.directory / "force-preparation-pending.json"
        if pending.exists() and not args.force:
            raise RuntimeError("Forced cache preparation was interrupted; rerun with --force before --resume.")
        if args.force:
            # Persist invalidation of both banks before any expensive work. A
            # cancelled first family must not restore an old second-family bank
            # or rebuild it from pre-force samples. Keep prior payloads as history.
            temporary = pending.with_suffix(".json.tmp")
            temporary.write_text(json.dumps({"run_report": str(run_path)}) + "\n")
            temporary.replace(pending)
            report["archived_numerical_directories"] = {}
            for topology in systems:
                for name in ("boundaries", "completed-samples"):
                    previous = args.directory / topology / name
                    if previous.exists():
                        archived = run_path.parent / f"{run_path.stem}-prior-numerical" / topology / name
                        archived.parent.mkdir(parents=True, exist_ok=True)
                        previous.rename(archived)
                        report["archived_numerical_directories"][f"{topology}/{name}"] = str(archived)
                BoundaryCache().save(args.directory / topology / "boundaries")
            save()
            pending.unlink()
        for topology, system in systems.items():
            point, sheets = native_point(system, topology)
            family_directory = args.directory / topology
            bank_path = family_directory / "boundaries"
            if not (bank_path / "physical-boundaries.bin").exists():
                bank = BoundaryCache()
            else:
                bank = BoundaryCache.load(bank_path)
            options = EvaluationOptions(
                digits=args.digits, guard_digits=args.guard_digits, series_order=args.order,
                workers=args.workers, max_steps=args.max_steps,
                max_precision_attempts=args.max_precision_attempts,
                cache_directory=family_directory / "exact-reductions",
                sample_cache_directory=family_directory / "completed-samples",
                reuse_samples=not args.force,
            )
            evaluator = IntegralEvaluator(options=options, reduction_batch_size=args.case_batch)
            family_started = perf_counter_ns()
            if args.cancel_file is not None and args.cancel_file.exists():
                control.cancel()
            future = pool.submit(system.generate_boundary, evaluator, bank, point, sheets,
                                 last=4, recompute=args.force, control=control)
            while True:
                if args.cancel_file is not None and args.cancel_file.exists():
                    control.cancel()
                report["progress"] = {
                    "topology": topology, "elapsed_ns": perf_counter_ns() - family_started,
                    "events": control.poll(),
                    "completed_sample_files": sum(1 for _ in
                        (family_directory / "completed-samples").glob("sample-*.bin")),
                }
                save()
                try:
                    result = future.result(timeout=5)
                    break
                except TimeoutError:
                    if future.done():
                        result = future.result()  # Preserve a native TimeoutError.
                        break
            check_native_evidence(result, system, topology, args.digits)
            bank.save(bank_path)
            results[topology] = result
            report["native"][topology] = {
                "identity": result.identity, "coordinates": ANCHORS[topology]["coordinates"],
                "root_sheets": "principal", "dimension": system.dimension,
                "leading_power": result.leading_power, "last_power": 4,
                "verified_digits": result.verified_digits, "working_bits": result.working_bits,
                "provenance": result.provenance, "system_provenance": system.provenance,
                "cache_hit": result.cache_hit, "elapsed_ns": perf_counter_ns() - family_started,
                "coefficients": [[{"real": str(v.real), "imaginary": str(v.imag)} for v in row]
                                 for row in result.coefficients],
                "comparison_errors": [[str(v) for v in row] for row in result.comparison_errors],
                "uncertainties_within_requested_budget": True,
            }
            save()
        # Numerical references are opened only after BOTH complete native calls.
        for topology, system in systems.items():
            report["comparisons"][topology] = compare_anchor(
                results[topology], system, topology, digits=args.digits,
            )
            report["references_loaded"] = True
            save()
        if not all(value["passed"] for value in report["comparisons"].values()):
            raise AssertionError("Native Euclidean anchors disagree with recorded comparison allowances.")
        verify_runtime_attestation(report["runtime_attestation"])
        if file_attestation(__file__) != report["anchor_runner"]:
            raise RuntimeError("The anchor runner changed during this run.")
        report["runtime_attestation_verified_at_completion"] = True
        report["checked_coefficients"] = 545
        report["status"] = "passed"
    except BaseException as exc:
        control.cancel()
        report["status"] = "incomplete_or_failed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        save()
        raise
    finally:
        pool.shutdown(wait=True)
        save()
    return report


def main():
    if not __debug__:
        raise RuntimeError("Acceptance requires enabled assertions; remove -O or PYTHONOPTIMIZE.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", "--cache-directory", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--digits", type=int, default=20)
    parser.add_argument("--guard-digits", type=int, default=40)
    parser.add_argument("--order", type=int, default=80)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-precision-attempts", type=int, default=3)
    parser.add_argument("--case-batch", type=int, default=16)
    parser.add_argument("--cancel-file", type=Path)
    args = parser.parse_args()
    if args.digits < 20 or min(args.guard_digits, args.order, args.workers, args.max_steps,
                              args.max_precision_attempts, args.case_batch) < 1:
        parser.error("At least 20 requested digits and positive resource/precision budgets are required.")
    if args.directory.exists() and any(args.directory.iterdir()) and not (args.resume or args.force):
        parser.error("Cold acceptance requires an empty directory; use --resume or --force.")
    run(args)


if __name__ == "__main__":
    main()
