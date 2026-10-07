"""Cold one-worker AMFlow and DiffExp for the kite notebook's physical family.

Run under native Python or exec this file in Pyodide. No numerical or reduction
cache is loaded. Package installation and imports are outside the compute timer.
"""

import json
import os
import sys
from time import perf_counter

from symbolica import E, S
from symbolica.community import hepkit as hep
from symbolica.community.hep import integration as flow


def benchmark(
    *,
    massless=False,
    mass_ratio="1",
    digits=8,
    guard_digits=24,
    extra_digits=None,
    series_order=70,
    sampled_reduction=None,
    mass_mode=None,
    finite_sample=False,
    seed_rho=None,
    destinations=None,
):
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    if extra_digits is None:
        extra_digits = 5 if massless else 6
    if sampled_reduction is None:
        sampled_reduction = not massless
    started = perf_counter()
    k, ell, p, rho, D, epsilon = S(
        "kite_timing::k",
        "kite_timing::ell",
        "kite_timing::p",
        "kite_timing::rho",
        "kite_timing::D",
        "kite_timing::epsilon",
    )
    kin = hep.Kinematics(D, momenta=[k, ell, p]).with_scalar_product(p, p, -rho)
    kk, ll = kin.scalar_product(k, k), kin.scalar_product(ell, ell)
    kp, lp, kl = (kin.scalar_product(a, b) for a, b in [(k, p), (ell, p), (k, ell)])
    family = hep.IntegralFamily(
        [k, ell],
        [p],
        [
            kk - E("0" if massless else "1"),
            kk - 2 * kp - rho,
            ll - E("0" if massless else mass_ratio),
            ll - 2 * lp - rho,
            kk + ll - 2 * kl,
        ],
        kinematics=kin,
    )
    options = flow.EvaluationOptions(
        digits=digits,
        guard_digits=guard_digits,
        series_order=series_order,
        workers=1,
        mass_mode=mass_mode or ("branch" if massless else "mass"),
        reuse_samples=False,
        sampled_reduction=sampled_reduction,
    )
    evaluator = flow.IntegralEvaluator(options=options)
    prepared = evaluator.prepare(
        family,
        [[1] * 5],
        [rho],
        epsilon,
        branch_domain="positive Euclidean virtuality",
    )
    if seed_rho is None:
        seed_rho = "5/2"
    seed_point = {rho: E(seed_rho)}
    leading, last = prepared.required_master_range(seed_point, 0, 0)
    preparation_seconds = perf_counter() - started
    print(
        f"Prepared {len(prepared.basis)} masters in {preparation_seconds:.3f}s",
        flush=True,
    )
    if finite_sample:
        samples = evaluator.evaluate_samples(
            family, prepared.basis, seed_point, epsilon, [E("1/23000")]
        )
        report = {
            "runtime": sys.platform,
            "seconds": perf_counter() - started,
            "mass_mode": options.mass_mode,
            "values": [[str(v) for v in sample] for sample in samples],
        }
        print(json.dumps(report, indent=2), flush=True)
        return report
    cache = flow.BoundaryCache()
    boundary = prepared.generate_boundary(
        cache, seed_point, last=last, extra_digits=extra_digits
    )
    seed_seconds = perf_counter() - started
    print(f"Verified AMFlow seed in {seed_seconds:.3f}s", flush=True)
    (seed,) = prepared.project_targets(boundary, 0, 0, digits=digits)
    results = []
    if destinations is None:
        destinations = ["2", "5/2", "3"]
    elif isinstance(destinations, str):
        destinations = destinations.split(",")
    for point in destinations:
        destination_cache = flow.BoundaryCache()
        destination_cache.extend(cache)
        transported = prepared.transport(
            destination_cache,
            {rho: E(point)},
            leading,
            last,
            admit_straight_path=True,
        )
        (target,) = prepared.project_targets(transported, 0, 0, digits=digits)
        value = -target.coefficients[0]
        record = {
            "rho": point,
            "real": float(value.real),
            "imaginary": float(value.imag),
            "absolute_error": float(target.absolute_errors[0]),
        }
        if massless:
            reference = float((E("6") * E("3").zeta() / E(point)).evaluate({}).real)
            assert abs(record["real"] - reference) <= 5 * 10**-digits * abs(reference), record
        elif mass_ratio == "1" and point in {"2", "5/2", "3"}:
            reference = {"2": 1.5103465711245225, "5/2": 1.341320994395849, "3": 1.2081735973141193}[point]
            assert abs(record["real"] - reference) <= 5 * 10**-digits * abs(reference), record
        results.append(record)
    report = {
        "runtime": sys.platform,
        "workers": options.workers,
        "mass_mode": options.mass_mode,
        "cold": True,
        "mass_ratio": "0" if massless else mass_ratio,
        "series_order": series_order,
        "sampled_reduction": sampled_reduction,
        "digits": digits,
        "guard_digits": guard_digits,
        "seed_digits": boundary.verified_digits,
        "seed_rho": seed_rho,
        "masters": prepared.basis,
        "epsilon_range": [leading, last],
        "preparation_seconds": preparation_seconds,
        "seed_seconds": seed_seconds,
        "total_seconds": perf_counter() - started,
        "seed_value": float(-seed.coefficients[0].real),
        "results": results,
    }
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--massless", action="store_true")
    parser.add_argument("--digits", type=int, default=8)
    parser.add_argument("--guard-digits", type=int, default=24)
    parser.add_argument("--mass-ratio", default="1")
    parser.add_argument("--series-order", type=int, default=70)
    parser.add_argument("--symbolic-reduction", dest="sampled_reduction", action="store_false", default=None)
    parser.add_argument("--sampled-reduction", dest="sampled_reduction", action="store_true")
    parser.add_argument("--extra-digits", type=int)
    parser.add_argument("--mass-mode")
    parser.add_argument("--finite-sample", action="store_true")
    parser.add_argument("--seed-rho")
    parser.add_argument("--destinations")
    args = parser.parse_args()
    benchmark(**vars(args))
