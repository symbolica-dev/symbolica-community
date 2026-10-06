"""Full native Laurent-vector, covariance and accepted-coverage views."""

from html import escape
import math
from .presentation import table,epsilon_label

def vector_rows(estimate):
    if estimate is None:
        return []
    if any(len(values) != len(estimate.orders) for values in (estimate.components, estimate.mean, estimate.standard_error)):
        raise ValueError("Native Laurent-vector shape mismatch")
    return [
        {"epsilon order": order, "component": component, "mean": mean,
         "standard error": error,
         "relative error": error / abs(mean) if mean != 0 else None}
        for order, component, mean, error in zip(
            estimate.orders, estimate.components, estimate.mean, estimate.standard_error
        )
    ]

def highest_order_rows(estimate):
    rows = vector_rows(estimate)
    highest = max((row["epsilon order"] for row in rows), default=None)
    return [row for row in rows if row["epsilon order"] == highest]

def highest_order_target(estimate, relative=0.001):
    """Display-only selected-order tolerance; native estimates remain untouched."""
    if estimate is None or not estimate.production_complete:
        return "Not assessed — complete production is required"
    rows = highest_order_rows(estimate)
    if not rows:
        return "Unavailable"
    return "Met" if all(row["standard error"] <= relative * abs(row["mean"]) for row in rows) else "Not met"

def sector_rows(snapshot):
    discrete = snapshot.method == "havana_discrete_mc"
    rows = []
    for sector in snapshot.sectors:
        row = {"sector": sector.id, "dimension": sector.dimension,
               "accepted points": sector.completed_points,
               "planned points": "as allocated" if sector.planned_points is None else sector.planned_points,
               "global batches seen" if discrete else "complete shifts": sector.complete_replicas,
               "planned global batches" if discrete else "planned shifts": sector.planned_replicas,
               "worker seconds": sector.worker_seconds}
        allocation = getattr(sector, "discrete_allocation", None)
        if allocation is not None:
            row.update({"native selection probability": allocation.probability,
                        "global points per batch": allocation.points_per_batch})
        rows.append(row)
    return rows


def coverage_summary(snapshot):
    if snapshot.method == "havana_discrete_mc":
        # Each sector carries the same global batch count, not an independent replica.
        first = next(iter(snapshot.sectors), None)
        return "Complete global batches", (first.complete_replicas if first else 0), (first.planned_replicas if first else 0)
    return "Complete sector shifts", sum(s.complete_replicas for s in snapshot.sectors), sum(s.planned_replicas for s in snapshot.sectors)

def covariance_rows(estimate):
    if estimate is None:
        return []
    if len(estimate.components) != len(estimate.orders) or len(estimate.covariance_of_mean) != len(estimate.orders) ** 2:
        raise ValueError("Native covariance shape mismatch")
    labels = [f"{epsilon_label(order)} {component}" for order, component in zip(estimate.orders, estimate.components)]
    n = len(labels)
    return [{"component": labels[i], **dict(zip(labels, estimate.covariance_of_mean[i * n:(i + 1) * n]))} for i in range(n)]

def history_plot(mo, history):
    """An SVG view of actual native means and one-standard-error intervals."""
    if not history:
        return mo.md("Waiting for a valid native estimate. History contains no provisional zeros.")
    chunks = []
    recorded_zero = []
    for component in dict.fromkeys(row["component"] for row in history):
        rows = [row for row in history if row["component"] == component]
        rows = [row for row in rows if math.isfinite(row["mean"]) and math.isfinite(row["standard error"])]
        if not rows:
            continue
        base = rows[-1]["mean"]
        lo = min(row["mean"] - row["standard error"] for row in rows)
        hi = max(row["mean"] + row["standard error"] for row in rows)
        margin = (hi - lo) * 0.1 or max(abs(lo) * 0.01, 1e-12)
        lo, hi = lo - margin, hi + margin
        xmin, xmax = rows[0]["accepted points"], rows[-1]["accepted points"]
        x = lambda point: 76 + (point - xmin) / max(xmax - xmin, 1) * 490
        y = lambda value: 172 - (value - lo) / (hi - lo) * 135
        marks, path = [], []
        for row in rows:
            xx, yy = x(row["accepted points"]), y(row["mean"])
            path.append(f"{xx:.2f},{yy:.2f}")
            marks.append(f'<path d="M {xx:.2f} {y(row["mean"] - row["standard error"]):.2f} V {y(row["mean"] + row["standard error"]):.2f}" stroke="#9276ce"/><circle cx="{xx:.2f}" cy="{yy:.2f}" r="3" fill="#6b46b0"><title>{escape(str(row))}</title></circle>')
        title = f"{epsilon_label(rows[-1]['epsilon order'])} {component}: mean ± 1 standard error"
        plot = mo.Html(f'<svg viewBox="0 0 620 220" role="img" aria-label="{escape(title)}" style="width:100%;color:var(--text-primary)"><text x="76" y="18" fill="currentColor" font-size="13">{escape(title)}</text><text x="76" y="30" fill="currentColor" font-size="10">Vertical offset from {base:.17g}</text><path d="M76 32 V172 H580" fill="none" stroke="currentColor" opacity=".3"/><text x="4" y="42" fill="currentColor" font-size="10">{hi - base:.3g}</text><text x="4" y="172" fill="currentColor" font-size="10">{lo - base:.3g}</text><polyline points="{" ".join(path)}" fill="none" stroke="#6b46b0" opacity=".6"/>{"".join(marks)}<text x="76" y="194" fill="currentColor" font-size="11">{xmin:,}</text><text x="566" y="194" text-anchor="end" fill="currentColor" font-size="11">{xmax:,}</text><text x="320" y="213" text-anchor="middle" fill="currentColor" font-size="11">Accepted points in this phase</text></svg>')
        if all(row["mean"] == 0 and row["standard error"] == 0 for row in rows):
            recorded_zero.append(plot)
        else:
            chunks.append(plot)
    if recorded_zero:
        chunks.append(mo.accordion({"Recorded zero-valued component histories": mo.vstack([
            mo.md("Every displayed mean and standard error in these recorded histories is zero. This does not establish symbolic exactness; all components remain in the Laurent vector and covariance."),
            *recorded_zero,
        ])}))
    return mo.vstack(chunks)

def result_view(mo, state):
    snapshot = state.snapshot
    if snapshot is None:
        if state.kernels is not None:
            return mo.callout("Kernels are ready. No integration session exists and zero points have been sampled. Select Integrate when ready.", kind="success")
        return mo.md("The Laurent vector will appear when native coverage permits an estimate.")
    estimate = snapshot.estimate
    coverage = sector_rows(snapshot)
    coverage_label, complete, planned = coverage_summary(snapshot)
    worker_seconds = snapshot.worker_seconds
    top = max(state.kernels.orders)
    target = highest_order_target(estimate)
    diagnostic = snapshot.evaluation_diagnostics
    details = {"Sector coverage": table(mo, coverage), "Full covariance of the mean": table(mo, covariance_rows(estimate))}
    status_rows = [{"field": "backend", "value": state.kernels.backend},
                   {"field": "method", "value": snapshot.method},
                   {"field": "stage", "value": snapshot.stage},
                   {"field": "uncertainty", "value": snapshot.uncertainty},
                   {"field": "stop_reason", "value": snapshot.stop_reason}]
    if snapshot.stop_detail is not None:
        status_rows.append({"field": "stop_detail", "value": snapshot.stop_detail})
    status_rows += [{"field": key, "value": str(value)} for key, value in state.configuration.items()]
    details["Run settings and native status"] = table(mo, status_rows)
    uncertainty_label = {"available": "Available", "exact": "Exact", "waiting_for_coverage": "Waiting for native coverage", "pilot_only": "Pilot only", "statistical_failure": "Statistical failure"}.get(snapshot.uncertainty, snapshot.uncertainty)
    stop_label = {None: "No native stop recorded", "planned_work_complete": "Allocation complete", "cancelled": "Cancelled", "target_reached": "Accuracy target reached", "work_limit": "Work limit", "time_limit": "Time limit", "numerical_failure": "Numerical failure"}.get(snapshot.stop_reason, snapshot.stop_reason)
    backend_label = {"native_o2": "Native O2", "portable_interpreted": "Portable interpreter"}.get(state.kernels.backend, state.kernels.backend)
    if diagnostic is not None:
        details["Native evaluation diagnostics"] = table(mo, [{name: getattr(diagnostic, name) for name in ("evaluations", "conditioning_checks", "rescues", "max_precision_bits", "failures", "weighted_checks", "additional_replays")}])
    return mo.vstack([
        mo.hstack([mo.stat(label="Accepted points", value=f"{snapshot.completed_points:,} / {snapshot.planned_points:,}"), mo.stat(label=coverage_label, value=f"{complete:,} / {planned:,}"), mo.stat(label=f"{epsilon_label(top)} · relative error ≤ 0.1%", value=target)], widths="equal"),
        mo.callout("Pilot training only. After completion, adapt another epoch or Freeze production. Pilot pauses retain the same session in memory; downloadable checkpoints require frozen production.", kind="info") if snapshot.stage == "pilot" else mo.md(""),
        mo.md(f"**Active wall time:** {state.integration_wall_seconds:.2f} s · **Native worker time:** {worker_seconds:.2f} s. Active time includes refresh waits and excludes caller-cancelled intervals."),
        mo.md(f"**Uncertainty:** {uncertainty_label} · **Native stop:** {stop_label} · **Backend:** {backend_label}"),
        mo.callout(snapshot.uncertainty_detail, kind="warn") if snapshot.uncertainty_detail else mo.md(""),
        table(mo, vector_rows(estimate)) if estimate is not None else mo.callout("The native session has no valid full-vector estimate yet. Missing uncertainty is not zero uncertainty.", kind="info"),
        mo.md(f"Native all-component target: **{'met' if estimate.meets(relative=0.001) else 'not met'}**. This is distinct from the selected highest-order target above.") if estimate is not None and snapshot.stage == "production" else mo.md(""),
        history_plot(mo, state.history),
        mo.accordion(details),
    ])


def previous_result_view(mo, state):
    snapshot = state.previous_snapshot
    if snapshot is None:
        return mo.md("")
    configuration = state.previous_configuration
    return mo.accordion({"Previous allocation · retained report": mo.vstack([
        mo.md(f"**{configuration['example']}** · {snapshot.method} · {snapshot.stage} · {state.previous_phase}. This is the saved prior allocation, not the current session."),
        mo.md(f"Accepted points: **{snapshot.completed_points:,}**. Its full report is available below."),
        table(mo, vector_rows(snapshot.estimate)) if snapshot.estimate is not None else mo.md("That allocation had no valid native estimate."),
    ])})
