"""Notebook presentation and caller scheduling; all science stays in HEPKit.

Native owners never cross threads. No estimator, graph renderer, CAS, or
integration rule is implemented here. Cancelling the caller stops scheduling;
it does not invent a native stopping reason or accept an unfinished package.
"""

from dataclasses import dataclass, field
from html import escape
import math
from time import perf_counter


EXAMPLES = {
    "Massive triangle": "triangle",
    "Massless box": "box",
    "Rank-two box numerator": "rank_two_box",
    "Coupled two-loop sunset": "sunset",
}


def prepare_input(builders, configuration):
    kind = configuration["example"]
    if kind == "triangle":
        return builders.massive_triangle(mass=configuration["mass"], s=configuration["s"])
    if kind == "box":
        return builders.massless_box(s12=configuration["s"], s23=configuration["t"])
    if kind == "rank_two_box":
        return builders.rank_two_box(s12=configuration["s"], s23=configuration["t"])
    if kind == "sunset":
        return builders.coupled_sunset(s=configuration["s"])
    raise ValueError("Unknown example")


def validate_configuration(value):
    if value is None:
        return None
    # marimo form validation receives raw frontend dropdown selections; the
    # submitted form.value is converted by the original UI elements afterwards.
    kind = value["example"]
    if isinstance(kind, list):
        kind = EXAMPLES.get(kind[0]) if kind else None
    if not math.isfinite(value["s"]) or value["s"] >= 0:
        return "Choose a finite negative s for this Euclidean example."
    if kind in {"box", "rank_two_box"} and (not math.isfinite(value["t"]) or value["t"] >= 0):
        return "Choose a finite negative t for the box."
    if kind == "triangle" and (not math.isfinite(value["mass"]) or value["mass"] <= 0):
        return "The triangle mass must be finite and positive."
    return None


def epsilon_label(order):
    return "ε" + str(order).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))


def vector_rows(estimate):
    if estimate is None:
        return []
    return [
        {"epsilon order": order, "component": component, "mean": mean,
         "standard error": error,
         "relative error": error / abs(mean) if mean != 0 else None}
        for order, component, mean, error in zip(
            estimate.orders, estimate.components, estimate.mean, estimate.standard_error, strict=True
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
    return [
        {"sector": sector.id, "dimension": sector.dimension,
         "accepted points": sector.completed_points, "planned points": sector.planned_points,
         "complete shifts": sector.complete_replicas, "planned shifts": sector.planned_replicas,
         "worker seconds": sector.worker_seconds}
        for sector in snapshot.sectors
    ]


def covariance_rows(estimate):
    if estimate is None:
        return []
    labels = [f"{epsilon_label(order)} {component}" for order, component in zip(estimate.orders, estimate.components, strict=True)]
    n = len(labels)
    return [{"component": labels[i], **dict(zip(labels, estimate.covariance_of_mean[i * n:(i + 1) * n], strict=True))} for i in range(n)]


def phase_rows(events):
    """Keep native event times; do not infer unobserved phase boundaries."""
    rows = []
    for event in events:
        if not rows or rows[-1]["phase"] != event.stage:
            rows.append({"phase": event.stage, "first event (s)": event.elapsed_seconds})
        rows[-1].update({"last event (s)": event.elapsed_seconds,
                         "completed": event.completed, "total": event.total,
                         "sectors": event.sectors, "kernels": event.kernels})
    return rows


def timing_rows(event):
    names = ["input", "parametrization", "domain", "geometry", "mapping", "symmetry", "subtraction", "laurent", "coefficient_expansion", "compilation", "total"]
    return [{"native phase": name, "seconds": getattr(event.timings, f"{name}_seconds")} for name in names]


@dataclass
class RunState:
    """One native run, explicitly advanced by its notebook caller."""

    kernels: object = None
    session: object = None
    snapshot: object = None
    configuration: dict | None = None
    events: list = field(default_factory=list)
    history: list = field(default_factory=list)
    active: bool = False
    active_started: float | None = None
    active_seconds: float = 0.0
    message: str = "Apply the inputs, then select Run."
    error: str | None = None
    checkpoint_bytes: bytes | None = None
    seen: dict = field(default_factory=lambda: {"run": 0, "cancel": 0, "resume": 0, "tick": ""})

    @property
    def integration_wall_seconds(self):
        return self.active_seconds + (perf_counter() - self.active_started if self.active_started is not None else 0.0)

    def stop_clock(self):
        self.active_seconds = self.integration_wall_seconds
        self.active_started = None

    def observe_generation(self, event, display=None):
        self.events.append(event)
        if display is not None:
            display(event)
        return True

    def start(self, fs, prepared, configuration, display=None):
        if self.active:
            self.message = "A run is active. Cancel it before starting another."
            return
        self.kernels = self.session = self.snapshot = None
        self.configuration = dict(configuration)
        self.events.clear()
        self.history.clear()
        self.checkpoint_bytes = None
        self.error = None
        self.active_seconds = 0.0
        self.active_started = None
        self.message = "Generating the submitted native integral…"
        try:
            integral = fs.Integral(**prepared.integral_arguments())
            observer = lambda event: self.observe_generation(event, display)
            generated = integral.generate(configuration["max_order"], observer=observer)
            self.kernels = generated.compile(observer=observer)
            settings = fs.QmcSettings(
                points=configuration["points"], shifts=configuration["shifts"],
                seed=configuration["seed"], package_points=configuration["package_points"],
                rule=configuration["rule"], periodization="korobov3",
            )
            self.session = self.kernels.session(settings)
            self.snapshot = self.session.snapshot()
            self.active = True
            self.active_started = perf_counter()
            self.message = "Integrating — one native package per refresh."
        except (Exception, KeyboardInterrupt) as error:
            self.fail(error)

    def fail(self, error):
        self.stop_clock()
        self.active = False
        stage = getattr(error, "stage", "caller")
        if isinstance(error, KeyboardInterrupt):
            self.error = None
            self.message = "Interrupted by the caller. Accepted integration coverage, if any, is saved."
        else:
            self.error = f"{type(error).__name__} [{stage}]: {error}"
            self.message = "Stopped; no numerical result has been substituted for this failure."
        if self.session is not None:
            self.snapshot = self.session.snapshot()
            self.checkpoint_bytes = self.session.checkpoint()

    def cancel(self):
        if self.session is None or not self.active:
            return
        self.stop_clock()
        self.active = False
        self.checkpoint_bytes = self.session.checkpoint()
        self.snapshot = self.session.snapshot()
        self.message = "Cancelled by the caller between packages. Accepted coverage is saved."

    def resume(self):
        if self.active or self.kernels is None or self.checkpoint_bytes is None:
            return
        try:
            self.session = self.kernels.restore(self.checkpoint_bytes)
            self.snapshot = self.session.snapshot()
            self.active = not self.session.complete
            self.active_started = perf_counter() if self.active else None
            self.error = None
            self.message = "Resumed from the native checkpoint." if self.active else "The saved allocation is already complete."
        except (Exception, KeyboardInterrupt) as error:
            self.fail(error)

    def advance(self):
        if not self.active:
            return
        try:
            self.snapshot = self.session.step(max_packages=1)
            if self.snapshot.estimate is not None:
                for row in highest_order_rows(self.snapshot.estimate):
                    self.history.append({"accepted points": self.snapshot.completed_points, **row})
            if self.session.complete:
                self.stop_clock()
                self.active = False
                self.checkpoint_bytes = self.session.checkpoint()
                self.message = "Planned allocation complete. Check the accuracy target separately."
        except (Exception, KeyboardInterrupt) as error:
            self.fail(error)


def table(mo, rows):
    if not rows:
        return mo.md("No native observations yet.")
    formats = {key: (lambda value: "—" if value is None else f"{value:.8g}")
               for key in rows[0] if any(isinstance(row.get(key), float) for row in rows)}
    if "relative error" in rows[0]:
        formats["relative error"] = lambda value: "—" if value is None else f"{100 * value:.4g}%"
    return mo.ui.table(rows, selection=None, show_column_summaries=False,
                       show_data_types=False, pagination=len(rows) > 12,
                       page_size=12, format_mapping=formats)


def generation_view(mo, state):
    if not state.events:
        return mo.md("The phase timeline appears after Run.")
    last = state.events[-1]
    summary = mo.md(f"**Generation {last.stage.replace('_', ' ')}** · {last.elapsed_seconds:.3f} s · {last.sectors} sectors · {last.kernels} kernels")
    if last.stage == "complete":
        timings = last.timings
        strip = mo.md(f"Geometry **{timings.geometry_seconds:.3f} s** · Mapping **{timings.mapping_seconds:.3f} s** · Laurent **{timings.laurent_seconds:.3f} s** · Compilation **{timings.compilation_seconds:.3f} s**")
        return mo.vstack([summary, strip, mo.accordion({
            "Generation event timeline": table(mo, phase_rows(state.events)),
            "All native phase timings": table(mo, timing_rows(last)),
        })])
    return mo.vstack([summary, table(mo, phase_rows(state.events))])


def history_plot(mo, history):
    """An SVG view of actual native means and one-standard-error intervals."""
    if not history:
        return mo.md("Waiting for a valid native estimate. History contains no provisional zeros.")
    chunks = []
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
        chunks.append(mo.Html(f'<svg viewBox="0 0 620 220" role="img" aria-label="{escape(title)}" style="width:100%;color:var(--text-primary)"><text x="76" y="18" fill="currentColor" font-size="13">{escape(title)}</text><text x="76" y="30" fill="currentColor" font-size="10">Vertical offset from {base:.17g}</text><path d="M76 32 V172 H580" fill="none" stroke="currentColor" opacity=".3"/><text x="4" y="42" fill="currentColor" font-size="10">{hi - base:.3g}</text><text x="4" y="172" fill="currentColor" font-size="10">{lo - base:.3g}</text><polyline points="{" ".join(path)}" fill="none" stroke="#6b46b0" opacity=".6"/>{"".join(marks)}<text x="76" y="194" fill="currentColor" font-size="11">{xmin:,}</text><text x="566" y="194" text-anchor="end" fill="currentColor" font-size="11">{xmax:,}</text><text x="320" y="213" text-anchor="middle" fill="currentColor" font-size="11">Accepted points, across all sectors and shifts</text></svg>'))
    return mo.vstack(chunks)


def result_view(mo, state):
    snapshot = state.snapshot
    if snapshot is None:
        return mo.md("The Laurent vector will appear when native coverage permits an estimate.")
    estimate = snapshot.estimate
    coverage = sector_rows(snapshot)
    complete = sum(row["complete shifts"] for row in coverage)
    planned = sum(row["planned shifts"] for row in coverage)
    top = max(state.kernels.orders)
    target = highest_order_target(estimate)
    diagnostic = snapshot.evaluation_diagnostics
    details = {"Sector coverage": table(mo, coverage), "Full covariance of the mean": table(mo, covariance_rows(estimate))}
    status_rows = [{"field": "backend", "value": state.kernels.backend},
                   {"field": "uncertainty", "value": snapshot.uncertainty},
                   {"field": "stop_reason", "value": snapshot.stop_reason}]
    status_rows += [{"field": key, "value": str(value)} for key, value in state.configuration.items()]
    details["Run settings and native status"] = table(mo, status_rows)
    uncertainty_label = {"available": "Available", "exact": "Exact", "waiting_for_coverage": "Waiting for complete shifts", "pilot_only": "Pilot only", "statistical_failure": "Statistical failure"}.get(snapshot.uncertainty, snapshot.uncertainty)
    stop_label = {None: "No native stop recorded", "planned_work_complete": "Allocation complete", "cancelled": "Cancelled", "target_reached": "Accuracy target reached", "work_limit": "Work limit", "time_limit": "Time limit", "numerical_failure": "Numerical failure"}.get(snapshot.stop_reason, snapshot.stop_reason)
    backend_label = {"native_o2": "Native O2", "portable_interpreted": "Portable interpreter"}.get(state.kernels.backend, state.kernels.backend)
    if diagnostic is not None:
        details["Native evaluation diagnostics"] = table(mo, [{name: getattr(diagnostic, name) for name in ("evaluations", "conditioning_checks", "rescues", "max_precision_bits", "failures", "weighted_checks", "additional_replays")}])
    return mo.vstack([
        mo.hstack([mo.stat(label="Accepted points", value=f"{snapshot.completed_points:,} / {snapshot.planned_points:,}"), mo.stat(label="Complete sector shifts", value=f"{complete:,} / {planned:,}"), mo.stat(label=f"{epsilon_label(top)} · relative error ≤ 0.1%", value=target)], widths="equal"),
        mo.md(f"**Integration wall time:** {state.integration_wall_seconds:.2f} s active · includes refresh waits, excludes caller-cancelled intervals."),
        mo.md(f"**Uncertainty:** {uncertainty_label} · **Native stop:** {stop_label} · **Backend:** {backend_label}"),
        mo.callout(snapshot.uncertainty_detail, kind="warn") if snapshot.uncertainty_detail else mo.md(""),
        table(mo, vector_rows(estimate)) if estimate is not None else mo.callout("The native session has no valid full-vector estimate yet. Missing uncertainty is not zero uncertainty.", kind="info"),
        mo.md(f"Native all-component target: **{'met' if estimate.meets(relative=0.001) else 'not met'}**. This is distinct from the selected highest-order target above.") if estimate is not None else mo.md(""),
        history_plot(mo, state.history),
        mo.accordion(details),
    ])
