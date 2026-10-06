"""Views of actual native generation events and timings."""

from .presentation import table

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

def generation_view(mo, state):
    if not state.events:
        return mo.md("The native phase timeline appears after Generate.")
    last = state.events[-1]
    progress = f"{last.completed:,} / {last.total:,}" if last.total is not None else f"{last.completed:,} observed"
    content = [
        mo.hstack([mo.stat(label=last.stage.replace("_", " ").title(), value=progress),
                   mo.stat(label="Native elapsed", value=f"{last.elapsed_seconds:.2f} s"),
                   mo.stat(label="Sectors / kernels", value=f"{last.sectors} / {last.kernels}")], widths="equal"),
        mo.md(last.detail),
    ]
    if last.total is not None and last.total > 0:
        content.append(mo.Html(f'<progress value="{last.completed}" max="{last.total}" style="width:100%;accent-color:#5b5bc4"></progress>'))
    detail = {"Observed phase timeline": table(mo, phase_rows(state.events)),
              "Native phase timings": table(mo, timing_rows(last))}
    coefficient = last.coefficient_expansion
    if coefficient is not None:
        requests = coefficient.requests
        detail["Coefficient expansion"] = table(mo, [{
            "sector": coefficient.sector, "stage": coefficient.stage,
            "requested method": coefficient.requested_method, "effective method": coefficient.effective_method,
            "attempt": coefficient.attempt, "relative width": coefficient.relative_width,
            "formal pieces": coefficient.formal_pieces,
            **{name: getattr(requests, name) for name in ("source_bodies", "unique_requests", "cached_partials", "aliases", "interleaved_requests", "fallback_requests")},
        }])
    if last.stage == "complete":
        t = last.timings
        content.append(mo.md(f"Geometry **{t.geometry_seconds:.3f} s** · Mapping **{t.mapping_seconds:.3f} s** · Symmetry **{t.symmetry_seconds:.3f} s** · Compilation **{t.compilation_seconds:.3f} s**"))
    content.append(mo.accordion(detail))
    return mo.vstack(content)


def input_view(mo, state):
    if state.prepared is None:
        return mo.md("The native diagram appears after Generate.")
    prepared = state.prepared
    details = {"Generated configuration": table(mo, [{"parameter": key, "value": str(value)} for key, value in state.configuration.items()]),
               "Input conventions": mo.md(
        "Native graph weights and the scalar numerator enter once, with measure "
        r"$\prod_\ell d^Dk_\ell/(i\pi^{D/2})$. No implicit Euler-gamma or scale factors are added."
    )}
    if state.input_events:
        details["HEPKit diagram-generation events"] = table(mo, [
            {"stage": e.stage, "completed": e.completed, "total": e.total}
            for e in state.input_events
        ])
    provenance = getattr(prepared, "provenance", None)
    if provenance:
        details["Input identity"] = mo.md(
            f"Live label **{provenance['actual_diagram_name']}**, audited Rust label "
            f"**{provenance['audited_diagram_name']}**; stable native ID `{prepared.raw_diagram.id}`. "
            "Only the cosmetic name differs; every physical serialized field is checked."
        )
    if state.numerator_view is not None:
        details["Native weighted scalar numerator"] = state.numerator_view
    return mo.vstack([
        mo.md(f"**{prepared.name}** · {prepared.diagram.loop_count} loop(s) · input preparation {state.preparation_seconds:.3f} s"),
        state.drawing if state.drawing is not None else mo.md("Native drawing has not been requested."),
        mo.accordion(details),
    ])
