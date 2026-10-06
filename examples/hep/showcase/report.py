"""Download native getter values without reconstructing estimates or expressions."""

import json


def _fields(owner, names):
    return {name: getattr(owner, name) for name in names.split()}


def run_report(state):
    """A presentation record; native kernel/checkpoint codecs remain separate."""
    result = {
        "schema": "fastsecdec-showcase-report-1",
        "configuration": state.configuration,
        "phase": state.phase,
        "active": state.active,
        "session_created": state.session is not None,
        "error": state.error,
        "checkpoint_warning": state.checkpoint_warning,
        "preparation_seconds": state.preparation_seconds,
        "completed_pilot_active_seconds": state.pilot_seconds,
        "integration_active_seconds": state.integration_wall_seconds,
        "time_semantics": "Caller active time includes refresh waits and excludes paused intervals; native worker time is separate.",
        "generation_events": [],
        "kernels": None,
        "generated": None,
        "snapshot": None,
        "history": state.history,
    }
    phases = "input parametrization domain geometry mapping symmetry subtraction laurent coefficient_expansion compilation total"
    for event in state.events:
        row = _fields(event, "stage completed total sectors kernels elapsed_seconds detail")
        row["timings"] = _fields(event.timings, " ".join(name + "_seconds" for name in phases.split()))
        coefficient = event.coefficient_expansion
        row["coefficient_expansion"] = None
        if coefficient is not None:
            row["coefficient_expansion"] = _fields(coefficient, "sector stage requested_method effective_method attempt relative_width formal_pieces")
            row["coefficient_expansion"]["requests"] = _fields(coefficient.requests, "source_bodies unique_requests cached_partials aliases interleaved_requests fallback_requests")
        result["generation_events"].append(row)
    if state.kernels is not None:
        result["kernels"] = _fields(state.kernels, "content_id backend orders components sector_count")
    if state.generated is not None:
        generated = state.generated
        result["generated"] = {"orders": generated.orders}
        if hasattr(generated, "metadata"):
            domain = generated.metadata.domain
            result["generated"]["domain"] = _fields(domain, "domain branch_policy caller_asserted relies_on_assertion")
            result["generated"]["domain"]["certificates"] = [_fields(f, "term_index factor_index certificate") for f in domain.factors]
            result["generated"]["sectors"] = [_fields(s, "index dimension coefficient_count alias_counts conditioning_basis cancellation_degree") for s in generated.sectors]
            result["generated"]["charts"] = [_fields(c, "source_index representative representative_permutation kernel_sector") for c in generated.metadata.charts]
    if state.snapshot is not None:
        snapshot = state.snapshot
        row = _fields(snapshot, "method stage completed_points planned_points complete_sectors worker_seconds uncertainty uncertainty_detail stop_reason stop_detail")
        row["sectors"] = []
        for sector in snapshot.sectors:
            sector_row = _fields(sector, "id dimension completed_points planned_points complete_replicas planned_replicas worker_seconds")
            allocation = getattr(sector, "discrete_allocation", None)
            sector_row["discrete_allocation"] = None if allocation is None else _fields(allocation, "probability points_per_batch")
            row["sectors"].append(sector_row)
        estimate = snapshot.estimate
        row["estimate"] = None if estimate is None else _fields(estimate, "orders components mean standard_error covariance_of_mean production_complete")
        diagnostics = snapshot.evaluation_diagnostics
        row["evaluation_diagnostics"] = None if diagnostics is None else _fields(diagnostics, "evaluations conditioning_checks rescues max_precision_bits failures weighted_checks additional_replays")
        result["snapshot"] = row
    return result


def report_bytes(state):
    return (json.dumps(run_report(state), indent=2, allow_nan=False) + "\n").encode("utf-8")
