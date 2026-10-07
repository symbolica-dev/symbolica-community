import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="Four-loop vacuum IBP laboratory")

with app.setup(hide_code=True):
    import marimo as mo


@app.cell(hide_code=True)
async def _():
    import hashlib as _hashlib
    import sys as _sys
    from pathlib import Path as _Path

    if _sys.platform == "emscripten":
        import micropip as _micropip
        from pyodide.http import pyfetch as _pyfetch

        _base = mo.notebook_location()
        _response = await _pyfetch(str(_base / "rustred-assets.json"))
        if _response.status != 200:
            raise RuntimeError("Export this notebook with scripts/export_rustred_wasm.py to include its WASM wheel and graph inputs.")
        _manifest = await _response.json()
        if _manifest["schema"] != "rustred-browser-assets-v1":
            raise ValueError("Unsupported browser asset manifest")
        _directory = _Path.cwd() / "rustred_notebook_inputs"
        _files = dict(_manifest["files"])
        _wheel = _manifest["wheel"]
        if _Path(_wheel).name != _wheel or not _wheel.endswith(".whl"):
            raise ValueError("Invalid browser wheel path")
        _files[_wheel] = _manifest["wheel_sha256"]
        for _name, _digest in _files.items():
            _relative = _Path(_name)
            if _relative.is_absolute() or ".." in _relative.parts:
                raise ValueError("Invalid browser asset path")
            _response = await _pyfetch(str(_base / _name))
            if _response.status != 200:
                raise RuntimeError(f"Cannot load browser input {_name}: HTTP {_response.status}")
            _payload = await _response.bytes()
            if _hashlib.sha256(_payload).hexdigest() != _digest:
                raise ValueError(f"Browser input checksum mismatch: {_name}")
            _path = _directory / _relative
            _path.parent.mkdir(parents=True, exist_ok=True)
            _path.write_bytes(_payload)
        del _payload
        await _micropip.install("emfs:" + str(_directory / _wheel))
        _sys.path.insert(0, str(_directory))

    from symbolica import E, N, S
    from symbolica.community import hepkit as hep
    rustred = getattr(hep, "rustred", None)

    from rustred_campaign_support import (
        FAMILY_NAMES,
        FourLoopCampaign,
        dot_sources,
        preferred_auxiliaries,
        summary_rows,
        parameter_rows,
        rule_coefficient_ids,
        rule_summary_rows,
        rule_view,
        terminal_rows,
        coefficient_view,
        normalization_summary_rows,
        normalization_relation_view,
        ExplicitNumeratorEvaluation,
    )

    return (
        E,
        ExplicitNumeratorEvaluation,
        FAMILY_NAMES,
        FourLoopCampaign,
        N,
        S,
        coefficient_view,
        dot_sources,
        hep,
        normalization_relation_view,
        normalization_summary_rows,
        parameter_rows,
        preferred_auxiliaries,
        rule_coefficient_ids,
        rule_summary_rows,
        rule_view,
        rustred,
        summary_rows,
        terminal_rows,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Four-loop vacuum IBP laboratory

    [Browse notebooks](/) · [Integral families](/?file=hep/integral_families.py)

    **DOT graph → routed family → native IBP search → inspectable artifact**

    Generate candidates for the **H, X, BMW and FG** unit-mass vacuum families,
    one after another, in symbolic dimension $d$. Every progress update and
    every candidate rule in the explorer comes from the live RustRed engine in
    HEPKit's shared Symbolica kernel. Candidate generation does not load a
    precomputed catalog. The separate, optional Vakint evaluation at the end
    explicitly uses its shipped reduction assets and numerical master inputs.

    This campaign generates candidate recurrences; it does **not** traverse a
    reduction graph or reduce a benchmark set of integrals. Finishing the
    bounded search is **not** a proof of full-family closure or a minimal
    master basis.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Setup and notebook helpers

    The folded setup block imports HEPKit and the campaign helpers. Expand its
    code to inspect the imports; the calculation starts with the graph below.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 1. Start with the graph

    The four packaged DOT files specify the physical edges and a loop-momentum
    basis. HEPKit supplies the routing, inverse propagators and independent
    completion. Expand the setup cell to see the imports; the longer helper
    module only manages the live UI and session lifecycle, not the algebra.
    """)
    return


@app.cell
def _(S, dot_sources, hep):
    dimension = S("d")
    scalar_model = hep.Model.phi_3_4()
    graph_inputs = dot_sources()
    diagrams = {
        name: hep.FeynmanDiagram.from_dot(scalar_model, dot)
        for name, dot in graph_inputs.items()
    }
    return diagrams, dimension, graph_inputs, scalar_model


@app.cell
def _(E, diagrams, dimension, hep, preferred_auxiliaries, scalar_model):
    families = {}
    auxiliary_indices = {}
    for _name, _diagram in diagrams.items():
        _physical = _diagram.propagator_family(kinematics=hep.Kinematics(dimension))
        _completed = _diagram.integral_family(
            independent_dot_products=preferred_auxiliaries(_name, _physical),
            kinematics=_physical.kinematics,
        )
        # Model parameter values do not replace symbolic masses automatically.
        _mass = scalar_model.particle("phi").mass
        _unit_mass = [den.replace(_mass, E("1")) for den in _completed.denominators]
        families[_name] = hep.IntegralFamily(
            _completed.loop_momenta, _completed.external_momenta, _unit_mass,
            kinematics=_completed.kinematics,
        )
        assert families[_name].is_complete and families[_name].is_independent
        auxiliary_indices[_name] = list(range(len(_physical.denominators), len(_unit_mass)))
    ibp_families = {name: hep.IBPFamily(family, name=name) for name, family in families.items()}
    return auxiliary_indices, families, ibp_families


@app.cell(hide_code=True)
def _(FAMILY_NAMES):
    graph_choice = mo.ui.dropdown(
        options=list(FAMILY_NAMES), value="H", label="Vacuum topology",
        allow_select_none=False,
    )
    graph_choice
    return (graph_choice,)


@app.cell(hide_code=True)
def _(
    auxiliary_indices,
    diagrams,
    families,
    graph_choice,
    graph_inputs,
    ibp_families,
    parameter_rows,
):
    _name = graph_choice.value
    _family = families[_name]
    _auxiliary_count = len(auxiliary_indices[_name])
    mo.vstack([
        diagrams[_name],
        mo.md(
            f"**{_name}:** four loops · {len(_family.denominators) - len(auxiliary_indices[_name])} physical "
            f"propagators · {_auxiliary_count} auxiliary "
            f"{'coordinate' if _auxiliary_count == 1 else 'coordinates'} · $m=1$"
        ),
        mo.accordion({
            "DOT input": mo.md(f"```dot\n{graph_inputs[_name]}\n```"),
            "Routed complete family": _family,
            "Native parameter legend": mo.ui.table(
                parameter_rows(ibp_families[_name].parameter_bindings),
                selection=None, show_column_summaries=False, show_download=False,
            ) if hasattr(ibp_families[_name], "parameter_bindings") else mo.md(
                "The parameter legend is available with the native session API."
            ),
        }),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Run the native search

    The default is **one native worker**, with the full positive-sector
    downset for each physical parent—not one preselected easy sector.
    Auxiliary ISP powers remain nonpositive; the per-sector numerical search
    depth is explicitly set to **2**, with the native exact sparse backend.
    Native hosts return from the start button immediately and stream progress.
    Pyodide runs the explicit queue synchronously with one worker; the dashboard
    receives completed evidence when the calculation returns.

    A real four-loop campaign can take substantial time; no runtime estimate
    is assumed here. Browser memory limits may prevent large four-loop searches;
    browser acceptance covers the smaller three-loop example. On native hosts,
    **Cancel** requests a stop at native safe points; an
    in-flight sector may need time to drain. Keep refresh enabled to advance
    from one completed family to the next. Pausing refresh does not pause
    native work.
    """)
    return


@app.cell
def _(FourLoopCampaign, auxiliary_indices, ibp_families, rustred):
    generation_options = {
        "n_cores": 1,
        "exact_backend": "sparse",
        "numerical_depth": 2,
        "event_capacity": 256,
        # Serialization/inspection allowance, not a larger algebra search.
        "bundle_max_entries": 10_000_000,
    }
    _capabilities = rustred.execution_capabilities() if (
        rustred is not None and hasattr(rustred, "execution_capabilities")
    ) else None
    campaign = FourLoopCampaign(ibp_families, auxiliary_indices, capabilities=_capabilities)
    native_generation_available = rustred is not None and all(
        hasattr(family, "start_generation") for family in ibp_families.values()
    )
    return campaign, generation_options, native_generation_available


@app.cell(hide_code=True)
def _(campaign, generation_options, native_generation_available):
    start_generation = mo.ui.button(
        label="Generate H → X → BMW → FG",
        on_click=lambda value: campaign.start(**generation_options),
        kind="success", disabled=not native_generation_available,
    )
    cancel_generation = mo.ui.button(
        label="Cancel after safe point", on_click=lambda value: campaign.cancel(),
        disabled=not native_generation_available or not campaign.capabilities["cancellation_in_flight"],
    )
    heartbeat = (mo.ui.refresh(options=["1s", "3s", "10s"], default_interval="1s")
                 if campaign.capabilities["live_event_polling"] else None)
    _buttons = ([start_generation, cancel_generation, heartbeat]
                if heartbeat is not None else [start_generation])
    _controls = [mo.hstack(
        _buttons, justify="start", wrap=True
    )]
    if not campaign.capabilities["background_sessions"]:
        _controls.append(mo.callout(
            "Single-thread browser execution: this button runs the queue synchronously. "
            "Live progress and in-flight cancellation are unavailable. Large four-loop "
            "searches may exceed browser memory; start with the three-loop example.",
            kind="info",
        ))
    if not native_generation_available:
        _controls.insert(0, mo.callout(
            "This installed HEPKit host does not include the native generation "
            "session API yet. Graph setup is available; generation is disabled. "
            "Install a community build with RustRed session support to run it.",
            kind="warn",
        ))
    mo.vstack(_controls)
    return cancel_generation, heartbeat, start_generation


@app.cell(hide_code=True)
def _(campaign, cancel_generation, heartbeat, start_generation, summary_rows):
    _ = heartbeat.value if heartbeat is not None else None, start_generation.value, cancel_generation.value
    live_snapshot = campaign.poll()
    _state = live_snapshot["state"]
    _elapsed = live_snapshot["elapsed_seconds"]
    _panels = [
        mo.hstack([
            mo.stat(_state.replace("_", " ").title(), label="Campaign state"),
            mo.stat(f"{live_snapshot['completed_families']} / 4", label="Artifacts completed"),
            mo.stat(f"{_elapsed:,.1f} s", label="Observed wall time"),
        ], widths="equal", gap=1),
        mo.ui.table(summary_rows(live_snapshot), selection=None, pagination=False,
                    show_column_summaries=False, show_download=False),
        mo.md(f"Artifact directory: `{live_snapshot['output_directory']}`"),
    ]
    if _state == "cancelling":
        _panels.append(mo.callout("Cancellation requested; native work has not yet drained.", kind="warn"))
    if live_snapshot["last_error"]:
        _panels.append(mo.callout(live_snapshot["last_error"], kind="danger"))
    if live_snapshot["evidence_error"]:
        _panels.append(mo.callout("Evidence logging failed: " + live_snapshot["evidence_error"], kind="warn"))
    mo.vstack(_panels)
    return (live_snapshot,)


@app.cell(hide_code=True)
def _(live_snapshot):
    mo.accordion({
        "Active native jobs": mo.json([
            {"family": row["family"], "jobs": row.get("active_jobs", []),
             "details_truncated": row.get("details_truncated", False)}
            for row in live_snapshot["families"] if row.get("active_jobs")
        ]),
        "Recent native events · bounded history": mo.ui.table(
            live_snapshot["events"][-20:], selection=None, pagination=False,
            show_column_summaries=False, show_download=False,
        ),
        "Coalesced or dropped native events": mo.json({
            row["family"]: row.get("dropped_events", 0)
            for row in live_snapshot["families"]
        }),
        "What the counters mean": mo.md("""
            Counts come from the native aggregate snapshot, not from the number
            of displayed events. Sectors means **generated / planned**; residuals
            means finite residual records. The event buffer is bounded: slow viewers may
            miss intermediate updates, reported as coalesced/dropped events.
            Sector progress is not a linear estimate of remaining algebra time.

            Wall time includes native preparation, generation and artifact
            assembly plus UI polling gaps. No traversal or cold closure check
            is performed by this notebook. Finite residual records are not
            automatically independent master integrals.
        """),
    })
    return


@app.cell(hide_code=True)
def _(FAMILY_NAMES):
    _intro = mo.md("""
    ## 3. Explore without loading every expression

    Open any **completed** family while later ones are still running. The
    explorer has its own controls: progress refreshes do not reset your
    selection. It requests a single page of sector/rule metadata, then the
    selected rule's structure. Coefficients are a separate, explicit request:
    decoding shares the native kernel. While generation is active, structural
    browsing remains available but coefficient rendering is disabled. Reopen
    the artifact after the campaign has stopped to enable that control.
    **Table search filters the current fetched page only** (up to 25 rows),
    not the entire artifact. Change the page offset to browse another page.
    Encoded bytes and structural records still occupy memory. Lazy browsing
    avoids eager coefficient decoding and large initial displays; it is not
    an on-disk database that loads no artifact data.
    """)
    artifact_family = mo.ui.dropdown(
        options=list(FAMILY_NAMES), value="H", label="Completed family",
        allow_select_none=False,
    )
    open_artifact = mo.ui.run_button(label="Open completed artifact")
    mo.vstack([_intro, mo.hstack([artifact_family, open_artifact], justify="start")])
    return artifact_family, open_artifact


@app.cell(hide_code=True)
def _(artifact_family, campaign, open_artifact):
    mo.stop(not open_artifact.value, mo.md("Generate a family, then open its artifact here."))
    mo.stop(artifact_family.value not in campaign.artifacts,
            mo.callout("This family has no completed artifact yet. Refresh the dashboard and try again.", kind="info"))
    artifact = campaign.artifacts[artifact_family.value]
    metadata = campaign.observe_view(artifact_family.value, "metadata", artifact.metadata)
    mo.accordion({"Native artifact metadata and authority": mo.json(metadata)})
    return artifact, metadata


@app.cell(hide_code=True)
def _(metadata):
    mo.stop(not metadata["total_sectors"], mo.md("This artifact contains no saved sectors."))
    sector_page_start = mo.ui.number(
        start=0, stop=max(0, metadata["total_sectors"] - 1), step=25,
        value=0, label="Sector page offset",
    )
    sector_page_start
    return (sector_page_start,)


@app.cell(hide_code=True)
def _(artifact, artifact_family, campaign, sector_page_start):
    sector_page = campaign.observe_view(artifact_family.value, "sectors", lambda:
        artifact.sectors(start=int(sector_page_start.value), limit=25))
    sector_table = mo.ui.table(
        sector_page["items"], selection="single", initial_selection=[0]
        if sector_page["items"] else [], pagination=False,
        show_column_summaries=False, show_download=False,
        label=f"Sectors · {sector_page['total']:,} total · select one row",
    )
    sector_table
    return (sector_table,)


@app.cell(hide_code=True)
def _(sector_table):
    mo.stop(not sector_table.value)
    _sector = sector_table.value[0]
    selected_sector = _sector["ordinal"]
    selected_sector_mask = _sector["sector"]
    rule_page_start = mo.ui.number(
        start=0, stop=max(0, _sector["total_rules"] - 1), step=25,
        value=0, label="Rule page offset",
    )
    terminal_page_start = mo.ui.number(
        start=0, stop=max(0, _sector["total_terminals"] - 1), step=25,
        value=0, label="Terminal page offset",
    )
    mo.hstack([rule_page_start, terminal_page_start], justify="start")
    return (
        rule_page_start,
        selected_sector,
        selected_sector_mask,
        terminal_page_start,
    )


@app.cell(hide_code=True)
def _(
    artifact,
    artifact_family,
    campaign,
    rule_page_start,
    rule_summary_rows,
    selected_sector,
):
    rule_page = campaign.observe_view(artifact_family.value, "rules", lambda:
        artifact.rules(selected_sector, start=int(rule_page_start.value), limit=25))
    rule_table = mo.ui.table(
        rule_summary_rows(rule_page["items"]), selection="single", initial_selection=[0]
        if rule_page["items"] else [], pagination=False,
        show_column_summaries=False, show_download=False,
        label=f"Saved rules · {rule_page['total']:,} in this sector · metadata only",
    )
    rule_table
    return (rule_table,)


@app.cell(hide_code=True)
def _():
    rule_detail_budget = mo.ui.dropdown(
        options={"64 KiB (default)": 65536, "256 KiB": 262144, "1 MiB": 1048576},
        value="64 KiB (default)", label="Selected rule structure budget",
        allow_select_none=False,
    )
    rule_detail_budget
    return (rule_detail_budget,)


@app.cell(hide_code=True)
def _(
    artifact,
    artifact_family,
    campaign,
    selected_sector,
    terminal_page_start,
    terminal_rows,
):
    terminal_page = campaign.observe_view(artifact_family.value, "terminals", lambda:
        artifact.terminals(selected_sector, start=int(terminal_page_start.value), limit=25))
    mo.accordion({
        f"Finite terminal records · {terminal_page['total']:,} in this sector": mo.vstack([
            mo.ui.table(terminal_rows(terminal_page), selection=None, pagination=False,
                        show_column_summaries=False, show_download=False),
            mo.md("Stored finite residual integrals, not a claim of master independence."),
            mo.accordion({"Raw native terminal page": mo.json(terminal_page)}),
        ]),
    })
    return


@app.cell(hide_code=True)
def _(
    artifact,
    artifact_family,
    campaign,
    rule_detail_budget,
    rule_table,
    selected_sector,
):
    mo.stop(not rule_table.value)
    rule_detail = None
    _error = None
    try:
        rule_detail = campaign.observe_view(artifact_family.value, "rule", lambda:
            artifact.rule(selected_sector, rule_table.value[0]["ordinal"],
                          max_output_bytes=rule_detail_budget.value))
    except Exception as _exception:
        _error = str(_exception)
    mo.stop(rule_detail is None, mo.callout(
        f"Rule detail not loaded: {_error}. Select a smaller rule or explicitly increase "
        "the structure budget; the initial HTML remains paged.", kind="warn"))
    return (rule_detail,)


@app.cell(hide_code=True)
def _(rule_detail):
    rhs_page_start = mo.ui.number(start=0, stop=max(0, len(rule_detail["rhs"]) - 1),
        step=10, value=0, label="RHS preview offset")
    condition_page_start = mo.ui.number(start=0,
        stop=max(0, len(rule_detail["excluded_all_zero_conjunctions"]) - 1),
        step=10, value=0, label="Excluded-branch preview offset")
    mo.hstack([rhs_page_start, condition_page_start], justify="start")
    return condition_page_start, rhs_page_start


@app.cell(hide_code=True)
def _(
    condition_page_start,
    rhs_page_start,
    rule_detail,
    rule_view,
    selected_sector_mask,
):
    rule_view(mo, rule_detail, selected_sector_mask,
        rhs_start=int(rhs_page_start.value), condition_start=int(condition_page_start.value))
    return


@app.cell(hide_code=True)
def _(
    campaign,
    condition_page_start,
    metadata,
    rhs_page_start,
    rule_coefficient_ids,
    rule_detail,
):
    _ids = rule_coefficient_ids(rule_detail, rhs_start=int(rhs_page_start.value),
                                condition_start=int(condition_page_start.value))
    mo.stop(not metadata["total_coefficients"], mo.md("This artifact has no coefficient IDs."))
    coefficient_id = mo.ui.number(
        start=0, stop=metadata["total_coefficients"] - 1, step=1,
        value=_ids[0] if _ids else 0, label="Coefficient or condition · native payload-local ID",
    )
    coefficient_budget = mo.ui.dropdown(
        options={"8 KiB preview": 8192, "64 KiB": 65536, "1 MiB": 1048576},
        value="8 KiB preview", label="Explicit coefficient print budget", allow_select_none=False,
    )
    _busy = campaign.state in {"running", "cancelling"}
    load_coefficient = mo.ui.run_button(label="Render this coefficient", disabled=_busy)
    mo.vstack([
        mo.md("IDs in the current preview: " + ", ".join(f"c_{i}" for i in _ids[:20])
              + (f" … ({len(_ids)} preview IDs)" if len(_ids) > 20 else "")
              + ". You may enter another payload-local ID; the native API validates it."),
        mo.hstack([coefficient_id, coefficient_budget, load_coefficient], justify="start", wrap=True),
        mo.md("Native algebra is active. Reopen the artifact after it drains to render a coefficient."
              if _busy else "Rendering is an explicit request; changing an ID or budget does not decode it."),
    ])
    return coefficient_budget, coefficient_id, load_coefficient


@app.cell
def _(
    artifact,
    artifact_family,
    campaign,
    coefficient_budget,
    coefficient_id,
    coefficient_view,
    load_coefficient,
):
    mo.stop(not load_coefficient.value)
    mo.stop(campaign.state in {"running", "cancelling"}, mo.callout(
        "Native algebra is active; coefficient rendering waits until the campaign drains.", kind="info"))
    # Display-only: never parse these strings back into a rule or certificate.
    coefficient = None
    _error = None
    try:
        coefficient = campaign.observe_view(artifact_family.value, "coefficient", lambda:
            artifact.coefficient(int(coefficient_id.value), max_output_bytes=coefficient_budget.value))
    except Exception as _exception:
        _error = str(_exception)
    mo.stop(coefficient is None, mo.callout(
        f"Coefficient not rendered: {_error}. Increase the explicit print budget and click "
        "Render again if a larger view is needed.", kind="warn"))
    coefficient_view(mo, coefficient)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. What do the terminal counts mean?

    A saved **finite residual** is where this bounded search stopped. It is
    not automatically an independent master: the same integral may appear in
    different routings, and numerator identities can relate distinct keys.

    The explicit operation below uses RustRed's existing exact, family-local
    unit aliases and weighted quadratic-numerator normalization. Unsupported
    shapes remain outputs; it does not run a new IBP search or certify closure.
    Counts are measured from **your completed artifacts**, not a shipped table.
    Do not add family-local counts and call the sum an independent basis:
    H, X, BMW and FG overlap, and equal raw vectors from different families
    are not an equivalence test.

    For comparison, FMFT ultimately uses the **19 symbolic representatives**
    PR0–PR15 and PR4d, PR9d, PR11d in
    [Czakon's basis, Fig. 1](https://arxiv.org/pdf/hep-ph/0411261), as referenced
    by [FMFT §2.4](https://arxiv.org/html/1707.01710v2).
    That convention includes factorizable products and dotted integrals.
    It is neither a count of residual keys nor 19 independent numerical
    constants: numerical Laurent data also depend on the required ε order.
    A smaller normalized count below is useful, but is not a proof that this
    notebook has reproduced that basis or its numerical values.
    """)
    return


@app.cell(hide_code=True)
def _(campaign, ibp_families):
    _available = all(hasattr(family, "normalize_candidate_terminals")
                     for family in ibp_families.values())
    normalize_terminals = mo.ui.button(
        value=0, label="Normalize completed terminal sets", kind="neutral", disabled=not _available,
        on_click=lambda value: (campaign.normalize_completed(), value + 1)[1],
    )
    mo.vstack([normalize_terminals, mo.md(
        "Explicit native algebra, only after generation drains. Already normalized families are reused."
        if _available else "This host does not yet include the native terminal-normalization API.")])
    return (normalize_terminals,)


@app.cell(hide_code=True)
def _(campaign, normalization_summary_rows, normalize_terminals):
    _ = normalize_terminals.value
    normalization_snapshot = dict(campaign.normalization_rows)
    _panels = [mo.ui.table(normalization_summary_rows(campaign), selection=None,
                           pagination=False, show_column_summaries=False, show_download=False)]
    if campaign.normalization_error:
        _panels.append(mo.callout(campaign.normalization_error, kind="warn"))
    for _name, _row in normalization_snapshot.items():
        if _row["state"] == "refused":
            _panels.append(mo.callout(f"{_name}: {_row['error']}", kind="warn"))
    mo.vstack(_panels)
    return (normalization_snapshot,)


@app.cell(hide_code=True)
def _(normalization_snapshot):
    _names = [name for name, row in normalization_snapshot.items() if row["state"] == "completed"]
    mo.stop(not _names, mo.md("No normalized result yet. Generation and normalization are separate explicit actions."))
    normalized_family = mo.ui.dropdown(options=_names, value=_names[0],
        label="Normalized family", allow_select_none=False)
    normalized_family
    return (normalized_family,)


@app.cell(hide_code=True)
def _(campaign, normalization_snapshot, normalized_family):
    normalized = campaign.normalizations[normalized_family.value]
    normalized_metadata = normalization_snapshot[normalized_family.value]["metadata"]
    _skipped = normalized_metadata["skipped"]
    mo.accordion({
        "Normalization authority and skipped shapes": mo.vstack([
            mo.md("Exact within this family; not ordinary-IBP source replay, closure, or master minimality."),
            mo.ui.table(_skipped[:10], selection=None, show_download=False, show_column_summaries=False),
            mo.json({key: value for key, value in normalized_metadata.items()
                     if key not in {"family_fingerprint", "skipped"}}),
        ]),
    })
    return normalized, normalized_metadata


@app.cell(hide_code=True)
def _(normalized_metadata):
    normalized_key_start = mo.ui.number(start=0,
        stop=max(0, normalized_metadata["canonical_terminals"] - 1), step=25,
        value=0, label="Canonical terminal page offset")
    normalized_relation_start = mo.ui.number(start=0,
        stop=max(0, normalized_metadata["total_relations"] - 1), step=25,
        value=0, label="Normalization relation page offset")
    mo.hstack([normalized_key_start, normalized_relation_start], justify="start")
    return normalized_key_start, normalized_relation_start


@app.cell(hide_code=True)
def _(normalized, normalized_key_start, terminal_rows):
    _page = normalized.terminals(start=int(normalized_key_start.value), limit=25)
    mo.ui.table(terminal_rows(_page), selection=None, pagination=False,
        show_download=False, show_column_summaries=False,
        label=f"Canonical outputs · {_page['total']} total · page-local search")
    return


@app.cell(hide_code=True)
def _(normalized, normalized_relation_start):
    _page = normalized.relations(start=int(normalized_relation_start.value), limit=25)
    normalization_relation_table = mo.ui.table(_page["items"], selection="single",
        initial_selection=[0] if _page["items"] else [], pagination=False,
        show_download=False, show_column_summaries=False,
        label=f"Normalization relations · {_page['total']} total · structure only")
    normalization_relation_table
    return (normalization_relation_table,)


@app.cell(hide_code=True)
def _(normalization_relation_table, normalized):
    mo.stop(not normalization_relation_table.value)
    normalization_relation = None
    _error = None
    try:
        normalization_relation = normalized.relation(
            normalization_relation_table.value[0]["ordinal"], max_output_bytes=65536)
    except Exception as _exception:
        _error = str(_exception)
    mo.stop(normalization_relation is None, mo.callout(
        f"Selected relation not loaded under the 64 KiB detail budget: {_error}", kind="warn"))
    normalized_rhs_start = mo.ui.number(start=0,
        stop=max(0, len(normalization_relation["rhs"]) - 1), step=10,
        value=0, label="Normalized relation RHS offset")
    normalized_rhs_start
    return normalization_relation, normalized_rhs_start


@app.cell(hide_code=True)
def _(
    normalization_relation,
    normalization_relation_view,
    normalized_rhs_start,
):
    normalization_relation_view(mo, normalization_relation, start=int(normalized_rhs_start.value))
    return


@app.cell(hide_code=True)
def _(normalization_relation, normalized_metadata, normalized_rhs_start):
    mo.stop(not normalized_metadata["total_coefficients"])
    _terms = normalization_relation["rhs"][int(normalized_rhs_start.value):int(normalized_rhs_start.value) + 10]
    normalized_coefficient_id = mo.ui.number(start=0,
        stop=normalized_metadata["total_coefficients"] - 1, step=1,
        value=_terms[0]["coefficient_id"] if _terms else 0,
        label="Normalization coefficient ID · separate local table")
    render_normalized_coefficient = mo.ui.run_button(label="Render normalization coefficient")
    mo.hstack([normalized_coefficient_id, render_normalized_coefficient], justify="start")
    return normalized_coefficient_id, render_normalized_coefficient


@app.cell(hide_code=True)
def _(
    coefficient_view,
    normalized,
    normalized_coefficient_id,
    render_normalized_coefficient,
):
    mo.stop(not render_normalized_coefficient.value)
    _detail = None
    _error = None
    try:
        _detail = normalized.coefficient(int(normalized_coefficient_id.value), max_output_bytes=8192)
    except Exception as _exception:
        _error = str(_exception)
    mo.stop(_detail is None, mo.callout(f"No coefficient preview under 8 KiB: {_error}", kind="warn"))
    coefficient_view(mo, _detail)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Evaluate a numerator with Vakint

    Generation, terminal normalization, and numerical integration are different
    tasks. This optional **explicit** calculation uses the H graph with

    $N=(k_1\!\cdot k_2)^2+(p_1\!\cdot k_3)(k_3\!\cdot p_2)
       +(p_1\!\cdot p_2)((k_2+k_1)\!\cdot k_2)$.

    We construct a separate symbolic-mass family from that graph; the unit-mass
    candidate family above is unchanged. Vakint uses native FeynKit tensor
    reduction and its **shipped RustRed assets**, not these freshly generated
    candidate files. The FORM executable is deliberately unavailable for this
    calculation. Numerical master data are still required; FORM-free does not
    mean master-data-free.

    The comparison below is the existing H rank-four Vakint reference in
    $\overline{\mathrm{MS}}$, with $m^2=3$, $\mu^2=5$, 32-digit arithmetic and
    five Laurent coefficients. The external vectors retain the original test's
    floating-point input values. Passing this one example establishes neither
    arbitrary-index closure nor a new numerical-master computation.
    """)
    return


@app.cell
def _(N, S, diagrams, dimension, hep, scalar_model):
    from symbolica import Replacement
    try:
        from symbolica.community.hepkit import vakint
    except ImportError:
        vakint = None
    mo.stop(vakint is None or not hasattr(vakint, "integral_from_diagram"), mo.callout(
        "This section requires the native HEPKit Vakint graph adapter.", kind="warn"))
    _base = diagrams["H"].integral_family(kinematics=hep.Kinematics(dimension))
    _mass_squared = S("vakint::muvsq")
    h_mass_substitutions = {scalar_model.particle("phi").mass: _mass_squared ** (N(1) / 2)}
    _replacements = [Replacement(left, right) for left, right in h_mass_substitutions.items()]
    h_evaluation_family = hep.IntegralFamily(
        _base.loop_momenta, [],
        [den.replace_multiple(_replacements) for den in _base.denominators],
        kinematics=_base.kinematics,
    )
    return h_evaluation_family, h_mass_substitutions, vakint


@app.cell
def _(
    S,
    diagrams,
    dimension,
    h_evaluation_family,
    h_mass_substitutions,
    hep,
    vakint,
):
    _p1, _p2 = S("h_reference_p1", "h_reference_p2")
    _kinematics = hep.Kinematics(dimension, momenta=[*h_evaluation_family.loop_momenta, _p1, _p2])
    _k1, _k2, _k3, _k4 = h_evaluation_family.loop_momenta
    _sp = _kinematics.scalar_product
    h_numerator = (_sp(_k1, _k2) ** 2 + _sp(_p1, _k3) * _sp(_k3, _p2)
                   + _sp(_p1, _p2) * _sp(_k2 + _k1, _k2))
    h_vakint_integral = vakint.integral_from_diagram(
        diagrams["H"], h_evaluation_family, h_numerator,
        parameter_substitutions=h_mass_substitutions, external_momenta=(_p1, _p2),
    )
    mo.vstack([mo.md("**Native numerator**"), h_numerator.formatted(max_terms=8),
               mo.accordion({"Evaluation family · symbolic mass, not the generation family":
                             h_evaluation_family,
                             "Native Vakint integral · bounded rich preview":
                             h_vakint_integral.formatted(max_terms=8)})])
    return (h_vakint_integral,)


@app.cell(hide_code=True)
def _(ExplicitNumeratorEvaluation):
    h_evaluation = ExplicitNumeratorEvaluation()
    return (h_evaluation,)


@app.cell(hide_code=True)
def _(campaign, h_evaluation, h_vakint_integral, live_snapshot):
    evaluate_h = mo.ui.button(
        value=0, label="Evaluate H numerator once",
        disabled=live_snapshot["state"] != "completed" or h_evaluation.state != "ready",
        on_click=lambda value: (h_evaluation.run(campaign, h_vakint_integral), value + 1)[1],
    )
    mo.vstack([evaluate_h, mo.md(
        "Enabled after all four generation sessions have drained. This synchronous native "
        "calculation runs once; changing an explorer selection does not rerun it."
    )])
    return (evaluate_h,)


@app.cell(hide_code=True)
def _(evaluate_h, h_evaluation):
    _ = evaluate_h.value
    mo.stop(h_evaluation.state == "ready", mo.md("No numerical evaluation requested."))
    mo.stop(h_evaluation.result is None, mo.callout(
        f"Vakint evaluation {h_evaluation.state}: {h_evaluation.error}", kind="warn"))
    _result = h_evaluation.result
    _metrics = _result["metrics"]
    mo.vstack([
        mo.callout("Reference comparison passed." if _metrics["reference_matches"]
                   else "Reference comparison did not pass.",
                   kind="success" if _metrics["reference_matches"] else "warn"),
        mo.md(f"Native symbolic evaluation: **{_metrics['symbolic_seconds']:.3f} s** · "
              f"numerical substitution: **{_metrics['numerical_seconds']:.3f} s**"),
        mo.md("**Computed Laurent series**"), _result["result"].formatted(max_terms=5, precision=32),
        mo.accordion({"Stored reference · comparison input, not a computed output":
                       _result["reference"].formatted(max_terms=5, precision=32),
                       "Numerical settings and measured timings": mo.json(_metrics)}),
        mo.md(h_evaluation.error or "Evidence saved in `vakint-h-numerator.json`."),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Save, reopen, and interpret

    Completed payloads are saved under the printed directory as
    `<family>/candidate.rrbin`. In another session, reopen trusted artifacts
    with `rustred.CandidateArtifact.open_file(path, bundle_max_entries=10_000_000)`;
    use the same explicit transport allowance as generation. Browsing metadata does
    not rerun generation. Native formats are for trusted local data.

    The same fresh directory records bounded native poll batches in
    `progress.jsonl`, the latest `snapshot.json`, per-family native timing
    reports, and explicit explorer-call timings in `views.jsonl`. Dropped
    events remain visible; this is not a complete native event journal.

    The DOT inputs and reference-only TOMLs are included with this example;
    no RustRed checkout, CLI subprocess, FORM process, or second Symbolica
    shared library is required by this workflow. Full-source verification and family closure
    are separate steps, not implied by an artifact's existence.
    """)
    return


if __name__ == "__main__":
    app.run()
