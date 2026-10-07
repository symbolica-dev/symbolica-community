import marimo

__generated_with = "0.24.2"
app = marimo.App(
    width="medium",
    app_title="Three-loop massive vacuum reduction",
)

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

    from symbolica import E, S
    from symbolica.community import hepkit as hep
    from three_loop_reduction_support import (
        ThreeLoopRun, assert_source_matches_family, graph_inputs, tomllib,
    )
    from rustred_campaign_support import (
        coefficient_view, integral_notation, rule_coefficient_ids,
        rule_summary_rows, rule_view, terminal_rows,
    )

    rustred = getattr(hep, "rustred", None)
    return (
        E,
        S,
        ThreeLoopRun,
        assert_source_matches_family,
        coefficient_view,
        graph_inputs,
        hep,
        integral_notation,
        rule_coefficient_ids,
        rule_summary_rows,
        rule_view,
        rustred,
        terminal_rows,
        tomllib,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Three-loop massive vacuum reduction

    [Browse notebooks](/) · [Integral families](/?file=hep/integral_families.py) ·
    [Four-loop candidate search](/?file=hep/four_loop_reduction.py)

    **Draw the graph → generate IBP rules → certify closure → reduce to masters.**

    The equal-mass **Mercedes graph** has four vertices, six edges and three
    loops. We compute exact coefficients in symbolic dimension $d$, including
    raised propagators, pinches and numerator insertions. The recursive native
    reducer uses the artifact generated and certified in this notebook.

    The certificate retains **38 raw terminal keys**, not 38 independent
    masters. Equivalent loop-momentum routings group them into **five named
    topology types**, shown below. The reductions keep the original keys;
    no numerical master values are needed, and this notebook does not prove
    that a terminal basis is minimal or linearly independent.
    The folded setup imports the native HEP objects and small UI helpers;
    generation starts only when you click **Generate**.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Setup and notebook helpers

    The folded setup block imports HEPKit, native RustRed and the small session
    and display helpers. Expand its code to inspect the imports. The visible
    cells below route the graph, certify the rules and request exact reductions.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Route the graph

    With $M=m^2$ and $D_i=q_i^2-M$, define
    $I_M(n)=\int_{k_1,k_2,k_3}\prod_{i=1}^{6}D_i^{-n_i}$ in a common
    dimensionally regulated measure. Positive powers are denominators,
    zero removes an edge, and negative powers insert numerator factors.
    Set $M=1$ for the IBP calculation and restore it by homogeneity below.
    All six scalar products are covered: no auxiliary propagator is needed.
    """)
    return


@app.cell
def _(E, S, assert_source_matches_family, graph_inputs, hep):
    dimension, mass_squared, integral = S("d", "M", "I")
    scalar_model = hep.Model.phi_3_4()
    dot_input, generation_source = graph_inputs()
    diagram = hep.FeynmanDiagram.from_dot(scalar_model, dot_input)
    routed = diagram.integral_family(kinematics=hep.Kinematics(dimension))
    model_mass = scalar_model.particle("phi").mass
    family = hep.IntegralFamily(
        routed.loop_momenta, routed.external_momenta,
        [den.replace(model_mass, E("1")) for den in routed.denominators],
        kinematics=routed.kinematics,
    )
    assert_source_matches_family(generation_source, family)
    # The source API preserves d; explicitly map its native symbol for display.
    parameter_bindings = [(S("rustred::d"), dimension)]
    mo.vstack([diagram, family])
    return (
        dimension,
        dot_input,
        generation_source,
        integral,
        mass_squared,
        parameter_bindings,
    )


@app.cell(hide_code=True)
def _(dot_input, generation_source, parameter_bindings):
    _plain = {"color_top_level_sum": False, "color_builtin_symbols": False,
              "bracket_level_colors": None, "show_namespaces": True}
    _bindings = [{"Artifact variable": internal.format(**_plain),
                  "Original HEPKit expression": original.format(**_plain)}
                 for internal, original in parameter_bindings]
    mo.vstack([
        mo.md("""
        **Generation input.** The current closure verifier accepts the explicit
        native source below. HEPKit routes the DOT graph; the preceding assertion
        checks that every source denominator equals its routed native scalar
        product, in the same order. The source contains the family definition
        and an ordinary target, with no saved rules. Its dimension is `d`;
        the table records the native symbol used in reduction coefficients.
        """),
        mo.accordion({
            "DOT graph": mo.md(f"```dot\n{dot_input}\n```"),
            "Checked native generation source": mo.md(f"```toml\n{generation_source}\n```"),
            "Coefficient parameter legend": mo.ui.table(_bindings,
                selection=None, show_column_summaries=False, show_download=False),
        }),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 2. Generate, then certify

    One worker searches every sector of the six-propagator family, including
    nonpositive powers. Native hosts provide live events and cooperative
    cancellation. Pyodide runs synchronously: Generate returns when the search
    completes, with no intermediate progress or in-flight cancellation.
    This session generates once; rerun the notebook for a fresh session.

    Generation produces candidates. **Certify generated rules** replays their
    sources and verifies coverage and termination before making an artifact
    available to the recursive reducer. This is a separate, explicit operation.
    """)
    return


@app.cell
def _(ThreeLoopRun, generation_source, rustred):
    generation_options = {
        "n_cores": 1, "exact_backend": "sparse", "numerical_depth": 2,
        "event_capacity": 256,
    }
    run = ThreeLoopRun(rustred, generation_source)
    native_available = rustred is not None and all(hasattr(rustred, method) for method in (
        "start_family_candidates", "certify_candidates",
        "inspect_closing_artifact", "reduce_with_closing_artifact",
    ))
    return generation_options, native_available, run


@app.cell(hide_code=True)
def _(generation_options, native_available, run):
    generate = mo.ui.button(label="Generate", kind="success",
        on_click=lambda value: run.start(**generation_options), disabled=not native_available)
    cancel = mo.ui.button(label="Cancel", on_click=lambda value: run.cancel(),
        disabled=not native_available or not run.capabilities["cancellation_in_flight"])
    heartbeat = (mo.ui.refresh(options=["1s", "3s", "10s"], default_interval="1s")
                 if run.capabilities["live_event_polling"] else None)
    _controls = [generate, cancel, heartbeat] if heartbeat is not None else [generate]
    mo.vstack([
        mo.hstack(_controls, justify="start", wrap=True),
        mo.md("Generate starts the calculation. " + (
            "Keep refresh enabled to collect live native progress."
            if run.capabilities["live_event_polling"] else
            "This browser runs one worker synchronously; progress appears after completion."
        )) if native_available
        else mo.callout("This HEPKit installation lacks the native closing-artifact API. "
                        "Install a Community build with RustRed support to run the reduction.", kind="warn"),
    ])
    return cancel, generate, heartbeat


@app.cell(hide_code=True)
def _(cancel, generate, heartbeat, run):
    _ = cancel.value, generate.value, heartbeat.value if heartbeat is not None else None
    live = run.poll()
    _counts = live["counts"]
    mo.vstack([
        mo.hstack([
            mo.stat(live["state"].title(), label="Generation"),
            mo.stat(f"{_counts.get('generated', 0)} / {_counts.get('sectors_total', '—')}", label="Sectors"),
            mo.stat(_counts.get("rules", 0), label="Candidate rules"),
            mo.stat(f"{live['elapsed_seconds']:.2f} s", label="Observed time"),
        ], widths="equal"),
        mo.callout(live["error"], kind="danger") if live["error"] else mo.md(""),
        mo.accordion({"Native execution evidence · bounded history": mo.json({
            "execution_mode": run.capabilities["execution_mode"],
            "counts": _counts, "active_jobs": live["active_jobs"],
            "recent_events": live["events"], "dropped_events": live["dropped_events"],
        })}),
    ])
    return


@app.cell(hide_code=True)
def _(native_available):
    certify = mo.ui.run_button(label="Certify generated rules", disabled=not native_available)
    certify
    return (certify,)


@app.cell
def _(certify, run, rustred, tomllib):
    mo.stop(not certify.value, mo.md("Generate first, then certify the resulting bundle."))
    run.poll()
    mo.stop(run.result is None, mo.callout("Generation has not completed successfully yet.", kind="info"))
    if run.closing is None:
        run.closing = rustred.certify_candidates(run.result.bundle)
        assert run.closing.status == "generated-durable"
    closing_artifact = run.closing.artifact
    if run.inspection is None:
        _inspection = rustred.inspect_closing_artifact(closing_artifact)
        assert _inspection.status == "inspected"
        run.inspection = tomllib.loads(_inspection.to_toml())
    inspection = run.inspection
    candidate_artifact = run.candidate
    master_powers = {tuple(item["powers"]) for item in inspection["artifact"]["masters"]}
    assert inspection["artifact"]["arity"] == 6 and master_powers
    return candidate_artifact, closing_artifact, inspection, master_powers


@app.cell(hide_code=True)
def _(closing_artifact, inspection, master_powers, run):
    mo.vstack([
        mo.callout(f"Closure certified: {len(master_powers)} raw terminal keys. "
                   "These are not independent masters. Every successful reduction "
                   "below ends entirely in this set.", kind="success"),
        mo.accordion({
            "Native certificate and replay evidence": mo.json(inspection),
            "Download this run's artifacts": mo.hstack([
                mo.download(run.result.bundle, filename="three-loop-candidates.rrbin",
                            label="Candidate bundle"),
                mo.download(closing_artifact, filename="three-loop-certified.rr",
                            label="Certified closing artifact"),
            ], justify="start"),
        }),
    ])
    return


@app.cell(hide_code=True)
def _(integral_notation, master_powers):
    _types = [
        {"Type": "T3,1", "Name": "Three one-loop tadpoles", "Raw keys": 16,
         "Representative": "I(1,1,1,0,0,0)"},
        {"Type": "T4,1", "Name": "Two-loop sunset × one-loop tadpole", "Raw keys": 12,
         "Representative": "I(1,1,1,1,0,0)"},
        {"Type": "T4,2", "Name": "Three-loop basketball (four-line banana)", "Raw keys": 3,
         "Representative": "I(0,1,1,1,1,0)"},
        {"Type": "T5,1", "Name": "Connected five-line vacuum", "Raw keys": 6,
         "Representative": "I(0,1,1,1,1,1)"},
        {"Type": "T6,1", "Name": "Mercedes (tetrahedron / K4)", "Raw keys": 1,
         "Representative": "I(1,1,1,1,1,1)"},
    ]
    _rows = [{"Raw terminal key": integral_notation(powers), "Total power": sum(powers)}
             for powers in sorted(master_powers)]
    mo.vstack([
        mo.md(r"""
        ### Five topology types, 38 raw terminal keys

        For this equal-mass input, changes of loop-momentum variables identify
        the raw keys in the following groups: $16+12+3+6+1=38$.
        The labels follow [R. N. Lee, Figure 2](https://arxiv.org/pdf/1203.4868#page=5).
        They name graph types, not a conversion to that paper's normalization.
        Each representative uses this notebook's denominator order, not a
        relabeling of the artifact. The first two types factorize into
        lower-loop integrals; the last three are connected three-loop graphs.
        """),
        mo.ui.table(_types, selection=None, pagination=False,
                    show_column_summaries=False, show_download=False),
        mo.md("""
        **The certified artifact is unchanged.** The following table and the
        reductions retain all 38 raw keys: the display does not apply the
        topology identifications, merge coefficients, or supply an additional
        proof of master independence. Closure means reduction terminates in
        these keys; independence and basis minimization are different questions.
        """),
        mo.ui.table(_rows, selection=None, pagination=True, page_size=10,
                    show_column_summaries=False, show_download=False),
    ])
    return


@app.cell(hide_code=True)
def _(candidate_artifact):
    _sectors = candidate_artifact.sectors(start=0, limit=64)["items"]
    _options = {
        f"{row['ordinal']} · {''.join('1' if b else '0' for b in row['sector'])} · {row['total_rules']} rules": row["ordinal"]
        for row in _sectors
    }
    _default = next((row["ordinal"] for row in _sectors
                     if all(row["sector"]) and row["total_rules"]), _sectors[0]["ordinal"])
    sector_choice = mo.ui.dropdown(options=_options,
        value=next(label for label, ordinal in _options.items() if ordinal == _default),
        label="Sector", allow_select_none=False)
    mo.vstack([
        mo.md("""
        ## 3. Inspect a generated recurrence

        Browse ten rules at a time from the same candidate bundle that was
        certified above. A selected rule shows bounded structure; coefficient
        rendering is a separate request. Native index labels start at zero.
        """),
        sector_choice,
    ])
    return (sector_choice,)


@app.cell(hide_code=True)
def _(candidate_artifact, sector_choice):
    _sector = candidate_artifact.sectors(start=int(sector_choice.value), limit=1)["items"][0]
    rule_offset = mo.ui.number(start=0, stop=max(0, _sector["total_rules"] - 1),
        step=10, value=0, label="Rule page offset")
    rule_offset
    return (rule_offset,)


@app.cell(hide_code=True)
def _(
    candidate_artifact,
    rule_offset,
    rule_summary_rows,
    sector_choice,
    terminal_rows,
):
    _sector = int(sector_choice.value)
    _page = candidate_artifact.rules(_sector, start=int(rule_offset.value), limit=10)
    rule_table = mo.ui.table(rule_summary_rows(_page["items"]), selection="single",
        initial_selection=[0] if _page["items"] else [], pagination=False,
        show_column_summaries=False, show_download=False,
        label=f"Rules · {_page['total']} in this sector")
    _terminals = candidate_artifact.terminals(_sector, start=0, limit=10)
    mo.vstack([rule_table, mo.accordion({
        f"Candidate terminal preview · first 10 of {_terminals['total']}": mo.ui.table(
            terminal_rows(_terminals), selection=None, pagination=False,
            show_column_summaries=False, show_download=False),
    })])
    return (rule_table,)


@app.cell(hide_code=True)
def _(candidate_artifact, rule_table, sector_choice):
    mo.stop(not rule_table.value, mo.md("Select a rule on a populated page."))
    rule_detail = candidate_artifact.rule(int(sector_choice.value),
        rule_table.value[0]["ordinal"], max_output_bytes=65536)
    rhs_offset = mo.ui.number(start=0, stop=max(0, len(rule_detail["rhs"]) - 1),
        step=10, value=0, label="RHS term offset")
    rhs_offset
    return rhs_offset, rule_detail


@app.cell(hide_code=True)
def _(candidate_artifact, rhs_offset, rule_detail, rule_view, sector_choice):
    _sector = candidate_artifact.sectors(start=int(sector_choice.value), limit=1)["items"][0]
    rule_view(mo, rule_detail, _sector["sector"], rhs_start=int(rhs_offset.value))
    return


@app.cell(hide_code=True)
def _(rhs_offset, rule_coefficient_ids, rule_detail):
    _ids = rule_coefficient_ids(rule_detail, rhs_start=int(rhs_offset.value))
    mo.stop(not _ids, mo.md("This preview contains no coefficients."))
    coefficient_id = mo.ui.dropdown(options={f"c_{cid}": cid for cid in _ids},
        value=f"c_{_ids[0]}",
        label="Coefficient in this preview", allow_select_none=False)
    render_coefficient = mo.ui.run_button(label="Render coefficient")
    mo.hstack([coefficient_id, render_coefficient], justify="start")
    return coefficient_id, render_coefficient


@app.cell
def _(
    candidate_artifact,
    coefficient_id,
    coefficient_view,
    render_coefficient,
):
    mo.stop(not render_coefficient.value)
    coefficient_view(mo, candidate_artifact.coefficient(int(coefficient_id.value),
                                                      max_output_bytes=65536))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Reduce completely and restore the mass

    Choose an integral and click **Reduce to certified masters**. The native
    reducer recursively follows the certified rules, combines coefficients
    exactly and returns only raw terminal keys. Results are cached per target.

    For three loops, $I_M(n)=M^{3d/2-\sum_i n_i}I_1(n)$. Thus a unit-mass
    coefficient $c_a(d)$ becomes
    $c_a(d)\,M^{\sum_i a_i-\sum_i n_i}$ multiplying $I_M(a)$.
    The native result supplies that integer exponent; the notebook checks it.
    """)
    return


@app.cell(hide_code=True)
def _(closing_artifact):
    mo.stop(not closing_artifact)
    targets = {
        "Doubled propagator · I(2,1,1,1,1,1)": (2, 1, 1, 1, 1, 1),
        "Two doubled propagators · I(2,2,1,1,1,1)": (2, 2, 1, 1, 1, 1),
        "Pinched graph with dots · I(2,2,1,0,0,0)": (2, 2, 1, 0, 0, 0),
        "Numerator insertion · I(1,1,1,-1,0,0)": (1, 1, 1, -1, 0, 0),
        "Scaleless sector · I(0,0,0,0,0,0)": (0, 0, 0, 0, 0, 0),
    }
    target_choice = mo.ui.dropdown(options=targets, value=next(iter(targets)),
        label="Target integral", allow_select_none=False)
    reduce_target = mo.ui.run_button(label="Reduce to certified masters")
    mo.hstack([target_choice, reduce_target], justify="start", wrap=True)
    return reduce_target, target_choice


@app.cell
def _(
    E,
    closing_artifact,
    inspection,
    master_powers,
    parameter_bindings,
    reduce_target,
    run,
    rustred,
    target_choice,
):
    mo.stop(not reduce_target.value, mo.md("Select a target and request its complete reduction."))
    selected_powers = tuple(target_choice.value)
    if selected_powers not in run.reductions:
        run.reductions[selected_powers] = rustred.reduce_with_closing_artifact(
            closing_artifact, list(selected_powers))
    reduction = run.reductions[selected_powers]
    assert reduction.status == "reduced"
    assert reduction.family_fingerprint == inspection["artifact"]["family_fingerprint"]
    reduced_terms = []
    for _term in reduction.terms:
        _master = tuple(_term.master_powers)
        assert _master in master_powers
        assert _term.common_mass_squared_power == sum(_master) - sum(selected_powers)
        # Native strings cross only the display boundary; native RustRed reduced them.
        _coefficient = E(_term.unit_mass_coefficient)
        for _internal, _original in parameter_bindings:
            _coefficient = _coefficient.replace(_internal, _original)
        reduced_terms.append((_master, _coefficient, _term.common_mass_squared_power))
    return reduced_terms, reduction, selected_powers


@app.cell(hide_code=True)
def _(reduced_terms):
    term_offset = mo.ui.number(start=0, stop=max(0, len(reduced_terms) - 1),
        step=10, value=0, label="Reduction term offset")
    term_offset
    return (term_offset,)


@app.cell(hide_code=True)
def _(
    integral,
    mass_squared,
    reduced_terms,
    reduction,
    selected_powers,
    term_offset,
    tomllib,
):
    _start = int(term_offset.value)
    _page = reduced_terms[_start:_start + 10]
    mo.vstack([
        mo.hstack([integral(*selected_powers), mo.md(
            f"**= sum of {len(reduced_terms)} raw terminal {'term' if len(reduced_terms) == 1 else 'terms'}**"
            if reduced_terms else "**= 0**")],
            justify="start"),
        *[mo.hstack([coefficient * mass_squared**power, mo.md(r"$\times$"), integral(*master)],
                    justify="start", wrap=True) for master, coefficient, power in _page],
        mo.md(f"Showing {_start + 1 if _page else 0}–{_start + len(_page)} of "
              f"{len(reduced_terms)} terms. Every displayed integral has common squared mass $M$."),
        mo.accordion({"Native traversal statistics and complete result": mo.vstack([
            mo.json(tomllib.loads(reduction.to_toml())["statistics"]),
            mo.download(reduction.to_toml(), filename="three-loop-reduction.toml",
                        label="Download all exact terms"),
        ])}),
    ])
    return


@app.cell
def _(E, closing_artifact, dimension, parameter_bindings, run, rustred):
    # Independent factorized check: three one-loop tadpoles with two raised powers.
    _target = (2, 2, 1, 0, 0, 0)
    if _target not in run.reductions:
        run.reductions[_target] = rustred.reduce_with_closing_artifact(closing_artifact, list(_target))
    _terms = run.reductions[_target].terms
    assert len(_terms) == 1 and _terms[0].master_powers == [1, 1, 1, 0, 0, 0]
    _coefficient = E(_terms[0].unit_mass_coefficient)
    for _internal, _original in parameter_bindings:
        _coefficient = _coefficient.replace(_internal, _original)
    assert (_coefficient - ((dimension - 2) / 2)**2).together() == 0
    assert _terms[0].common_mass_squared_power == -2
    mo.callout(r"Independent check passed: the factorized tadpole ratio is "
               r"$I_M(2,2,1,0,0,0)=\frac{(d-2)^2}{4M^2}I_M(1,1,1,0,0,0)$.", kind="success")
    return


if __name__ == "__main__":
    app.run()
