import marimo

__generated_with = "0.24.2"
app = marimo.App(
    width="medium",
    app_title="Three-loop massive vacuum reduction",
)

with app.setup(hide_code=True):
    from collections import deque
    from textwrap import dedent
    from time import monotonic

    import marimo as mo
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    def graph_inputs():
        """The graph and checked native source travel with this notebook."""
        return dedent('''\
            digraph Mercedes {
                // Three loop-basis chords; edge IDs fix the denominator order.
                B -> A [id=0, particle="phi", lmb_id=0];
                A -> C [id=1, particle="phi", lmb_id=1];
                C -> B [id=2, particle="phi", lmb_id=2];
                D -> B [id=3, particle="phi"];
                A -> D [id=4, particle="phi"];
                C -> D [id=5, particle="phi"];
            }
            '''), dedent('''\
            schema = "rustred.project.toml.v1"

            # The same ordered family as the graph, with common mass set to one.
            # Ordinary native input: no reduction rules or hints.
            [family]
            name = "rustred_three_loop_unit_mass_vacuum_k6_v1"
            loop_momenta = ["k1", "k2", "k3"]
            external_momenta = []
            dimension = "d"

            [[family.denominators]]
            id = "D1"
            expression = "k1^2-1"

            [[family.denominators]]
            id = "D2"
            expression = "k2^2-1"

            [[family.denominators]]
            id = "D3"
            expression = "k3^2-1"

            [[family.denominators]]
            id = "D4"
            expression = "(k1-k3)^2-1"

            [[family.denominators]]
            id = "D5"
            expression = "(k1-k2)^2-1"

            [[family.denominators]]
            id = "D6"
            expression = "(k2-k3)^2-1"

            [target]
            powers = [1, 1, 1, 1, 1, 1]
            numerator = "1"
            ''')

    def assert_source_matches_family(source, family):
        """Compare this fixed source to native routed scalar products exactly."""
        definition = tomllib.loads(source)["family"]
        assert definition["dimension"] == "d"
        assert definition["loop_momenta"] == ["k1", "k2", "k3"]
        assert definition["external_momenta"] == []
        assert [item["expression"] for item in definition["denominators"]] == [
            "k1^2-1", "k2^2-1", "k3^2-1", "(k1-k3)^2-1",
            "(k1-k2)^2-1", "(k2-k3)^2-1",
        ]
        assert len(family.loop_momenta) == 3 and not family.external_momenta
        assert family.is_complete and family.is_independent
        k1, k2, k3 = family.loop_momenta
        momenta = [k1, k2, k3, k1 - k3, k1 - k2, k2 - k3]
        assert len(family.denominators) == len(momenta)
        for denominator, momentum in zip(family.denominators, momenta):
            expected = family.kinematics.scalar_product(momentum, momentum) - 1
            assert (denominator - expected).expand() == 0

    def rule_expressions(artifact, rule, integral, parameter_bindings=()):
        """Materialize the selected source-input rule for Symbolica display.

        This conversion binds native n0, n1, ... and d to display symbols; it
        does not apply or certify a reduction rule.
        """
        from symbolica import E, S

        indices = [S(f"n_{axis}") for axis in range(len(rule["target"]["values"]))]
        bindings = [(S(f"rustred::n{axis}"), index)
                    for axis, index in enumerate(indices)] + list(parameter_bindings)
        coefficients = {}

        def coefficient(cid):
            if cid not in coefficients:
                detail = artifact.coefficient(cid, max_output_bytes=65536)
                value = (E(detail["numerator"], default_namespace="rustred")
                         / E(detail["denominator"], default_namespace="rustred"))
                for source, display in bindings:
                    value = value.replace(source, display)
                coefficients[cid] = value
            return coefficients[cid]

        def key_expression(key):
            return integral(*(indices[axis] + value if symbolic else value
                              for axis, (value, symbolic) in enumerate(
                                  zip(key["values"], key["symbolic"]))))

        return {
            "target": key_expression(rule["target"]),
            "rhs": sum((coefficient(term["coefficient_id"]) * key_expression(term["integral"])
                        for term in rule["rhs"]), E("0")),
            "affine": [coefficient(cid) for cid in rule["case"]["affine_zero_equations"]],
            "excluded": [[coefficient(cid) for cid in branch]
                         for branch in rule["excluded_all_zero_conjunctions"]],
        }

    def integral_notation(key):
        """Format native integer structure without parsing coefficient text."""
        if isinstance(key, dict):
            values, symbolic = key["values"], key["symbolic"]
            if len(values) != len(symbolic):
                raise ValueError("Integral flags and values have different arities")
        else:
            values, symbolic = key, [False] * len(key)
        powers = []
        for axis, (value, variable) in enumerate(zip(values, symbolic)):
            if type(value) is not int or type(variable) is not bool:
                raise ValueError("Expected exact native integer powers and boolean flags")
            if not variable:
                powers.append(str(value))
            else:
                suffix = f" + {value}" if value > 0 else f" - {-value}" if value < 0 else ""
                powers.append(f"n_{axis}{suffix}")
        return "I(" + ", ".join(powers) + ")"

    def rule_summary_rows(items):
        return [
            {"ordinal": row["ordinal"], "Target": integral_notation(row["target"]),
             "Case": ", ".join(f"n_{item['axis']} = {item['value']}"
                               for item in row["case"]["fixed"]) or row["case"]["kind"],
             "Affine equations": row["case"]["affine_equation_count"],
             "RHS terms": row["rhs_terms"], "Sources": row["retained_source_count"],
             "Excluded equations": row["guard_count"]}
            for row in items
        ]

    def terminal_rows(page):
        return [{"ordinal": page["start"] + index, "Integral": integral_notation(key)}
                for index, key in enumerate(page["items"])]

    class ThreeLoopRun:
        """Own one generation and cache the explicitly requested later algebra."""

        def __init__(self, native, source, *, clock=monotonic):
            self.native, self.source, self.clock = native, source, clock
            # Older native hosts predate this query. Browser builds provide it.
            self.capabilities = native.execution_capabilities() if (
                native is not None and hasattr(native, "execution_capabilities")
            ) else {
                "execution_mode": "background-coordinator", "background_sessions": True,
                "live_event_polling": True, "cancellation_in_flight": True,
                "max_workers": None,
            }
            self.state, self.session = "ready", None
            self.started, self.finished = None, None
            self.result, self.candidate = None, None
            self.closing, self.inspection = None, None
            self.reductions = {}
            self.events = deque(maxlen=20)
            self.counts, self.active_jobs = {}, []
            self.dropped_events, self.error = 0, None

        def start(self, **options):
            if self.state != "ready" or self.native is None:
                return False
            self.started, self.state = self.clock(), "running"
            try:
                self.session = self.native.start_family_candidates(
                    self.source, input_format="toml", **options)
                if not self.capabilities["background_sessions"]:
                    # WASM returns finished work; native hosts remain nonblocking.
                    self.poll()
            except Exception as error:
                self._fail(error)
                return False
            return True

        def cancel(self):
            if (not self.capabilities["cancellation_in_flight"] or self.session is None
                    or self.state not in {"running", "cancelling"}):
                return False
            self.session.cancel()
            self.state = "cancelling"
            return True

        def _fail(self, error):
            self.error, self.state = str(error), "failed"
            self.finished, self.session = self.clock(), None

        def poll(self):
            """Drain a bounded event batch, without waiting or decoding algebra."""
            if self.session is not None:
                try:
                    batch = self.session.poll_events(max_events=128, timeout=0.0)
                    native = batch["snapshot"]
                    self.counts = dict(native["counts"])
                    self.active_jobs = native["active_jobs"]
                    self.events.extend(batch["events"])
                    self.dropped_events = batch["dropped_events"]
                    if native["done"]:
                        if native["state"] == "completed":
                            self.result = self.session.result()
                            self.candidate = self.result.artifact()
                            self.state = "generated"
                        elif native["state"] == "cancelled":
                            self.state = "cancelled"
                        else:
                            raise RuntimeError(native.get("last_error") or "Generation failed")
                        self.finished, self.session = self.clock(), None
                except Exception as error:
                    self._fail(error)
            end = self.finished if self.finished is not None else self.clock()
            return {
                "state": self.state,
                "elapsed_seconds": 0 if self.started is None else end - self.started,
                "counts": dict(self.counts), "active_jobs": self.active_jobs,
                "events": list(self.events), "dropped_events": self.dropped_events,
                "error": self.error,
            }


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
            raise RuntimeError("Export this notebook with scripts/export_rustred_wasm.py to include its WASM wheel.")
        _manifest = await _response.json()
        if _manifest["schema"] != "rustred-browser-assets-v1":
            raise ValueError("Unsupported browser asset manifest")
        _directory = _Path.cwd() / "rustred_notebook_wheel"
        _wheel = _manifest["wheel"]
        if _Path(_wheel).name != _wheel or not _wheel.endswith(".whl"):
            raise ValueError("Invalid browser wheel path")
        _response = await _pyfetch(str(_base / _wheel))
        if _response.status != 200:
            raise RuntimeError(f"Cannot load browser wheel {_wheel}: HTTP {_response.status}")
        _payload = await _response.bytes()
        if _hashlib.sha256(_payload).hexdigest() != _manifest["wheel_sha256"]:
            raise ValueError(f"Browser wheel checksum mismatch: {_wheel}")
        _directory.mkdir(parents=True, exist_ok=True)
        (_directory / _wheel).write_bytes(_payload)
        del _payload
        await _micropip.install("emfs:" + str(_directory / _wheel))

    from symbolica import E, S
    from symbolica.community import hepkit as hep
    rustred = getattr(hep, "rustred", None)
    return E, S, hep, rustred


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
    This file includes its graph, generation source and small UI helpers;
    the folded startup cells load Symbolica and HEPKit. This small three-loop
    generation starts automatically.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Setup and notebook helpers

    The folded startup cells contain the graph, native generation source,
    session and display helpers, and Symbolica/HEPKit imports. Expand their code
    to inspect them. No neighboring helper or data files are required. The
    visible cells below route the graph, certify the rules and request reductions.
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
def _(E, S, hep):
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
    cancellation. Generation starts automatically. Pyodide runs synchronously:
    the call returns when the search completes, with no intermediate progress
    or in-flight cancellation.
    This session generates once; rerun the notebook for a fresh session.

    The session helper below calls this native Python API to derive the rules
    from the checked family input (no precomputed rules are loaded):

    ```python
    session = rustred.start_family_candidates(
        generation_source, input_format="toml", n_cores=1,
        exact_backend="sparse", numerical_depth=2, event_capacity=256,
    )
    ```

    Generation produces candidates. **Certify generated rules** replays their
    sources and verifies coverage and termination before making an artifact
    available to the recursive reducer. This is a separate, explicit operation.
    """)
    return


@app.cell
def _(generation_source, rustred):
    generation_options = {
        "n_cores": 1, "exact_backend": "sparse", "numerical_depth": 2,
        "event_capacity": 256,
    }
    run = ThreeLoopRun(rustred, generation_source)
    native_available = rustred is not None and all(hasattr(rustred, method) for method in (
        "start_family_candidates", "certify_candidates",
        "inspect_closing_artifact", "reduce_with_closing_artifact",
    ))
    if native_available:
        run.start(**generation_options)
    return native_available, run


@app.cell(hide_code=True)
def _(native_available, run):
    cancel = mo.ui.button(label="Cancel", on_click=lambda value: run.cancel(),
        disabled=not native_available or not run.capabilities["cancellation_in_flight"])
    heartbeat = (mo.ui.refresh(options=["1s", "3s", "10s"], default_interval="1s")
                 if native_available and run.capabilities["live_event_polling"] else None)
    _controls = [cancel, heartbeat] if heartbeat is not None else [cancel]
    mo.vstack([
        mo.hstack(_controls, justify="start", wrap=True),
        mo.md("Generation starts automatically. " + (
            "Keep refresh enabled to collect live native progress."
            if run.capabilities["live_event_polling"] else
            "This browser runs one worker synchronously; progress appears after completion."
        )) if native_available else mo.callout(
            "This HEPKit installation lacks the native closing-artifact API. "
            "Install a Community build with RustRed support to run the reduction.", kind="warn"),
    ])
    return cancel, heartbeat


@app.cell(hide_code=True)
def _(cancel, heartbeat, run):
    _ = cancel.value, heartbeat.value if heartbeat is not None else None
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
def _(certify, run, rustred):
    mo.stop(not certify.value, mo.md("Once automatic generation finishes, certify the resulting bundle."))
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
def _(master_powers):
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
        certified above. Selecting a rule materializes its full right-hand side
        as a Symbolica expression, with one integral term per line. Only the
        selected rule is decoded. Native index labels start at zero.
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
def _(candidate_artifact, rule_offset, sector_choice):
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
    return (rule_detail,)


@app.cell(hide_code=True)
def _(
    candidate_artifact,
    integral,
    parameter_bindings,
    rule_detail,
    sector_choice,
):
    _sector = candidate_artifact.sectors(start=int(sector_choice.value), limit=1)["items"][0]
    _expressions = rule_expressions(candidate_artifact, rule_detail, integral, parameter_bindings)
    _conditions = [mo.md("**Sector:** " + ", ".join(
        f"n_{axis} {'> 0' if active else '≤ 0'}" for axis, active in enumerate(_sector["sector"])))]
    _conditions.append(mo.md("**Fixed powers:** " + (", ".join(
        f"n_{item['axis']} = {item['value']}" for item in rule_detail["case"]["fixed"]) or "None")))
    for _equation in _expressions["affine"]:
        _conditions.append(mo.hstack([mo.md("Required zero:"),
                                     _equation.formatted(max_terms=None)], justify="start"))
    for _number, _branch in enumerate(_expressions["excluded"], 1):
        _conditions.append(mo.md(f"**Excluded branch {_number}:** all expressions below vanish"
                                 if _branch else f"**Excluded branch {_number}:** always true"))
        _conditions.extend(expr.formatted(max_terms=None) for expr in _branch)
    _conditions.append(mo.md("Any excluded branch forbids the rule; equations within a branch "
        "are joined by **AND**. Denominator poles, source conditions and native rule priority "
        "still govern applicability. This display does not replace the native dispatcher."))
    mo.vstack([
        mo.md(f"**Rule {rule_detail['ordinal']} · {len(rule_detail['rhs'])} RHS terms · "
              f"{rule_detail['retained_source_count']} retained sources**"),
        mo.hstack([_expressions["target"], mo.md("**=**")], justify="start"),
        _expressions["rhs"].formatted(max_terms=None, max_line_length=None, terms_on_new_line=True),
        mo.accordion({"Applicability conditions": mo.vstack(_conditions)}),
    ])
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


@app.cell
def _():
    return


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
