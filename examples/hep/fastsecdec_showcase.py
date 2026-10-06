import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="FastSecDec · from graph to Laurent vector")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="font-size:.78rem;letter-spacing:.16em;color:#7c699d">HEPKIT / FASTSECDEC</div>

    # From a Feynman graph to a Laurent vector

    Choose a native diagram, set its kinematics, and watch sector
    decomposition turn it into a numerical expansion in $\epsilon$.
    HEPKit and FastSecDec provide the native calculation. Every numerical value
    below comes from your explicit calculation. **Generate** prepares sectors and kernels; **Integrate** starts sampling separately.

    $D=4-2\epsilon$, with measure
    $\prod_\ell d^Dk_\ell/(i\pi^{D/2})$. Native graph weights and the scalar
    numerator enter **once**. There are no implicit Euler-gamma, scale or
    $4\pi$ factors.
    """)
    return


@app.cell(hide_code=True)
async def _(mo):
    import hashlib
    import io
    import json
    from pathlib import Path
    import sys
    import zipfile

    # Native: use the helper modules and input fixtures beside this notebook.
    # Browser: explicitly fetch, verify and mount the exported assets before
    # importing builders. A browser filesystem is never assumed to contain them.
    notebook_location = mo.notebook_location()
    browser_runtime = sys.platform == "emscripten"
    interrupt_isolated = False
    if browser_runtime:
        import micropip
        from js import globalThis

        interrupt_isolated = bool(globalThis.crossOriginIsolated)
        from pyodide.http import pyfetch

        _base = str(notebook_location).rstrip("/")
        _manifest_response = await pyfetch(f"{_base}/public/fastsecdec/manifest.json")
        if not _manifest_response.ok:
            raise RuntimeError("Missing FastSecDec asset manifest. Use the showcase export helper.")
        _manifest = await _manifest_response.json()
        asset_root = Path("/fastsecdec-showcase")
        asset_root.mkdir(exist_ok=True)
        for _kind in ("wheel", "assets"):
            _entry = _manifest[_kind]
            _response = await pyfetch(f"{_base}/public/fastsecdec/{_entry['filename']}")
            if not _response.ok:
                raise RuntimeError(f"Unable to fetch {_kind} from the explicit asset bundle")
            _data = await _response.bytes()
            if hashlib.sha256(_data).hexdigest() != _entry["sha256"]:
                raise RuntimeError(f"FastSecDec {_kind} hash does not match the manifest")
            if _kind == "wheel":
                _wheel_path = asset_root / _entry["filename"]
                _wheel_path.write_bytes(_data)
                await micropip.install(f"emfs:{_wheel_path}")
            else:
                with zipfile.ZipFile(io.BytesIO(_data)) as _archive:
                    if set(_archive.namelist()) != set(_entry["files"]):
                        raise RuntimeError("Asset archive differs from its explicit file manifest")
                    for _name in _archive.namelist():
                        _relative = Path(_name)
                        if _relative.is_absolute() or ".." in _relative.parts:
                            raise RuntimeError("Invalid asset archive path")
                        _destination = asset_root / _relative
                        _destination.parent.mkdir(parents=True, exist_ok=True)
                        _destination.write_bytes(_archive.read(_name))
        asset_description = "Verified browser bundle, mounted at /fastsecdec-showcase"
    else:
        if notebook_location is None:
            raise RuntimeError("Open this notebook with marimo so its asset location is explicit.")
        asset_root = Path(notebook_location)
        asset_description = f"Native assets: {asset_root}"
    if not (asset_root / "showcase/inputs.py").is_file() or not (asset_root / "fixtures/fastsecdec/scalar.json").is_file():
        raise RuntimeError(f"Incomplete FastSecDec assets at {asset_root}")
    sys.path.insert(0, str(asset_root))
    from showcase import inputs as builders
    from showcase import state as workflow, generation, integration, sectors, presentation, report
    from showcase import gghh as gghh_builder
    from symbolica.community import hepkit as hep
    return asset_description, browser_runtime, builders, gghh_builder, interrupt_isolated, workflow, generation, integration, sectors, presentation, report


@app.cell
def _(mo):
    # The canonical HEPKit API. Importing it starts no scientific work.
    from symbolica.community import hepkit as _hep
    if hasattr(_hep, "sector_decomposition"):
        import symbolica.community.hepkit.sector_decomposition as sd
    else:
        sd = None
    mo.show_code()
    return (sd,)


@app.cell(hide_code=True)
def _(mo, presentation):
    _choices = dict(presentation.EXAMPLES)
    _choices["gg → HH · extended run"] = "gghh"
    problem = mo.ui.dropdown(_choices, value="Massive triangle", label="Integral", allow_select_none=False)
    mo.vstack([mo.md("## 1 · Choose the input"), problem])
    return (problem,)


@app.cell(hide_code=True)
def _(browser_runtime, mo, problem):
    physics_controls = None
    if problem.value == "gghh":
        _input_panel = mo.vstack([
            mo.md(r"""### $gg\to HH$ · fixed physical point
One HEPKit-generated top double box with an internal gluon and $(+,+)$ helicities. This single projected diagram is not the full gauge-invariant amplitude."""),
            mo.hstack([
                mo.stat(label="Energy √s", value="300 GeV"), mo.stat(label="Higgs mass", value="125 GeV"),
                mo.stat(label="Top mass", value="172.5 GeV"), mo.stat(label="cos θ", value="4/5"),
            ], widths="equal", wrap=True),
            mo.accordion({"Conventions": mo.md(r"""
            Native HEPKit algebra closes the unnormalized color projection $\delta_{ab}$;
            shared GammaLoop wavefunctions supply the incoming helicities. Internal
            algebra retains $D=4-2\epsilon$ and external states are four-dimensional.
            Feynman gauge, generated weights and couplings, no spin/color average.

            The ordinary domain guard remains active. Generate prepares through the
            finite coefficient. Choose integration settings separately below.
            Completing an allocation need not meet the 0.1% target. Browser generation
            and interpreted integration can be substantially slower than native execution.
            """)}),
            mo.callout("Optional extended browser run: one CPU, with portable interpreted integration. Generate may take several minutes; completion and browser cost are not yet validated. No calculation begins until you select Generate.", kind="warn") if browser_runtime else mo.md(""),
        ])
    else:
        _fields = {"s": mo.ui.number(value=-1.0, step=0.1, label="s < 0")}
        if problem.value == "triangle":
            _fields["mass"] = mo.ui.number(start=0.0, value=1.0, step=0.1, label="Mass m")
        if problem.value in {"box", "rank_two_box"}:
            _fields["t"] = mo.ui.number(value=-1.0, step=0.1, label="t < 0")
        _fields["max_order"] = mo.ui.dropdown({"Finite term · ε⁰": 0, "Through ε¹": 1}, value="Through ε¹", label="Highest requested order")
        physics_controls = mo.ui.batch(mo.Html('<div style="display:flex;gap:1.5rem;flex-wrap:wrap">' + ''.join('<div>{' + name + '}</div>' for name in _fields) + '</div>'), _fields)
        _input_panel = mo.vstack([physics_controls,
                   mo.md("Editing these controls does not change an existing generated input or result. Select Generate to bind new physics; select Integrate separately to start sampling.")])
    _input_panel
    return (physics_controls,)


@app.cell(hide_code=True)
def _(mo):
    integration_method = mo.ui.dropdown({"Randomized lattice QMC": "qmc", "Havana Monte Carlo · sector importance": "havana_discrete_mc"}, value="Randomized lattice QMC", label="Integration method", allow_select_none=False)
    integration_method
    return (integration_method,)


@app.cell(hide_code=True)
def _(integration_method, mo, problem):
    _presets = {"Quick exploration": "quick"}
    if problem.value == "gghh":
        _presets["Native gg→HH accuracy observation · 15.7M points"] = "gghh_accuracy"
    allocation_preset = mo.ui.dropdown(_presets, value="Quick exploration", label="Integration preset", allow_select_none=False)
    allocation_preset if integration_method.value == "qmc" else mo.md("Havana uses global batches with native sector and continuous-grid importance sampling. A completed pilot must be frozen explicitly before production.")
    return (allocation_preset,)


@app.cell(hide_code=True)
def _(allocation_preset, browser_runtime, integration_method, mo, presentation):
    _preset = presentation.QMC_PRESETS[allocation_preset.value]
    _points = {f"{value:,}": value for value in (1024, 4096, 16384, 32768, 65536)}
    _shifts = {str(value): value for value in (4, 8, 16, 32, 64)}
    _rules = {"Kuo 33002": "kuo_33002", "Kuo 38005": "kuo_38005", "Kuo 39101": "kuo_39101", "HKKN α=3": "hkkn_alpha3"}
    if integration_method.value == "qmc":
        allocation_controls = mo.ui.batch(mo.Html('''
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:1rem">
        <div>{points}</div><div>{shifts}</div><div>{package_points}</div><div>{rule}</div><div>{periodization}</div><div>{seed}</div>
        </div>'''), {
            "points": mo.ui.dropdown(_points, value=f"{_preset['points']:,}", label="Points per sector and shift"),
            "shifts": mo.ui.dropdown(_shifts, value=str(_preset["shifts"]), label="Independent shifts"),
            "package_points": mo.ui.dropdown({"256": 256, "1,024": 1024}, value="1,024", label="Points per caller step"),
            "rule": mo.ui.dropdown(_rules, value=next(key for key, value in _rules.items() if value == _preset["rule"]), label="Published lattice"),
            "periodization": mo.ui.dropdown({"Korobov-3": "korobov3", "Korobov-2": "korobov2", "None": "none"}, value="Korobov-3", label="Periodization"),
            "seed": mo.ui.number(start=0, stop=2**32 - 1, step=1, value=_preset["seed"], label="Seed"),
        })
    else:
        _mc_fields = {
            "pilot_points": mo.ui.dropdown({"256": 256, "1,024": 1024, "4,096": 4096}, value="1,024", label="Global pilot points per batch"),
            "pilot_batches": mo.ui.number(start=2, stop=1024, step=1, value=4, label="Pilot batches per epoch"),
            "points_per_batch": mo.ui.dropdown({"1,024": 1024, "4,096": 4096, "16,384": 16384, "32,768": 32768}, value="4,096", label="Global production points per batch"),
            "batches": mo.ui.number(start=2, stop=65536, step=1, value=64, label="Production batches"),
            "seed": mo.ui.number(start=0, stop=2**32-1, step=1, value=20261007, label="Seed"),
            "bins": mo.ui.number(start=2, stop=256, step=1, value=32, label="Continuous bins per axis"),
            "minimum_probability_density": mo.ui.number(start=0.0001, stop=1, step=0.01, value=0.01, label="Minimum continuous probability density"),
            "maximum_sector_probability_ratio": mo.ui.number(start=1, stop=10000, step=1, value=100, label="Maximum sector probability ratio"),
        }
        allocation_controls = mo.ui.batch(mo.Html('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:1rem">' + ''.join('<div>{' + key + '}</div>' for key in _mc_fields) + '</div>'), _mc_fields)
    _allocation_view = [mo.accordion({"Integration settings · edit the chosen allocation": allocation_controls})]
    if integration_method.value == "qmc" and allocation_preset.value == "gghh_accuracy":
        _allocation_view.append(mo.callout("This 15,728,640-point gg→HH allocation reached 0.00937% finite-term relative standard error in the native eight-worker CLI observation (377.530 s). That is an observed result, not guaranteed precision or a browser runtime. The notebook uses one caller worker; sampling starts only with Integrate.", kind="warn" if browser_runtime else "info"))
    mo.vstack(_allocation_view)
    return (allocation_controls,)


@app.cell(hide_code=True)
def _(allocation_controls, integration_method, physics_controls, problem):
    if problem.value == "gghh":
        draft = {"example": "gghh", "max_order": 0, "method": integration_method.value, **allocation_controls.value}
    else:
        draft = {"example": problem.value, "method": integration_method.value, **physics_controls.value, **allocation_controls.value}
    return (draft,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### The scientific calls

    These visible functions are the code the controls execute. Defining them
    performs no generation, compilation or sampling. **Generate** invokes the
    diagram method and compiles its result; **Integrate** creates a session from
    those same kernels. Native observers supply the live dashboard.

    An explicit family member uses the same interface:
    `family.sector_decompose(powers=[...], numerator=weighted_scalar, regulator=eps)`.
    Its powers follow the native denominator order; zero powers omit a line and
    negative powers put that denominator in the numerator. A family carries no
    diagram weight or projector implicitly.
    """)
    return


@app.cell
def _(mo):
    def decompose_input(prepared, max_order, observer):
        # Native diagram, kinematics and Symbolica values are passed directly.
        diagram = prepared.diagram
        arguments = dict(prepared.integral_arguments())
        for name in ("diagram", "kinematics", "regulator", "dimension"):
            arguments.pop(name)
        generation_options = getattr(prepared, "generation_arguments", lambda: {})()
        return diagram.sector_decompose(
            kinematics=prepared.kinematics,
            regulator=prepared.regulator,
            dimension=prepared.dimension,
            max_order=max_order, observer=observer,
            **arguments, **generation_options,
        )

    def compile_sectors(generated, observer):
        # Compilation completes before Integrate can create a session.
        return generated.compile(observer=observer)

    mo.show_code()
    return compile_sectors, decompose_input


@app.cell
def _(mo, sd):
    def create_session(kernels, configuration):
        if configuration.get("method", "qmc") == "havana_discrete_mc":
            settings = sd.HavanaDiscreteSettings(
                points_per_batch=configuration["pilot_points"],
                batches=configuration["pilot_batches"],
                seed=configuration["seed"], bins=configuration["bins"],
                minimum_probability_density=configuration["minimum_probability_density"],
                maximum_sector_probability_ratio=configuration["maximum_sector_probability_ratio"],
            )
            # The pilot is trained explicitly, then frozen into production.
            return kernels.mc_session(settings, pilot=True)
        settings = sd.QmcSettings(
            points=configuration["points"], shifts=configuration["shifts"],
            seed=configuration["seed"], package_points=configuration["package_points"],
            rule=configuration["rule"], periodization=configuration.get("periodization", "korobov3"),
        )
        return kernels.session(settings)

    def advance_session(session, method):
        # The notebook owns the loop; each active refresh accepts at most one unit.
        if method == "havana_discrete_mc":
            return session.step(max_batches=1)
        return session.step(max_packages=1)

    mo.show_code()
    return advance_session, create_session


@app.cell(hide_code=True)
def _(mo, workflow):
    run_state = workflow.RunState(message="Choose the input, then Generate. No scientific work starts automatically.")
    get_sampling_active, set_sampling_active = mo.state(False)
    generate_button = mo.ui.button(value=0, on_click=lambda n: n+1, label="Generate", kind="success")
    integrate_button = mo.ui.button(value=0, on_click=lambda n: n+1, label="Integrate", kind="success")
    new_integration_button = mo.ui.button(value=0, on_click=lambda n: n+1, label="New integration")
    cancel_button = mo.ui.button(value=0, on_click=lambda n: n+1, label="Cancel", kind="danger")
    resume_button = mo.ui.button(value=0, on_click=lambda n: n+1, label="Resume")
    adapt_button = mo.ui.button(value=0, on_click=lambda n: n+1, label="Adapt another pilot")
    freeze_button = mo.ui.button(value=0, on_click=lambda n: n+1, label="Freeze production", kind="success")
    return adapt_button, cancel_button, freeze_button, generate_button, get_sampling_active, integrate_button, new_integration_button, resume_button, run_state, set_sampling_active


@app.cell(hide_code=True)
def _(get_sampling_active, mo):
    # The browser must have no automatic timer while synchronous generation runs.
    # Recreate this widget only when sampling starts or stops, never per package.
    refresh = mo.ui.refresh(options=["250ms", "1s", "5s"], default_interval="250ms", label="Caller step / refresh") if get_sampling_active() else None
    return (refresh,)


@app.cell(hide_code=True)
def _(adapt_button, browser_runtime, cancel_button, freeze_button, generate_button, integrate_button, interrupt_isolated, mo, new_integration_button, refresh, resume_button):
    mo.vstack([
        mo.md("## 2 · Generate → inspect → integrate"),
        mo.hstack([generate_button, integrate_button, cancel_button, resume_button, new_integration_button], justify="start", wrap=True),
        refresh if refresh is not None else mo.md("Automatic stepping is off. Integrate or Resume starts it."),
        mo.accordion({"Havana pilot actions": mo.vstack([
            mo.md("Integrate starts a pilot. After its allocation completes, choose another adaptation epoch or freeze both grids and start production. Pilot statistics never enter the production estimate. These actions do nothing during an active allocation."),
            mo.hstack([adapt_button, freeze_button], justify="start", wrap=True),
        ])}),
        mo.md("Generate includes native compilation and stops before sampling. Integrate starts the selected allocation from ready kernels. Cancel retains accepted coverage; Resume continues it. Havana pilots stay in memory until production is frozen."),
        mo.accordion({"Execution and cancellation": mo.vstack([
            mo.md("Cancel pauses integration between QMC packages or global Havana batches. Production coverage can be saved; pilots retain the same native session in memory. In marimo's editor, Stop (interrupt) / Ctrl-I (Cmd-I on macOS) requests KeyboardInterrupt. Generation checks it at native callback boundaries; integration checks every 256 points. Long algebra operations between checks can delay interruption. An interrupted package or batch does not enter accepted coverage. Only explicit Integrate, Resume or pilot actions enable further work; no allocation grows automatically."),
            mo.callout(
                "Browser interrupt is available in this isolated editor. Use Stop (interrupt) or Ctrl-I (Cmd-I on macOS); long native algebra calls may delay the response."
                if interrupt_isolated and mo.app_meta().mode == "edit" else
                "This view or host has no active KeyboardInterrupt control. Cancel is processed between integration packages. For browser interruption, export with --mode edit and use the documented isolated server. Reloading the page discards unsaved in-memory work.",
                kind="info" if interrupt_isolated and mo.app_meta().mode == "edit" else "warn",
            ) if browser_runtime else mo.md(""),
        ])}),
    ])
    return


@app.cell(hide_code=True)
def _(adapt_button, advance_session, allocation_controls, builders, cancel_button, compile_sectors, create_session, decompose_input, draft, freeze_button, generate_button, generation, gghh_builder, integrate_button, integration, mo, new_integration_button, presentation, refresh, resume_button, run_state, sd, set_sampling_active):
    _was_active = run_state.active
    _actions = {"generate": generate_button.value, "integrate": integrate_button.value,
                "cancel": cancel_button.value, "resume": resume_button.value,
                "adapt": adapt_button.value, "freeze": freeze_button.value,
                "new": new_integration_button.value, "tick": refresh.value if refresh is not None else ""}
    _changed = {key for key, value in _actions.items() if value != run_state.seen[key]}
    run_state.seen.update(_actions)
    if "cancel" in _changed:
        run_state.cancel()
    elif "generate" in _changed:
        _validation = None if draft["example"] == "gghh" else presentation.validate_configuration(draft)
        if _validation:
            run_state.message = _validation
        elif sd is None:
            run_state.message = "Install a community wheel with the FastSecDec API."
        else:
            _configuration = dict(draft)
            _prepare = (lambda observer: gghh_builder.prepare(observer=observer)) if draft["example"] == "gghh" else (lambda observer: builders.prepare(_configuration))
            run_state.generate(decompose_input, _prepare, _configuration, compile=compile_sectors,
                display=lambda event: mo.output.replace(generation.generation_view(mo, run_state)),
                input_display=lambda event: mo.output.replace(mo.vstack([
                    mo.md("**HEPKit diagram generation**"), presentation.table(mo, [{"native stage": event.stage, "completed": event.completed, "total": event.total}]),
                ])))
            if run_state.prepared is not None:
                try:
                    run_state.drawing = mo.as_html(run_state.prepared.diagram.render())
                except Exception as _drawing_error:
                    run_state.drawing = mo.callout(f"Native graph rendering is unavailable: {_drawing_error}", kind="warn")
    elif "new" in _changed:
        run_state.new_integration()
    elif "integrate" in _changed:
        # Draft physics never replaces generated provenance. Only the matching
        # case's explicit allocation controls are admitted at Integrate.
        _settings = None
        if run_state.configuration is not None and draft["example"] == run_state.configuration["example"]:
            _settings = {"method": draft["method"], **allocation_controls.value}
        run_state.integrate(create_session, _settings)
    elif "resume" in _changed:
        run_state.resume()
    elif "adapt" in _changed:
        run_state.pilot_action()
    elif "freeze" in _changed:
        run_state.pilot_action(freeze=True)
    elif "tick" in _changed:
        run_state.advance(advance_session)
    if run_state.active != _was_active:
        set_sampling_active(run_state.active)
    _content = [mo.callout(run_state.message, kind="danger" if run_state.error else "info")]
    if run_state.error:
        _content.append(mo.md(f"`{run_state.error}`"))
    if run_state.checkpoint_warning:
        _content.append(mo.callout(run_state.checkpoint_warning, kind="warn"))
    if run_state.configuration is not None:
        _c = run_state.configuration
        _content.append(mo.md(f"**Generated input:** {_c['example']} · requested $\\epsilon^{{{_c['max_order']}}}$. Draft edits do not alter this native owner."))
    _content.extend([generation.generation_view(mo, run_state), integration.result_view(mo, run_state), integration.previous_result_view(mo, run_state)])
    mo.output.replace(mo.vstack(_content))
    run_revision = (run_state.phase, run_state.snapshot, len(run_state.events), run_state.generated)
    return (run_revision,)


@app.cell(hide_code=True)
def _(generation, mo, run_revision, run_state, sectors):
    run_revision
    mo.vstack([
        mo.md("## 3 · Native input and generated sectors"),
        generation.input_view(mo, run_state),
        sectors.overview(mo, run_state.generated, run_state.kernels),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    sector_index = mo.ui.number(start=0, value=0, step=1, label="Sector index")
    coefficient_index = mo.ui.number(start=0, value=0, step=1, label="Coefficient index")
    alias_page = mo.ui.number(start=0, value=0, step=1, label="Alias page")
    chart_index = mo.ui.number(start=0, value=0, step=1, label="Chart index")
    term_index = mo.ui.number(start=0, value=0, step=1, label="Mapped term index")
    inspect_button = mo.ui.button(value=0, on_click=lambda n: n+1, label="Inspect sector")
    numerator_button = mo.ui.button(value=0, on_click=lambda n: n+1, label="Inspect weighted numerator")
    mo.vstack([mo.hstack([sector_index, coefficient_index, alias_page, chart_index, term_index], justify="start", wrap=True), mo.hstack([inspect_button, numerator_button], justify="start", wrap=True)])
    return alias_page, chart_index, coefficient_index, inspect_button, numerator_button, sector_index, term_index


@app.cell(hide_code=True)
def _(alias_page, chart_index, coefficient_index, inspect_button, mo, numerator_button, run_revision, run_state, sector_index, sectors, term_index):
    run_revision
    _actions = {"inspect": inspect_button.value, "numerator": numerator_button.value}
    _changed = {key for key, value in _actions.items() if value != run_state.seen[key]}
    run_state.seen.update(_actions)
    if "inspect" in _changed:
        try:
            if run_state.generated is None:
                raise ValueError("Generate an input first")
            run_state.inspected_sector = sectors.detail(mo, run_state.generated, int(sector_index.value), int(coefficient_index.value), int(alias_page.value), int(chart_index.value), int(term_index.value), run_state.kernels)
        except Exception as _inspection_error:
            run_state.inspected_sector = mo.callout(f"Sector inspection failed: {_inspection_error}", kind="warn")
    elif "numerator" in _changed:
        try:
            if run_state.prepared is None:
                raise ValueError("Generate an input first")
            run_state.numerator_view = run_state.prepared.scalar_numerator()
        except Exception as _numerator_error:
            run_state.numerator_view = mo.callout(f"Native numerator inspection failed: {_numerator_error}", kind="warn")
    mo.vstack([item for item in (run_state.inspected_sector, run_state.numerator_view) if item is not None])
    return


@app.cell(hide_code=True)
def _(asset_description, mo, report, run_revision, run_state):
    run_revision
    _downloads = []
    if run_state.previous_report_bytes is not None:
        _downloads.append(mo.download(run_state.previous_report_bytes, filename="fastsecdec-previous-report.json", label="Download previous allocation report"))
    if run_state.configuration is not None and not run_state.active:
        _downloads.append(mo.download(lambda: report.report_bytes(run_state), filename="fastsecdec-report.json", label="Download run report"))
    if run_state.kernels is not None and not run_state.active:
        _downloads.append(mo.download(lambda: run_state.kernels.to_bytes(), filename="fastsecdec-kernels.bin", label="Download native kernels"))
    if run_state.checkpoint_bytes is not None and not run_state.active:
        _downloads.append(mo.download(run_state.checkpoint_bytes, filename="fastsecdec-checkpoint.json", label="Download checkpoint"))
    mo.vstack([
        mo.hstack(_downloads, justify="start") if _downloads else mo.md(""),
        mo.accordion({"Reading the result · provenance and limits": mo.md(f"""
        Full signed Laurent vectors and real/imaginary covariance come from native
        snapshots. Missing uncertainty stays missing. The highest requested order
        target is 0.1%; complete production is required. Completing an allocation
        does not guarantee that target. Successive history observations share samples.

        Generated sector coefficients may be complex, while compiled estimates split
        their real and imaginary components. Native backend labels distinguish O2
        kernels from the portable interpreter. Interactive active time includes
        refresh waits; native worker time is separate. No responsiveness guarantee
        or prepared numerical output is supplied.

        **Assets:** {asset_description}. gg → HH is an optional extended run;
        browser completion and cost remain unvalidated. See `README.md` for
        source/build provenance, explicit browser asset mounting and interruption.
        """)}),
    ])
    return


@app.cell
def _(mo, run_revision):
    from symbolica import get_citations

    run_revision  # Update the process bibliography as the computation advances.
    citations = get_citations()
    mo.vstack([
        mo.md("## References"),
        *[mo.as_html(citation) for citation in citations],
        mo.download("\n\n".join(citation.to_bibtex() for citation in citations).encode(),
                    filename="fastsecdec-references.bib", label="Download BibTeX"),
    ])
    return (citations,)


if __name__ == "__main__":
    app.run()
