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

    Choose a native diagram, set its Euclidean kinematics, and watch sector
    decomposition turn it into a numerical expansion in $\epsilon$.
    HEPKit and FastSecDec provide the native calculation. Every numerical value
    below comes from your run.

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

    # Native: explicitly use the checked-out examples/hep asset directory.
    # Browser: explicitly fetch, verify and mount the exported assets before
    # importing builders. A browser filesystem is never assumed to contain them.
    notebook_location = mo.notebook_location()
    if sys.platform == "emscripten":
        import micropip
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
        asset_root = Path(notebook_location) / "hep"
        asset_description = f"Native assets: {asset_root}"
    if not (asset_root / "fastsecdec_inputs.py").is_file() or not (asset_root / "fixtures/fastsecdec/scalar.json").is_file():
        raise RuntimeError(f"Incomplete FastSecDec assets at {asset_root}")
    sys.path.insert(0, str(asset_root))
    import fastsecdec_inputs as builders
    import fastsecdec_views as views
    from symbolica.community import hepkit as hep
    fs = getattr(hep, "fastsecdec", None)
    return asset_description, builders, fs, views


@app.cell(hide_code=True)
def _(mo, views):
    _introduction = mo.md("""
    ## 1 · Choose the integral

    **Triangle:** equal internal mass $m$, two lightlike legs, invariant $s$.
    **Boxes:** massless lines, $s=s_{12}$ and $t=s_{23}$.
    **Sunset:** two coupled loop momenta, external $p^2=s$.
    The numerator examples retain their native routed scalar products.

    Edit the form, then **Apply inputs**. That prepares the graph only.
    Integration starts when you select **Run** below.
    """)
    input_form = mo.ui.batch(mo.Html("""
    <div style="display:grid;gap:1.25rem;padding:.5rem">
      <div>{example}</div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem">
        <div>{mass}</div><div>{s}</div><div>{t}</div>
      </div>
      <div>{max_order}</div>
      <details><summary style="cursor:pointer;padding:.4rem 0">Sampling settings · fixed QMC allocation</summary>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:1rem;padding-top:1rem">
          <div>{points}</div><div>{shifts}</div><div>{package_points}</div>
          <div>{rule}</div><div>{seed}</div>
        </div>
      </details>
    </div>
    """), {
        "example": mo.ui.dropdown(views.EXAMPLES, value="Massive triangle", label="Integral", allow_select_none=False),
        "mass": mo.ui.number(start=0.0, value=1.0, step=0.1, label="Mass m"),
        "s": mo.ui.number(value=-1.0, step=0.1, label="s < 0"),
        "t": mo.ui.number(value=-1.0, step=0.1, label="t < 0"),
        "max_order": mo.ui.dropdown({"Finite term · ε⁰": 0, "Through ε¹": 1}, value="Through ε¹", label="Highest requested order"),
        "points": mo.ui.dropdown({"1,024": 1024, "4,096": 4096, "16,384": 16384}, value="1,024", label="Points per sector and shift"),
        "shifts": mo.ui.dropdown({"4": 4, "8": 8, "16": 16}, value="8", label="Independent shifts"),
        "package_points": mo.ui.dropdown({"256": 256, "1,024": 1024}, value="1,024", label="Points per caller step"),
        "rule": mo.ui.dropdown({"Kuo 33002": "kuo_33002", "Kuo 38005": "kuo_38005", "Kuo 39101": "kuo_39101", "HKKN α=3": "hkkn_alpha3"}, value="Kuo 33002", label="Published lattice"),
        "seed": mo.ui.number(start=0, stop=2**32 - 1, step=1, value=20261005, label="Seed"),
    }).form(submit_button_label="Apply inputs", validate=views.validate_configuration)
    mo.vstack([_introduction, input_form])
    return (input_form,)


@app.cell(hide_code=True)
def _(builders, input_form, mo, views):
    submitted = input_form.value
    prepared = None
    if submitted is not None:
        try:
            prepared = views.prepare_input(builders, submitted)
            _diagram = prepared.diagram
            try:
                _drawing = mo.Html(f'<div style="max-width:480px;margin:auto">{_diagram.render()}</div>')
            except Exception as _render_error:
                # Keep the native renderer boundary. Never redraw the graph
                # using a second topology/layout implementation.
                _drawing = mo.callout(f"Native graph rendering is unavailable: {_render_error}", kind="warn")
            mo.output.replace(mo.vstack([
                mo.md(f"### {prepared.name}\n{_diagram.loop_count} loop(s) · highest requested order **{views.epsilon_label(submitted['max_order'])}**"),
                _drawing,
                mo.accordion({"Native scalar numerator and conventions": mo.vstack([
                    prepared.scalar_numerator(),
                    mo.md("The displayed expression includes the native numerator, projector, numerator prefactor and overall factor. It is not reinserted into the integral. The integral receives the original native owners."),
                ]), "Submitted configuration": views.table(mo, [{"parameter": key, "value": str(value)} for key, value in submitted.items()])}),
            ]))
        except Exception as _input_error:
            mo.output.replace(mo.callout(f"Input preparation failed: {_input_error}", kind="danger"))
    else:
        mo.output.replace(mo.callout("Apply the form to inspect your native diagram.", kind="info"))
    return prepared, submitted


@app.cell(hide_code=True)
def _(mo, views):
    state = views.RunState()
    run_button = mo.ui.button(value=0, on_click=lambda count: count + 1, label="Run", kind="success")
    cancel_button = mo.ui.button(value=0, on_click=lambda count: count + 1, label="Cancel", kind="danger")
    resume_button = mo.ui.button(value=0, on_click=lambda count: count + 1, label="Resume checkpoint")
    refresh = mo.ui.refresh(options=["250ms", "1s", "5s"], default_interval="250ms", label="Step / refresh")
    mo.vstack([
        mo.md("""
        ## 2 · Generate, then integrate

        Run builds the submitted integral and starts a **fixed** QMC allocation
        with Korobov-3 periodization. Each refresh advances at most one native
        package. Cancel stops future packages and saves accepted coverage;
        Resume restores that checkpoint, including numerical replay state.

        Generation and compilation are synchronous. Their native observer events
        feed the timeline; use marimo's interrupt control during those phases.
        A long native operation can delay repaint or interruption.
        """),
        mo.hstack([run_button, cancel_button, resume_button, refresh], justify="start", wrap=True),
    ])
    return cancel_button, refresh, resume_button, run_button, state


@app.cell(hide_code=True)
def _(cancel_button, fs, mo, prepared, refresh, resume_button, run_button, state, submitted, views):
    _actions = {"run": run_button.value, "cancel": cancel_button.value, "resume": resume_button.value, "tick": refresh.value}
    _changed = {key for key, value in _actions.items() if value != state.seen[key]}
    state.seen.update(_actions)
    if "cancel" in _changed:
        state.cancel()
    elif "run" in _changed:
        if prepared is None or submitted is None:
            state.message = "Apply valid inputs before Run."
        elif fs is None:
            state.message = "This Symbolica build has no FastSecDec bridge. Install the community showcase wheel."
        else:
            state.start(fs, prepared, submitted, display=lambda event: mo.output.replace(views.generation_view(mo, state)))
    elif "resume" in _changed:
        state.resume()
    elif "tick" in _changed:
        state.advance()
    _notice = mo.callout(state.message, kind="danger" if state.error else "info")
    _error = mo.md(f"`{state.error}`") if state.error else mo.md("")
    _run_config = mo.md("")
    if state.configuration is not None:
        _run_config = mo.md(f"**Active result:** {state.configuration['example']} · s={state.configuration['s']} · highest order {views.epsilon_label(state.configuration['max_order'])}. Changing the input form does not alter this run.")
    mo.output.replace(mo.vstack([_notice, _error, _run_config, views.generation_view(mo, state), views.result_view(mo, state)]))
    run_revision = (state.seen.copy(), state.active, state.snapshot)
    return (run_revision,)


@app.cell(hide_code=True)
def _(mo, run_revision, state):
    run_revision
    _downloads = []
    if state.kernels is not None and not state.active:
        _downloads.append(mo.download(lambda kernels=state.kernels: kernels.to_bytes(), filename="fastsecdec-kernels.bin", label="Download native kernels"))
    if state.checkpoint_bytes is not None and not state.active:
        _downloads.append(mo.download(state.checkpoint_bytes, filename="fastsecdec-checkpoint.json", label="Download native checkpoint"))
    if _downloads:
        mo.output.replace(mo.hstack(_downloads, justify="start"))
    return


@app.cell(hide_code=True)
def _(asset_description, fs, mo):
    mo.accordion({"Reading the result · provenance and limits": mo.md(f"""
    The table retains every native **signed epsilon order** and real/imaginary
    component. Full covariance is available in the details. The chart uses the
    highest signed requested order, not the largest absolute pole order.
    Error bars are one native standard error; observations from successive
    allocations share samples and are not independent trials.

    **0.1% target:** assess the highest requested coefficient only after complete
    production, using error ≤ 0.001 × |mean| for each of its native components.
    The separately reported native `meets` check covers the full vector. A zero
    mean with nonzero error does not pass a relative-only target. Completing
    planned points does not imply either target was reached; no work is added
    automatically.

    A caller cancellation is shown separately from the immutable native stop
    reason. Numerical errors retain their stage and accepted prefix. An absent
    estimate stays absent.

    Native wheels use `native_o2`; portable builds use `portable_interpreted`.
    The backend displayed with a result comes from its actual kernels. This
    notebook contains no prepared numerical results. The advanced
    [gg → HH example](https://github.com/alphal00p/fastSecDec/blob/main/examples/gghh_double_box/README.md)
    passed native feasibility: 61.285 s generation and 8.781 s integration on
    eight workers, with about 1.03% relative standard error for the finite
    coefficient. Browser cost remains unmeasured, so it stays outside this
    notebook's selector.

    **Assets:** {asset_description}. **Bridge present:** {fs is not None}.
    Browser exports require the explicit bundled wheel and fixtures described
    in `hep/FASTSECDEC_SHOWCASE.md`; a native wheel cannot run in Pyodide.
    """)})
    return


if __name__ == "__main__":
    app.run()
