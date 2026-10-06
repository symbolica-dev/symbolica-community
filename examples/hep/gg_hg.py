import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Two-loop Higgs + jet")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Two-loop Higgs + jet

    Compute the light-quark two-loop electroweak contribution to $gg \to Hg$:
    coherent W/Z form factors, the EW square, the infinite-top QCD square, and
    their interference. HEPKit supplies the model and tensor contractions.

    We load **40-digit starting values**, generated independently with native
    auxiliary-mass flow, then **compute the physical transport live**. No
    destination values or amplitudes are bundled. On one core, the browser run
    takes about five minutes for transport and 25 seconds for amplitude assembly.
    Rerunning with the same boundary cache reuses completed work.

    The cells below show the actual API calls. Change the exact kinematics and
    rerun to evaluate a nearby point from the accumulated cache. Mathematica and
    the original plugin are not needed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Setup and notebook helpers

    Browser installation and input checks are folded below; the calculation is visible.
    """)
    return


@app.cell(hide_code=True)
async def _():
    import asyncio
    import importlib.util
    from time import perf_counter
    import hashlib as _hashlib
    import json
    import sys as _sys
    from pathlib import Path
    import marimo as mo

    if _sys.platform == "emscripten":
        import micropip as _micropip
        from pyodide.http import pyfetch as _pyfetch

        _base = mo.notebook_location()
        _response = await _pyfetch(str(_base / "gg-hg-assets.json"))
        if _response.status != 200:
            raise RuntimeError("Export this notebook with scripts/export_gg_hg_wasm.py to include its wheel and scientific inputs.")
        _manifest = await _response.json()
        if _manifest["schema"] != "higgs-jet-browser-assets-v1":
            raise ValueError("Unsupported browser asset manifest")
        _directory = Path.cwd() / "gg_hg_inputs"
        for _name, _digest in _manifest["files"].items():
            _relative = Path(_name)
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
        _wheel_name = _manifest["wheel"]
        if Path(_wheel_name).name != _wheel_name or not _wheel_name.endswith(".whl"):
            raise ValueError("Invalid browser wheel path")
        _response = await _pyfetch(str(_base / _wheel_name))
        if _response.status != 200:
            raise RuntimeError(f"Cannot load browser wheel: HTTP {_response.status}")
        _payload = await _response.bytes()
        if _hashlib.sha256(_payload).hexdigest() != _manifest["wheel_sha256"]:
            raise ValueError("Browser wheel checksum mismatch")
        _wheel_path = _directory / _wheel_name
        _wheel_path.write_bytes(_payload)
        del _payload
        await _micropip.install("emfs:" + str(_wheel_path))
    else:
        _directory = Path(__file__).resolve().parent

    from symbolica import ComplexFloat, E, Float
    from symbolica.community.hep.integration import (
        AccuracyError,
        BoundaryCache,
        EvaluationOptions,
        HiggsJetAmplitude,
        HiggsJetFormFactorProjector,
        HiggsJetIntegralSystem,
    )

    from symbolica.community.hepkit import Model

    _spec = importlib.util.spec_from_file_location("gg_hg_boundaries", _directory / "gg_hg_boundaries.py")
    _support = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_support)
    load_boundary_bundle = _support.load_boundary_bundle
    data_directory = _directory / "data" / "gg_hg"
    return (
        AccuracyError, BoundaryCache, ComplexFloat, E, EvaluationOptions, Float, HiggsJetAmplitude,
        HiggsJetFormFactorProjector, HiggsJetIntegralSystem, Model, Path, asyncio,
        data_directory, json, load_boundary_bundle, mo, perf_counter,
    )


@app.cell
def _(HiggsJetAmplitude, Model):
    model = Model.standard_model()
    # Add symbolic W/Z and HEFT vertices; compute their form factors below.
    model = HiggsJetAmplitude.with_form_factor_vertices(model)
    return (model,)


@app.cell
def _(E):
    s = E("7173070292440521/111284741846000")  # Try adding E("1/100000").
    t = E("-12058167788971/339319588980")
    mh2 = E("1")
    masses = {"W": E("5399/13074"), "Z": E("7775/14631")}
    return masses, mh2, s, t


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load reusable boundaries

    The 48 planar and 61 nonplanar masters use the canonical measure
    $e^{2\gamma_E\epsilon}\prod_{j=1}^2 d^D k_j/(i\pi^{D/2})$, $D=4-2\epsilon$,
    with $m_V^2=\mu^2=1$ and the $+i0$ prescription. The loader checks basis,
    coordinates, root sheets, uncertainties and provenance before admitting seeds.
    A binary cache preserves all that evidence and every accepted intermediate point.
    """)
    return


@app.cell
def _(BoundaryCache, HiggsJetIntegralSystem, Path, data_directory, load_boundary_bundle):
    systems = {name: HiggsJetIntegralSystem(name) for name in ("planar", "nonplanar")}
    configurations = [(name, c) for name, system in systems.items() for c in system.configurations()]
    cache_directory = Path.cwd() / "gg_hg_visible_cache"
    if (cache_directory / "physical-boundaries.bin").exists():
        cache = BoundaryCache.load(cache_directory)
    else:
        cache = BoundaryCache()
    _seeds, _seed_results = load_boundary_bundle(data_directory, systems)
    cache.extend(_seeds)
    cache.save(cache_directory)
    return cache, cache_directory, configurations, systems


@app.cell
def _(EvaluationOptions):
    options = EvaluationOptions(digits=20, guard_digits=20, series_order=16, workers=1)
    return (options,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Transport in physical kinematics

    Each mass needs four crossed configurations. `evaluate` selects a compatible
    cached starting point, continues the master integrals, checks the requested
    accuracy and inserts accepted points into the same cache. We checkpoint after
    each configuration; rerunning an interrupted cell reuses completed results.
    """)
    return


@app.cell
async def _(asyncio, cache, cache_directory, configurations, masses, mh2, mo, options, perf_counter, s, systems, t):
    results, destinations = {}, {}
    _started = perf_counter()
    cache.save(cache_directory)
    _crossings = [(s, mh2 - s - t), (s, t), (mh2 - s - t, t), (t, s)]
    for _topology, _configuration in mo.status.progress_bar(configurations, title="Transporting masters"):
        _system = systems[_topology]
        _a, _b = _crossings[_configuration.permutation - 1]
        _mass = masses[_configuration.mass]
        _point = dict(zip(_system.coordinates, [_a / _mass, _b / _mass, mh2 / _mass]))
        destinations[_configuration.label] = _point
        _result = _system.evaluate(
            cache, _point, _configuration.root_sheets, options=options,
        )
        results[_configuration.label] = _result
        if not _result.cache_hit or _result.inserted_points:
            cache.save(cache_directory)
        await asyncio.sleep(0.05)  # Let the browser display progress between calls.
    transport_seconds = perf_counter() - _started
    mo.md(f"Transport: **{transport_seconds:.2f} s**; {sum(r.cache_hit for r in results.values())}/16 exact cache hits.")
    return destinations, results, transport_seconds


@app.cell(hide_code=True)
def _(mo, observables, results):
    # Marimo discovers dependencies in the body, not only in the signature.
    assert observables is not None  # Defer table RPCs until amplitude work is done.
    mo.ui.table([
        {"configuration": name, "checked digits": r.verified_digits,
         "steps": r.steps, "cached points added": r.inserted_points,
         "selected source": str(r.starting_coordinates)}
        for name, r in results.items()
    ], label="Transport accuracy and boundary reuse")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Project and assemble the coherent amplitude

    The kinematics-dependent projection includes $-1/[m_V^4(4\pi)^4]$ once.
    We use $\alpha=1/128$, $\alpha_s=118/1000$, and the incoming spin/color
    average $1/256$. Observables include no flux or phase-space factor.
    Finite-top QCD is outside this example.
    """)
    return


@app.cell
def _(HiggsJetAmplitude, HiggsJetFormFactorProjector, model):
    projector = HiggsJetFormFactorProjector()
    amplitude = HiggsJetAmplitude(model)
    return amplitude, projector


@app.cell
def _(configurations, masses, mh2, projector, results, s, t):
    form_factors = {}
    for _mass in ("W", "Z"):
        _blocks = {
            _topology: [results[c.label] for c in sorted(
                (c for name, c in configurations if name == _topology and c.mass == _mass),
                key=lambda c: c.permutation,
            )] for _topology in ("planar", "nonplanar")
        }
        form_factors[_mass] = projector.evaluate(s, t, mh2, masses[_mass], _blocks["planar"], _blocks["nonplanar"])
    return (form_factors,)


@app.cell
def _(AccuracyError, E, amplitude, form_factors, masses, mh2, model, s, t):
    _mw2, _mz2 = masses["W"], masses["Z"]
    parameters = {
        model.parameter("aEWM1").symbol: E("128"),
        model.parameter("aS").symbol: E("118/1000"),
        model.parameter("MZ").symbol: _mz2.sqrt(),
        model.parameter("Gf").symbol: E("𝜋") * E("1/128") * _mz2 / (E("2").sqrt() * _mw2 * (_mz2 - _mw2)),
    }
    _w, _z = form_factors["W"], form_factors["Z"]
    observables = amplitude.evaluate(
        s, t, mh2, _w.values, _z.values, _w.absolute_errors, _z.absolute_errors,
        parameters=parameters, digits=20, provenance=_w.provenance + "; " + _z.provenance,
    )
    if any(d is None or d < 20 for d in observables.verified_relative_digits.values()):
        raise AccuracyError("Refine the starting boundaries to reach 20 observable digits.")
    return observables, parameters


@app.cell(hide_code=True)
def _(ComplexFloat, E, Float, HiggsJetAmplitude, Model, data_directory, form_factors, json, masses, mh2, mo, model, observables, parameters, s, t):
    _result = observables
    # Comparison data is loaded only here, after the native result exists.
    _reference = json.loads((data_directory / "amplitude-validation.json").read_text())
    _mw2, _mz2 = E("5399/13074"), E("7775/14631")
    _same_point = ([s, t, mh2] == [E(value) for value in _reference["physical_s_t_MH_squared"]]
                   and masses == {"W": _mw2, "Z": _mz2})
    _same_observables = _same_point and parameters == {
        model.parameter("aEWM1").symbol: E("128"),
        model.parameter("aS").symbol: E("118/1000"),
        model.parameter("MZ").symbol: _mz2.sqrt(),
        model.parameter("Gf").symbol: E("𝜋") * E("1/128") * _mz2 / (E("2").sqrt() * _mw2 * (_mz2 - _mw2)),
    } and model.to_json() == HiggsJetAmplitude.with_form_factor_vertices(Model.standard_model()).to_json()
    _factor_reference = {block["mass"]: block["values"] for block in _reference["form_factors"]}
    _factor_rows = []
    for _mass in ("W", "Z"):
        _factors = form_factors[_mass]
        for _index, (_value, _error, _digits, _expected) in enumerate(zip(
            _factors.values, _factors.absolute_errors, _factors.verified_relative_digits,
            _factor_reference[_mass], strict=True,
        ), start=1):
            _reference_value = ComplexFloat(_expected["real"], _expected["imaginary"], decimal_digits=100)
            _reference_error = Float(_expected["absolute_error"], decimal_digits=100)
            _factor_rows.append({
                "form factor": f"{_mass}{_index}",
                "native result": f"{_value:.20e}",
                "propagated absolute uncertainty": f"{_error:.20e}",
                "achieved relative digits": _digits,
                "reference at recorded point": f"{_reference_value:.20e}" if _same_point else "different kinematics",
                "reference absolute uncertainty": f"{_reference_error:.20e}" if _same_point else "—",
                "absolute difference": f"{abs(_value - _reference_value):.20e}" if _same_point else "—",
            })
    _rows = []
    for _name, _value in _result.values.items():
        _ref = Float(_reference["expected_observables"][_name], decimal_digits=80)
        _accuracy = _reference["reference_accuracy"][_name]
        _ref_error = Float(_accuracy["input_absolute_error"], decimal_digits=80) + Float(
            _accuracy["rounding_absolute_error"], decimal_digits=80
        )
        _rows.append({
            "observable": _name,
            "native result": f"{_value:.20e}",
            "propagated absolute uncertainty": f"{_result.absolute_errors[_name]:.20e}",
            "achieved relative digits": _result.verified_relative_digits[_name],
            "reference at recorded point": f"{_ref:.20e}" if _same_observables else "different inputs",
            "reference absolute uncertainty": f"{_ref_error:.20e}" if _same_observables else "—",
            "absolute difference": f"{abs(_value - _ref):.20e}" if _same_observables else "—",
        })
    mo.vstack([
        mo.md("## W/Z form factors"),
        mo.md(f"Current exact kinematics: s = {s}, t = {t}, m_H² = {mh2}."),
        mo.ui.table(_factor_rows, label="Native W/Z form factors and propagated uncertainties"),
        mo.md("## Coherent observables"),
        mo.ui.table(_rows, label="Coherent observables and propagated uncertainties"),
        mo.md("The archived reference has its own input-accuracy limits. Extra printed digits do not strengthen its uncertainty. Nearby evaluations use new kinematics-dependent projections; the original-point comparison is then omitted."),
        mo.accordion({"Numerical provenance": mo.plain_text(_result.provenance)}),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Binary restart and exact cache reuse

    Reload the cache and repeat one configuration: no ODE steps are needed.
    Editing `s` above reruns transport and projection from nearby cached points.
    Browser files live for this page's lifetime; native files survive reopening.
    """)
    return


@app.cell
def _(BoundaryCache, cache, cache_directory, configurations, destinations, options, results, systems):
    cache.save(cache_directory)
    reloaded_cache = BoundaryCache.load(cache_directory)
    _topology, _configuration = configurations[0]
    repeated = systems[_topology].evaluate(
        reloaded_cache, destinations[_configuration.label], _configuration.root_sheets, options=options,
    )
    assert repeated.cache_hit and repeated.steps == 0
    assert repeated.coefficients == results[_configuration.label].coefficients
    return reloaded_cache, repeated


@app.cell(hide_code=True)
def _(mo, repeated):
    mo.md(f"Exact repeated cache hit: **{repeated.cache_hit}**, ODE steps: **{repeated.steps}**.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The physical families and exact canonical maps follow
    [Becchetti, Moriello and Schweitzer](https://arxiv.org/abs/2112.07578).
    All 4,360 transport coefficients and the coherent observables have independent
    [native validation](https://github.com/alphal00p/RustFlow/blob/main/docs/python-notebook-status.md).
    Numerical references above are loaded only after evaluation. The EW-square
    reference supports 19 relative comparison digits; printed precision is not
    an accuracy claim.

    Native boundary regeneration is available through
    `system.generate_boundary(evaluator, cache, configuration.start, configuration.root_sheets)`;
    the separate `gg_hg_acceptance.py` runner exercises empty-cache generation,
    cancellation, forced recomputation and precision refinement. That substantial
    calculation is deliberately separate from this live transport demonstration.
    """)
    return


@app.cell
def _(observables, repeated):
    from symbolica import get_citations

    _ = observables, repeated  # Collect after numerical work has registered its citations.
    get_citations()
    return


if __name__ == "__main__":
    app.run()
