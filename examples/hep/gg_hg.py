import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Two-loop Higgs + jet")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Two-loop Higgs + jet

    Evaluate the light-quark mixed QCD–electroweak contribution to
    $g g\to H g$, including coherent $W/Z$ form factors, its square, and
    interference with the infinite-top effective QCD amplitude.

    The integral evaluator uses exact physical propagators and the published,
    symbolically certified basis maps and differential equations for 48 planar
    and 61 nonplanar masters. It computes boundary values through native auxiliary-mass flow
    and recursive boundary reductions. The same series solver then transports
    these values in physical kinematics. Accepted intermediate points stay in a
    reusable binary boundary cache.

    **Run the stages in order.** Boundary generation is a substantial native
    computation. Opening this notebook starts no integral evaluation. Cancellation
    retains completed epsilon samples and completed boundary configurations.
    Native typed errors remain visible if a reduction or accuracy check fails.
    No Mathematica installation or plugin runtime is used.

    **Current validation:** the native cold calculation generated all sixteen
    physical starting configurations from empty numerical caches. All 4,360
    transport coefficients passed 20-digit mixed absolute/relative comparisons.
    Eight W/Z form factors and three observables agreed within the retained
    native and reference uncertainties; the EW-square reference supports
    19 relative comparison digits. Independent 40-digit regeneration and restart
    acceptance are tracked in
    [the validation report](https://github.com/alphal00p/RustFlow/blob/main/docs/python-notebook-status.md).
    Published numerical reference values are comparison data only.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Setup and notebook helpers

    Imports, the stage controller and controls are folded below. Opening the
    notebook only loads exact inputs and existing caches; use the controls to
    start numerical work.
    """)
    return


@app.cell(hide_code=True)
def _():
    import importlib.util
    import json
    from pathlib import Path
    import marimo as mo
    from symbolica import ComplexFloat, E, Float
    from symbolica.community.hep.integration import (
        BoundaryCache,
        EvaluationOptions,
        IntegralEvaluator,
        KinematicTransport,
    )

    _directory = Path(__file__).resolve().parent
    _spec = importlib.util.spec_from_file_location("gg_hg_support", _directory / "gg_hg_support.py")
    _support = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_support)
    CalculationSession = _support.CalculationSession
    data_directory = _directory / "data" / "gg_hg"
    return (
        BoundaryCache, CalculationSession, ComplexFloat, E, EvaluationOptions, Float,
        IntegralEvaluator, KinematicTransport, Path, data_directory, json, mo,
    )


@app.cell(hide_code=True)
def _(CalculationSession, Path, data_directory):
    session = CalculationSession(
        data_directory / "native-model.json",
        Path.cwd() / "gg_hg_cache",
        seed_digits=30,
        digits=20,
        workers=1,
        boundary_workers=1,
    )
    return (session,)


@app.cell(hide_code=True)
def _(mo, session):
    mo.md(rf"""
    ## Exact inputs and conventions

    A single native HEPKit model is shared by the effective diagrams and tensor
    contractions. The physical point, in units $m_H^2=1$, is
    $$s={session.point[0]},\qquad t={session.point[1]},\qquad u=m_H^2-s-t.$$
    We use $m_W^2=5399/13074$, $m_Z^2=7775/14631$, $\alpha=1/128$ and
    $\alpha_s=118/1000$. Every input is an exact Symbolica expression.

    The canonical integral measure is

    $$e^{{2\gamma_E\epsilon}}\prod_{{j=1}}^2\frac{{d^D k_j}}{{i\pi^{{D/2}}}},\qquad D=4-2\epsilon.$$

    Canonical kinematics use $m_V^2=\mu^2=1$ and the $+i0$ prescription.
    Each physical form factor includes
    $-1/[m_V^4(4\pi)^4]$ exactly once. HEPKit supplies the model couplings,
    Lorentz/color contractions and incoming spin/color average ($1/256$).
    Observables contain no phase-space or flux factor. Finite-top QCD is outside
    this demonstration.

    Requested observable accuracy is **20 decimal digits**. Seeds start at
    **30 digits**, and the displayed uncertainty is propagated from independent
    sample/precision checks. Working precision alone is not an accuracy claim.
    """)
    return


@app.cell(hide_code=True)
def _(mo, session):
    def launch(stage, **kwargs):
        session.submit(stage, **kwargs)
        return f"Started {stage}"

    boundary_load = mo.ui.button(label="1 · Load or compute boundaries", on_click=lambda _: launch("boundaries"))
    boundary_force = mo.ui.button(label="Recompute boundaries", on_click=lambda _: launch("boundaries", recompute=True))
    transport_load = mo.ui.button(label="2 · Load or compute transport", on_click=lambda _: launch("transport"))
    transport_force = mo.ui.button(label="Recompute transport", on_click=lambda _: launch("transport", recompute=True))
    amplitude_load = mo.ui.button(label="3 · Load or assemble amplitude", on_click=lambda _: launch("amplitude"))
    amplitude_force = mo.ui.button(label="Reassemble amplitude", on_click=lambda _: launch("amplitude", recompute=True))
    restart = mo.ui.button(label="Binary reload and exact repeated hit", on_click=lambda _: launch("restart"))
    nearby = mo.ui.button(label="Transport to nearby s + 1/100000", on_click=lambda _: launch("transport", nearby=True))
    cancel = mo.ui.button(label="Cancel current stage", kind="warn", on_click=lambda _: session.cancel())
    refresh = mo.ui.refresh(options=[1, 5, 10], default_interval=5, label="Progress refresh (seconds)")
    mo.vstack([
        mo.hstack([boundary_load, boundary_force]),
        mo.hstack([transport_load, transport_force]),
        mo.hstack([amplitude_load, amplitude_force]),
        mo.hstack([restart, nearby]),
        mo.hstack([cancel, refresh]),
        mo.md("Forced boundary recomputation bypasses numerical samples and boundary values while reusing exact reductions. Forced transport starts from the saved seed-only bank. Reassembly reruns the native contractions' scalar evaluation and uncertainty propagation."),
    ])
    return (refresh,)


@app.cell(hide_code=True)
def _(mo, refresh, session):
    refresh.value
    state = session.snapshot()
    mo.vstack([
        mo.md(f"**{state['status']}**"),
        mo.ui.table(state["timings"], label="Stage timings; nanoseconds, including failed/interrupted work"),
        mo.accordion({"Recent native progress": mo.plain_text("\n".join(state["events"][-30:]))}),
    ])
    return (state,)


@app.cell(hide_code=True)
def _(mo, session, state):
    mo.stop(not state["done"], mo.md("Computation is running; progress and cancellation remain available above."))
    _rows = [
        {
            "configuration": label,
            "checked digits": result.verified_digits,
            "input checked digits": result.input_verified_digits,
            "working bits": result.working_bits,
            "exact cache hit": result.cache_hit,
            "steps": result.steps,
            "inserted points": result.inserted_points,
            "selected source": str(result.starting_coordinates),
            "source history": result.provenance,
        }
        for label, result in session.results.items()
    ]
    mo.ui.table(_rows, label="Transport and accumulated boundary reuse")
    return


@app.cell(hide_code=True)
def _(mo, session, state):
    mo.stop(not state["done"] or session.amplitude is None, mo.md("Native effective diagrams appear after amplitude preparation."))
    mo.vstack([mo.md("## Native HEPKit diagrams"), *session.amplitude.diagrams])
    return


@app.cell(hide_code=True)
def _(ComplexFloat, E, Float, data_directory, json, mo, session, state):
    mo.stop(not state["done"] or session.observables is None, mo.md("Assemble the amplitude to view observables and uncertainties."))
    _result = session.observables
    # Comparison data is loaded only here, after the native result exists.
    _reference = json.loads((data_directory / "amplitude-validation.json").read_text())
    _same_point = session.point == [E(value) for value in _reference["physical_s_t_MH_squared"]]
    _factor_reference = {block["mass"]: block["values"] for block in _reference["form_factors"]}
    _factor_rows = []
    for _mass in ("W", "Z"):
        _factors = session.form_factors[_mass]
        for _index, (_value, _error, _digits, _expected) in enumerate(zip(
            _factors.values, _factors.absolute_errors, _factors.verified_relative_digits,
            _factor_reference[_mass], strict=True,
        ), start=1):
            _reference_value = ComplexFloat(_expected["real"], _expected["imaginary"], decimal_digits=100)
            _reference_error = Float(_expected["absolute_error"], decimal_digits=100)
            _factor_rows.append({
                "form factor": f"{_mass}{_index}",
                "native result": str(_value),
                "propagated absolute uncertainty": str(_error),
                "achieved relative digits": _digits,
                "reference at recorded point": str(_reference_value) if _same_point else "different kinematics",
                "reference absolute uncertainty": str(_reference_error) if _same_point else "—",
                "absolute difference": str(abs(_value - _reference_value)) if _same_point else "—",
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
            "native result": str(_value),
            "propagated absolute uncertainty": str(_result.absolute_errors[_name]),
            "achieved relative digits": _result.verified_relative_digits[_name],
            "reference at recorded point": str(_ref) if _same_point else "different kinematics",
            "reference absolute uncertainty": str(_ref_error) if _same_point else "—",
            "absolute difference": str(abs(_value - _ref)) if _same_point else "—",
        })
    mo.vstack([
        mo.md("## W/Z form factors"),
        mo.md(f"Current exact kinematics: s = {session.point[0]}, t = {session.point[1]}, m_H² = {session.point[2]}."),
        mo.ui.table(_factor_rows, label="Native W/Z form factors and propagated uncertainties"),
        mo.md("## Coherent observables"),
        mo.ui.table(_rows),
        mo.md("The archived reference has its own input-accuracy limits. Extra printed digits do not strengthen its uncertainty. Nearby evaluations use new kinematics-dependent projections; the original-point comparison is then omitted."),
        mo.accordion({"Numerical provenance": mo.plain_text(_result.provenance)}),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reproduction

    The exact physical families and canonical transformations follow
    [Becchetti, Moriello and Schweitzer, arXiv:2112.07578](https://arxiv.org/abs/2112.07578).
    Basis transformations are checked with native exact Symbolica arithmetic.
    Reference numerical coefficients are comparison data only.

    For a restart, reopen the notebook with the same cache directory and run
    the load-or-compute controls. The explicit binary-reload control verifies
    bit-for-bit endpoint equality and zero transport steps. Nearby transport
    retains and searches the growing collection of physical intermediate points.
    Long headless acceptance is separate from ordinary notebook smoke checks.
    """)
    return


if __name__ == "__main__":
    app.run()
