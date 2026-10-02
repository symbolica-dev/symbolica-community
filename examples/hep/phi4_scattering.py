import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Finite one-loop phi4 scattering")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Finite one-loop $\phi\phi\to\phi\phi$
    [Browse all notebooks](/) · [Generated counterterms](/?file=hep/phi4_renormalization.py)

    Use the three generated channels and counterterms from the companion notebook. Marimo composition shares that calculation. OneLOop evaluates the massive and massless finite correction, with physical $s+t+u=4m^2$ and the Feynman $+i0$ prescription.

    Independent Feynman-parameter integrals check six kinematic points, including the two-particle threshold and massless scattering. The symbolic massless branches yield the [FeynCalc high-energy logarithm](https://feyncalc.github.io/FeynCalcExamples/Phi4/OneLoop/PhiPhi-PhiPhi) at fixed negative $t$.

    The displayed quantity is the one-loop correction to the amputated amplitude, including its factor $i$. It excludes the tree term $-ig$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Setup and notebook helpers

    Imports and supporting routines are folded below. Expand a cell’s code to
    inspect or edit it; the calculation that follows shows the HEP operations.
    """)
    return


@app.cell(hide_code=True)
def _():
    import cmath
    import math

    import marimo as mo
    import numpy as np
    from symbolica import E, Replacement, S, Symbol
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hep import oneloop

    _set_namespace("phi4_one")
    return E, Replacement, S, Symbol, cmath, math, mo, np, oneloop


@app.cell(hide_code=True)
def _():
    from phi4_renormalization import app as renormalization_app

    return (renormalization_app,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell
async def _(renormalization_app):
    _shared_calculation = await renormalization_app.embed()
    M = _shared_calculation.defs["M"]
    s = _shared_calculation.defs["s"]
    t = _shared_calculation.defs["t"]
    u = _shared_calculation.defs["u"]
    mu2 = _shared_calculation.defs["mu2"]
    coupling = _shared_calculation.defs["coupling"]
    vertex_finite = _shared_calculation.defs["vertex_finite"]
    channel_coefficients = _shared_calculation.defs["channel_coefficients"]
    zero = _shared_calculation.defs["zero"]
    one = _shared_calculation.defs["one"]
    pi = _shared_calculation.defs["pi"]
    diagrams = _shared_calculation.defs["diagrams"]
    residues = _shared_calculation.defs["residues"]
    return (
        M,
        channel_coefficients,
        coupling,
        diagrams,
        mu2,
        one,
        pi,
        residues,
        s,
        t,
        u,
        vertex_finite,
        zero,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Evaluate the loop amplitude

    Compile the native master expression, then check it against independent parameter quadratures at several physical points.
    """)
    return


@app.cell
def _(M, cmath, math, mu2, np, oneloop, s, t, u, vertex_finite):
    finite_evaluator = oneloop.compile_native([vertex_finite], [M, s, t, u, mu2])
    _points = [
        (0.0, 5.0, -1.0, 2.0),
        (0.0, 20.0, -6.0, 1.0),
        (1.0, 4.0, 0.0, 1.0),
        (1.0, 5.0, -0.25, 1.0),
        (1.0, 12.0, -3.0, 2.0),
        (4.0, 25.0, -4.0, 3.0),
    ]
    _nodes, _weights = np.polynomial.legendre.leggauss(256)
    numeric_checks = []
    for _mass_squared, _sv, _tv, _scale in _points:
        _uv = 4 * _mass_squared - _sv - _tv
        _reference = 0j
        for _qv in (_sv, _tv, _uv):
            if _mass_squared == 0:
                _expected = 2 - cmath.log(complex(-_qv / _scale, -0.0))
            elif _qv >= 4 * _mass_squared:
                _beta = math.sqrt(1 - 4 * _mass_squared / _qv)
                _roots = [(1 - _beta) / 2, (1 + _beta) / 2]
                _expected = (
                    -math.log(_qv / _scale)
                    - sum(
                        (1 - _r) * math.log(1 - _r) + _r * math.log(_r) - 1
                        for _r in _roots
                    )
                    + 1j * math.pi * _beta
                )
            else:
                _x = (_nodes + 1) / 2
                _expected = (
                    -np.dot(
                        _weights, np.log((_mass_squared - _qv * _x * (1 - _x)) / _scale)
                    )
                    / 2
                )
            _actual = complex(oneloop.b0(_qv, _mass_squared, _mass_squared, _scale)[0])
            assert abs(_actual - _expected) < 2e-10, (
                _qv,
                _mass_squared,
                _actual,
                _expected,
            )
            _reference += _expected / 2
        _actual = complex(
            finite_evaluator.evaluate_complex(
                [[complex(_v) for _v in (_mass_squared, _sv, _tv, _uv, _scale)]]
            )[0, 0]
        )
        assert abs(_actual - _reference) < 4e-10, (_actual, _reference)
        numeric_checks.append(
            (_mass_squared, _sv, _tv, _uv, _scale, _actual, _reference)
        )
    return finite_evaluator, numeric_checks


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the massless continuation
    """)
    return


@app.cell
def _(
    E,
    Replacement,
    S,
    Symbol,
    channel_coefficients,
    coupling,
    mu2,
    one,
    oneloop,
    pi,
    s,
    t,
    u,
    zero,
):
    log_function, _absolute = S("log", "abs")
    physical_finite = zero
    for _invariant, _sign in [(s, 1), (t, -1), (u, -1)]:
        _massless_master = oneloop.B0(_invariant, zero, zero, mu2)
        _branch = oneloop.select_branch(
            oneloop.get_expression(_massless_master, coefficient=0),
            [Replacement(_invariant, E(str(_sign))), Replacement(mu2, one)],
        )
        _branch = _branch.replace(
            _absolute(-_invariant / mu2),
            (_sign * _invariant / mu2).replace(u, -s - t).expand(),
        )
        physical_finite += channel_coefficients[_invariant] / coupling**2 * _branch
    physical_finite = physical_finite.replace(u, -s - t)
    _expected_massless = (
        6
        - log_function(s / mu2)
        - log_function(-t / mu2)
        - log_function(((s + t) / mu2).expand())
        + Symbol.I * pi
    ) / 2
    assert (physical_finite - _expected_massless).expand() == zero
    return log_function, physical_finite


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Extract the high-energy logarithm
    """)
    return


@app.cell
def _(
    S,
    Symbol,
    coupling,
    log_function,
    mu2,
    one,
    physical_finite,
    pi,
    s,
    zero,
):
    _asymptotic_parameter = S("inverse_s")
    logarithmic_coefficient = (
        (s * physical_finite.derivative(s))
        .replace(s, 1 / _asymptotic_parameter)
        .series(_asymptotic_parameter, 0, 0)
        .to_expression()
    )
    assert logarithmic_coefficient == -one
    leading_amplitude = (
        Symbol.I
        * coupling**2
        / (16 * pi**2)
        * logarithmic_coefficient
        * log_function(s / mu2)
    )
    assert (
        leading_amplitude
        + Symbol.I * coupling**2 * log_function(s / mu2) / (16 * pi**2)
    ).together() == zero
    return (leading_amplitude,)


@app.cell(hide_code=True)
def _(diagrams, leading_amplitude, mo, physical_finite):
    mo.vstack(
        [
            mo.md("## Generated channels"),
            mo.hstack(diagrams["vertex"]),
            mo.md("**Physical massless finite coefficient**"),
            physical_finite,
            mo.md("**Leading high-energy amplitude**"),
            leading_amplitude,
            mo.md(
                "The finite coefficient is in units of ig²/(16π²). Its logarithmic derivative approaches −1 at fixed t; the complex branch retains the positive s-channel cut."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    _presets = {
        "Massless · s=5": 0,
        "Massless · s=20": 1,
        "Massive threshold": 2,
        "Massive · just above threshold": 3,
        "Massive · s=12": 4,
        "Heavier scalar · s=25": 5,
    }
    kinematic_point = mo.ui.dropdown(
        _presets, value="Massless · s=5", label="Kinematics"
    )
    subtraction_scheme = mo.ui.dropdown(
        ["MSbar", "MS"], value="MSbar", label="Subtraction scheme"
    )
    coupling_value = mo.ui.slider(
        0.1, 2.0, step=0.1, value=1.0, label="Quartic coupling g"
    )
    mo.vstack([kinematic_point, subtraction_scheme, coupling_value])
    return coupling_value, kinematic_point, subtraction_scheme


@app.cell
def _(
    coupling_value,
    finite_evaluator,
    kinematic_point,
    math,
    mo,
    numeric_checks,
    residues,
    subtraction_scheme,
):
    _selected_point = numeric_checks[kinematic_point.value]
    _M_value, _s_value, _t_value, _u_value, _mu_value, _, _reference = _selected_point
    _native = complex(
        finite_evaluator.evaluate_complex(
            [
                [
                    complex(_x)
                    for _x in (_M_value, _s_value, _t_value, _u_value, _mu_value)
                ]
            ]
        )[0, 0]
    )
    assert abs(_native - _reference) < 4e-10
    _scheme_shift = (
        0
        if subtraction_scheme.value == "MSbar"
        else float(residues[2]) * (math.log(4 * math.pi) - 0.5772156649015329)
    )
    _normalization = 1j * coupling_value.value**2 / (16 * math.pi**2)
    selected_amplitude = _normalization * (_native + _scheme_shift)
    selected_reference = _normalization * (_reference + _scheme_shift)
    finite_error = abs(selected_amplitude - selected_reference)
    assert finite_error < 1e-10
    mo.vstack(
        [
            mo.md("## Renormalized finite correction"),
            mo.ui.table(
                [
                    {
                        "m²": _M_value,
                        "s": _s_value,
                        "t": _t_value,
                        "u": _u_value,
                        "μ²": _mu_value,
                    }
                ],
                selection=None,
            ),
            mo.md(f"**Generated result:** `{selected_amplitude:.12g}`"),
            mo.md(f"**Parameter-integral reference:** `{selected_reference:.12g}`"),
            mo.md(f"**Absolute difference:** `{finite_error:.3g}`"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
