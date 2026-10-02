import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Electron Pauli form factor across threshold",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Electron Pauli form factor across threshold
    [Browse all notebooks](/) · [Generated vertex and g−2](/?file=hep/gminus2.py) ·
    [IBP differential equations](/?file=hep/ibp_differential_equations.py) ·
    [Massless triangle](/?file=hep/ibp_triangle.py)

    The [FeynCalc reference](https://feyncalc.github.io/FeynCalcExamples/QED/OneLoop/El-GaEl)
    computes $F_2(0)=\alpha/(2\pi)$. Here we continue the same generated
    vertex to nonzero momentum transfer, including the physical timelike sheet.
    The tree normalization, spin projectors, Dirac algebra and native IBP
    reduction come from the g−2 notebook through Marimo composition.

    Write $r=t/m^2$ and $R(r)=F_2(t)/(\alpha/(2\pi))$. Spacelike scattering
    has $r<0$. Timelike values continue the crossed electron-pair current;
    its pair-production cut opens at $r=4$. We use $t+i0$, so the imaginary
    part is positive. Exactly at threshold the one-loop form factor diverges.
    This notebook evaluates the Pauli form factor; it does not compute the
    renormalized Dirac form factor $F_1$.
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
    import marimo as mo
    import numpy as np
    from symbolica import S, Symbol
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hepkit import oneloop

    _set_namespace("pauli")
    return S, Symbol, mo, np, oneloop


@app.cell(hide_code=True)
def _():
    from gminus2 import app as vertex_app

    return (vertex_app,)


@app.cell(hide_code=True)
def _(finite, finite_a, finite_b, mass, np, oneloop, transfer):
    def generated_pauli(ratio, electron_mass=1.0, scale_ratio=1.0):
        m2 = electron_mass**2
        scale = scale_ratio * m2
        return (
            -complex(
                finite.evaluate(
                    {
                        transfer: ratio * m2,
                        mass: electron_mass,
                        finite_a: complex(oneloop.a0(m2, scale)[0]),
                        finite_b: complex(oneloop.b0(ratio * m2, m2, m2, scale)[0]),
                    }
                )
            )
            / 2
        )

    def analytic_pauli(ratio):
        if ratio < 0:
            q = -ratio
            return 4 * np.arcsinh(np.sqrt(q) / 2) / np.sqrt(q * (q + 4))
        if ratio == 0:
            return 1.0
        if ratio < 4:
            return (
                4
                * np.arctan(np.sqrt(ratio / (4 - ratio)))
                / np.sqrt(ratio * (4 - ratio))
            )
        beta = np.sqrt(1 - 4 / ratio)
        return 2 * (np.log((1 - beta) / (1 + beta)) + 1j * np.pi) / (ratio * beta)

    _nodes, _weights = np.polynomial.legendre.leggauss(1024)
    _nodes, _weights = (_nodes + 1) / 2, _weights / 2

    def parameter_pauli(ratio, deformation=1.0):
        # The endpoints stay fixed and Im Delta is negative at both poles.
        z = _nodes + 1j * deformation * _nodes * (1 - _nodes) * (1 - 2 * _nodes)
        jacobian = 1 + 1j * deformation * (1 - 6 * _nodes + 6 * _nodes**2)
        return complex(np.dot(_weights, jacobian / (1 - ratio * z * (1 - z))))

    return analytic_pauli, generated_pauli, parameter_pauli


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell
async def _(vertex_app):
    _shared = await vertex_app.embed()
    finite = _shared.defs["finite"]
    finite_a = _shared.defs["finite_a"]
    finite_b = _shared.defs["finite_b"]
    mass = _shared.defs["mass"]
    transfer = _shared.defs["transfer"]
    anomalous_moment = _shared.defs["anomalous_moment"]
    return anomalous_moment, finite, finite_a, finite_b, mass, transfer


@app.cell
def _(anomalous_moment, finite, mo):
    mo.vstack(
        [
            mo.md(
                "**Generated finite coefficient:** $F_2=-\u03b1 b/(4\u03c0)$, with $b=$"
            ),
            finite,
            mo.md("**Exact zero-transfer limit:**"),
            anomalous_moment,
            mo.md(r"""
        The independent Feynman-parameter representation is
        $$R(r)=\int_0^1\frac{dx}{1-rx(1-x)-i0}
          =1+t\,\partial_t B_0(t,m^2,m^2).$$
        The apparent $\mu^2$ dependence cancels between the finite tadpole
        and bubble. Vary the mass and renormalization scale to check this.

        For $r>4$, set $\beta=\sqrt{1-4/r}$. Integrating the parameter formula gives
        $$R(r)=\frac{2}{r\beta}
          \left[\log\frac{1-\beta}{1+\beta}+i\pi\right],
          \qquad \operatorname{Im}R=\frac{2\pi}{r\beta}.$$
        The imaginary part also follows by summing the residues at
        $x=(1\pm\beta)/2$; it is an independent check of the branch sign.
        """),
        ]
    )
    return


@app.cell
def _(analytic_pauli, generated_pauli, mo, np, oneloop, parameter_pauli):
    ratios = [-100.0, -10.0, -1.0, 0.0, 1.0, 3.9, 4.01, 5.0, 10.0, 100.0]
    validation_rows = []
    for _r in ratios:
        _references = [
            analytic_pauli(_r),
            parameter_pauli(_r),
            parameter_pauli(_r, 2.0),
        ]
        for _m in (0.5, 1.0, 2.0):
            for _scale_ratio in (0.1, 10.0):
                _generated = generated_pauli(_r, _m, _scale_ratio)
                _derivative = 1 + _r * _m**2 * complex(
                    oneloop.db0(_r * _m**2, _m**2, _m**2, _scale_ratio * _m**2)[0]
                )
                _error = max(abs(_generated - _v) for _v in [*_references, _derivative])
                assert _error < 5e-10 * max(1, abs(_generated))
                _cut = 2 * np.pi / (_r * np.sqrt(1 - 4 / _r)) if _r > 4 else 0.0
                assert abs(_generated.imag - _cut) < 2e-10
                validation_rows.append(
                    {
                        "t/m²": _r,
                        "m": _m,
                        "μ²/m²": _scale_ratio,
                        "Maximum absolute error": _error,
                    }
                )
    mo.accordion(
        {
            "60 checks: closed form, two contours, bubble derivative and cut": mo.ui.table(
                validation_rows, selection=None
            )
        }
    )
    return (ratios,)


@app.cell
def _(S, Symbol, generated_pauli, np):
    x, r, beta = S("x", "r", "beta")
    _primitive = (
        (1 / (1 - r * x * (1 - x)))
        .series(r, 0, 4)
        .to_expression()
        .expand()
        .integrate(x)
    )
    low_energy = (_primitive.replace(x, 1) - _primitive.replace(x, 0)).expand()
    assert low_energy == 1 + r / 6 + r**2 / 30 + r**3 / 140 + r**4 / 630
    _R = (
        (1 - beta**2)
        / (2 * beta)
        * (((1 - beta) / (1 + beta)).log() + Symbol.I * Symbol.PI)
    )
    _r = 4 / (1 - beta**2)
    ode_check = (
        _r * (_r - 4) * _R.derivative(beta) / _r.derivative(beta) + (_r - 2) * _R + 2
    ).together()
    assert ode_check == 0
    threshold_rows = []
    for _delta in (1e-2, 1e-4, 1e-6):
        _below, _above = generated_pauli(4 - _delta), generated_pauli(4 + _delta)
        assert abs(np.sqrt(_delta) * _below.real - np.pi) < 2 * np.sqrt(_delta)
        assert abs(np.sqrt(_delta) * _above.imag - np.pi) < _delta
        assert abs(_above.real + 1) < _delta
        threshold_rows.append(
            {
                "δ": _delta,
                "√δ Re R(4−δ) → π": np.sqrt(_delta) * _below.real,
                "√δ Im R(4+δ) → π": np.sqrt(_delta) * _above.imag,
                "Re R(4+δ) → −1": _above.real,
            }
        )
    return low_energy, threshold_rows


@app.cell(hide_code=True)
def _(low_energy, mo, threshold_rows):
    mo.vstack(
        [
            mo.md(r"""
        **Regular low-energy expansion** (through $r^4$):
        """),
            low_energy,
            mo.md(r"""
        The analytic continuation obeys
        $r(r-4)R'(r)+(r-2)R(r)+2=0$; Symbolica verifies this exactly.
        The boundary condition is $R(0)=1$. The threshold itself is singular:
        $\sqrt\delta\,\operatorname{Re}R(4-\delta)\to\pi$ and
        $\sqrt\delta\,\operatorname{Im}R(4+\delta)\to\pi$.
        """),
            mo.ui.table(threshold_rows, selection=None),
        ]
    )
    return


@app.cell
def _(mo, ratios):
    point_choice = mo.ui.dropdown(
        {f"t/m² = {r:g}": r for r in ratios},
        value="t/m² = 5",
        label="Momentum transfer",
    )
    mass_choice = mo.ui.dropdown(
        {"m = 0.5": 0.5, "m = 1": 1.0, "m = 2": 2.0}, value="m = 1", label="Mass"
    )
    scale_choice = mo.ui.dropdown(
        {"μ²/m² = 0.1": 0.1, "μ²/m² = 10": 10.0},
        value="μ²/m² = 0.1",
        label="Renormalization scale",
    )
    mo.hstack([point_choice, mass_choice, scale_choice], justify="start")
    return mass_choice, point_choice, scale_choice


@app.cell
def _(
    analytic_pauli,
    generated_pauli,
    mass_choice,
    mo,
    parameter_pauli,
    point_choice,
    scale_choice,
):
    selected_value = generated_pauli(
        point_choice.value, mass_choice.value, scale_choice.value
    )
    selected_error = max(
        abs(selected_value - reference)
        for reference in [
            analytic_pauli(point_choice.value),
            parameter_pauli(point_choice.value),
        ]
    )
    assert selected_error < 5e-10 * max(1, abs(selected_value))
    mo.vstack(
        [
            mo.md("**Normalized Pauli form factor** $R=F_2/(\u03b1/(2\u03c0))$"),
            mo.ui.table(
                [
                    {
                        "Re R": selected_value.real,
                        "Im R": selected_value.imag,
                        "Absolute validation error": selected_error,
                    }
                ],
                selection=None,
            ),
            mo.md(
                r"Multiply by $\alpha/(2\pi)$ to obtain $F_2$. At zero transfer this gives $a_e=(g-2)/2$."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
