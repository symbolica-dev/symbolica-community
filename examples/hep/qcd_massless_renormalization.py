import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Massless QCD renormalization")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Massless QCD: infrared rearrangement and counterterms

    Set the quark mass to zero **before** adding the auxiliary mass. This is the
    order used by the [FeynCalc massless reference](https://feyncalc.github.io/FeynCalcExamples/QCD/OneLoop/RenormalizationMassless).
    It changes which propagators are massified and therefore changes the
    auxiliary gluon mass counterterm. The physical field and coupling constants
    remain the same.

    This notebook reuses the symbolic calculation in the
    [massive QCD notebook](?file=hep/qcd_renormalization.py). Marimo's notebook
    composition supplies the generated diagrams, tensor reduction, native IBP
    solution and counterterm equations; the physics implementation is shared.
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
    from qcd_renormalization import app as qcd_app

    return mo, qcd_app


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell(hide_code=True)
async def _(qcd_app):
    qcd_calculation = await qcd_app.embed()
    shared = qcd_calculation.defs
    massless_result = shared["irr_results"]["massless"]
    massive_result = shared["irr_results"]["massive"]
    constant_names = ["Zq", "ZA", "Zξ", "Zc", "Zg", "ZAm", "Zcm"]
    assert massless_result["indices"] == [0, 2, 3, 4, 5, 6, 7]
    assert len(massless_result["residues"]) == len(constant_names)
    return constant_names, massive_result, massless_result, shared


@app.cell(hide_code=True)
def _(massless_result, mo, shared):
    mo.vstack(
        [
            mo.md(r"""
    ## Generated diagrams and primitive denominators

    The diagrams are generated with the same color and ghost conventions as the
    massive case. The raw numerators and denominator masses are then evaluated
    at $m_q=0$, before any Taylor expansion or infrared rearrangement.
    Each massless denominator becomes $q^2-M$. Extra longitudinal gluon factors
    retain their separate denominator powers, even when two internal gluons are
    present. Recombining the components reconstructs each original numerator.
    """),
            mo.ui.tabs(
                {
                    _kind: mo.hstack(_diagrams)
                    for _kind, _diagrams in shared["diagrams"].items()
                    if _kind != "tree"
                }
            ),
            mo.md(r"""
    ## One tadpole mass and seven equations

    Vacuum tensor reduction and Symbolica partial fractions reconstruct every
    scalar integrand before reduction. Native IBP reduces the five requested
    powers to $A_0(M)$, whose supplied pole is $M/\epsilon$ and is checked against
    shared OneLOop. An explicit marker on the first Taylor term beyond the
    superficial divergence degree must disappear from the UV poles.

    There is no quark mass operator. Removing its row and column from the matrix
    of generated CT coefficients leaves seven equations. Substituting their
    solution cancels the complete projected quark, gluon, ghost and vertex poles.
    The local operator basis and tadpole pole are inputs; no finite loop amplitude
    or general subtraction-forest generation is claimed.
    """),
            massless_result["solution"],
            massless_result["matrix"],
            massless_result["constants"],
            mo.md(r"""
    The quark loop now contributes an auxiliary term
    $2N_f M g^{\mu\nu}/\epsilon$. The generated local gluon operator has
    coefficient $2\delta Z_{Am}$, so
    $$\delta Z_{Am}=-\frac{C_A(1+3\xi)+8N_f}{8\epsilon},
    \qquad \delta Z_{cm}=0.$$
    Relative to the massive prescription, the residue changes by $-N_f$.
    The controls below verify that difference and the quark–gluon coupling
    relation as the gauge and color/flavor counts vary.
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    gauge_parameter = mo.ui.slider(
        0.0, 3.0, step=0.25, value=1.0, label="Gauge parameter ξ"
    )
    flavor_count = mo.ui.slider(0, 20, step=1, value=5, label="Quark flavors Nf")
    color_count = mo.ui.slider(2, 5, step=1, value=3, label="Colors Nc")
    subtraction_scheme = mo.ui.dropdown(
        ["MSbar", "MS"], value="MSbar", label="Subtraction scheme"
    )
    mo.vstack(
        [mo.hstack([gauge_parameter, flavor_count, color_count]), subtraction_scheme]
    )
    return color_count, flavor_count, gauge_parameter, subtraction_scheme


@app.cell(hide_code=True)
def _(
    color_count,
    constant_names,
    flavor_count,
    gauge_parameter,
    massive_result,
    massless_result,
    mo,
    shared,
    subtraction_scheme,
):
    _n = color_count.value
    _parameters = {
        shared["CA"]: _n,
        shared["CF"]: (_n**2 - 1) / (2 * _n),
        shared["Nf"]: flavor_count.value,
        shared["xi"]: gauge_parameter.value,
    }
    selected_residues = [
        complex(_value.evaluate(_parameters)).real
        for _value in massless_result["residues"]
    ]
    _massive_aux = complex(massive_result["residues"][6].evaluate(_parameters)).real
    auxiliary_shift = selected_residues[5] - _massive_aux
    assert abs(auxiliary_shift + flavor_count.value) < 1e-12
    _vertex = complex(
        (shared["epsilon"] * massless_result["poles"]["vertex"])
        .together()
        .evaluate(_parameters)
    ).real
    vertex_error = abs(
        selected_residues[0] + selected_residues[4] + selected_residues[1] / 2 + _vertex
    )
    beta0 = -2 * selected_residues[4]
    assert vertex_error < 1e-12
    assert selected_residues[1] == selected_residues[2]
    assert abs(beta0 - (11 * _n - 2 * flavor_count.value) / 3) < 1e-12
    _Symbol = shared["Symbol"]
    _c_delta = complex(((4 * _Symbol.PI).log() - _Symbol.EULER_GAMMA).evaluate({})).real
    _finite_factor = _c_delta if subtraction_scheme.value == "MSbar" else 0.0
    mo.vstack(
        [
            mo.md(
                r"The pole column gives coefficients of $a_4/\epsilon$. In MSbar the finite subtraction column gives $c_\Delta$ times the residue, with $c_\Delta=\log(4\pi)-\gamma_E$ in the common loop measure. It is a subtraction constant, not a finite loop amplitude."
            ),
            mo.ui.table(
                [
                    {
                        "Constant": _name,
                        "1/ε coefficient": _value,
                        "Finite subtraction coefficient": _finite_factor * _value,
                    }
                    for _name, _value in zip(
                        constant_names, selected_residues, strict=True
                    )
                ],
                selection=None,
            ),
            mo.md(
                rf"Auxiliary residue shift from the massive prescription: **{auxiliary_shift:.6g}**.  "
                + rf"Vertex cancellation residual: **{vertex_error:.1e}**.  "
                + rf"One-loop $\beta_0$: **{beta0:.6g}**."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
