import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Polarization sums in D dimensions")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Polarization sums in $D$ dimensions

    [Browse all notebooks](/) · [Light-cone soft radiation](/?file=hep/soft_function.py) · [Gluon scattering](/?file=hep/gluon_scattering.py) ·
    [Massive Dirac spin states](/?file=hep/polarized_spin.py) ·
    [Gluon scattering from quarks](/?file=hep/qcd_gluons.py)

    Query a particle by name and call `particle.spin_sum(p, i, j, dimension=D)`.
    Massless vectors have $D-2$ physical states; massive vectors have $D-1$.
    `average=True` divides by this count. Four dimensions remain the default.

    The massless projector uses an axial reference $n$ with $p\cdot n=\chi\ne0$.
    Its $n^2=\zeta$ term is retained, so the reference need not be null.
    The massive projector includes the longitudinal Proca polarization.

    A fixed four-dimensional averaging factor is also useful when comparing
    dimensional continuations. It changes only the normalization: the projector
    still uses $D$-dimensional Lorentz slots. For example, the
    [four-gluon FeynCalc reference](https://feyncalc.github.io/FeynCalcExamples/QCD/Tree/GlGl-GlGl)
    divides each incoming $D$-dimensional gluon sum by two.
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
    from symbolica.community import tensor as sp
    import marimo as mo
    from symbolica import E, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hep import Kinematics, Model
    from symbolica.community.tensor import TensorExpression

    _set_namespace("polarization")
    return E, Kinematics, Model, S, TensorExpression, mo, sp


@app.cell(hide_code=True)
def _():
    from symbolica.community.hep import Symbols

    return (Symbols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell
def _(mo):
    species = mo.ui.dropdown(
        {"Gluon": "g", "Photon": "a", "Massive Z": "Z"}, value="Gluon", label="Particle"
    )
    dimension_choice = mo.ui.dropdown(
        ["Symbolic D", "4", "6"], value="Symbolic D", label="Lorentz dimension"
    )
    normalization = mo.ui.dropdown(
        [
            "Sum over states",
            "Average in D dimensions",
            "Average over four-dimensional states",
        ],
        value="Average in D dimensions",
        label="Normalization",
    )
    mo.hstack([species, dimension_choice, normalization])
    return dimension_choice, normalization, species


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Declare the momentum and reference vector
    """)
    return


@app.cell
def _(E, Kinematics, Model, S, dimension_choice, sp, species):
    model = Model.standard_model()
    particle = model.particle(species.value)
    D = S("D") if dimension_choice.value == "Symbolic D" else E(dimension_choice.value)
    lorentz = sp.Representation.mink(
        D if dimension_choice.value == "Symbolic D" else int(dimension_choice.value)
    )
    p, n, i, j, chi, zeta = (
        sp.TensorName.vector("p").to_expression(),
        sp.TensorName.vector("n").to_expression(),
        S("i"),
        S("j"),
        S("chi"),
        S("zeta"),
    )
    mink, metric = (sp.Representation.mink, sp.TensorName.g().to_expression())
    mass_squared = particle.mass**2
    kinematics = (
        Kinematics(D)
        .with_scalar_product(p, p, mass_squared)
        .with_scalar_product(p, n, chi)
        .with_scalar_product(n, n, zeta)
    )
    return D, i, j, kinematics, lorentz, metric, n, p, particle


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Construct the spin projector
    """)
    return


@app.cell
def _(D, E, TensorExpression, i, j, n, normalization, p, particle):
    options = {
        "dimension": D,
        "reference": n if particle.is_massless else None,
        "average": normalization.value == "Average in D dimensions",
    }
    scale = E("1")
    if normalization.value == "Average over four-dimensional states":
        scale /= 2 if particle.is_massless else 3
    projector = scale * particle.spin_sum(p, i, j, **options)
    state_count = D - (2 if particle.is_massless else 1)
    density = TensorExpression(projector)
    return density, options, projector, scale, state_count


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Check the number of physical states
    """)
    return


@app.cell
def _(
    E,
    TensorExpression,
    i,
    j,
    kinematics,
    lorentz,
    metric,
    options,
    projector,
    scale,
    sp,
    state_count,
):
    trace = (
        kinematics.apply(
            TensorExpression(
                (
                    projector
                    * metric(
                        sp.PortPattern.exact(lorentz, i),
                        sp.PortPattern.exact(lorentz, j),
                    )
                ).expand()
            )
            .contract()
            .to_dots()
        )
        .to_expression()
        .together()
    )
    expected_trace = -scale if options["average"] else -scale * state_count
    assert (trace - expected_trace).together() == E("0")
    return (trace,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Check transversality
    """)
    return


@app.cell
def _(E, TensorExpression, i, kinematics, lorentz, p, projector, sp):
    transverse = (
        kinematics.apply(
            TensorExpression((projector * p(sp.PortPattern.exact(lorentz, i))).expand())
            .contract()
            .to_dots()
        )
        .to_expression()
        .together()
    )
    assert transverse == E("0")
    return (transverse,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Check the axial reference condition
    """)
    return


@app.cell
def _(E, TensorExpression, i, kinematics, lorentz, n, particle, projector, sp):
    if particle.is_massless:
        axial = (
            kinematics.apply(
                TensorExpression(
                    (projector * n(sp.PortPattern.exact(lorentz, i))).expand()
                )
                .contract()
                .to_dots()
            )
            .to_expression()
            .together()
        )
        assert axial == E("0")
    return


@app.cell
def _(density, mo, particle, state_count, trace, transverse):
    mo.vstack(
        [
            mo.md(f"**{particle.name}: projector and physical state count**"),
            density,
            mo.tree(
                {
                    "Physical states": state_count,
                    "Lorentz trace": trace,
                    "Momentum contraction": transverse,
                }
            ),
            mo.md(
                r"With the $(+,-,\ldots,-)$ metric, the projector trace is the negative state count. Both the momentum contraction and, for massless vectors, the reference contraction vanish exactly."
            ),
        ]
    )
    return


@app.cell
def _(
    E,
    Symbols,
    i,
    j,
    lorentz,
    mo,
    options,
    p,
    particle,
    projector,
    scale,
    sp,
):
    _ket, _bra = (Symbols.polarization, Symbols.polarization_conjugate)
    _pair = _ket(7, sp.PortPattern.exact(lorentz, i)) * _bra(
        7, sp.PortPattern.exact(lorentz, j)
    )
    _spectator = _ket(8, sp.PortPattern.exact(lorentz, i))
    sewn = scale * particle.sum_spins(_pair + _spectator, p, edge=7, **options)
    assert (sewn - projector - scale * _spectator).expand() == E("0")
    mo.vstack(
        [
            mo.md(
                "**Sewing-aware completeness**\n\n`sum_spins(..., edge=7, dimension=D)` replaces the paired wavefunctions on edge 7. The unpaired wavefunction on edge 8 remains present. This is the same replacement used by GammaLoop."
            ),
            mo.tree({"Before": _pair + _spectator, "After": sewn}),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
