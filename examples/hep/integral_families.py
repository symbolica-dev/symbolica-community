import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Integral families and Feynman parameters",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Integral families and Feynman parameters

    `diagram.integral_family()` preserves the routed propagators and appends
    auxiliary scalar products to complete the basis. Supply preferred dot
    products as its optional argument, or omit them to choose automatically.
    The original scalar integral has zero powers for these auxiliary entries.
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
    from symbolica import E, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hepkit import (
        FeynmanDiagram,
        IntegralFamily,
        Kinematics,
        Model,
    )

    _set_namespace("parameters")
    return E, FeynmanDiagram, IntegralFamily, Kinematics, Model, S, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    [Browse all notebooks](/) · [Two-loop topology preparation](/?file=hep/topology_preparation.py)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell
def _(FeynmanDiagram, Model, mo):
    _sunrise = FeynmanDiagram.from_dot(
        Model.phi4(),
        """digraph sunrise {
            ext [style=invis];
            ext -> a [particle="phi"];
            a -> b [particle="phi", lmb_id=0];
            a -> b [particle="phi", lmb_id=1];
            a -> b [particle="phi"];
            b -> ext [particle="phi"];
        }""",
    )
    automatic_family = _sunrise.integral_family()
    _p = automatic_family.external_momenta[0]
    _products = [
        automatic_family.kinematics.scalar_product(_k, _p)
        for _k in automatic_family.loop_momenta
    ]
    chosen_family = _sunrise.integral_family(_products)
    assert automatic_family.is_complete and chosen_family.is_complete
    assert chosen_family.denominators[3:] == _products
    mo.vstack(
        [
            mo.md("**Automatic basis**"),
            automatic_family,
            mo.md("**Preferred dot products**"),
            chosen_family,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feynman-parameter polynomials

    For the two-loop denominators $(k+l)^2-m^2$ and $(k-l)\cdot p+\delta$,
    with $p^2=s$, the quadratic loop matrix is singular. Feynkit still returns
    its algebraic polynomials $U=0$ and $F=sxy^2$.
    These polynomials alone do not define an integration formula or establish
    that the integral vanishes.
    """)
    return


@app.cell
def _(E, IntegralFamily, Kinematics, S, mo):
    _k, _l, _p, _s, _m2, _delta, _x, _y = S(
        "k",
        "l",
        "p",
        "s",
        "m2",
        "delta",
        "x",
        "y",
    )
    _kin = Kinematics(momenta=[_k, _l, _p]).with_scalar_product(_p, _p, _s)
    parameter_family = IntegralFamily(
        [_k, _l],
        [_p],
        [
            _kin.scalar_product(_k + _l, _k + _l) - _m2,
            _kin.scalar_product(_k - _l, _p) + _delta,
        ],
        kinematics=_kin,
    )
    parameter_u, parameter_f = parameter_family.symanzik([_x, _y])
    assert parameter_u == E("0")
    assert (parameter_f - _s * _x * _y**2).expand() == E("0")
    mo.vstack(
        [
            mo.md("**Integral family**"),
            parameter_family,
            mo.md("**First polynomial, U**"),
            parameter_u,
            mo.md("**Second polynomial, F**"),
            parameter_f,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
