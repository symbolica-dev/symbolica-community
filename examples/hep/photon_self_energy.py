import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Photon vacuum polarization")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Photon vacuum polarization

    Generate the electron loop and reduce its numerator in $D=4-2\epsilon$.
    The result is transverse: $(p^2g^{\mu\nu}-p^\mu p^\nu)F(p^2)$.
    Keeping $D$ symbolic until after reduction retains the finite rational term.

    The scalar masters use the shared OneLOop evaluator. The expressions below
    omit the common loop factor $i/(16\pi^2)$; the UV coefficient is
    $-4e^2/3$. Numerical values use $e=m^2=\mu^2=1$ and show the physical
    imaginary part above the pair-production threshold $p^2=4m^2$.
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
    from symbolica.community import hep
    from symbolica.community import tensor as sp
    import marimo as mo
    from symbolica import E, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hep import Kinematics, Model, TensorReducer, oneloop

    _set_namespace("photon_self_energy")
    return E, Kinematics, Model, S, TensorReducer, hep, mo, oneloop, sp


@app.cell
def _(Model):
    qed_model = Model.standard_model()
    return (qed_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    [Browse all notebooks](/) · [Born currents and real radiation](/?file=hep/photon_radiation.py)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate the electron loop

    The diagram retains routing and fermion-loop weights. Keep the Lorentz dimension symbolic until after reduction.
    """)
    return


@app.cell
def _(S, hep, qed_model, sp):
    _generated = qed_model.process(
        ["a"], ["a"], vertex_allow=["V_98"]
    ).generate_diagrams(
        loops=1,
        max_vertices=2,
        maximum_bridges=None,
        numerator_grouping=None,
        progress=None,
    )
    assert len(_generated.diagrams) == 1
    diagram = _generated.diagrams[0]
    D, s = S("D", "s")
    _electron = qed_model.particle("e-")
    mass, charge = _electron.mass, -_electron.electric_charge
    mu, nu = S("mu", "nu")
    K, P, _mink, metric = (
        hep.Kinematics.loop_momentum,
        hep.Kinematics.external_momentum,
        sp.Representation.mink,
        sp.TensorName.g().to_expression(),
    )
    return D, K, P, charge, diagram, mass, metric, mu, nu, s


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Label the two external photon ports

    The generated projector identifies the external legs. Relabel only those ports as $\mu$ and $
    u$.
    """)
    return


@app.cell
def _(D, S, diagram, mu, nu, qed_model, sp):
    _index, _wave = S("index_", "wave_")
    numerator = qed_model.expand_couplings(
        diagram.numerator_expression(in_lmb=True)
    ).with_lorentz_dimension(D)
    for _edge in diagram.external_edges:
        _matches = list(
            diagram.projector_expression().match(
                _wave(
                    _edge.id, sp.PortPattern.exact(sp.Representation.mink(4), _index)
                ),
                max_level=0,
            )
        )
        assert len(_matches) == 1
        _port = dict(_matches[0])[_index]
        numerator = sp.TensorExpression(numerator).rename_indices(
            {_port: (mu, nu)[_edge.external_index]}
        )
    return (numerator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Reduce the Dirac trace

    Use the shared tensor algebra before applying the angular integration identities.
    """)
    return


@app.cell
def _(numerator):
    trace = (
        numerator.simplify_algebra(contract="dots", gamma=True, epsilon=True)
        .expand()
        .to_expression()
    )
    return (trace,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Apply covariant tensor reduction

    `TensorReducer` integrates the loop-vector directions; the external momentum is held fixed. `integral_family()` supplies the same routed propagator basis.
    """)
    return


@app.cell
def _(D, K, Kinematics, P, TensorReducer, diagram, s, sp, trace):
    kin = Kinematics(D, momenta=[K(0), P(0)]).with_scalar_product(P(0), P(0), s)
    photon_family = diagram.integral_family(kinematics=kin)
    reducer = TensorReducer(
        D,
        integrated=[K(0, sp.PortPattern.exact(sp.Representation.mink(D)))],
        external=[P(0, sp.PortPattern.exact(sp.Representation.mink(D)))],
    )
    reduced = kin.apply(reducer.reduce(trace))
    return photon_family, reduced


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Identify scalar masters

    Equal-mass tadpoles differ by a loop-momentum shift. Polynomial moments are scaleless, leaving $A_0$ and $B_0$.
    """)
    return


@app.cell
def _(E, S, diagram, photon_family, reduced, s):
    d0, d1, a0, b0 = S("d0", "d1", "A0", "B0")
    scalar = photon_family.rewrite_numerator(reduced, [d0, d1]) / (d0 * d1)
    scalar *= (
        diagram.overall_factor_expression(evaluate=True)
        * diagram.numerator_prefactor_expression()
    )
    # Equal-mass tadpoles are related by a loop-momentum shift. Polynomial
    # moments are scaleless; the bubble's remaining scalar master is B0.
    _weights = {
        E("1"): E("0"),
        1 / d0: a0,
        1 / d1: a0,
        d0 / d1: s * a0,
        d1 / d0: s * a0,
        1 / (d0 * d1): b0,
    }
    _parts = scalar.expand().coefficient_list(d0, d1)
    integrated = sum(
        _coefficient * _weights[_monomial] for _monomial, _coefficient in _parts
    )
    return a0, b0, integrated


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check transversality

    The complete result must be proportional to $p^2g^{\mu
    u}-p^\mu p^
    u$. Verify this before selecting its coefficient.
    """)
    return


@app.cell
def _(D, P, integrated, metric, mu, nu, s, sp):
    metric_tensor = metric(
        sp.PortPattern.exact(sp.Representation.mink(D), mu),
        sp.PortPattern.exact(sp.Representation.mink(D), nu),
    )
    photon_form_factor = (integrated.expand().coefficient(metric_tensor) / s).together()
    _transverse = s * metric_tensor - P(
        0, sp.PortPattern.exact(sp.Representation.mink(D), mu)
    ) * P(0, sp.PortPattern.exact(sp.Representation.mink(D), nu))
    assert (integrated - photon_form_factor * _transverse).together() == 0
    return (photon_form_factor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extract the UV pole

    Expand at $D=4-2\epsilon$ with the master poles still present; dimension-dependent coefficients contribute to the finite part.
    """)
    return


@app.cell
def _(D, S, a0, b0, charge, mass, photon_form_factor):
    eps, af, bf = S("eps", "A0_finite", "B0_finite")
    laurent = (
        photon_form_factor.replace(D, 4 - 2 * eps)
        .replace(a0, af + mass**2 / eps)
        .replace(b0, bf + 1 / eps)
        .series(eps, 0, 0)
        .to_expression()
    )
    photon_uv = laurent.coefficient(eps**-1)
    assert (photon_uv + 4 * charge**2 / 3).together() == 0
    return af, bf, eps, laurent, photon_uv


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Evaluate the finite masters

    The native OneLoop backend supplies the finite coefficients below and above threshold.
    """)
    return


@app.cell
def _(E, af, bf, charge, eps, laurent, mass, oneloop, s):
    _finite = dict(laurent.expand().coefficient_list(eps))[E("1")]
    _a_coeffs = oneloop.master_coefficients(oneloop.A0(mass**2, E("1")))
    _b_coeffs = oneloop.master_coefficients(oneloop.B0(s, mass**2, mass**2, E("1")))
    _finite = _finite.replace(af, _a_coeffs[0]).replace(bf, _b_coeffs[0])
    photon_finite_values = [
        {
            "p²": _point,
            "Finite coefficient": str(
                _finite.evaluate({s: _point, mass: 1.0, charge: 1.0})
            ),
        }
        for _point in (-1.0, 3.0, 5.0)
    ]
    return (photon_finite_values,)


@app.cell(hide_code=True)
def _(
    diagram,
    mo,
    photon_family,
    photon_finite_values,
    photon_form_factor,
    photon_uv,
):
    mo.vstack(
        [
            diagram,
            photon_family,
            mo.md("**Transverse form factor in terms of A₀ and B₀**"),
            photon_form_factor,
            mo.md("**Coefficient of 1/ε**"),
            photon_uv,
            mo.ui.table(photon_finite_values, selection=None),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
