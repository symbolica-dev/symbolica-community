import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="QED annihilation: unpolarized and polarized",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Generated QED cross section

    Generate $e^-e^+\to\mu^-\mu^+$, sum and average the incoming spins,
    and contract the generated numerator with Idenso. The massive squared
    matrix element is retained; its massless limit is integrated over two-body
    phase space below. This reproduces the unpolarized result in the
    [FeynCalc example](https://feyncalc.github.io/FeynCalcExamples/QED/Tree/ElAel-MuAmu).
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
    from symbolica import E, S, Symbol
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hep import Kinematics, Model
    from symbolica.community.tensor import (
        Representation,
        TensorExpression,
        TensorRule,
        chain,
    )

    _set_namespace("qed_example")
    return (
        E,
        Kinematics,
        Model,
        Representation,
        S,
        Symbol,
        TensorExpression,
        TensorRule,
        chain,
        hep,
        mo,
    )


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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    [Browse all notebooks](/)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate the sewn annihilation graph
    """)
    return


@app.cell
def _(Model):
    qed_model = Model.standard_model()
    _process = qed_model.process(
        ["e-", "e+"], ["mu-", "mu+"], vertex_allow=["V_98", "V_99"]
    )
    _result = _process.generate_cross_section(
        loops=1,
        max_vertices=4,
        maximum_bridges=None,
        numerator_grouping=None,
        progress=None,
    )
    assert len(_result.diagrams) == 1
    qed_diagram = _result.diagrams[0]
    annihilation_cut = qed_diagram.cuts[0]
    return annihilation_cut, qed_diagram, qed_model


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Define its physical cut and kinematics
    """)
    return


@app.cell
def _(Kinematics, S, Symbols, annihilation_cut, hep, qed_diagram, qed_model):
    edge_momentum, external_momentum, loop_momentum = (
        Symbols.edge_momentum,
        hep.Kinematics.external_momentum,
        hep.Kinematics.loop_momentum,
    )
    qed_p1, qed_p2 = external_momentum(0), external_momentum(1)
    qed_k1, qed_k2 = S("k1", "k2")
    _loop_edge = qed_diagram.loop_momentum_basis.loop_edges[0]
    _loop_particle = next(
        _particle.name
        for _edge, _particle in zip(annihilation_cut.edges, annihilation_cut.particles)
        if _edge.id == _loop_edge
    )
    _physical_momentum = qed_k1 if _loop_particle == "mu-" else qed_k2
    _index = S("index_")
    qed_loop_pattern = loop_momentum(0, _index)
    qed_loop_replacement = annihilation_cut.orientations[
        _loop_edge
    ] * _physical_momentum(_index)
    qed_s = S("s", is_positive=True)
    qed_t, qed_u = S("t", "u")
    qed_me = qed_model.particle("e-").mass
    qed_mm = qed_model.particle("mu-").mass
    qed_e = -qed_model.particle("e-").electric_charge
    annihilation_kin = Kinematics.mandelstam(
        [qed_p1, qed_p2, qed_k1, qed_k2],
        [qed_me**2, qed_me**2, qed_mm**2, qed_mm**2],
        [qed_s, qed_t, qed_u],
    )
    return (
        annihilation_kin,
        edge_momentum,
        qed_e,
        qed_k1,
        qed_k2,
        qed_loop_pattern,
        qed_loop_replacement,
        qed_me,
        qed_mm,
        qed_p1,
        qed_p2,
        qed_s,
        qed_t,
        qed_u,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Sum spins and evaluate the trace
    """)
    return


@app.cell
def _(
    annihilation_kin,
    edge_momentum,
    qed_diagram,
    qed_loop_pattern,
    qed_loop_replacement,
    qed_model,
):
    _projector = qed_diagram.projector_expression()
    for _edge in qed_diagram.external_edges:
        _projector = qed_model.particle(_edge.particle_name).sum_spins(
            _projector,
            edge_momentum(_edge.id),
            edge=_edge.id,
            average=True,
        )
    _numerator = qed_model.expand_couplings(
        (qed_diagram.numerator_expression() * _projector).contract().to_dots()
    )
    annihilation_trace = (
        _numerator.expand()
        .simplify_algebra(contract="dots", gamma=True, epsilon=True)
        .expand()
        .to_expression()
    )
    annihilation_trace = annihilation_kin.apply(
        qed_diagram.loop_momentum_basis.route_expression(annihilation_trace).replace(
            qed_loop_pattern, qed_loop_replacement
        )
    )
    return (annihilation_trace,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Restore the scalar propagators
    """)
    return


@app.cell
def _(
    S,
    Symbols,
    annihilation_cut,
    annihilation_kin,
    annihilation_trace,
    qed_diagram,
):
    annihilation_denominator = qed_diagram.denominator_expression(
        edge_powers={_edge.id: 0 for _edge in annihilation_cut.edges},
        dimension=4,
        in_lmb=True,
    ).to_expression()
    _a, _b, _c, _inverse = S("a_", "b_", "c_", "inverse_")
    annihilation_denominator = annihilation_kin.apply(
        annihilation_denominator.replace(
            Symbols.denominator(_a, _b, _c, _inverse), _inverse
        )
    )
    qed_squared = (
        qed_diagram.overall_factor_expression(evaluate=True)
        * qed_diagram.numerator_prefactor_expression()
        * annihilation_trace
        / annihilation_denominator
    ).together()
    return (qed_squared,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the massive reference
    """)
    return


@app.cell
def _(E, qed_e, qed_me, qed_mm, qed_s, qed_squared, qed_t, qed_u):
    _expected = (
        2
        * qed_e**4
        / qed_s**2
        * (
            2 * qed_me**2 * (2 * qed_mm**2 + qed_s - qed_t - qed_u)
            + 2 * qed_me**4
            + 2 * qed_mm**4
            + 2 * qed_mm**2 * (qed_s - qed_t - qed_u)
            + qed_t**2
            + qed_u**2
        )
    )
    assert (qed_squared - _expected).replace(
        qed_u, 2 * qed_me**2 + 2 * qed_mm**2 - qed_s - qed_t
    ).together() == E("0")
    return


@app.cell(hide_code=True)
def _(mo, qed_diagram, qed_squared):
    mo.vstack(
        [
            mo.md("**Generated graph**"),
            qed_diagram,
            mo.md("**Massive, spin-averaged squared matrix element**"),
            qed_squared,
        ]
    )
    return


@app.cell
def _(
    E,
    Kinematics,
    S,
    Symbol,
    qed_e,
    qed_k1,
    qed_k2,
    qed_me,
    qed_mm,
    qed_p1,
    qed_p2,
    qed_s,
    qed_squared,
    qed_t,
    qed_u,
):
    qed_cos_theta, qed_alpha = S("cos_theta", "alpha")
    _pi = Symbol.PI
    _kin = Kinematics.mandelstam(
        [qed_p1, qed_p2, qed_k1, qed_k2],
        [E("0")] * 4,
        [qed_s, qed_t, qed_u],
    )
    _angular = (
        qed_squared.replace(qed_me, E("0"))
        .replace(qed_mm, E("0"))
        .replace(qed_t, -qed_s * (1 - qed_cos_theta) / 2)
        .replace(qed_u, -qed_s * (1 + qed_cos_theta) / 2)
        .replace(qed_e**4, (4 * _pi * qed_alpha) ** 2)
    )
    qed_differential = (
        _angular * _kin.two_body_phase_space(qed_k1, qed_k2) / _kin.flux(qed_p1, qed_p2)
    ).expand()
    _primitive = (
        qed_differential.to_polynomial().integrate(qed_cos_theta).to_expression()
    )
    qed_total = (
        2
        * _pi
        * (
            _primitive.replace(qed_cos_theta, E("1"))
            - _primitive.replace(qed_cos_theta, E("-1"))
        )
    ).together()
    assert (
        qed_differential - qed_alpha**2 * (1 + qed_cos_theta**2) / (4 * qed_s)
    ).together() == E("0")
    assert (qed_total - 4 * _pi * qed_alpha**2 / (3 * qed_s)).together() == E("0")
    # The Jacobian d(cos(theta))/dt = 2/s gives the second gallery coordinate.
    qed_differential_t = (
        2 * qed_differential.replace(qed_cos_theta, 1 + 2 * qed_t / qed_s) / qed_s
    ).expand()
    _expected_t = (
        qed_alpha**2 * (qed_s**2 + 2 * qed_s * qed_t + 2 * qed_t**2) / qed_s**4
    )
    assert (qed_differential_t - _expected_t).together() == 0
    _primitive_t = qed_differential_t.to_polynomial().integrate(qed_t).to_expression()
    _total_t = (
        2 * _pi * (_primitive_t.replace(qed_t, 0) - _primitive_t.replace(qed_t, -qed_s))
    )
    assert (_total_t - qed_total).together() == 0
    return qed_differential, qed_differential_t, qed_total


@app.cell(hide_code=True)
def _(mo, qed_differential, qed_differential_t, qed_total):
    mo.vstack(
        [
            mo.md(r"**Massless angular distribution** $d\sigma/d\Omega$"),
            qed_differential,
            mo.md(r"**Mandelstam distribution** $d\sigma/(dt\,d\varphi)$"),
            qed_differential_t,
            mo.md(
                r"**Both coordinates give the integrated cross section** $\sigma=4\pi\alpha^2/(3s)$"
            ),
            qed_total,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generated polarized QED cross section

    Spenso constructs the chiral projectors and Idenso evaluates their traces.
    The currents below select right-chiral fermion wavefunctions; in the
    massless limit these describe right-handed particles and left-handed
    antiparticles. No initial spin average is taken for specified states.
    Apply the projectors to the generated interaction vertices. Cut particle
    identities and orientations fix the physical outgoing momenta.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Select right-handed interaction vertices
    """)
    return


@app.cell
def _(Representation, S, TensorExpression, TensorRule, chain, qed_diagram):
    _a, _b, _ell = S("chiral::a_", "chiral::b_", "chiral::ell_")
    _pattern = TensorExpression.dirac_gamma(4)(_a, _b, _ell)
    _right_current = chain(
        Representation.bis(4)(_a),
        Representation.bis(4)(_b),
        TensorExpression.dirac_gamma(4)(_a, "middle", _ell),
        TensorExpression.projp(4)("middle", _b),
    )
    _right_handed = TensorRule(_pattern, _right_current)
    chiral_numerator = TensorExpression(1)
    for _vertex in qed_diagram.vertices:
        chiral_numerator *= _vertex.numerator_expression().replace(_right_handed)
    for _edge in qed_diagram.internal_edges:
        chiral_numerator *= _edge.numerator_expression()
    return (chiral_numerator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Close the incoming spin sums
    """)
    return


@app.cell
def _(Symbols, chiral_numerator, qed_diagram, qed_model):
    _Q = Symbols.edge_momentum
    _projector = qed_diagram.projector_expression()
    for _edge in qed_diagram.external_edges:
        _projector = qed_model.particle(_edge.particle_name).sum_spins(
            _projector, _Q(_edge.id), edge=_edge.id
        )
    chiral_spin_sum = qed_model.expand_couplings(
        (chiral_numerator * _projector).contract().to_dots()
    )
    return (chiral_spin_sum,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Reduce the chiral trace
    """)
    return


@app.cell
def _(
    Kinematics,
    chiral_spin_sum,
    qed_diagram,
    qed_k1,
    qed_k2,
    qed_loop_pattern,
    qed_loop_replacement,
    qed_me,
    qed_mm,
    qed_p1,
    qed_p2,
    qed_s,
    qed_t,
    qed_u,
):
    chiral_trace = (
        chiral_spin_sum.expand()
        .simplify_algebra(contract="dots", gamma=True, epsilon=True)
        .expand()
        .to_expression()
    )
    chiral_kin = Kinematics.mandelstam(
        [qed_p1, qed_p2, qed_k1, qed_k2],
        [qed_me**2, qed_me**2, qed_mm**2, qed_mm**2],
        [qed_s, qed_t, qed_u],
    )
    chiral_trace = chiral_kin.apply(
        qed_diagram.loop_momentum_basis.route_expression(chiral_trace).replace(
            qed_loop_pattern, qed_loop_replacement
        )
    )
    return chiral_kin, chiral_trace


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Restore propagators and graph factors
    """)
    return


@app.cell
def _(
    S,
    Symbols,
    chiral_kin,
    chiral_trace,
    qed_diagram,
    qed_me,
    qed_mm,
    qed_s,
    qed_t,
    qed_u,
):
    _cut = qed_diagram.cuts[0]
    _denominator = qed_diagram.denominator_expression(
        edge_powers={_edge.id: 0 for _edge in _cut.edges}, dimension=4, in_lmb=True
    ).to_expression()
    _q, _m, _power, _inverse = S(
        "chiral::q_", "chiral::m_", "chiral::power_", "chiral::inverse_"
    )
    _denominator = chiral_kin.apply(
        _denominator.replace(Symbols.denominator(_q, _m, _power, _inverse), _inverse)
    )
    qed_polarized = (
        (
            qed_diagram.overall_factor_expression(evaluate=True)
            * qed_diagram.numerator_prefactor_expression()
            * chiral_trace
            / _denominator
        )
        .replace(qed_t, 2 * qed_me**2 + 2 * qed_mm**2 - qed_s - qed_u)
        .factor()
    )
    return (qed_polarized,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the chiral massive and massless results
    """)
    return


@app.cell
def _(E, qed_e, qed_me, qed_mm, qed_polarized, qed_s, qed_t, qed_u):
    assert (
        qed_polarized - 4 * qed_e**4 * (qed_me**2 + qed_mm**2 - qed_u) ** 2 / qed_s**2
    ).replace(qed_u, 2 * qed_me**2 + 2 * qed_mm**2 - qed_s - qed_t).together() == E("0")
    qed_polarized_massless = qed_polarized.replace(qed_me, E("0")).replace(
        qed_mm, E("0")
    )
    assert (qed_polarized_massless - 4 * qed_e**4 * qed_u**2 / qed_s**2).replace(
        qed_u, -qed_s - qed_t
    ).together() == E("0")
    return (qed_polarized_massless,)


@app.cell(hide_code=True)
def _(mo, qed_polarized, qed_polarized_massless):
    mo.vstack(
        [
            mo.md("**Generated massive chiral-projected squared matrix element**"),
            qed_polarized,
            mo.md("**Generated massless polarized squared matrix element**"),
            qed_polarized_massless,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
