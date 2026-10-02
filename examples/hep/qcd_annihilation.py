import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="QCD annihilation: color sums and cross sections",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Different-flavor QCD annihilation

    [Browse all notebooks](/) · [Elastic quark scattering](/?file=hep/quark_scattering.py)

    Generate $b\bar b\to t\bar t$ through gluon exchange, retaining both
    quark masses. Initial spins and colors are averaged; final spins and colors
    are summed. `Particle.color_sum` supplies the shared completeness tensor,
    and Idenso contracts the color and Dirac algebra.

    The symbolic $SU(N_c)$ result uses $T_R=1/2$ and agrees with the
    [FeynCalc reference](https://feyncalc.github.io/FeynCalcExamples/QCD/Tree/QiQibar-QjQjbar).
    The massless $SU(3)$ limit is carried through two-body phase space below.
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
    from symbolica.community import hepkit as hep
    from symbolica.community import tensor as sp
    import marimo as mo
    from symbolica import E, S, Symbol
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hepkit import Kinematics, Model
    from symbolica.community.tensor import TensorExpression

    _set_namespace("qcd_annihilation")
    return E, Kinematics, Model, S, Symbol, TensorExpression, hep, mo, sp


@app.cell(hide_code=True)
def _():
    from symbolica.community.hepkit import Symbols

    return (Symbols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell
def _(Model):
    qcd_model = Model.standard_model()
    return (qcd_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate and inspect the sewn graph
    """)
    return


@app.cell
def _(qcd_model):
    _result = qcd_model.process(
        ["b", "b~"], ["t", "t~"], vertex_allow=["V_76", "V_137"]
    ).generate_cross_section(
        loops=1,
        max_vertices=4,
        maximum_bridges=None,
        numerator_grouping=None,
        progress=None,
    )
    assert len(_result.diagrams) == 1
    qcd_diagram = _result.diagrams[0]
    assert len(qcd_diagram.cuts) == 1
    cut = qcd_diagram.cuts[0]
    assert sorted(_p.name for _p in cut.particles) == ["t", "t~"]
    assert all(_side.loop_count == 0 for _side in (cut.left, cut.right))
    return cut, qcd_diagram


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Define the massive scattering kinematics
    """)
    return


@app.cell
def _(Kinematics, S, Symbols, hep, qcd_model):
    Q, P, K = (
        Symbols.edge_momentum,
        hep.Kinematics.external_momentum,
        hep.Kinematics.loop_momentum,
    )
    s, t, u = S("s", "t", "u")
    mb = qcd_model.particle("b").mass
    mt = qcd_model.particle("t").mass
    gs = qcd_model.parameter("G").symbol
    k1, k2 = S("k1", "k2")
    kinematics = Kinematics.mandelstam(
        [P(0), P(1), k1, k2], [mb**2, mb**2, mt**2, mt**2], [s, t, u]
    )
    return K, P, Q, gs, k1, k2, kinematics, mb, mt, s, t, u


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Sum incoming spins
    """)
    return


@app.cell
def _(Q, qcd_diagram, qcd_model):
    _projector = qcd_diagram.projector_expression()
    for _edge in qcd_diagram.external_edges:
        _projector = qcd_model.particle(_edge.particle_name).sum_spins(
            _projector, Q(_edge.id), edge=_edge.id, average=True
        )
    numerator = qcd_model.expand_couplings(
        qcd_diagram.numerator_expression().to_expression() * _projector
    )

    # Associate open color slots with the generated external wavefunction indices.
    # Slot duals determine the closure orientation, including the antiquark; no
    # fixed half-edge numbering or hand-written SU(3) averaging factor is needed.
    return (numerator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Close and average the incoming color ports
    """)
    return


@app.cell
def _(E, S, TensorExpression, numerator, qcd_diagram, qcd_model, sp):
    _bis, _wave, index_pattern, _metric, _left, _right = (
        sp.Representation.bis,
        S("wave_"),
        S("index_"),
        sp.TensorName.g().to_expression(),
        S("left_"),
        S("right_"),
    )
    _slots = TensorExpression(numerator).structure.slots
    color_projector = E("1")
    initial_color_states = 1
    for _edge in qcd_diagram.external_edges:
        _particle = qcd_model.particle(_edge.particle_name)
        initial_color_states *= abs(_particle.color)
        _indices = [
            dict(_match)[index_pattern]
            for _match in qcd_diagram.projector_expression().match(
                _wave(
                    _edge.id,
                    sp.PortPattern.exact(sp.Representation.bis(4), index_pattern),
                ),
                max_level=0,
            )
        ]
        _closure_slots = [
            _slot.dual().to_expression()
            for _slot in _slots
            if any(
                _slot.to_expression().replace(_i, E("0")) != _slot.to_expression()
                for _i in _indices
            )
        ]
        assert len(_closure_slots) == 2
        _closure = _metric(*_closure_slots)
        _color_indices = dict(
            next(_closure.match(_particle.color_sum(_left, _right), max_level=0))
        )
        color_projector *= _particle.color_sum(
            _color_indices[_left], _color_indices[_right], average=True
        )

    # Check singlet and adjoint sums through the same installed public interface.
    return color_projector, index_pattern, initial_color_states


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the particle color sums
    """)
    return


@app.cell
def _(E, S, TensorExpression, qcd_model):
    _i, _j = S("color_i", "color_j")
    assert qcd_model.particle("e-").color_sum(_i, _j, average=True) == E("1")
    for _name in ("b", "b~", "g"):
        _identity = qcd_model.particle(_name).color_sum(_i, _i, average=True)
        assert TensorExpression(_identity).contract(
            rank_one=False, collect_chains=False, collect_traces=False
        ).to_dots().to_expression() == E("1")

    # A named adjoint dimension is required by Spenso. Impose dA=Nc²-1 after
    # contraction, when the dimensions are ordinary scalar expressions.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Reduce color and Dirac traces
    """)
    return


@app.cell
def _(
    S,
    TensorExpression,
    color_projector,
    index_pattern,
    initial_color_states,
    numerator,
    sp,
):
    Nc, dA, _cof, _coad = (
        S("Nc"),
        S("dA"),
        sp.Representation.cof,
        sp.Representation.coad,
    )
    _generic = (
        (numerator * color_projector * initial_color_states / Nc**2)
        .replace(
            sp.PortPattern.exact(sp.Representation.cof(3), index_pattern),
            sp.PortPattern.exact(sp.Representation.cof(Nc), index_pattern),
        )
        .replace(
            sp.PortPattern.exact(sp.Representation.coad(8), index_pattern),
            sp.PortPattern.exact(sp.Representation.coad(dA), index_pattern),
        )
    )
    contracted = (
        TensorExpression(_generic.expand())
        .simplify_algebra(
            contract="dots",
            color=True,
            color_substitute_cof_dimension_invariants=True,
            gamma=True,
            epsilon=True,
        )
        .expand()
        .to_expression()
    )
    assert TensorExpression(contracted).is_scalar
    return Nc, contracted, dA


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Route the directed cut momentum
    """)
    return


@app.cell
def _(
    E,
    K,
    S,
    Symbols,
    contracted,
    cut,
    index_pattern,
    k1,
    k2,
    kinematics,
    qcd_diagram,
    s,
):
    _loop_edge = qcd_diagram.loop_momentum_basis.loop_edges[0]
    _loop_particle = next(
        _p.name for _edge, _p in zip(cut.edges, cut.particles) if _edge.id == _loop_edge
    )
    _physical_momentum = k1 if _loop_particle == "t" else k2
    routed = qcd_diagram.loop_momentum_basis.route_expression(contracted).replace(
        K(0, index_pattern),
        cut.orientations[_loop_edge] * _physical_momentum(index_pattern),
    )
    _a, _b, _c, _inverse, _denom = (
        S("a_"),
        S("b_"),
        S("c_"),
        S("inverse_"),
        Symbols.denominator,
    )
    denominator = kinematics.apply(
        qcd_diagram.denominator_expression(
            edge_powers={_edge.id: 0 for _edge in cut.edges}, dimension=4, in_lmb=True
        )
        .to_expression()
        .replace(_denom(_a, _b, _c, _inverse), _inverse)
    )
    assert (denominator - s**2).expand() == E("0")
    return denominator, routed


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Compare with the massive SU(N) reference
    """)
    return


@app.cell
def _(
    E,
    Nc,
    dA,
    denominator,
    gs,
    kinematics,
    mb,
    mt,
    qcd_diagram,
    routed,
    s,
    t,
    u,
):
    qcd_squared = (
        qcd_diagram.overall_factor_expression(evaluate=True)
        * qcd_diagram.numerator_prefactor_expression()
        * kinematics.apply(routed)
        / denominator
    ).replace(dA, Nc**2 - 1)
    _expected = (
        (Nc**2 - 1)
        * gs**4
        / (2 * Nc**2 * s**2)
        * (
            2 * mb**2 * (2 * mt**2 + s - t - u)
            + 2 * mb**4
            + 2 * mt**4
            + 2 * mt**2 * (s - t - u)
            + t**2
            + u**2
        )
    )
    assert (qcd_squared - _expected).replace(
        u, 2 * mb**2 + 2 * mt**2 - s - t
    ).together() == E("0")
    qcd_massless = qcd_squared.replace(mb, E("0")).replace(mt, E("0"))
    assert (
        qcd_massless - (Nc**2 - 1) * gs**4 * (t**2 + u**2) / (2 * Nc**2 * s**2)
    ).replace(u, -s - t).together() == E("0")
    su3_squared = qcd_massless.replace(Nc, E("3"))
    assert (su3_squared - 4 * gs**4 * (t**2 + u**2) / (9 * s**2)).replace(
        u, -s - t
    ).together() == E("0")

    # Carry the generated massless SU(3) result through physical two-body phase space.
    return qcd_massless, qcd_squared, su3_squared


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Integrate the massless cross section
    """)
    return


@app.cell
def _(E, Kinematics, P, S, Symbol, gs, k1, k2, s, su3_squared, t, u):
    _s_physical = S("s_physical", is_positive=True)
    _costheta, _alpha_s = S("costheta", "alpha_s")
    _pi = Symbol.PI
    _massless_kinematics = Kinematics.mandelstam(
        [P(0), P(1), k1, k2], [E("0")] * 4, [_s_physical, t, u]
    )
    _angular = (
        su3_squared.replace(s, _s_physical)
        .replace(t, -_s_physical * (1 - _costheta) / 2)
        .replace(u, -_s_physical * (1 + _costheta) / 2)
        .replace(gs**4, (4 * _pi * _alpha_s) ** 2)
    )
    qcd_differential = (
        _angular
        * _massless_kinematics.two_body_phase_space(k1, k2)
        / _massless_kinematics.flux(P(0), P(1))
    ).expand()
    assert (
        qcd_differential - _alpha_s**2 * (1 + _costheta**2) / (18 * _s_physical)
    ).together() == E("0")
    _primitive = qcd_differential.to_polynomial().integrate(_costheta).to_expression()
    qcd_total = (
        2
        * _pi
        * (
            _primitive.replace(_costheta, E("1"))
            - _primitive.replace(_costheta, E("-1"))
        )
    )
    assert (qcd_total - 8 * _pi * _alpha_s**2 / (27 * _s_physical)).together() == E("0")
    return qcd_differential, qcd_total


@app.cell(hide_code=True)
def _(
    E,
    Nc,
    color_projector,
    mb,
    mo,
    mt,
    qcd_diagram,
    qcd_differential,
    qcd_massless,
    qcd_squared,
    qcd_total,
    s,
    t,
    u,
):
    mo.vstack(
        [
            qcd_diagram,
            mo.md("**Initial color projector, including the color average**"),
            color_projector,
            mo.md("**Massive, spin- and color-averaged squared matrix element**"),
            qcd_squared.replace(u, 2 * mb**2 + 2 * mt**2 - s - t).together(),
            mo.md("**Massless SU(3) result**"),
            qcd_massless.replace(Nc, E("3")).together(),
            mo.md("**Massless angular distribution**"),
            qcd_differential,
            mo.md("**Integrated massless cross section**"),
            qcd_total.together(),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
