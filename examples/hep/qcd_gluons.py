import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Quark annihilation into gluons")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Quark annihilation into gluons

    [Browse all notebooks](/) · [Quark–gluon scattering](/?file=hep/quark_gluon_scattering.py)

    Generate $b\bar b\to gg$ with the bottom mass retained. The three diagrams
    include both quark-exchange channels and the three-gluon vertex. Summing
    amplitudes before forming their Dirac adjoint includes all interference.

    `Particle.color_sum` averages incoming colors and sums outgoing colors.
    `Particle.spin_sum` supplies fermion completeness and physical gluon
    polarizations. Idenso performs the color and Dirac algebra using Spenso
    representations, with $T_R=1/2$ and four-dimensional external spins.

    Two physical polarization references give the same massive $SU(N_c)$ result:
    the opposite final gluon and the incoming massive quark. The checks below
    compare against [FeynCalc](https://feyncalc.github.io/FeynCalcExamples/QCD/Tree/QQbar-GlGl)
    and verify the massless $SU(3)$ limit and exchange of the final gluons.
    The symbolic calculation takes a few minutes.
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
    from symbolica import E, Replacement, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hepkit import Kinematics, Model
    from symbolica.community.tensor import TensorExpression

    _set_namespace("qcd_gluons")
    return E, Kinematics, Model, Replacement, S, TensorExpression, hep, mo, sp


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
    ## Declare the massive kinematics and color group
    """)
    return


@app.cell
def _(E, Kinematics, S, hep, qcd_model, sp):
    P = hep.Kinematics.external_momentum
    s, t, u = S("s", "t", "u")
    mass = qcd_model.particle("b").mass
    gs = qcd_model.parameter("G").symbol
    Nc, dA, _cof, _coad = (
        sp.Nc,
        S("dA"),
        sp.Representation.cof,
        sp.Representation.coad,
    )
    kinematics = Kinematics.mandelstam(
        [P(0), P(1), P(2), P(3)], [mass**2, mass**2, E("0"), E("0")], [s, t, u]
    )
    return Nc, P, dA, gs, kinematics, mass, s, t, u


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate the coherent amplitude
    """)
    return


@app.cell
def _(S, qcd_model, sp):
    gluon_generated = qcd_model.process(
        ["b", "b~"], ["g", "g"], vertex_allow=["V_76", "V_36"]
    ).generate_amplitude(
        loops=0,
        max_vertices=2,
        maximum_bridges=None,
        numerator_grouping=None,
        progress=None,
    )
    assert len(gluon_generated.diagrams) == 3
    ports = S("external_0", "external_1", "external_2", "external_3")
    bra_ports = {port: S(f"bra_{i}") for i, port in enumerate(ports)}
    a, b, c, inverse, _wave, _rep, index = S(
        "a_", "b_", "c_", "inverse_", "wave_", "rep_", "index_"
    )
    conjugate, adjoint_index = (
        sp.BroadcastFunction.conj().to_expression(),
        S("adjoint_index"),
    )
    return (
        a,
        adjoint_index,
        b,
        bra_ports,
        c,
        conjugate,
        gluon_generated,
        index,
        inverse,
        ports,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Align the physical labels

    The native amplitude owns the diagram weights and relative signs. Relabel its external ports for the explicit projectors below.
    """)
    return


@app.cell
def _(
    Symbols,
    a,
    b,
    c,
    gluon_generated,
    inverse,
    kinematics,
    mass,
    ports,
    s,
    t,
    u,
):
    _denominators = [
        kinematics.apply(
            diagram.denominator_expression(dimension=4, in_lmb=True)
            .to_expression()
            .replace(Symbols.denominator(a, b, c, inverse), inverse)
        ).expand()
        for diagram in gluon_generated.diagrams
    ]
    assert set(_denominators) == {s, t - mass**2, u - mass**2}
    _port_map = {leg.tensor_index: ports[leg.index] for leg in gluon_generated.legs}
    operator = kinematics.apply(
        gluon_generated.expression().rename_indices(_port_map)
    ).expand()
    assert len(operator.structure.slots) == 8
    return (operator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Form the physical adjoint

    Give its external ports distinct labels and scope its internal dummy indices independently.
    """)
    return


@app.cell
def _(
    P,
    a,
    adjoint_index,
    b,
    bra_ports,
    conjugate,
    gs,
    mass,
    operator,
    s,
    sp,
    t,
    u,
):
    adjoint = (
        operator.dirac_adjoint()
        .simplify_algebra(
            contract="dots",
            color=False,
            gamma=True,
            gamma0=True,
            gamma_evaluate_traces=False,
        )
        .expand()
        .to_expression()
    )
    adjoint = adjoint.replace(conjugate(P(a, b)), P(a, b))
    for _real in (mass, gs, s, t, u):
        adjoint = adjoint.replace(conjugate(_real), _real)
    adjoint = (
        sp.TensorExpression(adjoint)
        .wrap_indices(adjoint_index, dummies_only=True)
        .rename_indices(bra_ports)
        .to_expression()
    )
    return (adjoint,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Sum the color states

    Match each particle completeness tensor to the dual external slot; this fixes quark and antiquark orientation.
    """)
    return


@app.cell
def _(E, S, bra_ports, operator, ports, qcd_model, sp):
    # Match the shared completeness tensor against the dual of each external slot.
    # Matching selects color slots and determines the quark/antiquark orientation.
    _left, _right, _metric = (
        S("left_"),
        S("right_"),
        sp.TensorName.g().to_expression(),
    )
    gluon_color_projector = E("1")
    initial_colors = 1
    for _position, _name in enumerate(("b", "b~", "g", "g")):
        _particle = qcd_model.particle(_name)
        if _position < 2:
            initial_colors *= abs(_particle.color)
        _matches = []
        for _slot in operator.structure.slots:
            _original = _slot.to_expression()
            if _original.replace(ports[_position], E("0")) == _original:
                continue
            _closure = _metric(
                _slot.dual().to_expression(),
                _original.replace(
                    ports[_position],
                    bra_ports[ports[_position]],
                ),
            )
            _match = next(
                _closure.match(_particle.color_sum(_left, _right), max_level=0), None
            )
            if _match is not None:
                _matches.append(_match)
        assert len(_matches) == 1
        _indices = dict(_matches[0])
        gluon_color_projector *= _particle.color_sum(
            _indices[_left], _indices[_right], average=_position < 2
        )
    return gluon_color_projector, initial_colors


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Promote to a symbolic color group

    Replace the incoming SU(3) average with $1/N_c^2$. Keep the adjoint dimension named until the color connections have contracted.
    """)
    return


@app.cell
def _(
    Nc,
    adjoint,
    dA,
    gluon_color_projector,
    index,
    initial_colors,
    operator,
    sp,
):
    # Spenso dimensions are integers or symbols. Keep dA symbolic until color
    # contraction, then impose the SU(N) relation and convert scalar Casimirs.
    generic = (
        (
            operator.to_expression()
            * adjoint
            * gluon_color_projector
            * initial_colors
            / Nc**2
        )
        .replace(
            sp.PortPattern.exact(sp.Representation.cof(3), index),
            sp.PortPattern.exact(sp.Representation.cof(Nc), index),
        )
        .replace(
            sp.PortPattern.exact(sp.Representation.coad(8), index),
            sp.PortPattern.exact(sp.Representation.coad(dA), index),
        )
    )
    return (generic,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Reduce color before adding polarizations

    This keeps the larger Lorentz sums factorized. Then impose $d_A=N_c^2-1$ in the scalar color invariants.
    """)
    return


@app.cell
def _(Nc, TensorExpression, dA, generic):
    colored = (
        TensorExpression(generic)
        .simplify_algebra(contract="dots", gamma=False, color=True)
        .to_expression()
        .replace(dA, Nc**2 - 1)
    )
    colored = TensorExpression(colored).simplify_algebra(
        contract="dots",
        gamma=False,
        color=True,
        color_substitute_cof_dimension_invariants=True,
    )
    return (colored,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Close the fermion line
    """)
    return


@app.cell
def _(P, bra_ports, colored, ports, qcd_model):
    # Dirac adjunction exchanges the two endpoints of the open fermion chain.
    # Reduce the fermion trace before introducing the physical gluon projectors.
    _spin_projector = qcd_model.particle("b").spin_sum(
        P(0),
        ports[0],
        bra_ports[ports[1]],
        average=True,
    ) * qcd_model.particle("b~").spin_sum(
        P(1),
        bra_ports[ports[0]],
        ports[1],
        average=True,
    )
    spin_summed = (
        (colored * _spin_projector)
        .simplify_algebra(contract="dots", color=False, gamma=True, epsilon=True)
        .expand()
        .to_expression()
    )
    return (spin_summed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## State the independent massive reference
    """)
    return


@app.cell
def _(Nc, gs, mass, s, t, u):
    expected = (
        (Nc**2 - 1)
        * gs**4
        * (
            mass**4 * (3 * t**2 + 14 * t * u + 3 * u**2)
            - mass**2 * (t**3 + 7 * t**2 * u + 7 * t * u**2 + u**3)
            - 6 * mass**8
            + t * u * (t**2 + u**2)
        )
        * (
            -2 * Nc**2 * mass**2 * (t + u)
            + 2 * Nc**2 * mass**4
            + Nc**2 * (t**2 + u**2)
            - s**2
        )
        / (2 * Nc**3 * s**2 * (u - mass**2) ** 2 * (t - mass**2) ** 2)
    )
    return (expected,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check two physical polarization references

    Add gluon completeness only after the fermion trace. Both reference choices must give the same scalar result.
    """)
    return


@app.cell
def _(
    E,
    P,
    TensorExpression,
    bra_ports,
    expected,
    kinematics,
    mass,
    ports,
    qcd_model,
    s,
    spin_summed,
    t,
    u,
):
    gluon_results = []
    for _references in ((P(3), P(2)), (P(0), P(0))):
        _polarizations = E("1")
        for _position, _reference in zip((2, 3), _references):
            _polarizations *= qcd_model.particle("g").spin_sum(
                P(_position),
                ports[_position],
                bra_ports[ports[_position]],
                reference=_reference,
            )
        _scalar = (
            TensorExpression(spin_summed * _polarizations)
            .contract()
            .to_dots()
            .expand()
            .to_expression()
        )
        assert TensorExpression(_scalar).is_scalar
        _squared = kinematics.apply(_scalar).replace(s, 2 * mass**2 - t - u).together()
        assert (_squared - expected.replace(s, 2 * mass**2 - t - u)).together() == E(
            "0"
        )
        gluon_results.append(_squared)
        print(
            f"Generated massive SU(N) quark annihilation into gluons: references={_references} passed"
        )
    assert gluon_results[0] == gluon_results[1]
    return (gluon_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the massless limit and Bose symmetry
    """)
    return


@app.cell
def _(E, Nc, Replacement, gluon_results, gs, mass, s, t, u):
    gluon_massless = gluon_results[0].replace(mass, E("0")).replace(Nc, E("3"))
    _expected_massless = (
        E("32/27") * gs**4 * (t**2 + u**2) / (t * u)
        - E("8/3") * gs**4 * (t**2 + u**2) / s**2
    )
    assert (gluon_massless - _expected_massless.replace(s, -t - u)).together() == E("0")
    # Bose exchange acts on the labeled squared amplitude. A cross section integrated
    # over both identical-gluon labels would additionally require a factor of 1/2!.
    assert (
        gluon_results[0]
        - gluon_results[0].replace_multiple([Replacement(t, u), Replacement(u, t)])
    ).together() == E("0")
    print(
        "Massive gauge-reference independence, Bose symmetry and massless SU(3) reference passed"
    )
    return (gluon_massless,)


@app.cell(hide_code=True)
def _(
    gluon_color_projector,
    gluon_generated,
    gluon_massless,
    gluon_results,
    mo,
):
    mo.vstack(
        [
            mo.md("**The three generated amplitudes**"),
            mo.hstack(list(gluon_generated.diagrams)),
            mo.md("**Color completeness, including the initial average**"),
            gluon_color_projector,
            mo.md("**Massive spin- and color-averaged squared amplitude**"),
            gluon_results[0].factor(),
            mo.md("**Massless SU(3) limit**"),
            gluon_massless.factor(),
            mo.md(
                "Both polarization references and the Bose-exchange check agree. This is a labeled squared amplitude; integrating over both identical final-gluon labels requires the usual factor of 1/2!."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
