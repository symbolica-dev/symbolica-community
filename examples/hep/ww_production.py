import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="W-pair production and electroweak cancellations",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # $e^-e^+\to W^-W^+$
    [Browse all notebooks](/) · [W and top decays](/?file=hep/weak_decays.py) · [Muon decay](/?file=hep/muon_decay.py)

    Generate all four tree diagrams with `Model.standard_model()`: photon, Z, neutrino and Higgs exchange. Shared particle spin sums average the incoming spins and include all physical W polarizations. The calculation keeps the electron mass and all sixteen interference products.

    The [FeynCalc example](https://feyncalc.github.io/FeynCalcExamples/EW/Tree/AnelEl-WW) provides the full massive squared amplitude and the massless-electron total rate. Both are checked below. The first symbolic contraction takes several minutes; subsequent changes to the numerical controls reuse it.

    Write $s=(p_{e^-}+p_{e^+})^2$ and $t=(p_{e^-}-p_{W^-})^2$, with $m_Z=m_W/\cos\theta_W$. Widths are zero. The marker $H$ multiplies only the Higgs diagram; the Standard Model is $H=1$.
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
    import numpy as np
    from symbolica import E, S, Symbol
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hepkit import Kinematics, Model, RenderSettings
    from symbolica.community.tensor import TensorExpression

    _set_namespace("ww")
    return (
        E,
        Kinematics,
        Model,
        RenderSettings,
        S,
        Symbol,
        TensorExpression,
        hep,
        mo,
        np,
        sp,
    )


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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate the four exchange diagrams
    """)
    return


@app.cell
def _(Model):
    model = Model.standard_model()
    _vertices = [
        _v
        for _v in model.vertex_rules
        if set(_v.particles) <= {"e-", "e+", "a", "Z", "W-", "W+", "ve", "ve~", "H"}
    ]
    generated = model.process(
        ["e-", "e+"], ["W-", "W+"], vertex_allow=_vertices
    ).generate_diagrams(
        max_vertices=2, maximum_bridges=None, numerator_grouping=None, progress=None
    )
    assert len(generated.diagrams) == 4
    assert {_d.internal_edges[0].particle_name for _d in generated.diagrams} == {
        "a",
        "Z",
        "H",
        "ve",
    }
    return generated, model


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Declare the masses and Mandelstam invariants
    """)
    return


@app.cell
def _(E, Kinematics, S, Symbol, hep, model):
    P = hep.Kinematics.external_momentum
    charge = -model.particle("e-").electric_charge
    sw = model.parameter("sw").symbol
    cw = model.parameter("cw").symbol
    mw = model.particle("W+").mass
    mz = model.particle("Z").mass
    mh = model.particle("H").mass
    me = model.particle("e-").mass
    ye = model.parameter("ye").symbol
    vev = model.parameter("vev").symbol
    yme = model.parameter("yme").symbol
    s, t, u, H = S("s", "t", "u", "Higgs")
    zero, one, pi = (E("0"), E("1"), Symbol.PI)
    kin = Kinematics.mandelstam(
        [P(_i) for _i in range(4)], [me**2, me**2, mw**2, mw**2], [s, t, u]
    )
    return (
        H,
        P,
        charge,
        cw,
        kin,
        me,
        mh,
        mw,
        mz,
        one,
        pi,
        s,
        sw,
        t,
        u,
        vev,
        ye,
        yme,
        zero,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Align the operators and remove the longitudinal Z term

    The explicit transverse-polarization check justifies dropping that term before forming the square.
    """)
    return


@app.cell
def _(E, S, Symbols, generated, kin, me, model, sp, vev, ye, yme):
    ports = S("port0", "port1", "port2", "port3")
    # Give the conjugate amplitude distinct external labels; scope only its dummies.
    bra_ports = {port: S(f"bra_{i}") for i, port in enumerate(ports)}
    a, b, c, inverse, wave, rep, index = S(
        "a_", "b_", "c_", "inverse_", "wave_", "rep_", "index_"
    )
    conjugate, wrapped = (sp.BroadcastFunction.conj().to_expression(), S("adjoint"))
    raw_diagram_terms = {}
    for _diagram in generated.diagrams:
        _numerator = model.expand_couplings(
            _diagram.numerator_expression(in_lmb=True).to_expression()
        )
        _numerator = (
            _numerator.replace(ye, model.parameter("ye").expression)
            .replace(yme, me)
            .replace(vev, model.parameter("vev").expression)
            .replace(E("1/2").sqrt(), E("2").sqrt() / 2)
        )
        for _edge in _diagram.external_edges:
            _matches = list(
                _diagram.projector_expression().match(
                    wave(_edge.id, rep(4, index)), max_level=0
                )
            )
            assert len(_matches) == 1
            _match = dict(_matches[0])
            _numerator = _numerator.replace(
                _match[rep](4, _match[index]),
                _match[rep](4, ports[_edge.external_index]),
            )
        _denominator = kin.apply(
            _diagram.denominator_expression(dimension=4, in_lmb=True)
            .to_expression()
            .replace(Symbols.denominator(a, b, c, inverse), inverse)
        ).expand()
        _particle_name = _diagram.internal_edges[0].particle_name
        _term = (
            _numerator
            * _diagram.overall_factor_expression(evaluate=True)
            * _diagram.numerator_prefactor_expression()
            / _denominator
        )
        raw_diagram_terms[_particle_name] = _term
    return a, b, bra_ports, conjugate, ports, raw_diagram_terms, wrapped


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Remove the longitudinal Z term

    Its contraction with physical W polarization sums must vanish.
    """)
    return


@app.cell
def _(
    P,
    S,
    TensorExpression,
    a,
    kin,
    me,
    model,
    mw,
    mz,
    ports,
    raw_diagram_terms,
    s,
    t,
    u,
    zero,
):
    diagram_terms = dict(raw_diagram_terms)
    _polar_ports = S("physical2", "physical3")
    _projection = model.particle("W-").spin_sum(
        P(2), ports[2], _polar_ports[0]
    ) * model.particle("W+").spin_sum(P(3), ports[3], _polar_ports[1])
    _longitudinal = (diagram_terms["Z"] * (s - mz**2)).expand().coefficient(
        mz ** (-2)
    ) * mz ** (-2)
    _transverse_test = (_longitudinal * _projection).replace(
        P(3, a), P(0, a) + P(1, a) - P(2, a)
    )
    _transverse_test = (
        TensorExpression(_transverse_test).contract().to_dots().expand().to_expression()
    )
    _transverse_test = (
        kin.apply(_transverse_test).replace(u, 2 * me**2 + 2 * mw**2 - s - t).expand()
    )
    assert _transverse_test == zero
    diagram_terms["Z"] = (
        (diagram_terms["Z"] - _longitudinal / (s - mz**2)).together().expand()
    )
    return (diagram_terms,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Prepare the spin sums and adjoints
    """)
    return


@app.cell
def _(
    H,
    P,
    TensorExpression,
    a,
    b,
    bra_ports,
    charge,
    conjugate,
    cw,
    diagram_terms,
    me,
    mh,
    model,
    mw,
    mz,
    ports,
    s,
    sw,
    t,
    u,
    wrapped,
):
    spins = model.particle("e-").spin_sum(
        P(0), ports[0], bra_ports[ports[1]], average=True
    ) * model.particle("e+").spin_sum(P(1), bra_ports[ports[0]], ports[1], average=True)
    density = model.particle("W-").spin_sum(
        P(2), ports[2], bra_ports[ports[2]]
    ) * model.particle("W+").spin_sum(P(3), ports[3], bra_ports[ports[3]])
    operators = {}
    adjoints = {}
    for _particle_name, _term in diagram_terms.items():
        _operator = TensorExpression(_term.expand())
        _adjoint = (
            _operator.dirac_adjoint()
            .simplify_algebra(
                contract="dots", gamma=True, gamma0=True, gamma_evaluate_traces=False
            )
            .expand()
            .to_expression()
            .replace(conjugate(P(a, b)), P(a, b))
        )
        for _real in (charge, sw, cw, mw, mz, mh, me, s, t, u, H):
            _adjoint = _adjoint.replace(conjugate(_real), _real)
        operators[_particle_name] = _operator
        adjoints[_particle_name] = (
            TensorExpression(_adjoint)
            .wrap_indices(wrapped, dummies_only=True)
            .rename_indices(bra_ports)
            .to_expression()
        )
    return adjoints, density, operators, spins


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Contract each interference pair
    """)
    return


@app.cell
def _(
    H,
    P,
    TensorExpression,
    a,
    adjoints,
    cw,
    density,
    kin,
    me,
    mw,
    mz,
    operators,
    s,
    spins,
    sw,
    t,
    u,
    zero,
):
    squared = zero
    pair_results = {}
    for _left, _left_operator in operators.items():
        for _right, _right_adjoint in adjoints.items():
            _paired = _left_operator * _right_adjoint * spins
            _paired = _paired.expand()
            _traced = (
                _paired.simplify_algebra(contract="dots", gamma=True, epsilon=True)
                .expand()
                .to_expression()
            )
            _scalar = (
                TensorExpression(_traced * density)
                .contract()
                .to_dots()
                .expand()
                .to_expression()
            )
            assert TensorExpression(_scalar).is_scalar
            _result = (
                kin.apply(_scalar)
                .replace(u, 2 * me**2 + 2 * mw**2 - s - t)
                .replace(mz, mw / cw)
                .replace(cw, (1 - sw**2).sqrt())
                .together()
            )
            _conserved = TensorExpression(_result).undo_dots().to_expression()
            _conserved = _conserved.replace(P(3, a), P(0, a) + P(1, a) - P(2, a))
            _result = (
                TensorExpression(_conserved)
                .simplify_algebra(
                    contract="dots", gamma=False, color=False, epsilon=True
                )
                .expand()
                .to_expression()
                .together()
            )
            # Epsilon reduction can introduce new dots after momentum conservation.
            _result = (
                kin.apply(_result).replace(u, 2 * me**2 + 2 * mw**2 - s - t).together()
            )
            pair_results[_left, _right] = _result
            squared += _result * H ** (int(_left == "H") + int(_right == "H"))
        print("Contracted interference row", _left, flush=True)
    squared = squared.together()
    return pair_results, squared


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check pair symmetry
    """)
    return


@app.cell
def _(pair_results, zero):
    for (_left, _right), _result in pair_results.items():
        assert (_result - pair_results[_right, _left]).together() == zero, (
            _left,
            _right,
        )
    print(
        "All four generated diagrams and sixteen interference products contracted",
        flush=True,
    )
    return


@app.cell
def _(RenderSettings, generated, mo, squared):
    # Native HEPKit settings and snapshots retain each diagram's physics styles.
    _config = RenderSettings()
    diagram_drawings = [diagram.render(config=_config) for diagram in generated.diagrams]
    mo.vstack(
        [
            mo.hstack(diagram_drawings),
            mo.md(
                "All sixteen products contracted; reversed interference pairs agree exactly. The longitudinal Z numerator was certified to vanish against the two physical W sums before removal."
            ),
            mo.accordion(
                {"Full squared amplitude, with Higgs marker H": mo.vstack([squared])}
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The massive high-energy cancellation
    The full generated result is compared exactly with the published massive expression. At fixed angle, inspect the coefficient growing as $s$ in the spin-averaged squared amplitude. It must vanish when the Higgs diagram is included. The electron's Yukawa coupling and mass are kept consistent in this check.
    """)
    return


@app.cell
def _(E, H, S, charge, cw, me, mh, mw, one, s, squared, sw, t, zero):
    reference = E(
        "\n- ((pi^2*alpha^2*((2*s^2*(s- mh^2)^2*me^8+ 4*s*(s- mh^2)*((- ((s- 4*t*sw^2)*mw^2)-\n2*s*t*(sw^2- 1))*mh^2+ s*((- 4*t*sw^2+ s+ 2*t)*mw^2+ s*t*(2*sw^2- 1)))*me^6+\n2*(((96*t^2*sw^4- 16*s*t*sw^2- 3*s^2)*mw^4- 2*s*t*(16*t*sw^4+ 4*(2*s- 3*t)*sw^2+\n3*s)*mw^2+ s^2*t*(8*t*sw^4- 12*t*sw^2+ s+ 6*t))*mh^4- 2*s*((96*t^2*sw^4- 16*t*(s+\n3*t)*sw^2+ s*(2*t- 3*s))*mw^4- s*t*(32*t*sw^4+ 8*(2*s- 3*t)*sw^2+ 9*s+ 4*t)*mw^2+\ns^2*t*(8*t*sw^4- 8*t*sw^2+ s+ 4*t))*mh^2+ s^2*((96*t^2*sw^4- 16*t*(s+ 6*t)*sw^2- 3*s^2+\n24*t^2+ 4*s*t)*mw^4- 4*s*t*(8*t*sw^4+ (4*s- 6*t)*sw^2+ 3*s+ 4*t)*mw^2+ s^2*t*(8*t*sw^4-\n4*t*sw^2+ s+ 4*t)))*me^4- (4*(2*(- 48*t^2*sw^4- 2*s*t*sw^2+ s^2)*mw^6+ 2*t*sw^2*(s*(5*s-\n4*t)- 24*(s- 2*t)*t*sw^2)*mw^4+ s*t*(8*(s- 4*t)*t*sw^4+ 12*t^2*sw^2- s*(2*s+ 3*t))*mw^2+\ns^2*t^2*(4*(s+ 2*t)*sw^4- 2*(s+ 3*t)*sw^2+ s+ 2*t))*mh^4- 4*s*(4*(- 48*t^2*sw^4- 2*(s-\n6*t)*t*sw^2+ s*(s+ t))*mw^6+ 2*t*(- 48*(s- 2*t)*t*sw^4+ 2*(5*s^2- 10*t*s- 12*t^2)*sw^2-\ns*(s+ t))*mw^4- s*t*(- 16*(s- 4*t)*t*sw^4+ 4*(s- 6*t)*t*sw^2+ 4*s^2+ 2*t^2+ 5*s*t)*mw^2+\ns^2*t^2*(8*(s+ 2*t)*sw^4- 2*(s+ 4*t)*sw^2+ s+ 3*t))*mh^2+ s^2*(8*(- 48*t^2*sw^4- 2*(s-\n12*t)*t*sw^2+ s*(s+ 2*t))*mw^6+ 4*t*(- 48*(s- 2*t)*t*sw^4+ 2*(5*s^2- 16*t*s-\n24*t^2)*sw^2+ s*(t- 2*s))*mw^4- 4*s*t*(- 8*(s- 4*t)*t*sw^4+ 4*(s- 3*t)*t*sw^2+ 2*s^2+\n2*t^2+ 3*s*t)*mw^2+ s^2*t^2*(16*(s+ 2*t)*sw^4- 8*t*sw^2+ s+ 4*t)))*me^2+ 2*(s-\nmh^2)^2*(4*(24*t^2*sw^4+ 4*s*t*sw^2+ s^2)*mw^8- 8*t*(4*t*(s+ 6*t)*sw^4+ s*(3*t-\n4*s)*sw^2+ s^2)*mw^6+ t*(8*t*(17*s^2+ 20*t*s+ 12*t^2)*sw^4- 20*s^2*t*sw^2+ s^2*(4*s+\n5*t))*mw^4- 2*s*t^2*(8*(2*s^2+ 3*t*s+ 2*t^2)*sw^4- 4*(2*s^2+ 2*t*s+ t^2)*sw^2+ s*(2*s+\nt))*mw^2+ s^2*t^3*(s+ t)*(8*sw^4- 4*sw^2+ 1)))*mw^4- 2*s*(1- sw^2)*(2*s^2*(s-\nmh^2)^2*me^8+ 2*s*(s- mh^2)*(((4*t*sw^2- 2*s+ 2*t)*mw^2+ s*t*(3- 2*sw^2))*mh^2+ s*(2*(-\n2*t*sw^2+ s+ t)*mw^2+ s*t*(2*sw^2- 1)))*me^6+ 2*(((s*(2*t- 3*s)- 8*(s-\n3*t)*t*sw^2)*mw^4+ s*t*((4*t- 8*s)*sw^2- 5*s+ 6*t)*mw^2+ s^2*t*(- 4*t*sw^2+ s+\n3*t))*mh^4+ s*(2*(3*s^2+ 8*t*sw^2*s- 4*t*s+ 6*t^2)*mw^4+ 4*s*t*((4*s- 2*t)*sw^2+ 4*s-\nt)*mw^2+ s^2*t*(4*t*sw^2- 2*s- 3*t))*mh^2+ s^2*((- 3*s^2+ 6*t*s+ 12*t^2- 8*t*(s+\n3*t)*sw^2)*mw^4- s*t*((8*s- 4*t)*sw^2+ 11*s+ 10*t)*mw^2+ s^2*t*(s+ 2*t)))*me^4+ (-\n2*(4*(s*(s+ t)- t*(s+ 12*t)*sw^2)*mw^6+ 2*t*((5*s^2- 16*t*s+ 24*t^2)*sw^2+ s*(5*s+\nt))*mw^4- s*t*(4*s^2+ t*s- 6*t^2+ 4*t*(t- s)*sw^2)*mw^2+ s^2*t^2*(- 2*t*sw^2+ s+\nt))*mh^4+ s*(8*(2*s^2+ 4*t*s+ 3*t^2- 2*t*(s+ 6*t)*sw^2)*mw^6+ 4*t*(8*s^2- 3*t*s- 6*t^2+\n2*(5*s^2- 22*t*s+ 12*t^2)*sw^2)*mw^4- 2*s*t*(8*s^2+ t*s- 8*t^2- 4*(s- 2*t)*t*sw^2)*mw^2+\ns^2*t^2*(4*s*sw^2+ s+ 2*t))*mh^2- 4*s^2*(2*(s^2- t*sw^2*s+ 3*t*s+ 3*t^2)*mw^6- t*(-\n3*s^2+ (28*t- 5*s)*sw^2*s+ t*s+ 6*t^2)*mw^4- s*t*(2*s^2+ t*s- t^2+ 2*t^2*sw^2)*mw^2+\ns^2*t^2*(s+ t)*sw^2))*me^2+ 4*(s- mh^2)^2*mw^2*(2*(2*t*(s+ 3*t)*sw^2+ s*(s+ t))*mw^6-\nt*((- 8*s^2+ 10*t*s+ 24*t^2)*sw^2+ 3*s*t)*mw^4+ 2*t*(s^3+ 2*t*(3*s^2+ 5*t*s+\n3*t^2)*sw^2)*mw^2- s*t^3*(s+ t)*(2*sw^2- 1)))*mw^2+ (2*s^2*(s- mh^2)^2*me^8+ 4*s*(s-\nmh^2)*((s*t- (s- 2*t)*mw^2)*mh^2+ s^2*mw^2)*me^6+ 2*(((- 3*s^2+ 4*t*s+ 12*t^2)*mw^4-\n4*s*(s- 2*t)*t*mw^2+ s^2*t*(s+ t))*mh^4- 2*s^2*(- 3*(s- 2*t)*mw^4+ t*(4*t- 7*s)*mw^2+\ns^2*t)*mh^2+ s^2*((- 3*s^2+ 8*t*s+ 12*t^2)*mw^4- 2*s*t*(5*s+ 4*t)*mw^2+ s^2*t*(s+\nt)))*me^4+ (- ((8*s*(s+ 2*t)*mw^6+ 4*t*(10*s^2+ 13*t*s+ 12*t^2)*mw^4- 4*s*t*(2*s^2+ t*s-\n2*t^2)*mw^2+ s^3*t^2)*mh^4)- 8*s*mw^2*(- 2*(s^2+ 3*t*s+ 3*t^2)*mw^4- 3*t*(3*s^2+ 3*t*s+\n2*t^2)*mw^2+ s*t*(2*s^2+ t*s- t^2))*mh^2+ 8*s^2*mw^2*(- ((s^2+ 4*t*s+ 6*t^2)*mw^4)-\n4*s*t*(s+ t)*mw^2+ s^2*t*(s+ t)))*me^2+ 8*(s- mh^2)^2*mw^4*((s^2+ 2*t*s+ 3*t^2)*mw^4+\n2*t*(s^2- 2*t*s- 3*t^2)*mw^2+ t*(s^3+ 3*t*s^2+ 5*t^2*s+ 3*t^3)))*(s-\ns*sw^2)^2))/(2*s^2*t^2*(s- mh^2)^2*mw^4*(mw^2- s*cw^2)^2*sw^4))\n"
    )
    for _name, _symbol in [
        ("s", s),
        ("t", t),
        ("me", me),
        ("mw", mw),
        ("mh", mh),
        ("sw", sw),
        ("cw", cw),
    ]:
        reference = reference.replace(S(_name), _symbol)
    reference = (
        reference.replace(S("pi"), one)
        .replace(S("alpha"), charge**2 / 4)
        .replace(cw, (1 - sw**2).sqrt())
    )
    full = squared.replace(H, one)
    assert (full - reference).together() == zero
    print("PASS massive FeynCalc amplitude", flush=True)
    z, _angle = S("inverse_s", "angle")
    high_energy = (
        squared.replace(t, -s * (1 - _angle) / 2)
        .replace(s, 1 / z)
        .series(z, 0, 0)
        .to_expression()
        .expand()
    )
    assert high_energy.replace(H, one).coefficient(z ** (-2)).together() == zero
    assert high_energy.replace(H, one).coefficient(z ** (-1)).together() == zero
    print("HIGH ENERGY GROWTH", high_energy.coefficient(z ** (-1)).factor(), flush=True)
    assert (
        high_energy.coefficient(z ** (-1))
        - charge**4 * me**2 * (H - 1) ** 2 / (32 * mw**4 * sw**4)
    ).together() == zero
    _massless = full.replace(me, zero).together()
    growth_coefficient = high_energy.coefficient(z ** (-1)).factor()
    return full, growth_coefficient, reference


@app.cell(hide_code=True)
def _(growth_coefficient, mo):
    mo.vstack(
        [
            mo.md("**Coefficient of s at fixed angle**"),
            growth_coefficient,
            mo.md(
                "The s² coefficient vanishes. The s coefficient is proportional to mₑ²(H−1)²: it also vanishes in the massless-electron limit."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Total rate with the physical massive flux
    `Kinematics.flux` and `two_body_phase_space` give
    $$\frac{d\sigma}{dt}=\frac{\overline{|\mathcal M|^2}}{16\pi s(s-4m_e^2)}.$$
    Symbolica integrates the rational $t$ dependence and verifies the primitive by differentiation. Both endpoints have $t<0$, so the logarithmic difference is evaluated using $\log(-t)$.

    The reference retains the electron mass in the amplitude and boundaries but uses $1/(16\pi s^2)$ for this prefactor. Its massive plotted rate is therefore the physical rate here multiplied by $1-4m_e^2/s$. The two agree in the massless limit. Nine independent quadratures check the integrated rates, and three also match the published massless total formula.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Integrate the massive matrix element
    """)
    return


@app.cell
def _(S, full, t, zero):
    _coefficients = (full * t**2).together().expand().coefficient_list(t)
    _C = S("C")
    _compressed = sum(
        (_C(_i) * _power / t**2 for _i, (_power, _) in enumerate(_coefficients)), zero
    )
    primitive = _compressed.integrate(t)
    for _i, (_, _coefficient) in enumerate(_coefficients):
        primitive = primitive.replace(_C(_i), _coefficient.together())
    assert (primitive.derivative(t) - full).together() == zero
    primitive = primitive.replace(t.log(), (-t).log())
    print("PASS full massive antiderivative", flush=True)
    return (primitive,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Derive the flux and phase-space normalization
    """)
    return


@app.cell
def _(Kinematics, hep, me, mw, one, pi, primitive, s, t, u, zero):
    _P = hep.Kinematics.external_momentum
    _kin = Kinematics.mandelstam(
        [_P(_i) for _i in range(4)], [me**2, me**2, mw**2, mw**2], [s, t, u]
    )
    _flux = _kin.flux(_P(0), _P(1))
    _phase_space = _kin.two_body_phase_space(_P(2), _P(3))
    integration_span = ((s - 4 * me**2) * (s - 4 * mw**2)).sqrt()
    native_dt_factor = 2 * pi * _phase_space / _flux / (integration_span / 2)
    dt_factor = one / (16 * pi * s * (s - 4 * me**2))
    assert (native_dt_factor**2 - dt_factor**2).together() == zero
    print("PASS massive flux and phase-space normalization", flush=True)
    _t_center = me**2 + mw**2 - s / 2
    _t_upper, _t_lower = (
        _t_center + integration_span / 2,
        _t_center - integration_span / 2,
    )
    total = dt_factor * (
        primitive.replace(t, _t_upper) - primitive.replace(t, _t_lower)
    )
    return dt_factor, native_dt_factor, total


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## State the massless-electron reference
    """)
    return


@app.cell
def _(charge, mw, pi, s, sw):
    _beta = (1 - 4 * mw**2 / s).sqrt()
    _den = mw**2 - s * (1 - sw**2)
    reference_massless = (
        charge**4
        / (16 * pi)
        * (
            _beta
            * (
                16 * (3 * mw**2 + 8 * s) * (2 * mw**4 + s**2) * sw**2
                - 3 * s * (-20 * s * mw**2 + 32 * mw**4 + 21 * s**2)
                - 4
                * (8 * s**2 * mw**2 + 160 * s * mw**4 + 96 * mw**6 + 15 * s**3)
                * sw**4
            )
            / (96 * s**2 * sw**4 * _den**2)
            + ((s - 2 * mw**2 - s * _beta) / (s - 2 * mw**2 + s * _beta)).log()
            * (
                24 * s * (s * mw**2 + 4 * mw**4 + s**2)
                - 24 * (2 * s**2 * mw**2 + 10 * s * mw**4 + 4 * mw**6 + s**3) * sw**2
            )
            / (96 * s**3 * sw**4 * _den)
        )
    )
    return (reference_massless,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the integral by independent quadrature
    """)
    return


@app.cell
def _(
    H,
    charge,
    dt_factor,
    me,
    mh,
    mw,
    native_dt_factor,
    np,
    reference,
    reference_massless,
    s,
    sw,
    t,
    total,
):
    _nodes, _weights = np.polynomial.legendre.leggauss(256)
    numeric_checks = []
    for _electron_mass in (0.0, 0.1, 0.4):
        for _sv in (5.0, 10.0, 25.0):
            _values = {
                charge: 1.0,
                sw: np.sqrt(0.23),
                mw: 1.0,
                mh: 1.5,
                me: _electron_mass,
                s: _sv,
                H: 1.0,
            }
            _native_factor = complex(native_dt_factor.evaluate(_values))
            _physical_factor = complex(dt_factor.evaluate(_values))
            assert (
                _native_factor.real > 0
                and abs(_native_factor - _physical_factor) < 1e-12
            )
            _integration_span = np.sqrt((_sv - 4 * _electron_mass**2) * (_sv - 4))
            _center = _electron_mass**2 + 1 - _sv / 2
            _norm = 1 / (16 * np.pi * _sv * (_sv - 4 * _electron_mass**2))
            _quadrature = (
                sum(
                    (
                        float(_w)
                        * complex(
                            reference.evaluate(
                                {
                                    **_values,
                                    t: _center + _integration_span * float(_x) / 2,
                                }
                            )
                        )
                        for _x, _w in zip(_nodes, _weights)
                    )
                )
                * _integration_span
                / 2
                * _norm
            )
            _actual = complex(total.evaluate(_values))
            assert abs(_actual - _quadrature) < 1e-10, (_values, _actual, _quadrature)
            if _electron_mass == 0:
                _published = complex(reference_massless.evaluate(_values))
                assert abs(_actual - _published) < 1e-12, (_actual, _published)
            assert _actual.real > 0 and abs(_actual.imag) < 1e-12
            numeric_checks.append(
                (_electron_mass, _sv, _actual.real, abs(_actual - _quadrature))
            )
    print(
        "PASS all nine massive rates and massless total reference",
        numeric_checks,
        flush=True,
    )
    return (numeric_checks,)


@app.cell(hide_code=True)
def _(dt_factor, mo, numeric_checks):
    mo.vstack(
        [
            mo.md("**Derived dσ/dt prefactor**"),
            dt_factor,
            mo.ui.table(
                [
                    {
                        "mₑ/mW": _r[0],
                        "s/mW²": _r[1],
                        "σ mW² (e=1)": _r[2],
                        "Quadrature error": _r[3],
                    }
                    for _r in numeric_checks
                ],
                selection=None,
            ),
        ]
    )
    return


@app.cell
def _(mo):
    electron_mass = mo.ui.dropdown(
        {
            "Massless": 0.0,
            "Electron (0.511 MeV)": 0.000511,
            "Illustrative lepton (20 GeV)": 20.0,
        },
        value="Electron (0.511 MeV)",
        label="Incoming lepton mass",
    )
    cm_energy = mo.ui.dropdown(
        {"180 GeV": 180.0, "250 GeV": 250.0, "500 GeV": 500.0, "2000 GeV": 2000.0},
        value="250 GeV",
        label="Center-of-mass energy",
    )
    scattering_angle = mo.ui.dropdown(
        {"Backward, cosθ=-0.5": -0.5, "Central, cosθ=0": 0.0, "Forward, cosθ=0.5": 0.5},
        value="Central, cosθ=0",
        label="W− angle",
    )
    mo.vstack([electron_mass, cm_energy, scattering_angle])
    return cm_energy, electron_mass, scattering_angle


@app.cell
def _(
    H,
    charge,
    cm_energy,
    electron_mass,
    me,
    mh,
    mw,
    np,
    reference,
    s,
    scattering_angle,
    squared,
    sw,
    t,
    total,
):
    _mass_value = electron_mass.value
    _s_value = cm_energy.value**2
    _W_mass = 80.4
    numeric_parameters = {
        charge: np.sqrt(4 * np.pi / 137),
        sw: np.sqrt(0.231),
        mw: _W_mass,
        mh: 125.0,
        me: _mass_value,
        s: _s_value,
    }
    _span_value = np.sqrt((_s_value - 4 * _mass_value**2) * (_s_value - 4 * _W_mass**2))
    _t_value = (
        _mass_value**2
        + _W_mass**2
        - _s_value / 2
        + _span_value * scattering_angle.value / 2
    )
    numeric_parameters[t] = _t_value
    selected_squared = complex(squared.evaluate({**numeric_parameters, H: 1.0})).real
    without_higgs = complex(squared.evaluate({**numeric_parameters, H: 0.0})).real
    _published_squared = complex(reference.evaluate(numeric_parameters)).real
    relative_error = abs(selected_squared - _published_squared) / max(
        1.0, abs(selected_squared)
    )
    assert relative_error < 1e-09
    _pb_per_GeV2 = 389379365.6
    angular_factor = (
        _span_value
        / (32 * np.pi * _s_value * (_s_value - 4 * _mass_value**2))
        * _pb_per_GeV2
    )
    selected_differential = selected_squared * angular_factor
    selected_total = complex(total.evaluate(numeric_parameters)).real * _pb_per_GeV2
    assert selected_squared > 0 and selected_total > 0
    return (
        angular_factor,
        numeric_parameters,
        relative_error,
        selected_differential,
        selected_total,
        without_higgs,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Resolve the numerical interference contributions
    """)
    return


@app.cell
def _(angular_factor, numeric_parameters, pair_results, selected_differential):
    pair_table = []
    _labels = {"a": "γ", "Z": "Z", "ve": "νₑ", "H": "Higgs"}
    for _left, _left_label in _labels.items():
        for _right, _right_label in _labels.items():
            if _left > _right:
                continue
            _weight = 1 if _left == _right else 2
            _value = (
                _weight
                * complex(pair_results[_left, _right].evaluate(numeric_parameters)).real
                * angular_factor
            )
            pair_table.append(
                {
                    "Exchange pair": _left_label + " × " + _right_label,
                    "dσ/dcosθ [pb]": _value,
                }
            )
    assert abs(
        sum(_row["dσ/dcosθ [pb]"] for _row in pair_table) - selected_differential
    ) < 1e-08 * max(1.0, abs(selected_differential))
    return (pair_table,)


@app.cell(hide_code=True)
def _(
    angular_factor,
    mo,
    pair_table,
    relative_error,
    selected_differential,
    selected_total,
    without_higgs,
):
    mo.vstack(
        [
            mo.md("## Numerical rate"),
            mo.md(
                "Illustrative tree-level inputs: mW=80.4 GeV, mH=125 GeV, sin²θW=0.231, α=1/137. θ is the W− angle from the electron beam."
            ),
            mo.ui.table(
                [
                    {
                        "SM dσ/dcosθ [pb]": selected_differential,
                        "SM total σ [pb]": selected_total,
                        "Higgs omitted dσ/dcosθ [pb]": without_higgs * angular_factor,
                        "Relative reference error": relative_error,
                    }
                ],
                selection=None,
            ),
            mo.md("**Diagram contributions and interference**"),
            mo.ui.table(pair_table, selection=None),
            mo.md(
                "Omitting Higgs exchange is a diagnostic alteration of the model. Its high-energy growth becomes visible for the illustrative heavy incoming lepton."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
