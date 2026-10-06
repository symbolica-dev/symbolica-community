import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Virtual photons and QCD radiation")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Virtual photons and QCD radiation

    [Browse notebooks](/) · [Light-cone soft radiation](/?file=hep/soft_function.py) ·
    [Photon–gluon currents](/?file=hep/photon_gluon.py) ·
    [Quark–gluon scattering](/?file=hep/quark_gluon_scattering.py)

    Generate $\gamma^*\to\mu^-\mu^+$, $\gamma^*\to q\bar q$ and
    both diagrams for $\gamma^*\to q\bar qg$. The massive squared
    amplitudes are checked before taking the massless limit used for the
    Dalitz distribution and regulated integral.

    This covers FeynCalc's [muon-pair](https://feyncalc.github.io/FeynCalcExamples/QED/Tree/Ga-MuAmu),
    [quark-pair](https://feyncalc.github.io/FeynCalcExamples/QCD/Tree/Ga-QQbar), and
    [real-radiation](https://feyncalc.github.io/FeynCalcExamples/QCD/Tree/Ga-QQbarGl)
    examples. All particles are selected by name.

    The virtual-photon current is contracted with $-g_{\mu\nu}$ without
    an initial spin average, matching those references. Its invariant mass is
    $Q=\sqrt{q^2}>0$. These are timelike current rates, not decays of an
    on-shell photon.
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
    from symbolica.community.hepkit import Kinematics, Model
    from symbolica.community.tensor import TensorExpression

    _set_namespace("radiation")
    return E, Kinematics, Model, S, Symbol, TensorExpression, hep, mo, np, sp


@app.cell(hide_code=True)
def _():
    from symbolica.community.hepkit import Symbols

    return (Symbols,)


@app.cell(hide_code=True)
def _(
    A,
    B,
    C,
    E,
    Nc,
    P,
    S,
    Symbols,
    TensorExpression,
    dA,
    ee,
    gs,
    mass,
    model,
    sp,
):
    def radiation_channel(outgoing, kin):
        allowed = [
            v
            for v in model.vertex_rules
            if sorted(v.particles)
            in [
                sorted(["b", "b~", "a"]),
                sorted(["b", "b~", "g"]),
                sorted(["mu-", "mu+", "a"]),
            ]
        ]
        generated = model.process(
            ["a"], outgoing, vertex_allow=allowed
        ).generate_diagrams(
            max_vertices=len(outgoing) - 1,
            maximum_bridges=None,
            numerator_grouping=None,
            progress=None,
        )
        assert len(generated.diagrams) == len(outgoing) - 1
        ports = S("i0", "i1", "i2", "i3")
        # Give the conjugate amplitude distinct external labels; scope only its dummies.
        bra_ports = {
            port: S(f"bra_{i}") for i, port in enumerate(ports[: len(outgoing) + 1])
        }
        a, b, c, inv, index, left, right = S(
            "a_", "b_", "c_", "inv_", "index_", "left_", "right_"
        )
        conj, wrap, metric = (
            sp.BroadcastFunction.conj().to_expression(),
            S("adjoint"),
            sp.TensorName.g().to_expression(),
        )
        terms = []
        for diagram in generated.diagrams:
            numerator = model.expand_couplings(
                diagram.numerator_expression(in_lmb=True).to_expression()
            ).replace(model.particle("mu-").mass, mass)
            for half in diagram.half_edges:
                edge = half.edge
                if edge.is_external:
                    numerator = sp.TensorExpression(numerator).rename_indices(
                        {Symbols.half_edge(half.id, 1): ports[edge.external_index]}
                    )
            denominator = kin.apply(
                diagram.denominator_expression(dimension=4, in_lmb=True)
                .to_expression()
                .replace(Symbols.denominator(a, b, c, inv), inv)
            ).expand()
            assert denominator in (E("1"), B, C), denominator
            terms.append(
                numerator
                * diagram.overall_factor_expression(evaluate=True)
                * diagram.numerator_prefactor_expression()
                / denominator
            )
        operator = sum(terms, E("0")).expand()
        adjoint = (
            sp.TensorExpression(operator)
            .dirac_adjoint()
            .simplify_algebra(
                contract="dots",
                color=False,
                gamma=True,
                gamma0=True,
                gamma_evaluate_traces=False,
            )
            .expand()
            .to_expression()
            .replace(conj(P(a, b)), P(a, b))
        )
        for real in (mass, ee, gs, A, B, C):
            adjoint = adjoint.replace(conj(real), real)
        adjoint = (
            sp.TensorExpression(adjoint)
            .wrap_indices(wrap, dummies_only=True)
            .rename_indices(bra_ports)
            .to_expression()
        )
        color_projector = E("1")
        for position, name in enumerate(["a", *outgoing]):
            particle = model.particle(name)
            if particle.color == 1:
                continue
            matches = []
            for slot in operator.structure.slots:
                original = slot.to_expression()
                if original.replace(ports[position], E("0")) == original:
                    continue
                closure = metric(
                    slot.dual().to_expression(),
                    original.replace(
                        ports[position],
                        bra_ports[ports[position]],
                    ),
                )
                match = next(
                    closure.match(particle.color_sum(left, right), max_level=0),
                    None,
                )
                if match is not None:
                    matches.append(dict(match))
            assert len(matches) == 1
            color_projector *= particle.color_sum(matches[0][left], matches[0][right])
        generic = (
            (operator.to_expression() * adjoint * color_projector)
            .replace(
                sp.PortPattern.exact(sp.Representation.cof(3), index),
                sp.PortPattern.exact(sp.Representation.cof(Nc), index),
            )
            .replace(
                sp.PortPattern.exact(sp.Representation.coad(8), index),
                sp.PortPattern.exact(sp.Representation.coad(dA), index),
            )
        )
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
        spins = model.particle(outgoing[0]).spin_sum(
            P(1), bra_ports[ports[2]], ports[1]
        ) * model.particle(outgoing[1]).spin_sum(P(2), ports[2], bra_ports[ports[1]])
        traced = (
            (colored * spins.replace(model.particle("mu-").mass, mass))
            .simplify_algebra(contract="dots", color=False, gamma=True, epsilon=True)
            .expand()
        )
        reduced = kin.apply(traced.contract().to_dots().expand())
        results = {}
        modes = (
            [
                "covariant",
                "quark reference",
                "antiquark reference",
                "gluon Ward",
                "photon Ward",
            ]
            if len(outgoing) == 3
            else ["covariant", "photon Ward"]
        )
        for mode in modes:
            photon = model.particle("a").spin_sum(
                P(0),
                ports[0],
                bra_ports[ports[0]],
                covariant=True,
            )
            gluon = (
                model.particle("g").spin_sum(
                    P(3),
                    ports[3],
                    bra_ports[ports[3]],
                    covariant=mode in ("covariant", "photon Ward", "gluon Ward"),
                    reference=P(1)
                    if mode == "quark reference"
                    else P(2)
                    if mode == "antiquark reference"
                    else None,
                )
                if len(outgoing) == 3
                else E("1")
            )
            if mode == "gluon Ward":
                gluon = P(
                    3, sp.PortPattern.exact(sp.Representation.mink(4), ports[3])
                ) * P(
                    3,
                    sp.PortPattern.exact(
                        sp.Representation.mink(4),
                        bra_ports[ports[3]],
                    ),
                )
            if mode == "photon Ward":
                photon = P(
                    0, sp.PortPattern.exact(sp.Representation.mink(4), ports[0])
                ) * P(
                    0,
                    sp.PortPattern.exact(
                        sp.Representation.mink(4),
                        bra_ports[ports[0]],
                    ),
                )
            contracted = (reduced * photon * gluon).contract().to_dots()
            assert contracted.is_scalar
            results[mode] = kin.apply(contracted).to_expression().together()
        for mode in [name for name in modes if name.endswith("Ward")]:
            assert results[mode] == E("0")
        for mode in [name for name in modes if name.endswith("reference")]:
            assert (results[mode] - results["covariant"]).together() == E("0")

        return generated, results

    return (radiation_channel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Define the masses and invariants
    """)
    return


@app.cell
def _(Model, S, hep, sp):
    model = Model.standard_model()
    P = hep.Kinematics.external_momentum
    mass = model.particle("b").mass
    ee = -model.particle("e-").electric_charge
    gs = model.parameter("G").symbol
    A, B, C = S("A", "B", "C")
    Nc, dA, cof, coad = (
        S("Nc"),
        S("dA"),
        sp.Representation.cof,
        sp.Representation.coad,
    )
    QQ = S("QQ", is_positive=True)
    x1, x2, x3 = S("x1", "x2", "x3")
    return A, B, C, Nc, P, QQ, dA, ee, gs, mass, model, x1, x2, x3


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate the radiative channel

    The preamble contains the repeated diagram and state-sum calculation. Keep the scalar kinematics explicit here.
    """)
    return


@app.cell
def _(A, B, C, E, Kinematics, P, QQ, mass, radiation_channel, x1, x2, x3):
    kin = Kinematics()
    for _i, _j, _value in [
        (1, 1, mass**2),
        (2, 2, mass**2),
        (3, 3, E("0")),
        (1, 2, A / 2),
        (1, 3, B / 2),
        (2, 3, C / 2),
        (0, 0, A + B + C + 2 * mass**2),
        (0, 1, mass**2 + (A + B) / 2),
        (0, 2, mass**2 + (A + C) / 2),
        (0, 3, (B + C) / 2),
    ]:
        kin = kin.with_scalar_product(P(_i), P(_j), _value)
    generated, results = radiation_channel(["b", "b~", "g"], kin)
    squared = (
        results["covariant"]
        .replace(A, QQ * (1 - x3))
        .replace(B, QQ * (1 - x2))
        .replace(C, QQ * (1 - x1))
        .together()
    )
    return generated, results, squared


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the independent massive formula
    """)
    return


@app.cell
def _(E, Nc, QQ, ee, gs, mass, squared, x1, x2, x3):
    expected = (
        8
        * ee**2
        * (Nc**2 - 1)
        / 2
        / 9
        * gs**2
        * (
            2
            * QQ
            * mass**2
            * (
                x1**3
                + x1**2 * (x2 + x3 - 5)
                + x1 * (x2**2 - 4 * x2 * x3 + 2 * x3 + 4)
                + x2**3
                + x2**2 * (x3 - 5)
                + 2 * x2 * (x3 + 2)
                - 2 * (x3 + 1)
            )
            - 8 * mass**4 * (x1**2 - 2 * x1 + x2**2 - 2 * x2 + 2)
            + QQ**2
            * (x1 - 1)
            * (x2 - 1)
            * (
                x1**2
                + 2 * x1 * (x3 - 2)
                + x2**2
                + 2 * x2 * (x3 - 2)
                + 2 * (x3 - 2) ** 2
            )
        )
        / (QQ**2 * (x1 - 1) ** 2 * (x2 - 1) ** 2)
    )
    assert (squared - expected).together() == E("0")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Take the massless limit
    """)
    return


@app.cell
def _(E, Nc, ee, gs, mass, squared, x1, x2, x3):
    massless = squared.replace(mass, 0).replace(x3, 2 - x1 - x2).together()
    assert (
        massless
        - 4 * ee**2 * gs**2 * (Nc**2 - 1) / 9 * (x1**2 + x2**2) / ((1 - x1) * (1 - x2))
    ).together() == E("0")
    return (massless,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate the Born normalization

    Compare the quark and muon channels with their physical color charges.
    """)
    return


@app.cell
def _(E, Kinematics, Nc, P, QQ, Symbol, ee, mass, radiation_channel):
    born_kin = Kinematics()
    for _i, _j, _value in [
        (0, 0, QQ),
        (1, 1, mass**2),
        (2, 2, mass**2),
        (1, 2, (QQ - 2 * mass**2) / 2),
        (0, 1, QQ / 2),
        (0, 2, QQ / 2),
    ]:
        born_kin = born_kin.with_scalar_product(P(_i), P(_j), _value)
    born_results = {}
    for _names, _color_charge in ((["b", "b~"], Nc / 9), (["mu-", "mu+"], E("1"))):
        born_generated, born_sums = radiation_channel(_names, born_kin)
        born_square = born_sums["covariant"]
        assert (
            born_square - 4 * ee**2 * _color_charge * (QQ + 2 * mass**2)
        ).together() == E("0")
        # The virtual-photon current is contracted with -g_mu_nu without averaging,
        # matching the gallery normalization. This is not an on-shell photon decay.
        width = (
            born_square
            * 4
            * Symbol.PI
            * born_kin.two_body_phase_space(P(1), P(2))
            / born_kin.flux(P(0))
        )
        massless_width = width.replace(mass, 0).together()
        assert (
            massless_width - ee**2 * _color_charge * QQ.sqrt() / (4 * Symbol.PI)
        ).together() == E("0")
        born_results[_names[0]] = (
            born_generated,
            born_square,
            width,
            massless_width,
        )
    assert (born_results["b"][3] / born_results["mu-"][3] - Nc / 9).together() == E("0")
    return (born_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Construct the three-body phase space
    """)
    return


@app.cell
def _(E, Kinematics, P, QQ, Symbol, x1, x2):
    massless_kin = Kinematics()
    for _i, _j, _value in [
        (0, 0, QQ),
        (1, 1, E("0")),
        (2, 2, E("0")),
        (3, 3, E("0")),
        (1, 2, QQ * (x1 + x2 - 1) / 2),
        (1, 3, QQ * (1 - x2) / 2),
        (2, 3, QQ * (1 - x1) / 2),
    ]:
        massless_kin = massless_kin.with_scalar_product(P(_i), P(_j), _value)
    phase = massless_kin.three_body_phase_space(P(1), P(2), P(3)).together()
    assert (phase - 1 / (128 * Symbol.PI**3 * QQ)).together() == E("0")
    # The transformation from pair invariant masses to energy fractions has Jacobian Q^4.
    return massless_kin, phase


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Normalize the energy distribution
    """)
    return


@app.cell
def _(
    E,
    Nc,
    P,
    QQ,
    S,
    Symbol,
    born_results,
    gs,
    massless,
    massless_kin,
    phase,
    x1,
    x2,
):
    alpha_s = S("alpha_s")
    distribution = (
        (massless * phase * QQ**2 / massless_kin.flux(P(0)) / born_results["b"][3])
        .replace(gs**2, 4 * Symbol.PI * alpha_s)
        .together()
    )
    cf = (Nc**2 - 1) / (2 * Nc)
    shape = (x1**2 + x2**2) / ((1 - x1) * (1 - x2))
    assert (distribution - alpha_s * cf / (2 * Symbol.PI) * shape).together() == E("0")
    # The full generated result must reproduce the independently audited soft limit.
    return alpha_s, cf, distribution, shape


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Verify the soft limit
    """)
    return


@app.cell
def _(A, B, C, E, Nc, S, cf, ee, gs, mass, results):
    lam = S("lambda")
    soft_ratio = results["covariant"].replace(mass, 0) / (4 * ee**2 * Nc * A / 9)
    soft_limit = (
        soft_ratio.replace(B, lam**2 * B)
        .replace(C, lam**2 * C)
        .series(lam, 0, -4)
        .to_expression()
        .replace(lam, 1)
    )
    assert (soft_limit - 4 * gs**2 * cf * A / (B * C)).together() == E("0")

    # Positive invariants y1=1-x1 and y2=1-x2 obey y1+y2<=1. The cut is
    # y1,y2>=beta, 0<beta<1/2, excluding both soft/collinear poles.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Integrate the first energy variable
    """)
    return


@app.cell
def _(E, S, Symbol, shape, x1, x2):
    y1, y2, beta = S("y1", "y2", "beta", is_positive=True)
    kernel = shape.replace(x1, 1 - y1).replace(x2, 1 - y2).together()
    argument = S("argument_")
    inner_primitive = kernel.integrate(y2).replace(
        Symbol.LOG(argument), lambda match: match[argument].expand().log()
    )
    assert (inner_primitive.derivative(y2) - kernel).together() == E("0")
    inner_cut = inner_primitive.replace(y2, 1 - y1) - inner_primitive.replace(y2, beta)
    # The triangular domain is symmetric under y1<->y2. Integrating the two
    # symmetric numerator terms therefore gives twice the first term's integral.
    return argument, beta, inner_cut, inner_primitive, y1


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Integrate the angular-cut kernel
    """)
    return


@app.cell
def _(E, Symbol, argument, beta, y1):
    outer_integrand = 2 * (1 - y1) ** 2 / y1 * ((1 - y1).log() - beta.log())
    outer_primitive = (
        outer_integrand.integrate(y1)
        .replace(
            Symbol.POLYLOG(2, argument),
            lambda match: match[argument].expand().polylog(2),
        )
        .replace(Symbol.LOG(argument), lambda match: match[argument].expand().log())
    )
    assert (outer_primitive.derivative(y1) - outer_integrand).together() == E("0")
    integrated_kernel = outer_primitive.replace(y1, 1 - beta) - outer_primitive.replace(
        y1, beta
    )
    # A real-branch form follows from Euler's dilogarithm reflection identity.
    return integrated_kernel, outer_primitive


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## State the closed cut-rate formula
    """)
    return


@app.cell
def _(E, Symbol, beta):
    closed = (
        2 * beta.log() ** 2
        + (3 - 4 * beta + beta**2) * (beta.log() - (1 - beta).log())
        + E("5/2")
        - 5 * beta
        + 4 * beta.polylog(2)
        - Symbol.PI**2 / 3
    )
    # Expand the analytic coefficients while retaining log(beta) symbolically.
    return (closed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Verify the small-cut logarithms
    """)
    return


@app.cell
def _(E, S, Symbol, beta, closed, inner_cut, y1):
    log_beta = S("log_beta")
    leading = (
        closed.replace(beta.log(), log_beta)
        .series(beta, 0, 0)
        .to_expression()
        .replace(log_beta, beta.log())
    )
    expected_leading = (
        2 * beta.log() ** 2 + 3 * beta.log() + E("5/2") - Symbol.PI**2 / 3
    )
    assert (leading - expected_leading).expand() == E("0")
    # Leibniz differentiation of the cut triangle provides an independent exact
    # check of the closed form, without relying on reflection simplification.
    boundary = inner_cut.replace(y1, beta)
    assert (closed.derivative(beta) + 2 * boundary).together() == E("0")
    return (leading,)


@app.cell
def _(
    Nc,
    QQ,
    alpha_s,
    beta,
    born_results,
    cf,
    closed,
    distribution,
    ee,
    generated,
    gs,
    inner_primitive,
    integrated_kernel,
    leading,
    mass,
    massless,
    outer_primitive,
    phase,
    results,
    squared,
    x1,
    x2,
):
    radiation = {
        "generated": generated,
        "results": results,
        "mass": mass,
        "ee": ee,
        "gs": gs,
        "QQ": QQ,
        "x1": x1,
        "x2": x2,
        "Nc": Nc,
        "squared": squared,
        "massless": massless,
        "born_results": born_results,
        "phase": phase,
        "alpha_s": alpha_s,
        "distribution": distribution,
        "cf": cf,
        "beta": beta,
        "inner_primitive": inner_primitive,
        "outer_primitive": outer_primitive,
        "integrated_kernel": integrated_kernel,
        "closed": closed,
        "leading": leading,
    }
    return (radiation,)


@app.cell
def _(mo, radiation):
    mo.vstack(
        [
            mo.md("## Generated diagrams"),
            mo.md("**Born quark and muon currents**"),
            mo.hstack(
                [
                    *radiation["born_results"]["b"][0].diagrams,
                    *radiation["born_results"]["mu-"][0].diagrams,
                ]
            ),
            mo.md("**Real QCD radiation: both emissions and their interference**"),
            mo.hstack(radiation["generated"].diagrams),
            mo.md(r"""
        The two massive Born squares are $4e^2(q^2+2m^2)$ and
        $4e^2 N_c Q_q^2(q^2+2m^2)$. Shared two-body phase space gives,
        in the massless limit, $\Gamma_\mu=\alpha Q$ and
        $\Gamma_q=N_c Q_q^2\alpha Q$.
        The model's bottom field has $Q_q=-1/3$; its charge cancels from
        the Born-normalized radiation distribution.
        """),
        ]
    )
    return


@app.cell
def _(mo, radiation):
    mo.vstack(
        [
            mo.md(r"""
        ## Exact amplitude and phase-space checks

        The full massive result agrees with the reference. Both photon and
        gluon Ward contractions vanish. Gluon polarization sums using either
        massive quark momentum as reference equal the covariant result.
        The full amplitude also reproduces the independently checked soft limit.

        Shared `Kinematics.three_body_phase_space(k1, k2, k3)` returns
        $$\frac{d\Phi_3}{ds_{12}ds_{23}}=\frac{1}{128\pi^3 q^2}.$$
        It integrates the overall orientation. Physical Dalitz boundaries,
        the decay factor $1/(2Q)$, and any identical-particle factors remain
        explicit. These three final particles are distinct.
        """),
            radiation["phase"],
            mo.md(r"""
        For massless final states define $x_i=2E_i/Q$,
        $0<x_i<1$ and $x_1+x_2+x_3=2$. Changing from pair invariants to
        $x_1,x_2$ has Jacobian $q^4$. The resulting distribution is
        $$\frac{1}{\Gamma_q}\frac{d\Gamma_{q\bar qg}}{dx_1dx_2}
        =\frac{\alpha_s C_F}{2\pi}
        \frac{x_1^2+x_2^2}{(1-x_1)(1-x_2)},\qquad
        C_F=\frac{N_c^2-1}{2N_c}.$$
        """),
            radiation["distribution"],
        ]
    )
    return


@app.cell
def _(mo):
    colors = mo.ui.dropdown(
        {f"SU({n})": n for n in (2, 3, 5)}, value="SU(3)", label="Color group"
    )
    cutoff = mo.ui.dropdown(
        {str(b): b for b in (0.01, 0.05, 0.1, 0.25, 0.4)},
        value="0.1",
        label="Invariant cut β",
    )
    coupling = mo.ui.slider(0.05, 0.2, step=0.001, value=0.118, label="αs")
    first_position = mo.ui.slider(
        0.05,
        0.95,
        step=0.05,
        value=0.5,
        label="First position in the allowed Dalitz region",
    )
    second_position = mo.ui.slider(
        0.05,
        0.95,
        step=0.05,
        value=0.5,
        label="Second position in the allowed Dalitz region",
    )
    mo.vstack(
        [
            mo.hstack([colors, cutoff, coupling]),
            mo.hstack([first_position, second_position]),
        ]
    )
    return colors, coupling, cutoff, first_position, second_position


@app.cell
def _(
    Symbol,
    colors,
    coupling,
    cutoff,
    first_position,
    radiation,
    second_position,
):
    _b = cutoff.value
    _y1 = _b + (1 - 2 * _b) * first_position.value
    _y2 = _b + (1 - _b - _y1) * second_position.value
    selected_x1, selected_x2 = 1 - _y1, 1 - _y2
    selected_x3 = _y1 + _y2
    assert _y1 >= _b and _y2 >= _b and _y1 + _y2 < 1
    _parameters = {
        radiation["Nc"]: colors.value,
        radiation["alpha_s"]: coupling.value,
        radiation["x1"]: selected_x1,
        radiation["x2"]: selected_x2,
        radiation["beta"]: _b,
    }
    selected_density = complex(radiation["distribution"].evaluate(_parameters)).real
    selected_rate = complex(
        (
            radiation["alpha_s"]
            * radiation["cf"]
            / (2 * Symbol.PI)
            * radiation["closed"]
        ).evaluate(_parameters)
    ).real
    automatic_rate = complex(
        (
            radiation["alpha_s"]
            * radiation["cf"]
            / (2 * Symbol.PI)
            * radiation["integrated_kernel"]
        ).evaluate(_parameters)
    ).real
    assert selected_density > 0 and selected_rate > 0
    assert abs(selected_rate - automatic_rate) < 1e-10
    return (
        selected_density,
        selected_rate,
        selected_x1,
        selected_x2,
        selected_x3,
    )


@app.cell(hide_code=True)
def _(
    mo,
    selected_density,
    selected_rate,
    selected_x1,
    selected_x2,
    selected_x3,
):
    mo.vstack(
        [
            mo.md(r"""
        ## Explore the physical Dalitz region

        Set $y_1=1-x_1$ and $y_2=1-x_2$. The cut
        $y_1,y_2\ge\beta$ with $y_1+y_2\le1$ excludes the soft/collinear
        singularities. The position controls stay inside this triangle.
        Energy fractions are normalized to half the virtual-photon mass.
        """),
            mo.md(
                f"**Energy fractions:** x₁ = {selected_x1:.4f}, x₂ = {selected_x2:.4f}, x₃ = {selected_x3:.4f}"
            ),
            mo.md(f"**Normalized differential density:** {selected_density:.8g}"),
            mo.md(
                f"**Integrated real-emission rate / Born rate:** {selected_rate:.8g}"
            ),
        ]
    )
    return


@app.cell
def _(Symbol, colors, coupling, np, radiation):
    _factor = radiation["alpha_s"] * radiation["cf"] / (2 * Symbol.PI)
    cut_rows = []
    _nodes, _weights = np.polynomial.legendre.leggauss(96)
    for _b in (0.01, 0.05, 0.1, 0.25, 0.4):
        _parameters = {
            radiation["Nc"]: colors.value,
            radiation["alpha_s"]: coupling.value,
            radiation["beta"]: _b,
        }
        _exact = complex((_factor * radiation["closed"]).evaluate(_parameters)).real
        _asymptotic = complex(
            (_factor * radiation["leading"]).evaluate(_parameters)
        ).real
        _first = _b + (_nodes + 1) * (1 - 2 * _b) / 2
        _quadrature = 0.0
        for _u, _weight in zip(_first, _weights):
            _second = _b + (_nodes + 1) * (1 - _u - _b) / 2
            _values = ((1 - _u) ** 2 + (1 - _second) ** 2) / (_u * _second)
            _quadrature += (
                _weight * (1 - _u - _b) / 2 * float(np.dot(_weights, _values))
            )
        _quadrature *= (
            (1 - 2 * _b)
            / 2
            * coupling.value
            * (colors.value**2 - 1)
            / (4 * np.pi * colors.value)
        )
        assert abs(_exact - _quadrature) < 2e-10
        cut_rows.append(
            {
                "β": _b,
                "Exact real / Born": _exact,
                "Independent quadrature": _quadrature,
                "Small-β expansion": _asymptotic,
            }
        )
    return (cut_rows,)


@app.cell(hide_code=True)
def _(cut_rows, mo, radiation):
    mo.vstack(
        [
            mo.md(r"""
        ## Regulated integral and infrared logarithms

        Symbolica computes the rational inner integral and the outer primitive,
        including its dilogarithm. Their derivatives are checked exactly.
        A real-branch form of the integrated kernel is displayed below; its
        cut derivative also agrees with the boundary of the integration region.
        Independent two-dimensional Gaussian quadrature checks the result.

        Multiply this kernel by $\alpha_s C_F/(2\pi)$.
        As $\beta\to0^+$ it approaches
        $$2\log^2\beta+3\log\beta+\frac52-\frac{\pi^2}{3}.$$
        The real contribution diverges as the cut is removed. Cancellation
        requires the virtual correction, which is not included here.
        """),
            radiation["closed"],
            mo.ui.table(cut_rows, selection=None),
            mo.accordion({"Symbolica's outer primitive": radiation["outer_primitive"]}),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
