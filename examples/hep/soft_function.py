import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Light-cone soft radiation")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Light-cone soft radiation

    [Browse all notebooks](/) · [Full QCD radiation](/?file=hep/photon_radiation.py) · [Quark–gluon scattering](/?file=hep/quark_gluon_scattering.py) ·
    [Polarization sums](/?file=hep/polarization_sums.py)

    Generate $\gamma^*\to q\bar q$ and both diagrams for
    $\gamma^*\to q\bar qg$, then take the ultrasoft limit $k\to\lambda^2 k$.
    The calculation follows
    [FeynCalc's SCET soft-function example](https://feyncalc.github.io/FeynCalcExamples/QCD/Tree/Ga-QQbar-SoftFunction).

    The quark and antiquark move along null directions $n,\bar n$, with
    $n\cdot\bar n=2$. Their independent large momentum components are $Q,\bar Q$.
    The gluon is massless. Lorentz dimension $D$ stays symbolic; the spinor
    representation has dimension four.

    The generated amplitudes retain open spinor and photon indices. Their
    leading terms factor onto the same Born current before any spin sum.
    Only gluon polarizations and final-state colors are summed.
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
    import numpy as np
    from symbolica import E, S, Symbol
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hep import Kinematics, Model
    from symbolica.community.tensor import Representation, TensorExpression, TensorName

    _set_namespace("soft")
    return (
        E,
        Kinematics,
        Model,
        Representation,
        S,
        Symbol,
        TensorExpression,
        TensorName,
        hep,
        mo,
        np,
        sp,
    )


@app.cell(hide_code=True)
def _():
    from symbolica.community.hep import Symbols

    return (Symbols,)


@app.cell(hide_code=True)
def _(TensorExpression, ordering):
    def reduce_collinear(expression, kinematics):
        return kinematics.apply(
            TensorExpression(expression)
            .expand()
            .simplify_algebra(contract="dots", **ordering, epsilon=True)
            .expand()
            .to_expression()
        ).expand()

    return (reduce_collinear,)


@app.cell(hide_code=True)
def _(D, gamma, n, nb, sp):
    def projector(left, right, middle):
        # P_minus = nbar_slash n_slash / 4; all spinor dimensions remain four.
        return (
            gamma(
                sp.PortPattern.exact(sp.Representation.bis(4), left),
                sp.PortPattern.exact(sp.Representation.bis(4), middle),
                nb(sp.PortPattern.exact(sp.Representation.mink(D))),
            )
            * gamma(
                sp.PortPattern.exact(sp.Representation.bis(4), middle),
                sp.PortPattern.exact(sp.Representation.bis(4), right),
                n(sp.PortPattern.exact(sp.Representation.mink(D))),
            )
            / 4
        )

    return (projector,)


@app.cell(hide_code=True)
def _(ports, projector, sm, sn, sp, sx, sy):
    def project(expression):
        # The collinear bra and anticollinear ket both select P_minus.
        expression = expression.replace(
            sp.PortPattern.exact(sp.Representation.bis(4), ports[1]),
            sp.PortPattern.exact(sp.Representation.bis(4), sx),
        ).replace(
            sp.PortPattern.exact(sp.Representation.bis(4), ports[2]),
            sp.PortPattern.exact(sp.Representation.bis(4), sy),
        )
        return projector(ports[1], sx, sm) * expression * projector(sy, ports[2], sn)

    return (project,)


@app.cell(hide_code=True)
def _(
    D,
    P,
    Q,
    Qbar,
    Symbols,
    TensorExpression,
    a,
    b,
    c,
    idx,
    inv,
    k,
    kin,
    lam,
    model,
    n,
    nb,
    one,
    ports,
    project,
    reduce_collinear,
    zero,
):
    def leading_soft_amplitude(_diagram, _outgoing):
        """Expand one generated amplitude at leading soft order and project its spinor endpoints."""
        _numerator = model.expand_couplings(
            _diagram.numerator_expression(in_lmb=True).to_expression()
        ).replace(model.particle("b").mass, zero)
        for _half in _diagram.half_edges:
            _edge = _half.edge.data
            if _edge.is_external:
                _numerator = _numerator.replace(
                    Symbols.half_edge(_half.data, 1),
                    ports[_edge.external_index],
                )
        _numerator = (
            TensorExpression(_numerator).with_lorentz_dimension(D).to_expression()
        )
        _denominator = (
            _diagram.denominator_expression(dimension=D, in_lmb=True)
            .to_expression()
            .replace(Symbols.denominator(a, b, c, inv), inv)
            .replace(model.particle("b").mass, zero)
        )
        # Expand the dot products before substituting linear vector combinations.
        _denominator = TensorExpression(_denominator).undo_dots().to_expression()
        for _position, _momentum in (
            (0, (Q * n(idx) + Qbar * nb(idx)) / 2 + lam**2 * k(idx)),
            (1, Q * n(idx) / 2),
            (2, Qbar * nb(idx) / 2),
            (3, lam**2 * k(idx)),
        ):
            _numerator = _numerator.replace(P(_position, idx), _momentum)
            _denominator = _denominator.replace(P(_position, idx), _momentum)
        _denominator = kin.apply(
            TensorExpression(_denominator)
            .contract(collect_chains=False, collect_traces=False)
            .to_dots()
            .expand()
            .to_expression()
        ).expand()

        _amplitude = (
            _numerator
            * _diagram.overall_factor_expression(evaluate=True)
            * _diagram.numerator_prefactor_expression()
            / _denominator
        )
        _leading = _amplitude.series(
            lam, 0, -2 if len(_outgoing) == 3 else 0
        ).to_expression()
        return _denominator, reduce_collinear(
            project(_leading.replace(lam, one)), kinematics=kin
        )

    return (leading_soft_amplitude,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Define the soft and collinear kinematics
    """)
    return


@app.cell
def _(E, Kinematics, Model, S, TensorName, hep, sp):
    model = Model.standard_model()
    vertices = [
        vertex
        for vertex in model.vertex_rules
        if sorted(vertex.particles)
        in [sorted(["b", "b~", "a"]), sorted(["b", "b~", "g"])]
    ]
    assert len(vertices) == 2
    P = hep.Kinematics.external_momentum
    D, Q, Qbar, kp, km, lam = S("D", "Q", "Qbar", "kp", "km", "lam")
    # Rank-one declarations let Spenso recognize contracted vectors and slashes.
    n, nb, k, ref = [
        TensorName.vector(name).to_expression() for name in ("n", "nb", "k", "r")
    ]
    mink, bis = (sp.Representation.mink, sp.Representation.bis)
    zero, one = E("0"), E("1")
    kin = Kinematics(D, momenta=[n, nb, k])
    for _vector in (n, nb, k):
        kin = kin.with_scalar_product(_vector, _vector, zero)
    kin = (
        kin.with_scalar_product(n, nb, E("2"))
        .with_scalar_product(n, k, kp)
        .with_scalar_product(nb, k, km)
    )
    return (
        D,
        P,
        Q,
        Qbar,
        k,
        kin,
        km,
        kp,
        lam,
        model,
        n,
        nb,
        one,
        ref,
        vertices,
        zero,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Declare the tensor ports and gamma convention
    """)
    return


@app.cell
def _(S, sp):
    ports = S("i0", "i1", "i2", "i3")
    a, b, c, inv, idx = S("a_", "b_", "c_", "inv_", "idx_")
    gamma, metric = (
        sp.TensorName.dirac_gamma().to_expression(),
        sp.TensorName.g().to_expression(),
    )
    si, sj, sx, sy, sm, sn = S("si", "sj", "sx", "sy", "sm", "sn")
    ordering = dict(gamma=True, gamma_ordering="canonical")

    # Check the large-component projector before applying it to generated amplitudes.
    return (
        a,
        b,
        c,
        gamma,
        idx,
        inv,
        metric,
        ordering,
        ports,
        si,
        sj,
        sm,
        sn,
        sx,
        sy,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Build complementary spin projectors
    """)
    return


@app.cell
def _(metric, projector, si, sj, sm, sp):
    pm = projector(si, sj, sm)
    identity = metric(
        sp.PortPattern.exact(sp.Representation.bis(4), si),
        sp.PortPattern.exact(sp.Representation.bis(4), sj),
    )
    pp = identity - pm
    return pm, pp


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the spin projectors
    """)
    return


@app.cell
def _(kin, pm, pp, projector, reduce_collinear, si, sj, sm, sn, sp, sx, zero):
    assert (
        reduce_collinear(
            pm.replace(
                sp.PortPattern.exact(sp.Representation.bis(4), sj),
                sp.PortPattern.exact(sp.Representation.bis(4), sx),
            )
            * projector(sx, sj, sn)
            - pm,
            kinematics=kin,
        )
        == zero
    )
    assert (
        reduce_collinear(
            pm.replace(
                sp.PortPattern.exact(sp.Representation.bis(4), sj),
                sp.PortPattern.exact(sp.Representation.bis(4), sx),
            )
            * pp.replace(
                sp.PortPattern.exact(sp.Representation.bis(4), si),
                sp.PortPattern.exact(sp.Representation.bis(4), sx),
            ).replace(
                sp.PortPattern.exact(sp.Representation.bis(4), sm),
                sp.PortPattern.exact(sp.Representation.bis(4), sn),
            ),
            kinematics=kin,
        )
        == zero
    )
    assert (
        reduce_collinear(
            pm.replace(
                sp.PortPattern.exact(sp.Representation.bis(4), sj),
                sp.PortPattern.exact(sp.Representation.bis(4), si),
            ),
            kinematics=kin,
        )
        == 2
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the light-cone annihilation identities
    """)
    return


@app.cell
def _(D, gamma, kin, n, nb, pm, reduce_collinear, si, sj, sp, sx, zero):
    assert (
        reduce_collinear(
            pm.replace(
                sp.PortPattern.exact(sp.Representation.bis(4), sj),
                sp.PortPattern.exact(sp.Representation.bis(4), sx),
            )
            * gamma(
                sp.PortPattern.exact(sp.Representation.bis(4), sx),
                sp.PortPattern.exact(sp.Representation.bis(4), sj),
                n(sp.PortPattern.exact(sp.Representation.mink(D))),
            ),
            kinematics=kin,
        )
        == zero
    )
    assert (
        reduce_collinear(
            gamma(
                sp.PortPattern.exact(sp.Representation.bis(4), si),
                sp.PortPattern.exact(sp.Representation.bis(4), sx),
                nb(sp.PortPattern.exact(sp.Representation.mink(D))),
            )
            * pm.replace(
                sp.PortPattern.exact(sp.Representation.bis(4), si),
                sp.PortPattern.exact(sp.Representation.bis(4), sx),
            ),
            kinematics=kin,
        )
        == zero
    )

    # The transverse metric projects onto the D-2 directions orthogonal to n,nbar.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the transverse metric
    """)
    return


@app.cell
def _(D, S, kin, metric, n, nb, reduce_collinear, sp, zero):
    mu, nu, rho = S("mu", "nu", "rho")
    transverse = (
        metric(
            sp.PortPattern.exact(sp.Representation.mink(D), mu),
            sp.PortPattern.exact(sp.Representation.mink(D), nu),
        )
        - (
            n(sp.PortPattern.exact(sp.Representation.mink(D), mu))
            * nb(sp.PortPattern.exact(sp.Representation.mink(D), nu))
            + nb(sp.PortPattern.exact(sp.Representation.mink(D), mu))
            * n(sp.PortPattern.exact(sp.Representation.mink(D), nu))
        )
        / 2
    )
    assert (
        reduce_collinear(
            transverse * n(sp.PortPattern.exact(sp.Representation.mink(D), nu)),
            kinematics=kin,
        )
        == zero
    )
    assert (
        reduce_collinear(
            transverse * nb(sp.PortPattern.exact(sp.Representation.mink(D), nu)),
            kinematics=kin,
        )
        == zero
    )
    assert (
        reduce_collinear(
            transverse
            * metric(
                sp.PortPattern.exact(sp.Representation.mink(D), mu),
                sp.PortPattern.exact(sp.Representation.mink(D), nu),
            ),
            kinematics=kin,
        )
        == D - 2
    )
    assert (
        reduce_collinear(
            transverse * transverse.replace(mu, rho) - transverse.replace(nu, rho),
            kinematics=kin,
        )
        == zero
    )
    return (transverse,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate the leading soft amplitudes
    """)
    return


@app.cell
def _(
    D,
    Q,
    Qbar,
    kin,
    km,
    kp,
    lam,
    leading_soft_amplitude,
    model,
    n,
    nb,
    one,
    ports,
    reduce_collinear,
    sp,
    vertices,
    zero,
):
    channels, denominators, generated_channels = [], [], []
    for _outgoing in (["b", "b~"], ["b", "b~", "g"]):
        _result = model.process(
            ["a"], _outgoing, vertex_allow=vertices
        ).generate_diagrams(
            max_vertices=len(_outgoing) - 1,
            maximum_bridges=None,
            numerator_grouping=None,
            progress=None,
        )
        assert len(_result.diagrams) == len(_outgoing) - 1
        generated_channels.append(_result)
        for _diagram in _result.diagrams:
            _denominator, _amplitude = leading_soft_amplitude(_diagram, _outgoing)
            denominators.append(_denominator)
            channels.append(_amplitude)
    assert set(denominators) == {one, Q * kp * lam**2, Qbar * km * lam**2}
    born = channels[0]
    assert (
        reduce_collinear(
            born * n(sp.PortPattern.exact(sp.Representation.mink(D), ports[0])),
            kinematics=kin,
        )
        == zero
    )
    assert (
        reduce_collinear(
            born * nb(sp.PortPattern.exact(sp.Representation.mink(D), ports[0])),
            kinematics=kin,
        )
        == zero
    )

    # Keep the photon and both spinor ports open; divide out only the Born color delta.
    return born, channels, generated_channels


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Extract the eikonal emission current
    """)
    return


@app.cell
def _(
    D,
    Q,
    Qbar,
    Representation,
    TensorExpression,
    born,
    channels,
    k,
    kin,
    km,
    kp,
    model,
    n,
    nb,
    ports,
    reduce_collinear,
    sp,
    zero,
):
    color_identity = TensorExpression.g(
        Representation.cof(3), Representation.cof(3).dual()
    )(ports[2], ports[1]).to_expression()
    current = (born / color_identity).together()
    generator = TensorExpression.color_t(8, 3)(
        ports[3], ports[2], ports[1]
    ).to_expression()
    gs = model.parameter("G").symbol
    # The generated convention gives a common minus relative to the reference's
    # emission current. Its relative quark/antiquark sign and square agree.
    for _emitted, _vector, _product, _sign in zip(
        channels[1:], (n, nb), (kp, km), (-1, 1)
    ):
        _expected = (
            _sign
            * gs
            * _vector(sp.PortPattern.exact(sp.Representation.mink(D), ports[3]))
            / _product
            * current
            * generator
        )
        assert reduce_collinear(_emitted - _expected, kinematics=kin).together() == zero
    eikonal = ((channels[1] + channels[2]) / (current * generator * gs)).together()
    assert (
        eikonal
        + n(sp.PortPattern.exact(sp.Representation.mink(D), ports[3])) / kp
        - nb(sp.PortPattern.exact(sp.Representation.mink(D), ports[3])) / km
    ).expand() == zero
    assert (
        reduce_collinear(
            eikonal * k(sp.PortPattern.exact(sp.Representation.mink(D), ports[3])),
            kinematics=kin,
        )
        == zero
    )
    assert eikonal.derivative(Q) == zero and eikonal.derivative(Qbar) == zero

    # Close color indices without summing or averaging the external spin states.
    return current, eikonal


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Contract the color factors
    """)
    return


@app.cell
def _(Representation, S, TensorExpression, ports):
    Nc, dA = S("Nc", "dA")
    fund = Representation.cof(Nc)
    color_born = TensorExpression.g(fund, fund.dual())(ports[2], ports[1])
    color_real = TensorExpression.color_t(dA, Nc)(ports[3], ports[2], ports[1])
    color_norms = []
    for _tensor in (color_born, color_real):
        _norm = (
            (_tensor * _tensor.dirac_adjoint())
            .simplify_algebra(contract="dots", gamma=False, color=True)
            .to_expression()
        )
        color_norms.append(
            TensorExpression(_norm.replace(dA, Nc**2 - 1))
            .simplify_algebra(
                contract="dots",
                gamma=False,
                color=True,
                color_substitute_cof_dimension_invariants=True,
            )
            .to_expression()
        )
    assert color_norms == [Nc, (Nc**2 - 1) / 2]
    cf = (color_norms[1] / color_norms[0]).together()

    # An arbitrary reference includes a nonzero r^2; no gauge term may survive.
    return Nc, cf, color_norms


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Sum physical gluon polarizations
    """)
    return


@app.cell
def _(
    D,
    S,
    eikonal,
    k,
    kin,
    km,
    kp,
    model,
    n,
    nb,
    ports,
    reduce_collinear,
    ref,
    zero,
):
    rn, rnb, rk, r2 = S("rn", "rnb", "rk", "r2")
    polarization_kin = (
        kin.with_scalar_product(ref, n, rn)
        .with_scalar_product(ref, nb, rnb)
        .with_scalar_product(ref, k, rk)
        .with_scalar_product(ref, ref, r2)
    )
    mu2 = S("mu2")
    polarization_results = {}
    for _label, _reference in (
        ("Covariant", None),
        ("n", n),
        ("nbar", nb),
        ("Arbitrary reference", ref),
    ):
        _density = model.particle("g").spin_sum(
            k,
            ports[3],
            mu2,
            dimension=D,
            reference=_reference,
            covariant=_reference is None,
        )
        squared = reduce_collinear(
            eikonal * eikonal.replace(ports[3], mu2) * _density,
            kinematics=polarization_kin,
        ).together()
        assert (squared - 4 / (kp * km)).together() == zero
        polarization_results[_label] = squared
    return polarization_results, squared


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Normalize the soft function
    """)
    return


@app.cell
def _(
    D,
    Nc,
    Q,
    Qbar,
    S,
    Symbol,
    born,
    cf,
    color_norms,
    current,
    eikonal,
    generated_channels,
    k,
    km,
    kp,
    n,
    nb,
    pm,
    polarization_results,
    squared,
    transverse,
    zero,
):
    alpha = S("alpha_s")
    soft_function = (
        squared * cf * 4 * Symbol.PI * alpha / (32 * Symbol.PI**2)
    ).together()
    assert (soft_function - cf * alpha / (2 * Symbol.PI * kp * km)).together() == zero
    eta = S("eta")
    assert (
        soft_function.replace(kp, eta * kp).replace(km, km / eta) - soft_function
    ).together() == zero
    energy, z = S("energy", "z")
    angular_soft = (
        soft_function.replace(kp, energy * (1 - z))
        .replace(km, energy * (1 + z))
        .together()
    )
    assert (
        angular_soft.replace(energy, 2 * energy) - angular_soft / 4
    ).together() == zero
    soft = {
        "generated_channels": generated_channels,
        "D": D,
        "n": n,
        "nb": nb,
        "k": k,
        "kp": kp,
        "km": km,
        "Q": Q,
        "Qbar": Qbar,
        "pm": pm,
        "transverse": transverse,
        "born": born,
        "current": current,
        "eikonal": eikonal,
        "colors": color_norms,
        "cf": cf,
        "Nc": Nc,
        "alpha": alpha,
        "energy": energy,
        "z": z,
        "soft_function": soft_function,
        "angular_soft": angular_soft,
        "polarization_results": polarization_results,
    }
    return (soft,)


@app.cell
def _(mo, soft):
    mo.vstack(
        [
            mo.md("## Generated amplitudes"),
            mo.md("One Born diagram and two real-emission diagrams:"),
            mo.hstack(soft["generated_channels"][0].diagrams),
            mo.hstack(soft["generated_channels"][1].diagrams),
        ]
    )
    return


@app.cell
def _(TensorExpression, mo, soft):
    mo.vstack(
        [
            mo.md(r"""
        ## Collinear projection

        The transverse metric is
        $g_\perp^{\mu\nu}=g^{\mu\nu}-
        (n^\mu\bar n^\nu+\bar n^\mu n^\nu)/2$.
        It is idempotent, annihilates both light-cone vectors and has trace $D-2$.

        Both the collinear bra and anticollinear ket select
        $P_-=\not{\bar n}\not n/4$.
        The calculation checks $P_-^2=P_-$, $P_-(1-P_-)=0$,
        $\mathrm{tr}P_-=2$ and the corresponding Dirac annihilation identities.
        """),
            TensorExpression(soft["transverse"]),
            TensorExpression(soft["pm"]),
            mo.md("**Projected Born amplitude:**"),
            TensorExpression(soft["born"]),
        ]
    )
    return


@app.cell
def _(TensorExpression, mo, soft):
    mo.vstack(
        [
            mo.md(r"""
        ## Factorization and gauge independence

        After a Laurent expansion through $\lambda^{-2}$, each generated
        emission diagram equals the Born current times its color generator
        and an eikonal factor. The independent hard scales $Q,\bar Q$ cancel.

        The current below is extracted from their sum. Its common sign depends
        on the emission convention and cancels from the square; the relative
        quark/antiquark sign is fixed by the generated diagrams.
        Contracting it with $k$ gives zero.
        """),
            TensorExpression(soft["eikonal"]),
            mo.md(r"""
        The Born and emitted color norms are $N_c$ and $(N_c^2-1)/2$,
        so their ratio is $C_F=(N_c^2-1)/(2N_c)$. The common open Born current
        cancels. All four polarization choices below agree exactly, including
        a generic reference vector with arbitrary nonzero norm.
        """),
            mo.ui.table(
                [
                    {"Polarization sum": label, "Squared eikonal current": str(value)}
                    for label, value in soft["polarization_results"].items()
                ],
                selection=None,
            ),
        ]
    )
    return


@app.cell
def _(mo, soft):
    colors = mo.ui.dropdown(
        {f"SU({n})": n for n in (2, 3, 5)}, value="SU(3)", label="Color group"
    )
    polarization = mo.ui.dropdown(
        list(soft["polarization_results"]),
        value="Arbitrary reference",
        label="Gluon polarization sum",
    )
    gluon_energy = mo.ui.slider(0.5, 5.0, step=0.5, value=2.0, label="Gluon energy")
    angle = mo.ui.slider(-0.9, 0.9, step=0.1, value=0.0, label="cos θ")
    coupling = mo.ui.slider(0.05, 0.2, step=0.001, value=0.118, label="αs")
    mo.vstack(
        [mo.hstack([colors, polarization]), mo.hstack([gluon_energy, angle, coupling])]
    )
    return angle, colors, coupling, gluon_energy, polarization


@app.cell
def _(Symbol, angle, colors, coupling, gluon_energy, np, polarization, soft):
    selected_soft = (
        soft["polarization_results"][polarization.value]
        * soft["cf"]
        * 4
        * Symbol.PI
        * soft["alpha"]
        / (32 * Symbol.PI**2)
    ).together()
    assert (selected_soft - soft["soft_function"]).together() == 0
    _parameters = {
        soft["Nc"]: colors.value,
        soft["alpha"]: coupling.value,
        soft["energy"]: gluon_energy.value,
        soft["z"]: angle.value,
    }
    selected_value = complex(soft["angular_soft"].evaluate(_parameters)).real
    assert selected_value > 0
    angular_rows = []
    for _z in np.linspace(-0.9, 0.9, 19):
        _point = {**_parameters, soft["z"]: float(_z)}
        _value = complex(soft["angular_soft"].evaluate(_point)).real
        _double_energy = complex(
            soft["angular_soft"].evaluate(
                {**_point, soft["energy"]: 2 * gluon_energy.value}
            )
        ).real
        assert abs(_double_energy - _value / 4) < 1e-12
        angular_rows.append(
            {
                "cos θ": round(float(_z), 2),
                "Soft function": _value,
                "At twice the energy": _double_energy,
            }
        )
    return angular_rows, selected_soft, selected_value


@app.cell(hide_code=True)
def _(angular_rows, mo, selected_soft, selected_value):
    mo.vstack(
        [
            mo.md(r"""
        ## Leading soft function

        With $g_s^2=4\pi\alpha_s$ and the reference's explicit
        prefactor $1/(32\pi^2)$,
        $$S^{(0)}(k)=\frac{C_F\alpha_s}{2\pi k^+k^-},\qquad
        k^+=n\cdot k,\quad k^-=\bar n\cdot k.$$

        In the back-to-back frame $k^\pm=E_g(1\mp\cos\theta)$.
        The table shows both the collinear enhancement and the $E_g^{-2}$
        soft scaling. Energy values use any consistent units; $S^{(0)}$ has
        inverse-energy-squared units.

        This is the unintegrated leading soft kernel with the stated prefactor.
        No phase-space integration or infrared regulator is included.
        """),
            selected_soft,
            mo.md(f"**Selected value: {selected_value:.8g}**"),
            mo.ui.table(angular_rows, selection=None),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
