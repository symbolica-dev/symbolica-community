import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Higgs to gluons: a finite triangle",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Higgs to gluons: a finite quark triangle

    [Browse notebooks](/) · [Tree-level Higgs decays](/?file=hep/higgs_decay.py) ·
    [Electron anomalous moment](/?file=hep/gminus2.py) ·
    [Odd photons](/?file=hep/odd_photons.py) ·
    [Tadpole mass insertions](/?file=hep/tadpole_mass_insertions.py)

    Generate both orientations of the massive top-quark loop in $H\to gg$,
    reduce the open Lorentz tensor and its scalar integrals with native IBP,
    and evaluate the remaining triangle with OneLOop. This follows
    [FeynCalc's Higgs-to-gluons example](https://feyncalc.github.io/FeynCalcExamples/QCD/OneLoop/H-GlGl).

    Both outgoing gluons are on shell. Write $s=m_H^2$, $m=m_q$ and
    $T^{\mu\nu}=(s/2)g^{\mu\nu}-k_2^\mu k_1^\nu$.
    The two orientations retain their fermion-loop signs and color traces.
    The calculation checks both Ward identities and the full tensor's UV poles.
    Physical gluon polarization sums then remove the tensor terms that vanish
    against transverse external states.
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
    import math

    import marimo as mo
    from symbolica import E, Replacement, S, Symbol
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hepkit import (
        IBPFamily,
        Kinematics,
        Model,
        TensorReducer,
        oneloop,
    )
    from symbolica.community.tensor import TensorExpression

    _set_namespace("hgg")
    return (
        E,
        IBPFamily,
        Kinematics,
        Model,
        Replacement,
        S,
        Symbol,
        TensorExpression,
        TensorReducer,
        hep,
        math,
        mo,
        oneloop,
        sp,
    )


@app.cell(hide_code=True)
def _(
    D,
    E,
    K,
    Nc,
    P,
    Q11,
    Q12,
    Q21,
    Q22,
    Qg,
    Replacement,
    TensorExpression,
    arguments,
    ca,
    cb,
    color,
    coordinates,
    dA,
    family,
    gs,
    index,
    kinematics,
    m,
    metric,
    model,
    mu,
    nu,
    p_mu,
    p_nu,
    q_mu,
    q_nu,
    reducer,
    sp,
    wave,
    y,
):
    def triangle_integrals(orientation, diagram):
        """Project one fermion-loop orientation and map it to the common triangle family."""
        assert diagram.overall_factor_expression(evaluate=True) == E("-1")
        assert diagram.numerator_prefactor_expression() == E("1")
        numerator = model.expand_couplings(
            diagram.numerator_expression(in_lmb=True).to_expression()
        ).replace(model.parameter("yt").symbol * E("1/2").sqrt(), y)
        for edge in diagram.external_edges:
            if edge.external_index == 0:
                continue
            port = dict(
                next(
                    diagram.projector_expression().match(
                        wave(
                            edge.id,
                            sp.PortPattern.exact(sp.Representation.mink(4), index),
                        ),
                        max_level=0,
                    )
                )
            )[index]
            numerator = numerator.replace(
                sp.PortPattern.exact(sp.Representation.mink(4), port),
                sp.PortPattern.exact(
                    sp.Representation.mink(D), (mu, nu)[edge.external_index - 1]
                ),
            )
            numerator = numerator.replace(
                sp.PortPattern.exact(sp.Representation.coad(8), port),
                sp.PortPattern.exact(
                    sp.Representation.coad(dA), (ca, cb)[edge.external_index - 1]
                ),
            )
        numerator = (
            numerator.replace(
                sp.PortPattern.exact(sp.Representation.mink(4), index),
                sp.PortPattern.exact(sp.Representation.mink(D), index),
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
        # Shared Idenso evaluates Tr(Ta Tb)=1/2 delta_ab, before summing the two
        # orientations. Promote Lorentz dimension before Dirac trace; tr(1)=4.
        trace = (
            TensorExpression(numerator.expand())
            .simplify_algebra(
                contract="dots",
                gamma=True,
                epsilon=True,
                color=True,
                color_substitute_cof_dimension_invariants=True,
            )
            .expand()
            .to_expression()
        )
        trace = (
            trace
            * diagram.overall_factor_expression(evaluate=True)
            * diagram.numerator_prefactor_expression()
        ).expand()
        assert (trace.coefficient(color) * color - trace).expand() == E("0")
        # Chirality makes the complete trace vanish at zero quark mass, even
        # with fixed Yukawa coupling. Check before dividing out the mass factor.
        assert trace.replace(m, E("0")).expand() == E("0")
        trace = trace.coefficient(color) / (gs**2 * y * m)
        mapping = diagram.integral_family(kinematics=kinematics).mapping_to(
            family, [(-1 if orientation else 1) * K(0)]
        )
        assert mapping is not None
        # IntegralMapping acts on scalar numerators: reduce free loop indices first.
        reduced = mapping.apply(kinematics.apply(reducer.reduce(trace))).expand()
        reduced = reduced.replace_multiple(
            [
                Replacement(
                    metric(
                        sp.PortPattern.exact(sp.Representation.mink(D), mu),
                        sp.PortPattern.exact(sp.Representation.mink(D), nu),
                    ),
                    Qg,
                ),
                Replacement(
                    P(0, sp.PortPattern.exact(sp.Representation.mink(D), mu)),
                    p_mu + q_mu,
                ),
                Replacement(
                    P(0, sp.PortPattern.exact(sp.Representation.mink(D), nu)),
                    p_nu + q_nu,
                ),
                Replacement(
                    P(1, sp.PortPattern.exact(sp.Representation.mink(D), mu)), p_mu
                ),
                Replacement(
                    P(1, sp.PortPattern.exact(sp.Representation.mink(D), nu)), p_nu
                ),
            ]
        ).expand()
        reduced = reduced.replace_multiple(
            [
                Replacement(p_mu * p_nu, Q11),
                Replacement(p_mu * q_nu, Q12),
                Replacement(q_mu * p_nu, Q21),
                Replacement(q_mu * q_nu, Q22),
            ]
        )
        polynomial = family.rewrite_numerator(reduced, coordinates).expand()

        terms = []
        for monomial, coefficient in polynomial.coefficient_list(*coordinates):
            powers = tuple(
                1 - monomial.to_polynomial(vars=coordinates).degree(label)
                for label in coordinates
            )
            assert not coefficient.matches(K(arguments))
            assert not coefficient.matches(P(arguments))
            assert all(
                coefficient.derivative(label).expand() == E("0")
                for label in coordinates
            )

            terms.append((powers, coefficient))
        return terms, polynomial

    return (triangle_integrals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Define the masses and tensor structures
    """)
    return


@app.cell
def _(Model, S, hep, sp):
    model = Model.standard_model()
    D, s = S("D", "s")
    m = model.particle("t").mass
    gs = model.parameter("G").symbol
    y, mu, nu, ca, cb, eps = S("y", "mu", "nu", "ca", "cb", "eps")
    K, P, mink, cof, coad, metric = (
        hep.Kinematics.loop_momentum,
        hep.Kinematics.external_momentum,
        sp.Representation.mink,
        sp.Representation.cof,
        sp.Representation.coad,
        sp.TensorName.g().to_expression(),
    )
    index, wave, arguments = S("index_", "wave_", "arguments__")
    Nc, dA = S("Nc", "dA")
    # The SM UFO stores yt=sqrt(2)*m/v separately from the propagator pole mass.
    # Define y=yt/sqrt(2); the comparison later imposes y=m/v explicitly.
    return (
        D,
        K,
        Nc,
        P,
        arguments,
        ca,
        cb,
        dA,
        eps,
        gs,
        index,
        m,
        metric,
        model,
        mu,
        nu,
        s,
        wave,
        y,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Fix the Yukawa convention at tree level
    """)
    return


@app.cell
def _(E, S, Symbol, metric, model, y):
    top, higgs, gluon = (model.particle(_name) for _name in ("t", "H", "g"))
    yukawa_vertices = [
        v
        for v in model.vertex_rules
        if sorted(v.particles) == sorted([top.antiname, top.name, higgs.name])
    ]
    gluon_vertices = [
        v
        for v in model.vertex_rules
        if sorted(v.particles) == sorted([top.antiname, top.name, gluon.name])
    ]
    assert len(yukawa_vertices) == len(gluon_vertices) == 1
    yukawa_tree_result = model.process(
        [higgs], [top, top.antiparticle], vertex_allow=yukawa_vertices
    ).generate_diagrams(max_vertices=1, numerator_grouping=None, progress=None)
    assert len(yukawa_tree_result.diagrams) == 1
    yukawa_tree = yukawa_tree_result.diagrams[0]
    yukawa_tree_kernel = model.expand_couplings(
        yukawa_tree.numerator_expression().to_expression()
    ).replace(model.parameter("yt").symbol * E("1/2").sqrt(), y)
    assert yukawa_tree.overall_factor_expression(evaluate=True) == E("1")
    assert yukawa_tree.numerator_prefactor_expression() == E("1")
    # Strip only the generated spin/color identity to establish the tree phase.
    identities = S("left_", "right_")
    yukawa_tree_coupling = yukawa_tree_kernel.replace(metric(*identities), E("1"))
    assert (yukawa_tree_coupling + Symbol.I * y).expand() == E("0")
    return gluon, gluon_vertices, higgs, yukawa_tree_result, yukawa_vertices


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate both fermion-loop orientations
    """)
    return


@app.cell
def _(gluon, gluon_vertices, higgs, model, yukawa_vertices):
    result = model.process(
        [higgs], [gluon, gluon], vertex_allow=gluon_vertices + yukawa_vertices
    ).generate_diagrams(
        loops=1,
        max_vertices=3,
        maximum_bridges=0,
        numerator_grouping=None,
        progress=None,
    )
    assert len(result.diagrams) == 2
    # P0=k1+k2 is incoming Higgs momentum and P1=k1 the first outgoing gluon.
    return (result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Define the triangle family and tensor projection
    """)
    return


@app.cell
def _(
    D,
    E,
    K,
    Kinematics,
    P,
    S,
    TensorReducer,
    ca,
    cb,
    dA,
    gluon,
    index,
    result,
    s,
    sp,
):
    kinematics = (
        Kinematics(D, momenta=[K(0), P(0), P(1)])
        .with_scalar_product(P(0), P(0), s)
        .with_scalar_product(P(1), P(1), E("0"))
        .with_scalar_product(P(0), P(1), s / 2)
    )
    reducer = TensorReducer(
        D,
        integrated=[K(0, sp.PortPattern.exact(sp.Representation.mink(D)))],
        external=[
            P(0, sp.PortPattern.exact(sp.Representation.mink(D))),
            P(1, sp.PortPattern.exact(sp.Representation.mink(D))),
        ],
    )
    family = result.diagrams[0].integral_family(kinematics=kinematics)
    coordinates = S("d0", "d1", "d2")
    Qg, Q11, Q12, Q21, Q22 = S("Qg", "Q11", "Q12", "Q21", "Q22")
    p_mu, p_nu, q_mu, q_nu = S("pmu", "pnu", "qmu", "qnu")
    color = gluon.color_sum(ca, cb).replace(
        sp.PortPattern.exact(sp.Representation.coad(8), index),
        sp.PortPattern.exact(sp.Representation.coad(dA), index),
    )
    return (
        Q11,
        Q12,
        Q21,
        Q22,
        Qg,
        color,
        coordinates,
        family,
        kinematics,
        p_mu,
        p_nu,
        q_mu,
        q_nu,
        reducer,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Project the two numerators

    The folded per-diagram routine performs color and Dirac reduction, tensor projection and the exact routing map. Check that both orientations agree.
    """)
    return


@app.cell
def _(E, result, triangle_integrals):
    terms_by_diagram, targets, routed_polynomials = [], set(), []
    for _orientation, _diagram in enumerate(result.diagrams):
        _terms, _polynomial = triangle_integrals(_orientation, _diagram)
        terms_by_diagram.append(_terms)
        routed_polynomials.append(_polynomial)
        targets.update(powers for powers, _ in _terms)
    assert (routed_polynomials[0] - routed_polynomials[1]).together() == E("0")
    return targets, terms_by_diagram


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Reduce the scalar integrals
    """)
    return


@app.cell
def _(IBPFamily, S, family, targets):
    solution = IBPFamily(family, name="higgs_gluon_triangle").reduce_laporta(
        [list(t) for t in sorted(targets)], max_depth=2
    )
    assert {tuple(p) for p in solution.residuals} == {
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 1, 1),
    }
    integral, a0, b0, c0 = S("I", "A0", "B0", "C0")
    # One-line pinches are shifted equal-mass tadpoles; the two-line pinch has
    # invariant s, and the three-line master has external invariants (0,0,s).
    for powers in ((0, 0, 1), (0, 1, 0), (1, 0, 0)):
        assert family.sector(powers).find_mapping(family.sector([0, 1, 0])) is not None
    # Symanzik polynomials independently identify the mass/invariant arguments
    # of the scalar B0(s;m^2,m^2) and C0(0,0,s;m^2,m^2,m^2) masters.
    return a0, b0, c0, integral, solution


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Identify the scalar masters

    Symanzik polynomials independently fix the mass and invariant arguments.
    """)
    return


@app.cell
def _(E, Replacement, S, a0, b0, c0, family, integral, m, s):
    x0, x1, x2 = S("x0", "x1", "x2")
    triangle_U, triangle_F = family.symanzik([x0, x1, x2])
    assert (triangle_U - x0 - x1 - x2).expand() == E("0")
    assert (triangle_F - m**2 * triangle_U**2 + s * x1 * x2).expand() == E("0")
    bubble_U, bubble_F = family.sector([0, 1, 1]).symanzik([x1, x2])
    assert (bubble_U - x1 - x2).expand() == E("0")
    assert (bubble_F - m**2 * bubble_U**2 + s * x1 * x2).expand() == E("0")
    masters = [Replacement(integral(*p), a0) for p in ((0, 0, 1), (0, 1, 0), (1, 0, 0))]
    masters += [Replacement(integral(0, 1, 1), b0), Replacement(integral(1, 1, 1), c0)]
    return (masters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Assemble the master coefficients

    Verify the Ward identities before choosing a physical transverse basis.
    """)
    return


@app.cell
def _(
    D,
    E,
    Q11,
    Q22,
    a0,
    b0,
    c0,
    eps,
    integral,
    masters,
    solution,
    terms_by_diagram,
):
    integrated_by_diagram = []
    for terms in terms_by_diagram:
        reduction = sum(
            (
                coefficient * solution.reduce(list(powers), integral=integral)
                for powers, coefficient in terms
            ),
            E("0"),
        )
        integrated_by_diagram.append(reduction.replace_multiple(masters).together())
    assert (integrated_by_diagram[0] - integrated_by_diagram[1]).together() == E("0")
    integrated = sum(integrated_by_diagram, E("0")).expand()
    # Truncating scalar masters at finite order is safe: their rational
    # coefficients have no hidden (D-4) poles.
    for master in (a0, b0, c0):
        assert (
            integrated.coefficient(master)
            .together()
            .replace(D, 4 - 2 * eps)
            .series(eps, 0, -1)
            .to_expression()
        ) == E("0")
    assert integrated.coefficient(Q11).together() == E("0")
    assert integrated.coefficient(Q22).together() == E("0")
    # On-shell open tensors can retain k1_mu*k2_nu. It is transverse to both
    # massless momenta and vanishes against their own physical polarizations.
    return (integrated,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Extract the D-dimensional form factor
    """)
    return


@app.cell
def _(D, E, Q12, Q21, Qg, b0, c0, integrated, m, s):
    physical = integrated.replace(Q12, E("0")).together()
    transverse = s * Qg / 2 - Q21
    form_factor = (physical.expand().coefficient(Qg) * 2 / s).together()
    assert (physical - form_factor * transverse).together() == E("0")
    expected_D = -4 / s * (2 * (4 - D) / (D - 2) * b0 + (8 * m**2 / (D - 2) - s) * c0)
    assert (form_factor - expected_D).together() == E("0")
    # Keep D dependence until after supplying the scalar pole residues. The
    # finite rational 2 is produced by (D-4)*B0; setting D=4 first loses it.
    return (form_factor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check ultraviolet finiteness
    """)
    return


@app.cell
def _(D, E, Qg, Replacement, S, a0, b0, c0, eps, integrated, m, s):
    Af, Bf, Cf = S("Af", "Bf", "Cf")
    laurent = (
        integrated.replace(D, 4 - 2 * eps)
        .replace_multiple(
            [
                Replacement(a0, m**2 / eps + Af),
                Replacement(b0, 1 / eps + Bf),
                Replacement(c0, Cf),
            ]
        )
        .series(eps, 0, 0)
        .to_expression()
        .expand()
    )
    assert laurent.coefficient(eps**-1).together() == E("0")
    assert laurent.coefficient(eps**-2).together() == E("0")
    finite = dict(laurent.coefficient_list(eps))[E("1")].together().expand()
    finite_factor = (finite.coefficient(Qg) * 2 / s).together()
    assert (finite_factor + 4 / s * (2 + (4 * m**2 - s) * Cf)).together() == E("0")
    # Scalar masters restore i/(16*pi^2). Thus native iM has coefficient
    # -i*gs^2*y*m/(4*pi^2*s) [2+(4m^2-s)C0] multiplying delta_ab*T_mu_nu.
    return Cf, finite, finite_factor


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Verify the heavy-quark limit
    """)
    return


@app.cell
def _(Cf, E, finite_factor, m, s):
    normalized = (-3 * m**2 * finite_factor / 4).together()
    assert (normalized - 3 * m**2 / s * (2 + (4 * m**2 - s) * Cf)).together() == E("0")
    small_s_triangle = (
        -1 / (2 * m**2) - s / (24 * m**4) - s**2 / (180 * m**6) - s**3 / (1120 * m**8)
    )
    heavy_series = (
        normalized.replace(Cf, small_s_triangle).series(s, 0, 2).to_expression()
    )
    assert (
        heavy_series - 1 - 7 * s / (120 * m**2) - s**2 / (168 * m**4)
    ).expand() == E("0")

    # Shared spin/color contractions establish the unaveraged tensor norm.
    return heavy_series, normalized


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check Bose symmetry and transversality
    """)
    return


@app.cell
def _(
    D,
    E,
    P,
    Replacement,
    TensorExpression,
    kinematics,
    metric,
    mu,
    nu,
    s,
    sp,
):
    tensor = s * metric(
        sp.PortPattern.exact(sp.Representation.mink(D), mu),
        sp.PortPattern.exact(sp.Representation.mink(D), nu),
    ) / 2 - (
        P(0, sp.PortPattern.exact(sp.Representation.mink(D), mu))
        - P(1, sp.PortPattern.exact(sp.Representation.mink(D), mu))
    ) * P(1, sp.PortPattern.exact(sp.Representation.mink(D), nu))
    exchanged = tensor.replace_multiple(
        [
            Replacement(
                P(0, sp.PortPattern.exact(sp.Representation.mink(D), mu)),
                P(0, sp.PortPattern.exact(sp.Representation.mink(D), nu)),
            ),
            Replacement(
                P(1, sp.PortPattern.exact(sp.Representation.mink(D), mu)),
                P(0, sp.PortPattern.exact(sp.Representation.mink(D), nu))
                - P(1, sp.PortPattern.exact(sp.Representation.mink(D), nu)),
            ),
            Replacement(
                P(1, sp.PortPattern.exact(sp.Representation.mink(D), nu)),
                P(0, sp.PortPattern.exact(sp.Representation.mink(D), mu))
                - P(1, sp.PortPattern.exact(sp.Representation.mink(D), mu)),
            ),
        ]
    )
    assert (tensor - exchanged).expand() == E("0")
    for contraction in (
        P(1, sp.PortPattern.exact(sp.Representation.mink(D), mu)),
        P(0, sp.PortPattern.exact(sp.Representation.mink(D), nu))
        - P(1, sp.PortPattern.exact(sp.Representation.mink(D), nu)),
    ):
        ward = (
            TensorExpression((tensor * contraction).expand())
            .contract()
            .to_dots()
            .to_expression()
        )
        assert kinematics.apply(ward).expand() == E("0")
    norm = kinematics.apply(TensorExpression((tensor**2).expand()).contract().to_dots())
    assert (norm.to_expression() - (D - 2) * s**2 / 4).expand() == E("0")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Normalize the color tensor
    """)
    return


@app.cell
def _(TensorExpression, color, dA):
    color_norm = (
        TensorExpression(color**2)
        .contract(rank_one=False, collect_chains=False, collect_traces=False)
        .to_dots()
        .to_expression()
    )
    assert color_norm == dA
    # Also evaluate the physical axial polarization sums, taking each gluon as
    # the other's null reference. Ward identities imply agreement with -g sums.
    return (color_norm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Contract physical gluon polarizations
    """)
    return


@app.cell
def _(
    E,
    Kinematics,
    P,
    Replacement,
    S,
    TensorExpression,
    gluon,
    metric,
    mu,
    nu,
    s,
    sp,
):
    rho, sigma = S("rho", "sigma")
    physical_kinematics = (
        Kinematics(momenta=[P(1), P(2)])
        .with_scalar_product(P(1), P(1), E("0"))
        .with_scalar_product(P(2), P(2), E("0"))
        .with_scalar_product(P(1), P(2), s / 2)
    )
    physical_tensor = s * metric(
        sp.PortPattern.exact(sp.Representation.mink(4), mu),
        sp.PortPattern.exact(sp.Representation.mink(4), nu),
    ) / 2 - P(2, sp.PortPattern.exact(sp.Representation.mink(4), mu)) * P(
        1, sp.PortPattern.exact(sp.Representation.mink(4), nu)
    )
    conjugate_tensor = physical_tensor.replace_multiple(
        [Replacement(mu, rho), Replacement(nu, sigma)]
    )
    polarization_sum = gluon.spin_sum(P(1), mu, rho, reference=P(2)) * gluon.spin_sum(
        P(2), nu, sigma, reference=P(1)
    )
    physical_norm = physical_kinematics.apply(
        TensorExpression(
            (physical_tensor * conjugate_tensor * polarization_sum).expand()
        )
        .contract()
        .to_dots()
    )
    assert (physical_norm.to_expression() - s**2 / 2).together() == E("0")
    # Contract the actual full finite open tensor, including its Q12 coefficient,
    # against both physical polarization projectors. Their difference is exactly
    # zero without setting the coefficient of Q12 to zero by hand.
    return (
        physical_kinematics,
        physical_norm,
        physical_tensor,
        polarization_sum,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the complete finite tensor
    """)
    return


@app.cell
def _(
    E,
    P,
    Q12,
    Q21,
    Qg,
    Replacement,
    TensorExpression,
    finite,
    finite_factor,
    metric,
    mu,
    nu,
    physical_kinematics,
    physical_tensor,
    polarization_sum,
    sp,
):
    full_finite_tensor = finite.replace_multiple(
        [
            Replacement(
                Qg,
                metric(
                    sp.PortPattern.exact(sp.Representation.mink(4), mu),
                    sp.PortPattern.exact(sp.Representation.mink(4), nu),
                ),
            ),
            Replacement(
                Q12,
                P(1, sp.PortPattern.exact(sp.Representation.mink(4), mu))
                * P(2, sp.PortPattern.exact(sp.Representation.mink(4), nu)),
            ),
            Replacement(
                Q21,
                P(2, sp.PortPattern.exact(sp.Representation.mink(4), mu))
                * P(1, sp.PortPattern.exact(sp.Representation.mink(4), nu)),
            ),
        ]
    )
    physical_projection_residual = (
        physical_kinematics.apply(
            TensorExpression(
                (
                    (full_finite_tensor - finite_factor * physical_tensor)
                    * polarization_sum
                ).expand()
            )
            .contract()
            .to_dots()
        )
        .to_expression()
        .together()
    )
    assert physical_projection_residual == E("0")
    # y=m/v and gs^2=4*pi*alpha_s. Color dimension is specialized after contraction.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Restore couplings and form the square
    """)
    return


@app.cell
def _(Nc, S, Symbol, color_norm, dA, physical_norm):
    alpha_s, vev, A, Abar = S("alpha_s", "vev", "A", "Abar")
    MH = S("MH", is_positive=True)
    squared = (
        physical_norm.to_expression()
        * color_norm
        * (alpha_s / (3 * Symbol.PI * vev)) ** 2
        * A
        * Abar
    ).replace(dA, Nc**2 - 1)
    return A, Abar, MH, alpha_s, squared, vev


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Integrate the two-body decay width
    """)
    return


@app.cell
def _(A, Abar, E, Kinematics, MH, Nc, P, Symbol, alpha_s, s, squared, vev):
    decay_kinematics = (
        Kinematics()
        .with_scalar_product(P(0), P(0), MH**2)
        .with_scalar_product(P(1), P(1), E("0"))
        .with_scalar_product(P(2), P(2), E("0"))
        .with_scalar_product(P(1), P(2), MH**2 / 2)
    )
    phase_space = decay_kinematics.two_body_phase_space(P(1), P(2)).expand()
    flux = decay_kinematics.flux(P(0))
    assert (phase_space - 1 / (32 * Symbol.PI**2)).together() == E("0")
    assert flux == 2 * MH
    # Integrate dOmega=4*pi and include 1/2! for identical final gluons.
    width = (
        squared.replace(s, MH**2) * 4 * Symbol.PI * phase_space / flux / 2
    ).together()
    assert (
        width
        - (Nc**2 - 1) * alpha_s**2 * MH**3 * A * Abar / (576 * Symbol.PI**3 * vev**2)
    ).together() == E("0")
    return (width,)


@app.cell(hide_code=True)
def _(
    family,
    form_factor,
    heavy_series,
    mo,
    result,
    solution,
    width,
    yukawa_tree_result,
):
    mo.vstack(
        [
            mo.md("**Generated Yukawa vertex and both quark-loop orientations**"),
            mo.hstack([*yukawa_tree_result.diagrams, *result.diagrams]),
            mo.md("**Three-propagator family and native IBP reduction**"),
            family,
            mo.ui.table([solution.stats], selection=None),
            mo.md(r"""
        The finite search leaves shifted tadpoles, a bubble and a triangle.
        Verified loop shifts identify the tadpoles; their coefficients cancel
        from the physical form factor. The displayed coefficient multiplies
        $g_s^2ym\,\delta^{ab}T^{\mu\nu}$ before the common loop measure
        $i/(16\pi^2)$, with Yukawa coupling $y=y_t/\sqrt2$.
        """),
            form_factor,
            mo.md(r"""
        **A finite rational term from dimensional regularization**

        At $D=4-2\epsilon$, the bubble's coefficient is proportional to
        $\epsilon$ while $B_0$ has a $1/\epsilon$ pole. Their product supplies
        the constant **2** in
        $$K=2+(4m^2-s)C_0(0,0,s;m^2,m^2,m^2).$$
        Setting $D=4$ before integrating would lose this term.

        With $y=m/v$ and $g_s^2=4\pi\alpha_s$, define
        $$A=\frac{3m^2}{s}K,\qquad
        i\mathcal M=-\frac{i\alpha_s}{3\pi v}\,A\,\delta^{ab}T_{\mu\nu}
          \epsilon_1^{*\mu}\epsilon_2^{*\nu}.$$
        The generated Yukawa vertex fixes the native phase. The reference page
        applies an explicit overall `PreFactor -> -1`; decay probabilities agree.

        **Heavy-quark expansion:** the normalized form factor approaches one.
        """),
            heavy_series,
            mo.md(r"""
        **Decay width from shared spin sums, color sums, flux and phase space**

        The identical-gluon factor $1/2!$ is applied once in phase space.
        There is no initial spin or color average for the Higgs.
        In the expression below, $\bar A$ denotes the complex conjugate of $A$.
        """),
            width,
        ]
    )
    return


@app.cell(hide_code=True)
def _(Cf, S, m, normalized, oneloop, s):
    scale2 = S("hgg_notebook::scale2")
    triangle_coefficients = oneloop.master_coefficients(
        oneloop.C0(0, 0, s, m**2, m**2, m**2, scale2)
    )
    normalized_oneloop = normalized.replace(Cf, triangle_coefficients[0])
    return normalized_oneloop, scale2, triangle_coefficients


@app.cell
def _(mo):
    mass_ratio = mo.ui.slider(
        0.05,
        5.0,
        step=0.05,
        value=0.35,
        label="Higgs mass / (2 × loop-quark mass)",
    )
    quark_mass = mo.ui.number(
        1.0, 500.0, step=0.5, value=172.5, label="Loop-quark mass [GeV]"
    )
    strong_coupling = mo.ui.number(0.01, 0.5, step=0.001, value=0.118, label="αs")
    mo.vstack([mass_ratio, mo.hstack([quark_mass, strong_coupling])])
    return mass_ratio, quark_mass, strong_coupling


@app.cell
def _(
    A,
    Abar,
    MH,
    Nc,
    alpha_s,
    m,
    mass_ratio,
    math,
    normalized_oneloop,
    quark_mass,
    s,
    scale2,
    strong_coupling,
    triangle_coefficients,
    vev,
    width,
):
    _ratio = mass_ratio.value
    _mass = quark_mass.value
    higgs_mass = 2 * _mass * _ratio
    _s = higgs_mass**2
    _point = {s: _s, m: _mass, scale2: _mass**2}
    form_factor_value = complex(normalized_oneloop.evaluate(_point))
    _tau = 1 / _ratio**2
    if _ratio <= 1:
        _f = math.asin(_ratio) ** 2
    else:
        _beta = math.sqrt(1 - _tau)
        _f = -0.25 * (math.log((1 + _beta) / (1 - _beta)) - 1j * math.pi) ** 2
    analytic_form_factor = 1.5 * _tau * (1 + (1 - _tau) * _f)
    form_factor_error = abs(form_factor_value - analytic_form_factor)
    assert math.isfinite(form_factor_error) and form_factor_error < 2e-8
    assert all(
        abs(complex(_pole.evaluate(_point))) < 1e-12
        for _pole in triangle_coefficients[1:]
    )
    scale_error = max(
        abs(
            complex(normalized_oneloop.evaluate({**_point, scale2: _factor * _mass**2}))
            - form_factor_value
        )
        for _factor in (0.25, 4.0)
    )
    assert scale_error < 2e-10
    decay_width = complex(
        width.evaluate(
            {
                Nc: 3.0,
                alpha_s: strong_coupling.value,
                MH: higgs_mass,
                vev: 246.22,
                A: form_factor_value,
                Abar: form_factor_value.conjugate(),
            }
        )
    )
    _reference_width = (
        strong_coupling.value**2
        * higgs_mass**3
        * abs(analytic_form_factor) ** 2
        / (72 * math.pi**3 * 246.22**2)
    )
    assert abs(decay_width - _reference_width) < 1e-8 * max(1.0, abs(_reference_width))
    assert decay_width.real >= 0 and abs(decay_width.imag) < 1e-12
    return (
        decay_width,
        form_factor_error,
        form_factor_value,
        higgs_mass,
        scale_error,
    )


@app.cell(hide_code=True)
def _(
    decay_width,
    form_factor_error,
    form_factor_value,
    higgs_mass,
    mo,
    scale_error,
):
    mo.vstack(
        [
            mo.md(r"""
        **Through the quark-production threshold**

        The triangle acquires an imaginary part for $m_H>2m$.
        With $\tau=4m^2/s$, the independent analytic reference is
        $$A=\frac32\tau[1+(1-\tau)f(\tau)],\quad
        f(\tau)=\begin{cases}
        \arcsin^2(1/\sqrt\tau),&\tau\ge1,\\
        -\frac14\left[\log\frac{1+\sqrt{1-\tau}}{1-\sqrt{1-\tau}}-i\pi\right]^2,&0<\tau<1.
        \end{cases}$$
        The width uses $|A|^2$, including the imaginary part above threshold.
        Numerical values use $N_c=3$ and $v=246.22$ GeV. This is the single
        quark-loop contribution with $y=m/v$; additional flavors and higher-order
        corrections are separate contributions.
        """),
            mo.ui.table(
                [
                    {
                        "mH [GeV]": higgs_mass,
                        "Re A (generated + IBP + OneLOop)": form_factor_value.real,
                        "Im A": form_factor_value.imag,
                        "|A|²": abs(form_factor_value) ** 2,
                        "Analytic comparison error": form_factor_error,
                        "Scale variation error": scale_error,
                        "Γ(H→gg) [GeV]": decay_width.real,
                    }
                ],
                selection=None,
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
