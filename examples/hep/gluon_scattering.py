import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Gluon scattering in D dimensions")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gluon–gluon scattering in $D$ dimensions

    [Browse all notebooks](/) · [Polarization sums](/?file=hep/polarization_sums.py) ·
    [Quark scattering](/?file=hep/quark_scattering.py) ·
    [Quark–gluon scattering](/?file=hep/quark_gluon_scattering.py)

    Generate $gg\to gg$ from the Standard Model: three exchange diagrams and
    the four-gluon contact interaction. Retain every interference term and
    keep both the Lorentz dimension $D$ and the SU($N_c$) color dimension symbolic.
    All particles are selected by model name.

    Each external gluon uses the shared physical axial projector. The two incoming
    momenta are each other's gauge references, as are the outgoing pair.
    Incoming colors are averaged; outgoing colors and polarizations are summed.
    The derivation reproduces the
    [FeynCalc four-gluon example](https://feyncalc.github.io/FeynCalcExamples/QCD/Tree/GlGl-GlGl)
    and its four-dimensional SU(3) reference exactly.

    The initial symbolic contraction can take several minutes. The controls below
    reuse that derived result.
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
    from symbolica.community.tensor import TensorExpression

    _set_namespace("gg")
    return E, Kinematics, Model, S, Symbol, TensorExpression, hep, mo, np, sp


@app.cell(hide_code=True)
def _():
    from symbolica.community.hep import Symbols

    return (Symbols,)


@app.cell(hide_code=True)
def _(D, E, Kinematics, N, P, S, Symbol, fixed_two, gs, np, s, t, u):
    def integrate_rate():
        z, cutoff, alpha = S("z", "cutoff", "alpha_s")
        kin4 = Kinematics.mandelstam([P(i) for i in range(4)], [E("0")] * 4, [s, t, u])
        phase = kin4.two_body_phase_space(P(2), P(3)) / kin4.flux(P(0), P(1))
        assert (phase - 1 / (64 * Symbol.PI**2 * s)).together() == 0
        # Azimuth integration supplies 2*pi and identical gluon events supply 1/2!.
        # Display s*sigma/alpha_s**2, a dimensionless angular-cut rate.
        density = (
            (fixed_two.replace(D, 4) * phase * Symbol.PI * s / alpha**2)
            .replace(gs**4, (4 * Symbol.PI * alpha) ** 2)
            .replace(t, -s * (1 - z) / 2)
            .together()
        )
        assert density.derivative(s).together() == 0
        primitive = density.integrate(z)
        assert (primitive.derivative(z) - density).together() == 0
        primitive = primitive.replace((z - 1).log(), (1 - z).log())
        cut_rate = primitive.replace(z, cutoff) - primitive.replace(z, -cutoff)
        assert cut_rate.replace(cutoff, 0).together() == 0
        assert (
            cut_rate.derivative(cutoff)
            - density.replace(z, cutoff)
            - density.replace(z, -cutoff)
        ).together() == 0
        nodes, weights = np.polynomial.legendre.leggauss(128)
        rate_checks = []
        for colors in (2, 3, 5):
            for cut in (0.2, 0.5, 0.8):
                exact = cut_rate.evaluate({N: colors, cutoff: cut})
                numeric = cut * sum(
                    weight * density.evaluate({N: colors, z: cut * node})
                    for node, weight in zip(nodes, weights, strict=True)
                )
                assert exact.real > 0 and abs(exact.imag) < 1e-10
                assert abs(exact - numeric) < 2e-11 * max(1, abs(exact))
                rate_checks.append((colors, cut, exact.real, abs(exact - numeric)))
        print(
            "Native flux, identical-event factor and nine angular-cut quadratures passed",
            flush=True,
        )
        return density, cut_rate, z, cutoff, rate_checks

    return (integrate_rate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate the four-gluon amplitude

    Keep the Lorentz dimension and color group symbolic.
    """)
    return


@app.cell
def _(E, Kinematics, Model, S, hep, sp):
    model = Model.standard_model()
    P = hep.Kinematics.external_momentum
    s = S("s", is_positive=True)
    t, u, D, N, dA = S("t", "u", "D", "N", "dA")
    gs = model.parameter("G").symbol
    coad, conj = (sp.Representation.coad, sp.BroadcastFunction.conj().to_expression())
    index, a, b, c, inv, wave, rep = S(
        "index_", "a_", "b_", "c_", "inv_", "wave_", "rep_"
    )
    ports = S("i0", "i1", "i2", "i3")
    # Give the conjugate amplitude distinct external labels; scope only its dummies.
    bra_ports = {port: S(f"bra_{_i}") for _i, port in enumerate(ports)}
    wrapped = S("adjoint")
    bars = S("j0", "j1", "j2", "j3")
    gluon = model.particle("g")
    vertices = [
        v
        for v in model.vertex_rules
        if v.particles.count(gluon.name) == len(v.particles)
        and len(v.particles) in (3, 4)
    ]
    assert len(vertices) == 2
    generated = model.process(
        ["g", "g"], ["g", "g"], vertex_allow=vertices
    ).generate_diagrams(
        max_vertices=2, maximum_bridges=None, numerator_grouping=None, progress=None
    )
    assert len(generated.diagrams) == 4
    kin = (
        Kinematics(D)
        .with_scalar_product(P(0), P(0), E("0"))
        .with_scalar_product(P(1), P(1), E("0"))
        .with_scalar_product(P(2), P(2), E("0"))
        .with_scalar_product(P(3), P(3), E("0"))
        .with_scalar_product(P(0), P(1), s / 2)
        .with_scalar_product(P(2), P(3), s / 2)
        .with_scalar_product(P(0), P(2), -t / 2)
        .with_scalar_product(P(1), P(3), -t / 2)
        .with_scalar_product(P(0), P(3), -u / 2)
        .with_scalar_product(P(1), P(2), -u / 2)
    )
    return (
        D,
        N,
        P,
        a,
        b,
        bars,
        bra_ports,
        c,
        conj,
        dA,
        generated,
        gluon,
        gs,
        index,
        inv,
        kin,
        model,
        ports,
        rep,
        s,
        t,
        u,
        wave,
        wrapped,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Align the diagram ports and propagators

    Retain all four weighted terms, including the contact vertex.
    """)
    return


@app.cell
def _(
    D,
    Symbols,
    a,
    b,
    c,
    generated,
    index,
    inv,
    kin,
    model,
    ports,
    rep,
    sp,
    wave,
):
    terms = []
    denominators = []
    for _diagram in generated.diagrams:
        _numerator = model.expand_couplings(
            _diagram.numerator_expression(in_lmb=True).to_expression()
        )
        for _edge in _diagram.external_edges:
            _match = dict(
                next(
                    _diagram.projector_expression().match(
                        wave(_edge.id, rep(4, index)), max_level=0
                    )
                )
            )
            _numerator = sp.TensorExpression(_numerator).rename_indices(
                {_match[index]: ports[_edge.external_index]}
            )
        _numerator = _numerator.with_lorentz_dimension(D)
        _denominator = kin.apply(
            _diagram.denominator_expression(dimension=D, in_lmb=True)
            .to_expression()
            .replace(Symbols.denominator(a, b, c, inv), inv)
        ).expand()
        denominators.append(_denominator)
        terms.append(
            _numerator
            * _diagram.overall_factor_expression(evaluate=True)
            * _diagram.numerator_prefactor_expression()
            / _denominator
        )
    return denominators, terms


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Collect the coherent operator
    """)
    return


@app.cell
def _(E, denominators, kin, s, t, terms, u):
    assert set(denominators) == {E("1"), s, t, u}
    raw_amplitude = sum(terms, E("0"))
    operator = kin.apply(
        raw_amplitude.contract(collect_chains=False, collect_traces=False)
        .to_dots()
        .expand()
    )
    assert len(operator.structure.slots) == 8
    return (operator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Form its physical adjoint
    """)
    return


@app.cell
def _(
    D,
    P,
    a,
    b,
    bars,
    bra_ports,
    conj,
    gs,
    operator,
    ports,
    s,
    sp,
    t,
    u,
    wrapped,
):
    adjoint = operator.dirac_adjoint().to_expression().replace(conj(P(a, b)), P(a, b))
    for _real in (s, t, u, D, gs):
        adjoint = adjoint.replace(conj(_real), _real)
    adjoint = (
        sp.TensorExpression(adjoint)
        .wrap_indices(wrapped, dummies_only=True)
        .rename_indices(bra_ports)
        .to_expression()
    )
    for _i in range(4):
        adjoint = adjoint.replace(bra_ports[ports[_i]], bars[_i])
    return (adjoint,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Sum and average the color states
    """)
    return


@app.cell
def _(E, bars, gluon, ports):
    color_projector = E("1")
    for _i in range(4):
        color_projector *= gluon.color_sum(ports[_i], bars[_i], average=_i < 2)
    return (color_projector,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Promote the color representation

    Use a named adjoint dimension until color contraction is complete.
    """)
    return


@app.cell
def _(adjoint, color_projector, dA, gluon, index, operator, sp):
    generic = (
        operator.to_expression() * adjoint * color_projector * gluon.color**2 / dA**2
    ).replace(
        sp.PortPattern.exact(sp.Representation.coad(8), index),
        sp.PortPattern.exact(sp.Representation.coad(dA), index),
    )
    return (generic,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Reduce the color factors
    """)
    return


@app.cell
def _(N, TensorExpression, dA, generic):
    print("color contraction", flush=True)
    colored = (
        TensorExpression(generic)
        .simplify_algebra(contract="dots", gamma=False, color=True)
        .to_expression()
        .replace(dA, N**2 - 1)
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
    ## Contract the physical polarization sums

    Pair incoming and outgoing axial references. Apply kinematics after each selected polarization to limit intermediate growth.
    """)
    return


@app.cell
def _(D, P, TensorExpression, bars, colored, gluon, kin, ports, s, t, u):
    # Physical axial references pair the two incoming and the two outgoing gluons.
    result = colored
    for _i, _j in enumerate((1, 0, 3, 2)):
        projector = gluon.spin_sum(
            P(_i), ports[_i], bars[_i], reference=P(_j), dimension=D
        ) / (2 if _i < 2 else 1)
        projector = kin.apply(projector)
        print("polarization", _i, "start", flush=True)
        expanded = (result * projector).expand()
        tensor = TensorExpression(expanded)
        tensor = tensor.contract().to_dots()
        result = kin.apply(tensor).to_expression().replace(u, -s - t)
        print("polarization", _i, "done", flush=True)
    return (result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the independent D-dimensional formula
    """)
    return


@app.cell
def _(D, N, gs, result, s, t, u):
    expected = (
        (D - 2) ** 2
        * N**2
        * gs**4
        * (t**2 + t * u + u**2) ** 3
        / ((N**2 - 1) * s**2 * t**2 * u**2)
    )
    delta = (result - expected.replace(u, -s - t)).together()
    assert delta == 0
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Compare the averaging conventions

    A fixed two-state initial average and a D-dimensional initial average agree at D=4.
    """)
    return


@app.cell
def _(D, E, N, gs, result, s, t, u):
    fixed_two = result.together()
    dimensional_average = (4 * fixed_two / (D - 2) ** 2).together()
    assert dimensional_average.derivative(D).together() == 0
    standard = E("9/2") * gs**4 * (3 - t * u / s**2 - s * u / t**2 - s * t / u**2)
    assert (
        fixed_two.replace(D, 4).replace(N, 3) - standard.replace(u, -s - t)
    ).together() == 0
    assert (fixed_two - fixed_two.replace(t, -s - t)).together() == 0
    print(
        "Generated four-gluon amplitude: symbolic D, SU(N), both averages and Bose exchange passed",
        flush=True,
    )
    return dimensional_average, fixed_two


@app.cell
def _(generated, mo):
    mo.vstack(
        [
            mo.md("**The four generated diagrams**"),
            mo.hstack(generated.diagrams, justify="space-around"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Two averaging conventions**

    The reference divides each incoming polarization sum by two:
    $$\overline{|\mathcal M|^2}_{2} =
    \frac{(D-2)^2 N_c^2 g_s^4(t^2+tu+u^2)^3}
    {(N_c^2-1)s^2t^2u^2},\qquad s+t+u=0.$$
    Averaging over all $D-2$ physical incoming states instead multiplies this
    result by $4/(D-2)^2$. These conventions agree at $D=4$.

    For $D=4$, $N_c=3$, the result equals
    $\frac92 g_s^4(3-tu/s^2-su/t^2-st/u^2)$.
    The exact comparison also checks exchange of the identical outgoing gluons.
    """)
    return


@app.cell
def _(mo):
    colors = mo.ui.dropdown(
        {"Nc = 2": 2, "Nc = 3": 3, "Nc = 5": 5}, value="Nc = 3", label="Color group"
    )
    dimension = mo.ui.dropdown(
        ["Symbolic D", "4", "6"], value="4", label="Lorentz dimension"
    )
    averaging = mo.ui.dropdown(
        ["Two-state reference average", "D-dimensional state average"],
        value="Two-state reference average",
        label="Initial spin average",
    )
    mo.hstack([colors, dimension, averaging])
    return averaging, colors, dimension


@app.cell
def _(
    D,
    E,
    N,
    averaging,
    colors,
    dimension,
    dimensional_average,
    fixed_two,
    gs,
    mo,
):
    _squared = (
        fixed_two
        if averaging.value == "Two-state reference average"
        else dimensional_average
    )
    shown_squared = (_squared / gs**4).replace(N, colors.value)
    if dimension.value != "Symbolic D":
        shown_squared = shown_squared.replace(D, E(dimension.value))
    shown_squared = shown_squared.factor()
    mo.vstack([mo.md(r"**Squared amplitude divided by $g_s^4$**"), shown_squared])
    return


@app.cell
def _(integrate_rate):
    density, cut_rate, z, cutoff, rate_checks = integrate_rate()
    return cut_rate, cutoff, density, rate_checks, z


@app.cell(hide_code=True)
def _(mo, rate_checks):
    assert len(rate_checks) == 9
    mo.md(r"""
    **Four-dimensional angular-cut event rates**

    The rate uses $D=4$ with native flux and two-body phase space, the full
    azimuth integral and the $1/2!$ factor for identical final gluons:
    $$\frac{d\sigma}{d\cos\theta}=\frac{\overline{|\mathcal M|^2}}{64\pi s}.$$
    This lower panel always uses four dimensions, independently of the amplitude
    display above. We show $s\sigma/\alpha_s^2$, with $g_s^2=4\pi\alpha_s$.
    The cut $|\cos\theta|<c<1$ excludes the massless exchange poles.
    Symbolica integrates the angular density; nine independent Gaussian
    quadratures verify the color and cut choices below.
    """)
    return


@app.cell
def _(mo):
    cut = mo.ui.dropdown(
        {"|cosθ| < 0.2": 0.2, "|cosθ| < 0.5": 0.5, "|cosθ| < 0.8": 0.8},
        value="|cosθ| < 0.8",
        label="Angular acceptance",
    )
    mo.hstack([cut])
    return (cut,)


@app.cell
def _(N, colors, cut, cut_rate, cutoff, density, mo, np, z):
    selected_rate = cut_rate.evaluate({N: colors.value, cutoff: cut.value})
    assert selected_rate.real > 0 and abs(selected_rate.imag) < 1e-10
    angular_rows = [
        {
            "cos(theta)": float(_angle),
            "s/alpha_s² × dσ/dcos(theta)": density.evaluate(
                {N: colors.value, z: float(_angle)}
            ).real,
        }
        for _angle in np.linspace(-cut.value, cut.value, 11)
    ]
    mo.vstack(
        [
            mo.md(f"Accepted event rate: **{selected_rate.real:.8g}** × αs²/s."),
            mo.ui.table(angular_rows, selection=None),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
