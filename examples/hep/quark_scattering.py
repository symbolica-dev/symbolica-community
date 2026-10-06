import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Massive quark scattering and color interference",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Massive quark scattering and color interference
    [Browse all notebooks](/) · [Quark annihilation](/?file=hep/qcd_annihilation.py) ·
    [Quarks to gluons](/?file=hep/qcd_gluons.py) ·
    [Identical leptons](/?file=hep/identical_leptons.py)

    Generate four elastic QCD processes with massive quarks, retaining every
    diagram and interference term. The references are
    [distinct quarks](https://feyncalc.github.io/FeynCalcExamples/QCD/Tree/QiQj-QiQj),
    [distinct quark–antiquark](https://feyncalc.github.io/FeynCalcExamples/QCD/Tree/QiQjbar-QiQjbar),
    [identical quarks](https://feyncalc.github.io/FeynCalcExamples/QCD/Tree/QiQi-QiQi), and
    [same-flavor quark–antiquark](https://feyncalc.github.io/FeynCalcExamples/QCD/Tree/QiQibar-QiQibar).

    The calculation uses the model's bottom and top fields as two generic massive
    flavors and selects only their quark–gluon vertices. Shared particle spin
    and color sums average the incoming states and sum the outgoing states.
    Spenso handles the color algebra for symbolic $N_c$; Idenso handles the
    Dirac traces. Color factors and relative diagram signs follow from the generated amplitudes.

    The full massive SU(N) results, their interference terms and massless limits
    are checked exactly. The event rates add native flux and two-body phase
    space. An angular cut excludes the poles from massless gluon exchange.
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
    from symbolica import E, Replacement, S, Symbol
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hepkit import Kinematics, Model, RenderSettings
    from symbolica.community.tensor import TensorExpression

    _set_namespace("qscatter")
    return (
        E,
        Kinematics,
        Model,
        RenderSettings,
        Replacement,
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
def _(E, M, Nc, Replacement, S, Symbol, gs, m, np, results, s, t):
    def integrate_rates():
        observables = {name: dict(result) for name, result in results.items()}

        # Carry the generated massive result through the native elastic flux and phase
        # space. Full angular integration needs a cut because of massless gluon exchange.
        z, cutoff, alpha_s, C = S("z", "cutoff", "alpha_s", "C")
        template = sum((C(i) * z**i for i in range(7)), E("0")) / (1 - z**2) ** 2
        primitive_template = template.integrate(z)
        assert (primitive_template.derivative(z) - template).together() == 0
        rate_checks = []
        nodes, weights = np.polynomial.legendre.leggauss(128)
        for name, result in observables.items():
            same = result["same_flavor"]
            second_mass = m if same else M
            kallen = (s - m**2 - second_mass**2) ** 2 - 4 * m**2 * second_mass**2
            angle_t = -kallen * (1 - z) / (2 * s)
            symmetry = E("1/2") if name == "qq" else E("1")
            prefactor = symmetry / (32 * Symbol.PI * s)
            density = (
                result["squared"]
                .replace(t, angle_t)
                .replace(gs**4, (4 * Symbol.PI * alpha_s) ** 2)
                * prefactor
            ).together()
            coefficient_polynomial = (density * (1 - z**2) ** 2).together().expand()
            coefficients = {
                monomial.to_polynomial(vars=[z]).degree(z): coefficient
                for monomial, coefficient in coefficient_polynomial.coefficient_list(z)
            }
            assert set(coefficients) <= set(range(7))
            primitive = primitive_template.replace_multiple(
                [Replacement(C(i), coefficients.get(i, E("0"))) for i in range(7)]
            )
            assert (primitive.derivative(z) - density).together() == 0
            # On -1<z<1, replacing log(z-1) by log(1-z) changes only an irrelevant
            # additive imaginary constant. This gives a manifestly real primitive.
            primitive = primitive.replace((z - 1).log(), (1 - z).log())
            cut_rate = primitive.replace(z, cutoff) - primitive.replace(z, -cutoff)
            assert cut_rate.replace(cutoff, 0).together() == 0
            if name == "qq":
                # One forward quark per event equals half the full labeled phase space.
                residual = (
                    primitive.replace(z, cutoff)
                    - primitive.replace(z, 0)
                    - cut_rate / 2
                ).together()
                # The real-interval identity can retain atanh(-cutoff) symbolically.
                # Equal derivatives and the value at zero establish the exact equality.
                assert residual.derivative(cutoff).together() == 0
                assert residual.replace(cutoff, 0).together() == 0
            result.update(
                density=density, cut_rate=cut_rate, angle_t=angle_t, symmetry=symmetry
            )
            for nv in (2, 3, 5):
                for fraction, other_fraction in [(0.0, 0.0), (0.1, 0.15), (0.2, 0.25)]:
                    for sv in (4.0, 25.0):
                        for cv in (0.2, 0.5, 0.8):
                            values = {
                                Nc: nv,
                                m: fraction * np.sqrt(sv),
                                M: other_fraction * np.sqrt(sv),
                                s: sv,
                                alpha_s: 0.118,
                            }
                            integrated = complex(
                                cut_rate.evaluate({**values, cutoff: cv})
                            )
                            quadrature = cv * sum(
                                weight
                                * complex(density.evaluate({**values, z: cv * node}))
                                for node, weight in zip(nodes, weights, strict=True)
                            )
                            assert abs(integrated.imag) < 1e-11
                            assert integrated.real > 0
                            error = abs(integrated - quadrature)
                            assert error < 2e-11 * max(1, abs(integrated)), (
                                name,
                                nv,
                                fraction,
                                sv,
                                cv,
                                error,
                            )
                            rate_checks.append((name, nv, fraction, sv, cv, error))
            print(
                name, "massive angular-cut rates and event counting passed", flush=True
            )
        print(
            len(rate_checks), "independent phase-space quadratures passed", flush=True
        )

        return observables, rate_checks, z, cutoff, alpha_s

    return (integrate_rates,)


@app.cell(hide_code=True)
def _(
    E,
    M,
    Nc,
    P,
    Symbol,
    Symbols,
    TensorExpression,
    a,
    b,
    bra_ports,
    c,
    conj,
    dA,
    gs,
    index,
    inverse,
    m,
    model,
    ports,
    rep,
    s,
    sp,
    t,
    u,
    wave,
    wrapped,
):
    def evaluate_quark_channel(name, particle_names, generation, same, kin, masses):
        """Reduce one channel and certify its massive, interference and massless references."""
        operators, denominators = [], []
        for diagram in generation.diagrams:
            numerator = model.expand_couplings(
                diagram.numerator_expression(in_lmb=True).to_expression()
            )
            for edge in diagram.external_edges:
                match = dict(
                    next(
                        diagram.projector_expression().match(
                            wave(edge.id, rep(4, index)), max_level=0
                        )
                    )
                )
                numerator = sp.TensorExpression(numerator).rename_indices(
                    {match[index]: ports[edge.external_index]}
                )
            denominator = kin.apply(
                diagram.denominator_expression(dimension=4, in_lmb=True)
                .to_expression()
                .replace(Symbols.denominator(a, b, c, inverse), inverse)
            ).expand()
            denominators.append(denominator)
            operators.append(
                numerator
                * diagram.overall_factor_expression(evaluate=True)
                * diagram.numerator_prefactor_expression()
                / denominator
            )
        assert set(denominators) == (
            {s, t} if name == "qaq" else {t, u} if name == "qq" else {t}
        )
        amplitude = sum(operators, E("0"))
        operator = amplitude.expand()
        assert len(operator.structure.slots) == 8
        adjoints = []
        for term in operators:
            adjoint = (
                sp.TensorExpression(term)
                .dirac_adjoint(preserve_indices=True)
                .to_expression()
            )
            for real in (s, t, u, m, M, gs):
                adjoint = adjoint.replace(conj(real), real)
            adjoints.append(
                sp.TensorExpression(adjoint)
                .wrap_indices(wrapped, dummies_only=True)
                .rename_indices(bra_ports)
                .to_expression()
            )
        colors = E("1")
        spins = E("1")
        initial_colors = 1
        for position, particle_name in enumerate(particle_names):
            particle = model.particle(particle_name)
            if position < 2:
                initial_colors *= abs(particle.color)
            # Signed particle representations choose quark/antiquark duals.
            # Incoming color ports close in the reverse order from outgoing ports.
            ci, cj = (
                (
                    bra_ports[ports[position]],
                    ports[position],
                )
                if position < 2
                else (
                    ports[position],
                    bra_ports[ports[position]],
                )
            )
            colors *= particle.color_sum(ci, cj, average=position < 2)
            column = (position < 2) == (not particle.is_antiparticle)
            i, j = (
                (
                    ports[position],
                    bra_ports[ports[position]],
                )
                if column
                else (
                    bra_ports[ports[position]],
                    ports[position],
                )
            )
            spins *= particle.spin_sum(P(position), i, j, average=position < 2)
        evaluated = []
        for raw in (
            amplitude.to_expression() * sum(adjoints, E("0")),
            sum(
                (
                    x.to_expression() * y
                    for x, y in zip(operators, adjoints, strict=True)
                ),
                E("0"),
            ),
        ):
            generic = (
                (raw * colors * initial_colors / Nc**2)
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
            colored = (
                TensorExpression(colored)
                .simplify_algebra(
                    contract="dots",
                    gamma=False,
                    color=True,
                    color_substitute_cof_dimension_invariants=True,
                )
                .to_expression()
            )
            scalar = (
                TensorExpression(colored * spins)
                .simplify_algebra(
                    contract="dots", color=False, gamma=True, epsilon=True
                )
                .expand()
                .to_expression()
            )
            assert TensorExpression(scalar).is_scalar
            value = kin.apply(scalar).replace(u, sum(masses) - s - t).together()
            evaluated.append(value)
        squared, diagonal = evaluated
        if same:
            x, y, z = (t, u, s) if name == "qq" else (s, t, u)
            expected = (
                (Nc**2 - 1)
                * gs**4
                * (
                    -4 * m**2 * (Nc * (x**3 + y**3) - 2 * s * t * u)
                    + 4 * m**4 * (Nc * (x**2 + y**2) - 3 * x * y)
                    + Nc * (x**4 + x**3 * y + x**2 * y**2 + x * y**3 + y**4)
                    - z**2 * x * y
                )
                / (Nc**3 * x**2 * y**2)
            )
            interference = (
                -(Nc**2 - 1)
                * gs**4
                * (z**2 - 8 * m**2 * z + 12 * m**4)
                / (Nc**3 * x * y)
            )
            massless_expected = (
                (Nc**2 - 1)
                * gs**4
                / (2 * Nc**2)
                * (
                    (s**2 + u**2) / t**2
                    + ((s**2 + t**2) / u**2 if name == "qq" else (t**2 + u**2) / s**2)
                )
            )
            massless_expected -= (Nc**2 - 1) * gs**4 * z**2 / (Nc**3 * x * y)
        else:
            expected = (
                (Nc**2 - 1)
                * gs**4
                * (
                    -4 * M**2 * (u - m**2)
                    + 2 * M**4
                    + 2 * m**4
                    - 4 * u * m**2
                    + t**2
                    + 2 * t * u
                    + 2 * u**2
                )
                / (2 * Nc**2 * t**2)
            )
            interference = E("0")
            massless_expected = (Nc**2 - 1) * gs**4 * (s**2 + u**2) / (2 * Nc**2 * t**2)
        residual = (squared - expected.replace(u, sum(masses) - s - t)).together()
        assert residual == 0
        assert (
            squared - diagonal - interference.replace(u, sum(masses) - s - t)
        ).together() == 0
        massless = squared.replace(m, E("0")).replace(M, E("0"))
        assert (massless - massless_expected.replace(u, -s - t)).together() == 0
        if name == "qq":
            assert (squared - squared.replace(t, 4 * m**2 - s - t)).together() == 0
        phase = kin.two_body_phase_space(P(2), P(3)) / kin.flux(P(0), P(1))
        assert (phase - 1 / (64 * Symbol.PI**2 * s)).together() == 0
        channel = {
            "squared": squared,
            "diagonal": diagonal,
            "interference": squared - diagonal,
            "diagrams": generation.diagrams,
            "kinematics": kin,
            "same_flavor": same,
            "massless": massless,
        }
        print(
            name,
            "massive SU(N), interference, massless and native phase-space checks passed",
            flush=True,
        )
        return channel

    return (evaluate_quark_channel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Define the quark masses and couplings
    """)
    return


@app.cell
def _(Model, S, hep, sp):
    model = Model.standard_model()
    P = hep.Kinematics.external_momentum
    s, t, u = S("s", "t", "u")
    m = model.particle("b").mass
    M = model.particle("t").mass
    gs = model.parameter("G").symbol
    Nc, dA, cof, coad = (
        S("Nc"),
        S("dA"),
        sp.Representation.cof,
        sp.Representation.coad,
    )
    return M, Nc, P, dA, gs, m, model, s, t, u


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Select the QCD interaction and tensor ports
    """)
    return


@app.cell
def _(S, model, sp):
    a, b, c, inverse, wave, rep, index = S(
        "a_", "b_", "c_", "inverse_", "wave_", "rep_", "index_"
    )
    ports = S("i0", "i1", "i2", "i3")
    # Give the conjugate amplitude distinct external labels; scope only its dummies.
    bra_ports = {port: S(f"bra_{i}") for i, port in enumerate(ports)}
    conj, wrapped = (sp.BroadcastFunction.conj().to_expression(), S("adjoint"))
    flavors = [model.particle(_name) for _name in ("b", "t")]
    gluon = model.particle("g")
    vertices = [
        v
        for v in model.vertex_rules
        if sorted(v.particles)
        in [sorted([q.name, q.antiname, gluon.name]) for q in flavors]
    ]
    assert len(vertices) == 2
    return (
        a,
        b,
        bra_ports,
        c,
        conj,
        index,
        inverse,
        ports,
        rep,
        vertices,
        wave,
        wrapped,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate the four flavor channels

    Each channel passes the same folded reduction routine. It checks the massive result, the interference, the massless limit and the native phase-space normalization.
    """)
    return


@app.cell
def _(Kinematics, M, P, evaluate_quark_channel, m, model, s, t, u, vertices):
    results = {}
    for _name, _particle_names in [
        ("qq_prime", ["b", "t", "b", "t"]),
        ("qaq_prime", ["b", "t~", "b", "t~"]),
        ("qq", ["b", "b", "b", "b"]),
        ("qaq", ["b", "b~", "b", "b~"]),
    ]:
        _same = _name in ("qq", "qaq")
        _masses = [m**2, (m if _same else M) ** 2] * 2
        _kin = Kinematics.mandelstam([P(i) for i in range(4)], _masses, [s, t, u])
        _generation = model.process(
            _particle_names[:2], _particle_names[2:], vertex_allow=vertices
        ).generate_diagrams(
            max_vertices=2,
            maximum_bridges=None,
            numerator_grouping=None,
            progress=None,
        )
        assert len(_generation.diagrams) == (2 if _same else 1)
        results[_name] = evaluate_quark_channel(
            _name, _particle_names, _generation, _same, _kin, _masses
        )
    return (results,)


@app.cell
def _(integrate_rates):
    observables, rate_checks, z, cutoff, alpha_s = integrate_rates()
    return alpha_s, cutoff, observables, rate_checks, z


@app.cell(hide_code=True)
def _(mo, rate_checks):
    mo.md(r"""
    **Massive elastic phase space**

    With $s=(p_1+p_2)^2$, $t=(p_1-k_1)^2$, and $z=\cos\theta$,
    $$t=-\frac{\lambda(s,m_1^2,m_2^2)}{2s}(1-z),\qquad
      \lambda=(s-m_1^2-m_2^2)^2-4m_1^2m_2^2.$$
    Native phase space divided by flux gives $1/(64\pi^2s)$ per solid angle.
    Thus
    $$\frac{d\sigma_{\rm event}}{dz}
      =\frac{S_f}{32\pi s}\overline{|\mathcal M|^2},\qquad
      S_f=\begin{cases}1/2! & q_iq_i\to q_iq_i,\\1&\text{otherwise}.\end{cases}$$
    For identical quarks, integrating one forward quark per event gives the
    same rate as this symmetry factor over the full symmetric angular cut.
    Symbolica verifies that identity and differentiates every antiderivative
    back to the generated density. **216 independent quadratures pass.**

    The controls show $s\sigma/\alpha_s^2$ and
    $(s/\alpha_s^2)d\sigma/dz$, so the displayed numbers are dimensionless.
    Masses are fractions of $\sqrt{s}$; for identical flavors both masses are equal.
    """)
    assert len(rate_checks) == 216
    return


@app.cell
def _(mo):
    channel = mo.ui.dropdown(
        {
            "Distinct quarks": "qq_prime",
            "Distinct quark–antiquark": "qaq_prime",
            "Identical quarks": "qq",
            "Same-flavor quark–antiquark": "qaq",
        },
        value="Identical quarks",
        label="Process",
    )
    colors = mo.ui.dropdown(
        {"Nc = 2": 2, "Nc = 3": 3, "Nc = 5": 5}, value="Nc = 3", label="Colors"
    )
    masses = mo.ui.dropdown(
        {"Massless": (0.0, 0.0), "Light": (0.1, 0.15), "Heavy": (0.2, 0.25)},
        value="Light",
        label="Mass fractions",
    )
    cut = mo.ui.dropdown(
        {"|cosθ| < 0.2": 0.2, "|cosθ| < 0.5": 0.5, "|cosθ| < 0.8": 0.8},
        value="|cosθ| < 0.8",
        label="Angular cut",
    )
    mo.hstack([channel, colors, masses, cut], justify="start")
    return channel, colors, cut, masses


@app.cell
def _(RenderSettings, channel, mo, results):
    mo.hstack(
        [
            diagram.render(config=RenderSettings())
            for diagram in results[channel.value]["diagrams"]
        ]
    )
    return


@app.cell
def _(channel, mo, results):
    _result = results[channel.value]
    mo.accordion(
        {
            "Generated symbolic amplitudes and interference": mo.vstack(
                [
                    mo.md(
                        "**Spin- and color-averaged squared amplitude, with labeled final momenta:**"
                    ),
                    _result["squared"].factor(),
                    mo.md("**Interference:**"),
                    _result["interference"].factor(),
                    mo.md("**Massless limit:**"),
                    _result["massless"].factor(),
                ]
            )
        }
    )
    return


@app.cell
def _(
    M,
    Nc,
    Symbol,
    alpha_s,
    channel,
    colors,
    cut,
    cutoff,
    gs,
    m,
    masses,
    np,
    observables,
    s,
    t,
    z,
):
    selected_result = observables[channel.value]
    m1, m2 = masses.value
    if selected_result["same_flavor"]:
        m2 = m1
    _values = {Nc: colors.value, m: m1, M: m2, s: 1.0, alpha_s: 1.0}
    selected_rate = complex(
        selected_result["cut_rate"].evaluate({**_values, cutoff: cut.value})
    )
    assert abs(selected_rate.imag) < 1e-10 and selected_rate.real > 0
    _nodes, _weights = np.polynomial.legendre.leggauss(128)
    _quad = cut.value * sum(
        weight
        * complex(selected_result["density"].evaluate({**_values, z: cut.value * node}))
        for node, weight in zip(_nodes, _weights, strict=True)
    )
    selected_error = abs(selected_rate - _quad)
    assert selected_error < 2e-11 * max(1, abs(selected_rate))
    _diagonal = (
        selected_result["diagonal"]
        .replace(t, selected_result["angle_t"])
        .replace(gs**4, (4 * Symbol.PI * alpha_s) ** 2)
        * selected_result["symmetry"]
        / (32 * Symbol.PI * s)
    ).together()
    angular_rows = []
    for _z in np.linspace(-cut.value, cut.value, 11):
        _full = complex(selected_result["density"].evaluate({**_values, z: _z}))
        _diag = complex(_diagonal.evaluate({**_values, z: _z}))
        assert abs(_full.imag) < 1e-10 and _full.real > 0
        angular_rows.append(
            {
                "cosθ": float(_z),
                "Full density": _full.real,
                "Diagonal terms": _diag.real,
                "Interference": (_full - _diag).real,
            }
        )
    return angular_rows, m1, m2, selected_error, selected_rate, selected_result


@app.cell(hide_code=True)
def _(
    angular_rows,
    m1,
    m2,
    mo,
    selected_error,
    selected_rate,
    selected_result,
):
    mo.vstack(
        [
            mo.md(f"**Masses:** $m_1/\\sqrt{{s}}={m1:g}$, $m_2/\\sqrt{{s}}={m2:g}$."),
            mo.ui.table(
                [
                    {
                        "s σ / αs²": selected_rate.real,
                        "Quadrature error": selected_error,
                        "Final-state factor": float(
                            complex(selected_result["symmetry"].evaluate({})).real
                        ),
                    }
                ],
                selection=None,
            ),
            mo.md(
                "**Angular distribution:** all columns use the same event-counting convention."
            ),
            mo.ui.table(angular_rows, selection=None),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
