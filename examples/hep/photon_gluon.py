import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Photon–gluon currents and crossing",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Photon–gluon currents and crossing

    [Browse notebooks](/) · [Real QCD radiation](/?file=hep/photon_radiation.py) ·
    [Quark–gluon scattering](/?file=hep/quark_gluon_scattering.py)

    Generate both diagrams for each process:
    $\gamma^*g\to q\bar q$, $q\gamma^*\to gq$ and
    $q\bar q\to\gamma^*g$. Keep the quark mass, photon virtuality and
    generic SU($N_c$) color. Model particles and vertices are selected by name.

    These calculations reproduce FeynCalc's
    [photon–gluon fusion](https://feyncalc.github.io/FeynCalcExamples/QCD/Tree/GaGl-QQbar),
    [QCD Compton](https://feyncalc.github.io/FeynCalcExamples/QCD/Tree/QGa-GlQ), and
    [annihilation](https://feyncalc.github.io/FeynCalcExamples/QCD/Tree/QQbar-GaGl)
    examples. Each channel is generated independently before checking crossing.

    The off-shell photon current is contracted with $-g_{\mu\nu}$ without
    a photon spin average, matching the references. Incoming quark/gluon
    spins and colors are averaged. This current contraction is not a physical
    virtual-photon cross section; a lepton tensor is needed for an observable
    involving a virtual photon.
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

    _set_namespace("photon_gluon")
    return E, Kinematics, Model, S, Symbol, TensorExpression, hep, mo, np, sp


@app.cell(hide_code=True)
def _():
    from symbolica.community.hepkit import Symbols

    return (Symbols,)


@app.cell(hide_code=True)
def _(
    E,
    Kinematics,
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
    s,
    sp,
    t,
    u,
    virtuality,
):
    def photon_gluon_channel(names, photon_position, gluon_position, fermion_ports):
        masses = [
            mass**2 if name in ("b", "b~") else virtuality if name == "a" else E("0")
            for name in names
        ]
        kin = Kinematics.mandelstam([P(i) for i in range(4)], masses, [s, t, u])
        allowed = [
            v
            for v in model.vertex_rules
            if sorted(v.particles)
            in [
                sorted(["b", "b~", "a"]),
                sorted(["b", "b~", "g"]),
            ]
        ]
        generated = model.process(
            names[:2], names[2:], vertex_allow=allowed
        ).generate_diagrams(
            max_vertices=2,
            maximum_bridges=None,
            numerator_grouping=None,
            progress=None,
        )
        assert len(generated.diagrams) == 2
        ports = S(
            "i0",
            "i1",
            "i2",
            "i3",
        )
        # Give the conjugate amplitude distinct external labels; scope only its dummies.
        bra_ports = {port: S(f"bra_{i}") for i, port in enumerate(ports)}
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
            )
            for half in diagram.half_edges:
                edge = half.edge.data
                if edge.is_external:
                    numerator = sp.TensorExpression(numerator).rename_indices(
                        {Symbols.half_edge(half.data, 1): ports[edge.external_index]}
                    )
            denominator = kin.apply(
                diagram.denominator_expression(dimension=4, in_lmb=True)
                .to_expression()
                .replace(Symbols.denominator(a, b, c, inv), inv)
            ).expand()
            assert denominator in (s - mass**2, t - mass**2, u - mass**2), denominator
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
        for real in (mass, ee, gs, s, t, u, virtuality):
            adjoint = adjoint.replace(conj(real), real)
        adjoint = (
            sp.TensorExpression(adjoint)
            .wrap_indices(wrap, dummies_only=True)
            .rename_indices(bra_ports)
            .to_expression()
        )
        color_projector = E("1")
        initial_colors = 1
        generic_initial_colors = E("1")
        for position, name in enumerate(names):
            particle = model.particle(name)
            if particle.color == 1:
                continue
            if position < 2:
                initial_colors *= abs(particle.color)
                generic_initial_colors *= dA if name == "g" else Nc
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
            color_projector *= particle.color_sum(
                matches[0][left], matches[0][right], average=position < 2
            )
        generic = (
            (
                operator.to_expression()
                * adjoint
                * color_projector
                * initial_colors
                / generic_initial_colors
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
        # Adjoint reverses the single open fermion chain. The explicit endpoint
        # pairing distinguishes pair creation, annihilation and Compton scattering.
        spins = E("1")
        for position, left_position, right_position, wrap_left in fermion_ports:
            left_port, right_port = ports[left_position], ports[right_position]
            if wrap_left:
                left_port = bra_ports[left_port]
            else:
                right_port = bra_ports[right_port]
            spins *= model.particle(names[position]).spin_sum(
                P(position), left_port, right_port, average=position < 2
            )
        traced = (
            (colored * spins)
            .simplify_algebra(contract="dots", color=False, gamma=True, epsilon=True)
            .expand()
        )
        reduced = kin.apply(traced.contract().to_dots().expand())
        results = {}
        modes = (
            "covariant",
            "photon reference",
            "quark reference",
            "gluon Ward",
            "photon Ward",
        )
        quark_position = next(i for i, name in enumerate(names) if name == "b")
        for mode in modes:
            photon = model.particle("a").spin_sum(
                P(photon_position),
                ports[photon_position],
                bra_ports[ports[photon_position]],
                covariant=True,
            )
            reference = (
                P(photon_position)
                if mode == "photon reference"
                else P(quark_position)
                if mode == "quark reference"
                else None
            )
            gluon = model.particle("g").spin_sum(
                P(gluon_position),
                ports[gluon_position],
                bra_ports[ports[gluon_position]],
                reference=reference,
                covariant=reference is None,
                average=gluon_position < 2,
            )
            if mode == "gluon Ward":
                gluon = P(
                    gluon_position,
                    sp.PortPattern.exact(
                        sp.Representation.mink(4), ports[gluon_position]
                    ),
                ) * P(
                    gluon_position,
                    sp.PortPattern.exact(
                        sp.Representation.mink(4),
                        bra_ports[ports[gluon_position]],
                    ),
                )
            if mode == "photon Ward":
                photon = P(
                    photon_position,
                    sp.PortPattern.exact(
                        sp.Representation.mink(4), ports[photon_position]
                    ),
                ) * P(
                    photon_position,
                    sp.PortPattern.exact(
                        sp.Representation.mink(4),
                        bra_ports[ports[photon_position]],
                    ),
                )
            contracted = (reduced * photon * gluon).contract().to_dots()
            assert contracted.is_scalar
            results[mode] = (
                kin.apply(contracted)
                .to_expression()
                .replace(u, 2 * mass**2 + virtuality - s - t)
                .together()
            )
        for mode in [name for name in modes if name.endswith("Ward")]:
            assert results[mode] == E("0")
        for mode in [name for name in modes if name.endswith("reference")]:
            assert (results[mode] - results["covariant"]).together() == E("0")

        return generated, results, kin

    return (photon_gluon_channel,)


@app.cell(hide_code=True)
def _(mass, virtuality):
    def reference_polynomial(x, y):
        return (
            -(mass**4)
            * (
                2 * virtuality**2
                - 2 * virtuality * (x + y)
                + 3 * x**2
                + 14 * x * y
                + 3 * y**2
            )
            + mass**2
            * (
                2 * virtuality**2 * (x + y)
                - 8 * virtuality * x * y
                + x**3
                + 7 * x**2 * y
                + 7 * x * y**2
                + y**3
            )
            + 6 * mass**8
            - x * y * (2 * virtuality**2 - 2 * virtuality * (x + y) + x**2 + y**2)
        ) / ((x - mass**2) ** 2 * (y - mass**2) ** 2)

    return (reference_polynomial,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Define the couplings and invariants
    """)
    return


@app.cell
def _(Model, S, hep, sp):
    model = Model.standard_model()
    P = hep.Kinematics.external_momentum
    mass = model.particle("b").mass
    ee = -model.particle("e-").electric_charge
    gs = model.parameter("G").symbol
    s, t, u, virtuality = S("s", "t", "u", "q2")
    Nc, dA, cof, coad = (
        S("Nc"),
        S("dA"),
        sp.Representation.cof,
        sp.Representation.coad,
    )

    # Reference polynomial shared by the crossed channels, independently stated
    # in invariant variables. No crossing rule is inserted into the amplitudes.
    return Nc, P, dA, ee, gs, mass, model, s, t, u, virtuality


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate and check the crossed channels

    Each channel uses the same folded amplitude-and-polarization routine. The analytic reference is independent of diagram generation.
    """)
    return


@app.cell
def _(
    E,
    Nc,
    ee,
    gs,
    mass,
    photon_gluon_channel,
    reference_polynomial,
    s,
    t,
    u,
    virtuality,
):
    channels = {}
    for _channel, _names, _photon, _gluon, _spin_ports, _prefactor, _pair in (
        (
            "fusion",
            ["a", "g", "b", "b~"],
            0,
            1,
            [(2, 3, 2, True), (3, 3, 2, False)],
            -2 * ee**2 * gs**2 / 9,
            (t, u),
        ),
        (
            "compton",
            ["b", "a", "g", "b"],
            1,
            2,
            [(0, 0, 3, False), (3, 0, 3, True)],
            2 * ee**2 * gs**2 * (Nc**2 - 1) / (9 * Nc),
            (s, t),
        ),
        (
            "annihilation",
            ["b", "b~", "a", "g"],
            2,
            3,
            [(0, 0, 1, False), (1, 0, 1, True)],
            -(ee**2) * gs**2 * (Nc**2 - 1) / (9 * Nc**2),
            (t, u),
        ),
    ):
        _generated, _results, _kin = photon_gluon_channel(
            _names, _photon, _gluon, _spin_ports
        )
        _target = (
            (_prefactor * reference_polynomial(*_pair))
            .replace(u, 2 * mass**2 + virtuality - s - t)
            .together()
        )
        _delta = (_results["covariant"] - _target).together()
        assert _delta == E("0"), _channel
        _x, _y = _pair
        _massless_target = (
            -_prefactor
            * (2 * virtuality**2 - 2 * virtuality * (_x + _y) + _x**2 + _y**2)
            / (_x * _y)
        )
        assert (
            _results["covariant"].replace(mass, 0)
            - _massless_target.replace(u, virtuality - s - t)
        ).together() == E("0")
        channels[_channel] = (_generated, _results, _kin)
    return (channels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Verify crossing and Bose symmetry
    """)
    return


@app.cell
def _(E, Nc, channels, mass, s, t, virtuality):
    fusion = channels["fusion"][1]["covariant"]
    compton = channels["compton"][1]["covariant"]
    annihilation = channels["annihilation"][1]["covariant"]
    # Crossing changes both the fermion-trace sign and the incoming color average.
    crossed_fusion = fusion.replace(s, 2 * mass**2 + virtuality - s - t)
    assert (compton + (Nc**2 - 1) / Nc * crossed_fusion).together() == E("0")
    assert (annihilation - (Nc**2 - 1) / (2 * Nc**2) * fusion).together() == E("0")
    assert (
        fusion - fusion.replace(t, 2 * mass**2 + virtuality - s - t)
    ).together() == E("0")
    # Check the reference spacelike continuation q^2=-Q^2 explicitly at zero mass.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the spacelike continuation
    """)
    return


@app.cell
def _(E, Nc, S, channels, ee, gs, mass, s, t, u, virtuality):
    Q2 = S("Q2", is_positive=True)
    for _channel, _x, _y, _prefactor in (
        ("fusion", t, u, 2 * ee**2 * gs**2 / 9),
        ("compton", s, t, -16 * ee**2 * gs**2 / 27),
    ):
        _reference = _prefactor * (
            _x / _y + _y / _x + 2 * Q2 * (_x + _y + Q2) / (_x * _y)
        )
        _result = (
            channels[_channel][1]["covariant"]
            .replace(mass, 0)
            .replace(virtuality, -Q2)
            .replace(Nc, 3)
        )
        assert (_result - _reference.replace(u, -Q2 - s - t)).together() == E("0")

    # Only for a real photon do we form ordinary two-body cross sections.
    # Incoming photons have two physical states, so include their missing 1/2;
    # the off-shell current convention used above intentionally has no such average.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Form the physical angular distributions
    """)
    return


@app.cell
def _(E, Nc, P, S, Symbol, channels, ee, gs, mass, np, s, t, virtuality):
    z, rho, beta = S("z", "rho", "beta")
    alpha, alpha_s = S("alpha", "alpha_s")
    angular = {}
    for _channel, (_, _sums, _kin) in channels.items():
        phase = (
            (_kin.two_body_phase_space(P(2), P(3)) / _kin.flux(P(0), P(1)))
            .replace(virtuality, 0)
            .together()
        )
        # Squaring the measure ratio checks its invariant normalization without
        # imposing a branch identity outside the physical region.
        velocity_squared = 1 - 4 * mass**2 / s
        factor_squared = (
            velocity_squared
            if _channel == "fusion"
            else 1 / velocity_squared
            if _channel == "annihilation"
            else E("1")
        )
        assert (
            phase**2 - factor_squared / (64 * Symbol.PI**2 * s) ** 2
        ).together() == E("0")
        _angle_t = (
            mass**2 - s * (1 - beta * z) / 2
            if _channel != "compton"
            else mass**2 - (s**2 - mass**4) / (2 * s) + (s - mass**2) ** 2 * z / (2 * s)
        )
        # Normalize out alpha*alpha_s/s. Final particles are distinct in all channels.
        _density = (
            _sums["covariant"].replace(virtuality, 0).replace(t, _angle_t)
            * phase
            * 2
            * Symbol.PI
            * s
            / (alpha * alpha_s)
        )
        _density = _density.replace(ee**2, 4 * Symbol.PI * alpha).replace(
            gs**2, 4 * Symbol.PI * alpha_s
        )
        if _channel != "annihilation":
            _density /= 2
        _density = _density.replace(mass, (rho * s).sqrt()).together()
        angular[_channel] = _density
        for nc in (2, 3, 5):
            for _r in (0.0, 0.02, 0.1):
                for _cosine in (-0.7, 0.0, 0.7):
                    _values = {
                        s: 100.0,
                        Nc: nc,
                        rho: _r,
                        beta: np.sqrt(1 - 4 * _r),
                        z: _cosine,
                    }
                    exact = complex(_density.evaluate(_values))
                    scaled = complex(_density.evaluate({**_values, s: 400.0}))
                    assert abs(exact.imag) < 1e-12 and exact.real > 0
                    assert abs(exact - scaled) < 1e-10

    # Off-shell currents at explicit center-of-mass invariants, with incoming
    # photons spacelike and the outgoing photon timelike.
    return angular, beta, rho, z


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the soft limits
    """)
    return


@app.cell
def _(
    Nc,
    angular,
    beta,
    channels,
    ee,
    gs,
    mass,
    np,
    rho,
    s,
    t,
    u,
    virtuality,
    z,
):
    for _channel, (_, _sums, _) in channels.items():
        for _r in (0.0, 0.02, 0.1):
            for v in (0.05, 0.2):
                v = v if _channel == "annihilation" else -v
                for _cosine in (-0.7, 0.0, 0.7):
                    kallen = 1 + _r * _r + v * v - 2 * _r - 2 * v - 2 * _r * v
                    _angle_t = (
                        _r - (1 - v) * (1 - np.sqrt(1 - 4 * _r) * _cosine) / 2
                        if _channel != "compton"
                        else _r
                        - (1 + _r - v) * (1 - _r) / 2
                        + (1 - _r) * np.sqrt(kallen) * _cosine / 2
                    )
                    point = {
                        s: 1,
                        t: _angle_t,
                        mass: np.sqrt(_r),
                        virtuality: v,
                        Nc: 3,
                        ee: 1,
                        gs: 1,
                    }
                    _values = [
                        complex(_sums[mode].evaluate(point))
                        for mode in (
                            "covariant",
                            "photon reference",
                            "quark reference",
                        )
                    ]
                    assert max(abs(value.imag) for value in _values) < 1e-10
                    assert max(abs(value - _values[0]) for value in _values) < 1e-8
                    assert all(np.isfinite(value.real) for value in _values)
    calculation = {
        "channels": channels,
        "angular": angular,
        "mass": mass,
        "ee": ee,
        "gs": gs,
        "s": s,
        "t": t,
        "u": u,
        "virtuality": virtuality,
        "Nc": Nc,
        "z": z,
        "rho": rho,
        "beta": beta,
    }
    return (calculation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checks before choosing a numerical point

    - Both diagrams and their interference reproduce the complete massive references.
    - Photon and gluon Ward contractions vanish in every channel.
    - Gluon references along the quark or virtual photon give the same current contraction.
    - Exact crossing includes the fermion sign and changed incoming color averages.
    - The massless spacelike-photon limits agree with the reference formulas.
    - Shared flux and two-body phase space supply the real-photon angular rates.

    Write $F(s,t,u)$ for the fusion current contraction. With the same invariant
    arguments, annihilation gives $(N_c^2-1)F/(2N_c^2)$; crossing $s\leftrightarrow u$
    gives the Compton current with factor $-(N_c^2-1)/N_c$.
    """)
    return


@app.cell
def _(mo):
    channel = mo.ui.dropdown(
        {
            "Photon–gluon fusion": "fusion",
            "QCD Compton": "compton",
            "Quark annihilation": "annihilation",
        },
        value="Photon–gluon fusion",
        label="Process",
    )
    colors = mo.ui.dropdown(
        {f"SU({n})": n for n in (2, 3, 5)}, value="SU(3)", label="Color group"
    )
    mass_ratio = mo.ui.slider(
        0, 0.2, step=0.01, value=0.02, label="Quark mass squared / s"
    )
    virtuality_ratio = mo.ui.slider(
        0, 0.3, step=0.01, value=0.2, label="Photon |q²| / s"
    )
    angle = mo.ui.slider(-0.8, 0.8, step=0.1, value=0.0, label="cos θ")
    mo.vstack(
        [mo.hstack([channel, colors]), mo.hstack([mass_ratio, virtuality_ratio, angle])]
    )
    return angle, channel, colors, mass_ratio, virtuality_ratio


@app.cell
def _(calculation, channel, mo):
    _diagrams = calculation["channels"][channel.value][0].diagrams
    mo.vstack(
        [
            mo.md("## Generated diagrams"),
            mo.hstack(_diagrams),
            mo.accordion(
                {
                    "Exact massive current contraction": calculation["channels"][
                        channel.value
                    ][1]["covariant"]
                }
            ),
        ]
    )
    return


@app.cell
def _(angle, calculation, channel, colors, mass_ratio, np, virtuality_ratio):
    numeric_r = mass_ratio.value
    signed_virtuality = virtuality_ratio.value * (
        1 if channel.value == "annihilation" else -1
    )
    numeric_v = signed_virtuality
    numeric_beta = np.sqrt(1 - 4 * numeric_r)
    numeric_kallen = (
        1
        + numeric_r**2
        + numeric_v**2
        - 2 * numeric_r
        - 2 * numeric_v
        - 2 * numeric_r * numeric_v
    )
    numeric_shape = (
        9
        * calculation["channels"][channel.value][1]["covariant"]
        / (calculation["ee"] ** 2 * calculation["gs"] ** 2)
    )
    numeric_shape = numeric_shape.together()
    numeric_rate = calculation["angular"][channel.value]
    angular_rows = []
    selected_current = None
    selected_density = None
    for _z in sorted(
        set(
            [round(float(v), 1) for v in np.linspace(-0.8, 0.8, 17)]
            + [round(angle.value, 1)]
        )
    ):
        if channel.value == "compton":
            _t = (
                numeric_r
                - (1 + numeric_r - numeric_v) * (1 - numeric_r) / 2
                + (1 - numeric_r) * np.sqrt(numeric_kallen) * _z / 2
            )
        else:
            _t = numeric_r - (1 - numeric_v) * (1 - numeric_beta * _z) / 2
        _point = {
            calculation["s"]: 1,
            calculation["t"]: _t,
            calculation["mass"]: np.sqrt(numeric_r),
            calculation["virtuality"]: numeric_v,
            calculation["Nc"]: colors.value,
        }
        _current = complex(numeric_shape.evaluate(_point))
        _density = complex(
            numeric_rate.evaluate(
                {
                    calculation["s"]: 1,
                    calculation["rho"]: numeric_r,
                    calculation["beta"]: numeric_beta,
                    calculation["Nc"]: colors.value,
                    calculation["z"]: _z,
                }
            )
        )
        assert (
            abs(_current.imag) < 1e-8
            and abs(_density.imag) < 1e-10
            and _density.real > 0
        )
        angular_rows.append(
            {
                "cos θ": _z,
                "Off-shell current / (e² gs² Qq²)": _current.real,
                "Real-photon s/(α αs) dσ/dcosθ": _density.real,
            }
        )
        if abs(_z - angle.value) < 1e-8:
            selected_current, selected_density = _current.real, _density.real
    assert selected_current is not None and selected_density is not None
    # Gaussian quadrature convergence checks the additional angular-cut rate.
    return (
        angular_rows,
        numeric_beta,
        numeric_r,
        numeric_rate,
        selected_current,
        selected_density,
        signed_virtuality,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the integrated angular distribution
    """)
    return


@app.cell
def _(calculation, colors, np, numeric_beta, numeric_r, numeric_rate):
    _integrals = []
    for _order in (48, 96):
        _nodes, _weights = np.polynomial.legendre.leggauss(_order)
        _values = [
            complex(
                numeric_rate.evaluate(
                    {
                        calculation["s"]: 1,
                        calculation["rho"]: numeric_r,
                        calculation["beta"]: numeric_beta,
                        calculation["Nc"]: colors.value,
                        calculation["z"]: 0.8 * _node,
                    }
                )
            ).real
            for _node in _nodes
        ]
        _integrals.append(0.8 * float(np.dot(_weights, _values)))
    assert abs(_integrals[0] - _integrals[1]) < 1e-9
    cut_rate = _integrals[1]
    return (cut_rate,)


@app.cell(hide_code=True)
def _(
    angular_rows,
    cut_rate,
    mo,
    selected_current,
    selected_density,
    signed_virtuality,
):
    mo.vstack(
        [
            mo.md("## Off-shell current and real-photon rate"),
            mo.md(
                f"**Photon q²/s = {signed_virtuality:.2f}.** Incoming photons are spacelike; outgoing photons are timelike."
            ),
            mo.md(f"**Selected current / (e² gs² Qq²):** {selected_current:.8g}"),
            mo.md(f"**Real-photon s/(α αs) dσ/dcosθ:** {selected_density:.8g}"),
            mo.md(
                f"**Real-photon angular-cut rate s σ/(α αs), |cosθ| < 0.8:** {cut_rate:.8g}"
            ),
            mo.md(r"""
        The angle is between the first incoming and first outgoing particle.
        The off-shell column uses the displayed virtuality. The rate column sets
        $q^2=0$ and averages an incoming photon's two physical polarizations.
        All final particles are distinct. To obtain a rate, multiply by
        $\alpha\alpha_s/s$; the model's bottom-quark charge is $Q_q=-1/3$.
        The angular cut keeps the massless collinear endpoints outside the integral.
        """),
            mo.ui.table(angular_rows, selection=None),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
