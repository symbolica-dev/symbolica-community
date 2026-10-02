import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Muon decay and three-body phase space",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Muon decay and three-body phase space
    [Browse notebooks](/) · [W and top decays](/?file=hep/weak_decays.py)

    Generate $\mu^-\to e^-\bar\nu_e\nu_\mu$ with the embedded Standard Model charged-current vertices. The existing W propagator supplies its complete unitary-gauge numerator, including the longitudinal term. Shared particle spin sums average the muon spin and sum all final spins.

    First compare the full squared amplitude with the [FeynCalc reference](https://feyncalc.github.io/FeynCalcExamples/EW/Tree/Mu-ElAnelNmu), then derive the Fermi limit. The shared `Kinematics.three_body_phase_space` supplies the Dalitz measure, independently checked by a recursive two-body derivation. Symbolica integrates the resulting spectrum and finite-mass correction.

    Write $M=m_\mu$, $m=m_e$, $s=(p_e+p_{\bar\nu_e})^2$ and $t=(p_e+p_{\nu_\mu})^2$. Both neutrinos are massless. The coupling convention is $e^4=32G_F^2m_W^4\sin^4\theta_W$.
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
    from symbolica.community.hepkit import Kinematics, Model
    from symbolica.community.tensor import TensorExpression

    _set_namespace("muon_decay")
    return (
        E,
        Kinematics,
        Model,
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
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Use the unitary-gauge W propagator
    """)
    return


@app.cell
def _(Model):
    import json

    _specification = json.loads(Model.standard_model().to_json())
    for _propagator in _specification["propagators"]:
        if _propagator["particle"] in ("W-", "W+"):
            _propagator["numerator"] = (
                "-1𝑖*(UFO::Metric(UFO::idx(1,1),UFO::idx(1,2))"
                "-UFO::P(UFO::idx(1,1))*UFO::P(UFO::idx(1,2))/UFO::MW^2)"
            )
    model = Model.from_json(json.dumps(_specification))
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate the weak decay
    """)
    return


@app.cell
def _(model):
    muon, electron = (model.particle("mu-"), model.particle("e-"))
    vertices = [
        _v
        for _v in model.vertex_rules
        if any(_p in ("W-", "W+") for _p in _v.particles)
        and any(_p in ("e-", "e+", "mu-", "mu+") for _p in _v.particles)
    ]
    assert len(vertices) == 4
    generated = model.process(
        ["mu-"], ["e-", "ve~", "vm"], vertex_allow=vertices
    ).generate_diagrams(
        max_vertices=2, maximum_bridges=None, numerator_grouping=None, progress=None
    )
    assert len(generated.diagrams) == 1
    diagram = generated.diagrams[0]
    assert len(diagram.internal_edges) == 1
    assert diagram.internal_edges[0].particle_name in ("W-", "W+")
    return diagram, electron, muon


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Define the masses and external ports
    """)
    return


@app.cell
def _(E, S, Symbol, hep, model, sp):
    P = hep.Kinematics.external_momentum
    charge = -model.particle("e-").electric_charge
    sw = model.parameter("sw").symbol
    mw = model.particle("W+").mass
    mm = model.particle("mu-").mass
    me = model.particle("e-").mass
    GF, s, t, z = S("GF", "s", "t", "inverse_W2")
    M = S("M", is_positive=True)
    mass = S("m", is_positive=True)
    conjugate, wrapped = (sp.BroadcastFunction.conj().to_expression(), S("adjoint"))
    ports = S("mu", "e", "antinue", "numu")
    # Give the conjugate amplitude distinct external labels; scope only its dummies.
    bra_ports = {port: S(f"bra_{i}") for i, port in enumerate(ports)}
    wave, rep, index = S("wave_", "rep_", "index_")
    zero, one, pi = (E("0"), E("1"), Symbol.PI)
    return (
        GF,
        M,
        P,
        bra_ports,
        charge,
        conjugate,
        index,
        mass,
        me,
        mm,
        mw,
        one,
        pi,
        ports,
        rep,
        s,
        sw,
        t,
        wave,
        wrapped,
        z,
        zero,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Align the generated numerator
    """)
    return


@app.cell
def _(diagram, index, model, ports, rep, sp, wave):
    numerator = model.expand_couplings(
        diagram.numerator_expression(in_lmb=True).to_expression()
    )
    for _edge in diagram.external_edges:
        _matches = list(
            diagram.projector_expression().match(
                wave(_edge.id, rep(4, index)), max_level=0
            )
        )
        assert len(_matches) == 1
        numerator = sp.TensorExpression(numerator).rename_indices(
            {dict(_matches[0])[index]: ports[_edge.external_index]}
        )
    operator = (
        numerator
        * diagram.overall_factor_expression(evaluate=True)
        * diagram.numerator_prefactor_expression()
    )
    return (operator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Construct the Dirac adjoint

    Its external labels are distinct and its internal dummy indices have their own scope.
    """)
    return


@app.cell
def _(
    P,
    S,
    bra_ports,
    charge,
    conjugate,
    index,
    me,
    mm,
    mw,
    operator,
    sp,
    sw,
    wrapped,
):
    adjoint = (
        operator.dirac_adjoint()
        .expand()
        .simplify_algebra(
            contract="dots", gamma=True, gamma0=True, gamma_evaluate_traces=False
        )
        .expand()
        .to_expression()
    )
    _momentum_index = S("momentum_index_")
    adjoint = adjoint.replace(
        conjugate(P(_momentum_index, index)), P(_momentum_index, index)
    )
    for _real in (charge, sw, mw, mm, me):
        adjoint = adjoint.replace(conjugate(_real), _real)
    adjoint = (
        sp.TensorExpression(adjoint)
        .wrap_indices(wrapped, dummies_only=True)
        .rename_indices(bra_ports)
        .to_expression()
    )
    return (adjoint,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Sum the external spins
    """)
    return


@app.cell
def _(
    P,
    TensorExpression,
    adjoint,
    bra_ports,
    electron,
    model,
    muon,
    operator,
    ports,
):
    projector = (
        muon.spin_sum(P(0), ports[0], bra_ports[ports[3]], average=True)
        * electron.spin_sum(P(1), bra_ports[ports[2]], ports[1])
        * model.particle("ve~").spin_sum(P(2), ports[2], bra_ports[ports[1]])
        * model.particle("vm").spin_sum(P(3), bra_ports[ports[0]], ports[3])
    )
    contracted = (
        (operator * adjoint * projector)
        .expand()
        .simplify_algebra(contract="dots", gamma=True, epsilon=True)
        .expand()
        .to_expression()
    )
    assert TensorExpression(contracted).is_scalar
    return (contracted,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Apply the Dalitz kinematics
    """)
    return


@app.cell
def _(Kinematics, P, S, Symbols, diagram, me, mm, mw, s, t, zero):
    kin = Kinematics()
    for _i, _mass_squared in enumerate((mm**2, me**2, zero, zero)):
        kin = kin.with_scalar_product(P(_i), P(_i), _mass_squared)
    for _i, _j, _value in [
        (1, 2, (s - me**2) / 2),
        (1, 3, (t - me**2) / 2),
        (2, 3, (mm**2 + me**2 - s - t) / 2),
        (0, 1, (s + t) / 2),
        (0, 2, (mm**2 - t) / 2),
        (0, 3, (mm**2 - s) / 2),
    ]:
        kin = kin.with_scalar_product(P(_i), P(_j), _value)
    _den = Symbols.denominator
    _edge_, _momentum_, _mass_, _quad_ = S(
        "edge_",
        "momentum_",
        "mass_",
        "quad_",
    )
    denominator = kin.apply(
        diagram.denominator_expression(dimension=4, in_lmb=True)
        .to_expression()
        .replace(_den(_edge_, _momentum_, _mass_, _quad_), _quad_)
    )
    assert (denominator - s + mw**2).expand() == zero
    return denominator, kin


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Restore the Fermi-constant convention
    """)
    return


@app.cell
def _(
    GF,
    M,
    Replacement,
    charge,
    contracted,
    denominator,
    kin,
    mass,
    me,
    mm,
    mw,
    sw,
):
    squared = (kin.apply(contracted) / denominator**2).together()
    squared = (
        squared.replace(charge**4, 32 * GF**2 * mw**4 * sw**4)
        .replace_multiple([Replacement(mm, M), Replacement(me, mass)])
        .together()
    )
    return (squared,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the full finite-W result
    """)
    return


@app.cell
def _(GF, M, mass, mw, s, squared, t, zero):
    _pk, _pq1, _pq2 = ((s + t) / 2, (M**2 - t) / 2, (M**2 - s) / 2)
    _kq1, _kq2, _q1q2 = (
        (s - mass**2) / 2,
        (t - mass**2) / 2,
        (M**2 + mass**2 - s - t) / 2,
    )
    _reference_full = (
        16
        * GF**2
        / (s - mw**2) ** 2
        * (
            -2 * mass**2 * _pq2 * _kq1**2
            - 2 * mass**2 * mw**2 * _kq2 * _pq1
            + 2 * mass**2 * mw**2 * _kq1 * _pq2
            - 2 * mass**2 * mw**2 * _pk * _q1q2
            - mass**4 * _kq1 * _pq2
            + 2 * mass**2 * _pk * _kq1 * _kq2
            + 2 * mass**2 * _kq1 * _kq2 * _pq1
            + 2 * mass**2 * _pk * _kq1 * _q1q2
            + 2 * mass**2 * _kq1 * _pq1 * _q1q2
            - 4 * mass**2 * mw**2 * _pq1 * _q1q2
            + 4 * mw**4 * _kq2 * _pq1
        )
    )
    assert (squared - _reference_full).together() == zero
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Take the low-energy limit
    """)
    return


@app.cell
def _(GF, M, mass, mw, s, squared, t, z, zero):
    fermi = squared.replace(mw, 1 / z.sqrt()).series(z, 0, 0).to_expression().expand()
    assert (fermi - 16 * GF**2 * (t - mass**2) * (M**2 - t)).together() == zero
    finite_W = (
        squared.replace(mass, zero)
        .replace(mw, 1 / z.sqrt())
        .series(z, 0, 1)
        .to_expression()
        .expand()
    )
    assert (finite_W - fermi.replace(mass, zero) * (1 + 2 * s * z)).together() == zero
    return fermi, finite_W


@app.cell(hide_code=True)
def _(diagram, fermi, mo, squared):
    mo.vstack(
        [
            diagram,
            mo.accordion(
                {"Full unitary-gauge squared amplitude": mo.vstack([squared])}
            ),
            mo.md("**Derived low-energy squared amplitude**"),
            fermi,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Obtain the three-body phase-space measure
    """)
    return


@app.cell
def _(M, P, kin, mm):
    dalitz_measure = (
        kin.three_body_phase_space(P(1), P(2), P(3)).replace(mm, M).together()
    )
    return (dalitz_measure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Derive the Dalitz measure
    Factor the decay into $P\to Q+\bar\nu_e$ and $Q\to e+\nu_\mu$, with $Q^2=t$. Integrate the global orientation and the inner azimuth, and include $dt/(2\pi)$. The pair-rest-frame energies determine $s(\cos\theta)$ and its Jacobian.

    On the physical domain $m^2<t<M^2$, the two positive square roots are $M^2-t$ and $t-m^2$. The derived boundaries are $m^2M^2/t\leq s\leq M^2+m^2-t$. All final particles are distinct.
    """)
    return


@app.cell
def _(Kinematics, M, P, S, dalitz_measure, fermi, mass, one, pi, t, zero):
    _Q = S("Q")
    _outer = (
        Kinematics()
        .with_scalar_product(_Q, _Q, t)
        .with_scalar_product(P(2), P(2), zero)
        .with_scalar_product(_Q, P(2), (M**2 - t) / 2)
    )
    _inner = (
        Kinematics()
        .with_scalar_product(P(1), P(1), mass**2)
        .with_scalar_product(P(3), P(3), zero)
        .with_scalar_product(P(1), P(3), (t - mass**2) / 2)
    )
    outer_measure = _outer.two_body_phase_space(_Q, P(2))
    inner_measure = _inner.two_body_phase_space(P(1), P(3))
    outer_measure = outer_measure.replace(
        (4 * ((M**2 - t) / 2).expand() ** 2).sqrt(), M**2 - t
    ).together()
    inner_measure = inner_measure.replace(
        (4 * ((t - mass**2) / 2).expand() ** 2).sqrt(), t - mass**2
    ).together()
    assert (outer_measure - (M**2 - t) / (32 * pi**2 * M**2)).together() == zero
    assert (inner_measure - (t - mass**2) / (32 * pi**2 * t)).together() == zero
    _cosine = S("cosine")
    _electron_energy = (t + mass**2) / (2 * t.sqrt())
    _electron_momentum = (t - mass**2) / (2 * t.sqrt())
    _spectator_energy = (M**2 - t) / (2 * t.sqrt())
    _s_of_cosine = mass**2 + 2 * _spectator_energy * (
        _electron_energy - _electron_momentum * _cosine
    )
    s_min = _s_of_cosine.replace(_cosine, one).together()
    s_max = _s_of_cosine.replace(_cosine, -one).together()
    assert (s_min - mass**2 * M**2 / t).together() == zero
    assert (s_max - M**2 - mass**2 + t).together() == zero
    _jacobian = -_s_of_cosine.derivative(_cosine)
    _recursive_measure = (
        outer_measure * inner_measure * (4 * pi) * (2 * pi) / (2 * pi * _jacobian)
    ).together()
    assert (dalitz_measure - _recursive_measure).together() == zero
    assert (dalitz_measure - 1 / (128 * pi**3 * M**2)).together() == zero
    flux = Kinematics().with_scalar_product(P(0), P(0), M**2).flux(P(0))
    assert flux == 2 * M
    density = (fermi * dalitz_measure / flux).together()
    return density, flux, inner_measure, outer_measure, s_max, s_min


@app.cell(hide_code=True)
def _(dalitz_measure, density, inner_measure, mo, outer_measure, s_max, s_min):
    mo.vstack(
        [
            mo.md("**Outer and inner two-body measures per solid angle**"),
            mo.hstack([outer_measure, inner_measure]),
            mo.md("**Derived s bounds**"),
            mo.hstack([s_min, s_max]),
            mo.md("**dΦ₃ / (ds dt)**"),
            dalitz_measure,
            mo.md("**dΓ / (ds dt) in the Fermi limit**"),
            density,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exact integrations
    Let $r=m^2/M^2$ and $\Gamma_0=G_F^2M^5/(192\pi^3)$. Integrating the massive Dalitz distribution derives $\Gamma/\Gamma_0=f(r)$, including the logarithmic term.

    For a massless electron, $x_E=2E_e/M$ runs from zero to one. Integrating at fixed $x_E$ derives the normalized Michel spectrum. Expanding the full generated W amplitude also gives its first propagator correction. The finite-mass result uses the Fermi limit; the finite-W result below uses a massless electron.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Integrate the massive decay density
    """)
    return


@app.cell
def _(GF, M, S, density, mass, one, pi, s_max, s_min, t, zero):
    tau = S("tau", is_positive=True)
    r = S("r", is_positive=True)
    normalization = GF**2 * M**5 / (192 * pi**3)
    _integrand = (
        (density * (s_max - s_min) * M**2 / normalization)
        .replace(t, M**2 * tau)
        .replace(mass, M * r.sqrt())
        .together()
    )
    assert (_integrand - 12 * (1 - tau) ** 2 * (tau - r) ** 2 / tau).together() == zero
    _primitive = _integrand.integrate(tau)
    mass_correction = (
        _primitive.replace(tau, one) - _primitive.replace(tau, r)
    ).expand()
    _reference_correction = 1 - 8 * r + 8 * r**3 - r**4 - 12 * r**2 * r.log()
    assert (mass_correction - _reference_correction).together() == zero
    assert mass_correction.replace(r, one) == zero
    _total_width = normalization * mass_correction
    return mass_correction, normalization, r, tau


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Recover the Michel spectrum
    """)
    return


@app.cell
def _(M, S, density, mass, normalization, one, t, tau, zero):
    xE = S("xE", is_positive=True)
    _massless_density = density.replace(mass, zero)
    _dimensionless_density = (
        (_massless_density * M**4 / normalization).replace(t, M**2 * tau).together()
    )
    _michel_primitive = _dimensionless_density.integrate(tau)
    michel_spectrum = (
        _michel_primitive.replace(tau, xE) - _michel_primitive.replace(tau, zero)
    ).expand()
    assert (michel_spectrum - 2 * xE**2 * (3 - 2 * xE)).together() == zero
    _michel_integral = michel_spectrum.integrate(xE)
    assert (
        _michel_integral.replace(xE, one) - _michel_integral.replace(xE, zero) - 1
    ).together() == zero
    return michel_spectrum, xE


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Extract the finite-W correction
    """)
    return


@app.cell
def _(M, dalitz_measure, finite_W, flux, mass, normalization, s, t, z, zero):
    _correction_density = (finite_W.coefficient(z) * dalitz_measure / flux).replace(
        mass, zero
    )
    _correction_primitive = _correction_density.integrate(s)
    _correction_t = (
        _correction_primitive.replace(s, M**2 - t)
        - _correction_primitive.replace(s, zero)
    ).together()
    _correction_primitive = _correction_t.integrate(t)
    W_coefficient = (
        _correction_primitive.replace(t, M**2) - _correction_primitive.replace(t, zero)
    ).together()
    assert (W_coefficient / normalization - 3 * M**2 / 5).together() == zero
    W_relative_coefficient = (W_coefficient / (normalization * M**2)).together()
    return W_coefficient, W_relative_coefficient


@app.cell(hide_code=True)
def _(
    W_coefficient,
    mass_correction,
    michel_spectrum,
    mo,
    model,
    normalization,
):
    mo.vstack(
        [
            mo.md("**Mass correction f(r)**"),
            mass_correction,
            mo.md("**Normalized massless Michel spectrum**"),
            michel_spectrum,
            mo.md("**Massless width through order M²/mW²**"),
            normalization + W_coefficient / model.particle("W+").mass ** 2,
        ]
    )
    return


@app.cell
def _(E, mass_correction, np, r):
    _nodes, _weights = np.polynomial.legendre.leggauss(160)
    numeric_checks = []
    for _rv in [0.0, (0.511 / 105.658) ** 2, 0.01, 0.1, 0.3, 0.7]:
        _value = (
            1.0
            if _rv == 0
            else complex(mass_correction.replace(r, E(str(_rv))).evaluate({}))
        )
        assert abs(complex(_value).imag) < 1e-14
        _actual = complex(_value).real
        _integral_value = sum(
            (
                float(_w)
                * (1 - _rv)
                / 2
                * 12
                * (1 - (_rv + (1 - _rv) * (float(_n) + 1) / 2)) ** 2
                * (_rv + (1 - _rv) * (float(_n) + 1) / 2 - _rv) ** 2
                / (_rv + (1 - _rv) * (float(_n) + 1) / 2)
                for _n, _w in zip(_nodes, _weights, strict=True)
            )
        )
        assert abs(_actual - _integral_value) < 2e-08, (_rv, _actual, _integral_value)
        assert 0 < _actual <= 1
        numeric_checks.append((_rv, _actual, _integral_value))
    return (numeric_checks,)


@app.cell
def _(mo):
    daughter_mass = mo.ui.dropdown(
        {
            "Massless electron": 0,
            "Illustrative electron/muon ratio": 1,
            "r=0.01": 2,
            "r=0.1": 3,
            "r=0.3": 4,
            "r=0.7": 5,
        },
        value="Illustrative electron/muon ratio",
        label="Daughter mass ratio r",
    )
    W_scale = mo.ui.dropdown(
        {"Fermi limit": 0.0, "M²/mW²=0.000002": 2e-06, "M²/mW²=0.01": 0.01},
        value="Fermi limit",
        label="Finite-W expansion",
    )
    mo.vstack([daughter_mass, W_scale])
    return W_scale, daughter_mass


@app.cell
def _(
    E,
    W_relative_coefficient,
    W_scale,
    daughter_mass,
    michel_spectrum,
    mo,
    np,
    numeric_checks,
    xE,
):
    selected_r, selected_correction, quadrature = numeric_checks[daughter_mass.value]
    assert abs(selected_correction - quadrature) < 2e-08
    relative_W_width = (
        1 + complex(W_relative_coefficient.evaluate({})).real * W_scale.value
    )
    _spectrum_samples = [
        {
            "xE": float(_v),
            "(dΓ/dxE)/Γ₀": complex(
                michel_spectrum.replace(xE, E(str(_v))).evaluate({})
            ).real,
        }
        for _v in np.linspace(0, 1, 11)
    ]
    mo.vstack(
        [
            mo.md("## Mass-corrected Fermi width"),
            mo.md(f"`r = {selected_r:.9g}` · **Γ/Γ₀ = {selected_correction:.12g}**"),
            mo.md(
                f"Independent Dalitz quadrature: `{quadrature:.12g}`; difference `{abs(selected_correction - quadrature):.3g}`."
            ),
            mo.md("## Finite-W correction, massless electron"),
            mo.md(f"**Γ/Γ₀ = {relative_W_width:.12g}** through first order in M²/mW²."),
            mo.md("## Massless Michel spectrum samples"),
            mo.ui.table(_spectrum_samples, selection=None),
            mo.md(
                "The spectrum integrates to one. Both mass and W corrections above are normalized to Γ₀; they are evaluated in their stated limits."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
