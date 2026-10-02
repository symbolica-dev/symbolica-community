import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Bhabha and Møller scattering")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bhabha and Møller scattering

    [Browse all notebooks](/) · [QED production](/?file=hep/qed_cross_section.py) ·
    [Diphoton annihilation](/?file=hep/diphoton.py)

    Generate the two tree diagrams for either **Bhabha scattering**,
    $e^-e^+\to e^-e^+$, or **Møller scattering**, $e^-e^-\to e^-e^-$.
    The calculation retains the electron mass and the interference between
    channels. It averages both incoming spins and sums both outgoing spins,
    with $s+t+u=4m_e^2$.

    The default generated amplitudes retain the labeled external particles
    and their relative fermion signs. In particular, Møller scattering
    requires both exchange diagrams before squaring the amplitude.

    Reference calculations: [FeynCalc Bhabha scattering](https://feyncalc.github.io/FeynCalcExamples/QED/Tree/ElAel-ElAel)
    and [FeynCalc Møller scattering](https://feyncalc.github.io/FeynCalcExamples/QED/Tree/ElEl-ElEl).
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
    from symbolica import E, Expression, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hepkit import Kinematics, Model
    from symbolica.community.tensor import TensorExpression

    _set_namespace("lepton_scattering")
    return E, Expression, Kinematics, Model, S, hep, mo


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
def _(Model, mo):
    lepton_model = Model.standard_model()
    reaction = mo.ui.dropdown(
        ["Bhabha", "Møller"], value="Bhabha", label="Scattering process"
    )
    mo.vstack([reaction])
    return lepton_model, reaction


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate both exchange channels

    The amplitude aligns the external states and retains the relative fermion sign.
    """)
    return


@app.cell
def _(Kinematics, S, hep, lepton_model, reaction):
    P = hep.Kinematics.external_momentum
    s = S("s", is_positive=True)
    t, u = S("t", "u")
    _electron = lepton_model.particle("e-")
    mass, charge = _electron.mass, -_electron.electric_charge
    names = ["e-", "e+", "e-", "e+"] if reaction.value == "Bhabha" else ["e-"] * 4
    kinematics = Kinematics.mandelstam(
        [P(i) for i in range(4)], [mass**2] * 4, [s, t, u]
    )
    lepton_generated = lepton_model.process(
        names[:2], names[2:], vertex_allow=["V_98"]
    ).generate_amplitude(max_vertices=2, progress=None)
    assert len(lepton_generated.diagrams) == 2
    lepton_generated
    return P, charge, kinematics, lepton_generated, mass, s, t, u


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the channel denominators
    """)
    return


@app.cell
def _(S, Symbols, kinematics, lepton_generated, reaction, s, t, u):
    _a, _b, _c, _inverse = S("a_", "b_", "c_", "inverse_")
    _denominators = [
        kinematics.apply(
            diagram.denominator_expression(dimension=4, in_lmb=True)
            .to_expression()
            .replace(Symbols.denominator(_a, _b, _c, _inverse), _inverse)
        ).expand()
        for diagram in lepton_generated.diagrams
    ]
    assert set(_denominators) == ({s, t} if reaction.value == "Bhabha" else {t, u})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Separate the coherent square and interference

    Square the full amplitude and each diagram with the same native spin average. Subtracting the diagonal contributions isolates the interference.
    """)
    return


@app.cell
def _(E, hep, kinematics, lepton_generated, mass, s, t, u):
    settings = dict(gamma=True, epsilon=True)
    _amplitudes = [lepton_generated] + [
        hep.Amplitude.from_diagram(d) for d in lepton_generated.diagrams
    ]
    _squares = []
    for _amplitude in _amplitudes:
        _scalar = (
            _amplitude.squared()
            .sum_spins(average_initial=True)
            .expression()
            .simplify_algebra(contract="dots", **settings)
        )
        assert _scalar.is_scalar
        _squares.append(
            kinematics.apply(_scalar)
            .to_expression()
            .replace(u, 4 * mass**2 - s - t)
            .together()
        )
    lepton_squared = _squares[0]
    lepton_interference = (lepton_squared - sum(_squares[1:], E("0"))).together()
    return lepton_interference, lepton_squared


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check crossing and identical-fermion symmetry
    """)
    return


@app.cell
def _(E, charge, lepton_interference, lepton_squared, mass, reaction, s, t, u):
    _massless_interference = (
        4 * charge**4 * u**2 / (s * t)
        if reaction.value == "Bhabha"
        else 4 * charge**4 * s**2 / (t * u)
    )
    assert (
        lepton_interference.replace(mass, E("0"))
        - _massless_interference.replace(u, -s - t)
    ).together() == E("0")
    if reaction.value == "Møller":
        assert (
            lepton_squared - lepton_squared.replace(t, 4 * mass**2 - s - t)
        ).together() == E("0")
    return


@app.cell(hide_code=True)
def _(lepton_interference, lepton_squared, mo):
    mo.vstack(
        [
            mo.md("**Massive, spin-averaged squared amplitude**"),
            lepton_squared,
            mo.md("**Interference between the two channels**"),
            lepton_interference,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Angular distribution

    In the center-of-mass frame let $x=\cos\theta$ and
    $\beta=\sqrt{1-4m_e^2/s}$. Then
    $t=-s\beta^2(1-x)/2$ and $u=-s\beta^2(1+x)/2$.
    The shared flux and phase-space methods give
    $d\Phi_2/(d\Omega\,\mathrm{flux})=1/(64\pi^2s)$.

    For Bhabha scattering, $\theta$ labels the outgoing electron.
    For Møller scattering either outgoing electron can carry that label:
    the displayed **event density includes $1/2!$ on the full sphere**.
    Equivalently, integrate the labeled density over one hemisphere.
    This factor belongs to phase-space counting; both interfering exchange
    amplitudes remain in the squared matrix element.
    """)
    return


@app.cell
def _(
    E,
    Expression,
    Kinematics,
    P,
    S,
    charge,
    lepton_interference,
    lepton_squared,
    mass,
    reaction,
    s,
    t,
    u,
):
    beta = S("beta", is_positive=True)
    cos_theta, alpha = S("cos_theta", "alpha")
    _mass_squared = s * (1 - beta**2) / 4
    _kinematics = Kinematics.mandelstam(
        [P(_position) for _position in range(4)], [_mass_squared] * 4, [s, t, u]
    )
    # Choose the physical branch s > 0, beta > 0 after the shared calculation.
    _flux = (
        _kinematics.flux(P(0), P(1)).expand().replace((s**2 * beta**2).sqrt(), s * beta)
    )
    _measure = (
        _kinematics.two_body_phase_space(P(2), P(3))
        .expand()
        .replace((s**2 * beta**2).sqrt(), s * beta)
    )
    assert (_measure / _flux - 1 / (64 * Expression.PI**2 * s)).together() == E("0")
    _counting = E("1") if reaction.value == "Bhabha" else E("1/2")
    _densities = []
    for _square in (lepton_squared, lepton_interference):
        _densities.append(
            (
                _square.replace(t, -s * beta**2 * (1 - cos_theta) / 2)
                .replace(mass, _mass_squared.sqrt())
                .replace(charge**4, (4 * Expression.PI * alpha) ** 2)
                * _measure
                / _flux
                * _counting
            ).together()
        )
    angular_density, angular_interference = _densities
    normalized_density = (angular_density * s / alpha**2).together()
    massless_kernel = (
        (3 + cos_theta**2) ** 2 / (4 * (1 - cos_theta) ** 2)
        if reaction.value == "Bhabha"
        else (3 + cos_theta**2) ** 2 / (2 * (1 - cos_theta**2) ** 2)
    )
    assert (normalized_density.replace(beta, E("1")) - massless_kernel).together() == E(
        "0"
    )
    return (
        alpha,
        angular_density,
        angular_interference,
        beta,
        cos_theta,
        massless_kernel,
        normalized_density,
    )


@app.cell(hide_code=True)
def _(angular_density, massless_kernel, mo):
    mo.vstack(
        [
            mo.md(r"**Event angular density $d\sigma/d\Omega$**"),
            angular_density,
            mo.md(r"**Massless limit, in units of $\alpha^2/s$**"),
            massless_kernel,
        ]
    )
    return


@app.cell
def _(mo):
    cm_speed = mo.ui.slider(
        0.1, 1.0, step=0.05, value=1.0, label="CM speed beta (1 = massless limit)"
    )
    scattering_cosine = mo.ui.slider(
        -0.95, 0.95, step=0.05, value=0.0, label="Scattering angle cos(theta)"
    )
    mo.vstack([cm_speed, scattering_cosine])
    return cm_speed, scattering_cosine


@app.cell
def _(
    E,
    alpha,
    angular_interference,
    beta,
    cm_speed,
    cos_theta,
    mo,
    normalized_density,
    s,
    scattering_cosine,
):
    _interference = (angular_interference * s / alpha**2).together()
    angular_value, interference_value = (
        _quantity.replace(beta, E(str(cm_speed.value)))
        .replace(cos_theta, E(str(scattering_cosine.value)))
        .to_float(16)
        for _quantity in (normalized_density, _interference)
    )
    assert float(angular_value) > 0
    mo.md(
        rf"Event density: **{angular_value}** × $\alpha^2/s$. "
        rf"The interference contributes **{interference_value}** × $\alpha^2/s$."
    )
    return


if __name__ == "__main__":
    app.run()
