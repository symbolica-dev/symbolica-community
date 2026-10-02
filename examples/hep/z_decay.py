import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Chiral Z decays")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chiral Z decays

    [Browse all notebooks](/) · [W and top decays](/?file=hep/weak_decays.py) · [Two-photon annihilation](/?file=hep/diphoton.py)

    Generate $Z\to f\bar f$ from the Standard Model and retain the fermion mass,
    left- and right-chiral couplings, and the Z's longitudinal polarization.
    Choose a neutrino, charged lepton, up-type quark or down-type quark to reproduce
    the four classes in the
    [FeynCalc example](https://feyncalc.github.io/FeynCalcExamples/EW/Tree/Z-FFbar).

    The three initial Z polarizations are averaged; final spins and colors are
    summed. Fermion and antifermion are distinct, so no identical-particle factor
    is needed.
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
    from symbolica import E, Expression, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hep import Kinematics, Model
    from symbolica.community.tensor import TensorExpression

    _set_namespace("z_decay")
    return E, Expression, Kinematics, Model, S, hep, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell
def _(Model, mo):
    model = Model.standard_model()
    channel = mo.ui.dropdown(
        ["Electron neutrinos", "Electrons", "Charm quarks", "Bottom quarks"],
        value="Electrons",
        label="Decay channel",
    )
    mo.vstack([channel])
    return channel, model


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Select the physical states

    Read masses and charges from the model. The dropdown changes the final-state
    particles; their spin and color representations stay with those particle objects.
    """)
    return


@app.cell
def _(channel, hep, model):
    P = hep.Kinematics.external_momentum
    charge = -model.particle("e-").electric_charge
    sw = model.parameter("sw").symbol
    cw = model.parameter("cw").symbol
    mz = model.particle("Z").mass
    particle_name = {
        "Electron neutrinos": "ve",
        "Electrons": "e-",
        "Charm quarks": "c",
        "Bottom quarks": "b",
    }[channel.value]
    fermion = model.particle(particle_name)
    mass = fermion.mass
    weak_isospin = fermion.weak_isospin
    electric_charge = fermion.charge
    assert weak_isospin is not None
    antifermion = fermion.antiparticle
    return (
        P,
        antifermion,
        charge,
        cw,
        electric_charge,
        fermion,
        mass,
        mz,
        particle_name,
        sw,
        weak_isospin,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate and square the amplitude

    `generate_amplitude()` keeps the diagram weights and aligns the external ports.
    The native spin and color sums also handle conjugation and fermion orientation.
    """)
    return


@app.cell
def _(antifermion, fermion, model):
    amplitude = model.process(["Z"], [fermion, antifermion]).generate_amplitude(
        max_vertices=1, progress=None
    )
    assert len(amplitude.diagrams) == 1
    diagram = amplitude.diagrams[0]
    operator = amplitude.expression()
    amplitude
    return amplitude, diagram, operator


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Reduce the spin and color sums

    Average the three initial Z polarizations and sum the final states.
    Enable the identity families explicitly, then pass the scalar expression to the
    external kinematics.
    """)
    return


@app.cell
def _(amplitude):
    settings = dict(gamma=True, color=True, epsilon=True)
    reduced = (
        amplitude.squared()
        .sum_spins(average_initial=True)
        .sum_colors()
        .expression()
        .simplify_algebra(contract="dots", **settings)
    )
    assert reduced.is_scalar
    scalar = reduced.to_expression()
    return (scalar,)


@app.cell
def _(Kinematics, P, mass, mz, scalar):
    kin = (
        Kinematics()
        .with_scalar_product(P(0), P(0), mz**2)
        .with_scalar_product(P(1), P(1), mass**2)
        .with_scalar_product(P(2), P(2), mass**2)
        .with_scalar_product(P(1), P(2), (mz**2 - 2 * mass**2) / 2)
        .with_scalar_product(P(0), P(1), mz**2 / 2)
        .with_scalar_product(P(0), P(2), mz**2 / 2)
    )
    squared = kin.apply(scalar).together()
    return (squared,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Verify the closed formula

    Compare with the independent massive reference before evaluating a width.
    The identical-particle factor belongs to phase space, not the amplitude.
    """)
    return


@app.cell
def _(
    E,
    charge,
    cw,
    electric_charge,
    fermion,
    mass,
    mz,
    particle_name,
    squared,
    sw,
    weak_isospin,
):
    colors = abs(fermion.color)
    cv, ca = weak_isospin - 2 * electric_charge * sw**2, weak_isospin
    expected = (
        colors
        * charge**2
        / (3 * sw**2 * cw**2)
        * (cv**2 * (mz**2 + 2 * mass**2) + ca**2 * (mz**2 - 4 * mass**2))
    )
    residual = (
        ((squared - expected) * sw**2 * cw**2 / charge**2)
        .expand()
        .replace(cw, (1 - sw**2).sqrt())
        .together()
    )
    assert residual == E("0"), (particle_name, residual.format_plain())
    return ca, colors, cv


@app.cell(hide_code=True)
def _(diagram, mo, operator, squared):
    mo.vstack(
        [
            mo.md("**Generated decay diagram and chiral current**"),
            diagram,
            operator,
            operator.to_expression(),
            mo.md("**Spin-averaged squared matrix element**"),
            squared.together(),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Physical phase space

    With $r=m_f^2/m_Z^2$ and $\beta=\sqrt{1-4r}$, the decay width is
    $$\Gamma=\frac{N_c G_F m_Z^3\beta}{6\pi\sqrt2}
    \left[c_V^2(1+2r)+c_A^2(1-4r)\right],$$
    where $c_V=T_3-2Q_f\sin^2\theta_W$ and $c_A=T_3$.
    Here $N_c=1$ for leptons and $3$ for quarks. The expression applies above
    threshold, $m_Z>2m_f$.

    Feynkit supplies the two-body measure and rest-frame decay flux.
    Integrating the isotropic spin-averaged density over solid angle gives the
    total width. In the massless neutrino channel it reduces to
    $G_Fm_Z^3/(12\pi\sqrt2)$.
    """)
    return


@app.cell
def _(E, Expression, Kinematics, P, S, mass):
    pi = Expression.PI
    # Express the physical branch with positive M and beta. The on-shell
    # relation m^2=M^2(1-beta^2)/4 describes 0<beta<=1 above threshold.
    physical_mass = S("M", is_positive=True)
    beta = E("1") if mass == E("0") else S("beta", is_positive=True)
    mass_squared = physical_mass**2 * (1 - beta**2) / 4
    physical_kin = (
        Kinematics()
        .with_scalar_product(P(0), P(0), physical_mass**2)
        .with_scalar_product(P(1), P(1), mass_squared)
        .with_scalar_product(P(2), P(2), mass_squared)
        .with_scalar_product(P(1), P(2), (physical_mass**2 - 2 * mass_squared) / 2)
    )
    measure = physical_kin.two_body_phase_space(P(1), P(2)).expand()
    # Symbolica keeps products under radicals intact; select the positive
    # root explicitly using M>0 and beta>0.
    measure = measure.replace(
        (physical_mass**4 * beta**2).sqrt(), physical_mass**2 * beta
    )
    flux = physical_kin.flux(P(0))
    assert (measure - beta / (32 * pi**2)).together() == E("0")
    assert flux == 2 * physical_mass
    return beta, flux, mass_squared, measure, physical_mass, pi


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Form the physical width

    The final fermion and antifermion are distinct; no identical-particle factor enters.
    """)
    return


@app.cell
def _(
    E,
    S,
    beta,
    charge,
    cw,
    flux,
    mass,
    measure,
    mz,
    physical_mass,
    pi,
    squared,
    sw,
):
    physical_squared = squared.replace(mz, physical_mass)
    if mass != E("0"):
        physical_squared = physical_squared.replace(
            mass, physical_mass * (1 - beta**2).sqrt() / 2
        )
    # f and fbar are distinct final particles, so no factorial is present.
    width = 4 * pi * measure * physical_squared / flux
    gf = S("G_F")
    width = width.replace(
        charge**2, 4 * E("2").sqrt() * gf * physical_mass**2 * sw**2 * cw**2
    )
    return gf, width


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Compare with vector and axial couplings
    """)
    return


@app.cell
def _(E, beta, ca, colors, cv, gf, mass_squared, physical_mass, pi):
    expected_width = (
        colors
        * gf
        * physical_mass**3
        * beta
        * E("2").sqrt()
        / (12 * pi)
        * (
            cv**2 * (1 + 2 * mass_squared / physical_mass**2)
            + ca**2 * (1 - 4 * mass_squared / physical_mass**2)
        )
    )
    return (expected_width,)


@app.cell
def _(
    E,
    beta,
    ca,
    colors,
    cv,
    cw,
    expected_width,
    gf,
    mass,
    physical_mass,
    pi,
    sw,
    width,
):
    width_residual = (
        (width - expected_width).expand().replace(cw, (1 - sw**2).sqrt()).together()
    )
    assert width_residual == E("0"), width_residual.format_plain()
    if mass != E("0"):
        massless_width = width.replace(beta, E("1"))
        expected_massless = (
            colors * gf * physical_mass**3 * (cv**2 + ca**2) * E("2").sqrt() / (12 * pi)
        )
        assert (massless_width - expected_massless).expand().replace(
            cw, (1 - sw**2).sqrt()
        ).together() == E("0")
    else:
        assert (
            width - gf * physical_mass**3 * E("2").sqrt() / (24 * pi)
        ).expand().replace(cw, (1 - sw**2).sqrt()).together() == E("0")

    z_decay_width = width.replace(cw, (1 - sw**2).sqrt()).together()
    return (z_decay_width,)


@app.cell(hide_code=True)
def _(mo, z_decay_width):
    mo.vstack(
        [
            mo.md(
                r"**Total decay width** in terms of $M=m_Z$ and the final-particle speed $\beta$. The massless neutrino channel has $\beta=1$."
            ),
            z_decay_width,
            mo.callout(
                "The generated squared amplitude, phase-space normalization and massless limit agree with the FeynCalc reference.",
                kind="success",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
