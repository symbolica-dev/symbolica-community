import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Massive Higgs decays")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Massive Higgs decays

    [Browse all notebooks](/) · [Chiral Z decays](/?file=hep/z_decay.py) ·
    [Higgs to gluons](/?file=hep/higgs_gluons.py)

    Generate $H\to f\bar f$, $H\to W^-W^+$ or $H\to ZZ$ from the Standard
    Model, retaining the final-particle masses. The calculation sums final spins
    and colors, including all three physical polarizations of each massive vector.
    A scalar Higgs needs no initial spin average.

    These are **on-shell two-body decays above threshold**, $m_H>2m_f$ or
    $m_H>2m_V$. The $WW$ and $ZZ$ results describe a Higgs above those thresholds;
    they do not describe the off-shell $WW^*$ and $ZZ^*$ decays at 125 GeV.

    The reference results are FeynCalc's
    [fermion](https://feyncalc.github.io/FeynCalcExamples/EW/Tree/H-FFbar),
    [WW](https://feyncalc.github.io/FeynCalcExamples/EW/Tree/H-WW), and
    [ZZ](https://feyncalc.github.io/FeynCalcExamples/EW/Tree/H-ZZ) examples.
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

    _set_namespace("higgs_decay")
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
        ["Electrons", "Charm quarks", "Bottom quarks", "W bosons", "Z bosons"],
        value="Bottom quarks",
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
    mh = model.particle("H").mass
    mw = model.particle("W+").mass
    mz = model.particle("Z").mass
    vev = model.parameter("vev").symbol
    particle_names, mass, yukawa, yukawa_mass = {
        "Electrons": (("e-", "e+"), model.particle("e-").mass, "ye", "yme"),
        "Charm quarks": (("c", "c~"), model.particle("c").mass, "yc", "ymc"),
        "Bottom quarks": (
            ("b", "b~"),
            model.particle("b").mass,
            "yb",
            "ymb",
        ),
        "W bosons": (("W-", "W+"), mw, None, None),
        "Z bosons": (("Z", "Z"), mz, None, None),
    }[channel.value]
    particles = [model.particle(name) for name in particle_names]
    return (
        P,
        charge,
        cw,
        mass,
        mh,
        mw,
        mz,
        particle_names,
        particles,
        sw,
        vev,
        yukawa,
        yukawa_mass,
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
def _(model, particles):
    amplitude = model.process(["H"], particles).generate_amplitude(
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

    The scalar Higgs needs no spin average. Sum all final polarizations and colors.
    Enable the identity families explicitly, then pass the scalar expression to the
    external kinematics.
    """)
    return


@app.cell
def _(amplitude):
    settings = dict(gamma=True, color=True, epsilon=True)
    reduced = (
        amplitude.squared()
        .sum_spins()
        .sum_colors()
        .expression()
        .simplify_algebra(contract="dots", **settings)
    )
    assert reduced.is_scalar
    scalar = reduced.to_expression()
    return (scalar,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## State the tree-level mass convention

    UFO Yukawa masses and pole masses are independent model inputs. This comparison
    sets them equal explicitly and substitutes the model's vacuum expectation value.
    """)
    return


@app.cell
def _(mass, model, scalar, vev, yukawa, yukawa_mass):
    scalar_with_masses = scalar
    if yukawa is not None:
        scalar_with_masses = scalar_with_masses.replace(
            model.parameter(yukawa).symbol, model.parameter(yukawa).expression
        ).replace(model.parameter(yukawa_mass).symbol, mass)
    scalar_with_masses = scalar_with_masses.replace(
        vev, model.parameter("vev").expression
    )
    return (scalar_with_masses,)


@app.cell
def _(Kinematics, P, mass, mh, scalar_with_masses):
    kin = (
        Kinematics()
        .with_scalar_product(P(0), P(0), mh**2)
        .with_scalar_product(P(1), P(1), mass**2)
        .with_scalar_product(P(2), P(2), mass**2)
        .with_scalar_product(P(1), P(2), (mh**2 - 2 * mass**2) / 2)
    )
    squared = kin.apply(scalar_with_masses).together()
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
    mass,
    mh,
    mw,
    mz,
    particle_names,
    particles,
    squared,
    sw,
    yukawa,
):
    colors = abs(particles[0].color)
    if yukawa is not None:
        expected = (
            colors * charge**2 * mass**2 * (mh**2 - 4 * mass**2) / (2 * mw**2 * sw**2)
        )
        assert squared.replace(mass, E("0")) == E("0")
    else:
        expected = (
            charge**2
            * (mh**4 - 4 * mh**2 * mass**2 + 12 * mass**4)
            / (4 * mw**2 * sw**2)
        )
        if particle_names == ("Z", "Z"):
            # This is the labeled amplitude: the identical-Z factor belongs to
            # phase space below, not to the amplitude or polarization sum.
            expected *= mw**4 / (mz**4 * cw**4)
    residual = (squared - expected).expand().replace(cw, (1 - sw**2).sqrt()).together()
    assert residual == E("0"), (particle_names, residual.format_plain())
    return (colors,)


@app.cell(hide_code=True)
def _(cw, diagram, mo, operator, squared, sw):
    mo.vstack(
        [
            mo.md("**Generated decay diagram and open amplitude**"),
            diagram,
            operator,
            mo.md("**Squared matrix element, summed over final spins and colors**"),
            squared.replace(cw, (1 - sw**2).sqrt()).together(),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Decay width

    Write $M=m_H$, $r=m^2/M^2$ and $\beta=\sqrt{1-4r}$. The shared two-body
    phase-space measure is $d\Phi_2/d\Omega=\beta/(32\pi^2)$ and the rest-frame
    decay flux is $2M$. Integrating over solid angle gives
    $\Gamma=S\beta\sum|\mathcal M|^2/(16\pi M)$.

    The identical-particle factor is **$S=1/2!$ for $ZZ$** and $S=1$ for the
    distinct $W^-W^+$ and $f\bar f$ final states. It is applied once, to the
    phase-space integral. For the $ZZ$ comparison we use $m_W=m_Z\cos\theta_W$;
    for fermions we identify the Yukawa mass with the kinematic mass at tree level.

    The gallery widths, with $\alpha=e^2/(4\pi)$, are
    $$\Gamma_{f\bar f}=\frac{N_c\alpha M m_f^2}{8m_W^2\sin^2\theta_W}\beta^3,$$
    $$\Gamma_{WW}=\frac{\alpha M^3}{16m_W^2\sin^2\theta_W}
      \beta(1-4r+12r^2),\qquad
      \Gamma_{ZZ}=\frac{\alpha M^3}{32m_W^2\sin^2\theta_W}
      \beta(1-4r+12r^2).$$
    Here $N_c=1$ for leptons and $3$ for quarks; use the selected final-particle
    mass in $r$ and $\beta$.
    """)
    return


@app.cell
def _(E, Expression, Kinematics, P, S):
    pi = Expression.PI
    alpha = S("alpha")
    physical_mass = S("M", is_positive=True)
    beta = S("beta", is_positive=True)
    mass_squared = physical_mass**2 * (1 - beta**2) / 4

    # M>0 and 0<beta<1 select the physical, above-threshold two-body branch.
    physical_kin = (
        Kinematics()
        .with_scalar_product(P(0), P(0), physical_mass**2)
        .with_scalar_product(P(1), P(1), mass_squared)
        .with_scalar_product(P(2), P(2), mass_squared)
        .with_scalar_product(P(1), P(2), (physical_mass**2 - 2 * mass_squared) / 2)
    )
    measure = physical_kin.two_body_phase_space(P(1), P(2)).expand()
    # Symbolica retains products under radicals; choose the positive root explicitly.
    measure = measure.replace(
        (physical_mass**4 * beta**2).sqrt(), physical_mass**2 * beta
    )
    flux = physical_kin.flux(P(0))
    assert (measure - beta / (32 * pi**2)).together() == E("0")
    assert flux == 2 * physical_mass
    return alpha, beta, flux, mass_squared, measure, physical_mass, pi


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Integrate the two-body phase space

    The identical-Z symmetry factor appears exactly once, in the width.
    """)
    return


@app.cell
def _(E, alpha, charge, flux, measure, particle_names, pi, squared):
    # Apply the explicit 1/2! only for the identical ZZ final state.
    symmetry = E("1/2") if particle_names == ("Z", "Z") else E("1")
    width = symmetry * 4 * pi * measure * squared / flux
    width = width.replace(charge**2, 4 * pi * alpha)
    return (width,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Compare with the independent width

    Keep the Yukawa and vector-boson formulas separate.
    """)
    return


@app.cell
def _(
    alpha,
    beta,
    colors,
    mass,
    mass_squared,
    mw,
    particle_names,
    physical_mass,
    sw,
    yukawa,
):
    if yukawa is not None:
        expected_width = (
            colors * alpha * physical_mass * mass**2 * beta**3 / (8 * mw**2 * sw**2)
        )
    else:
        expected_width = (
            alpha
            * physical_mass**3
            * beta
            / ((32 if particle_names == ("Z", "Z") else 16) * mw**2 * sw**2)
        )
        expected_width *= (
            1
            - 4 * mass_squared / physical_mass**2
            + 12 * mass_squared**2 / physical_mass**4
        )
    return (expected_width,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Impose the physical mass relations

    Use the tree-level electroweak relation for ZZ and check the threshold and massless limits.
    """)
    return


@app.cell
def _(
    E,
    beta,
    cw,
    expected_width,
    mass,
    mh,
    mw,
    mz,
    particle_names,
    physical_mass,
    sw,
    width,
    yukawa,
):
    physical_width = width
    physical_reference = expected_width
    if particle_names == ("Z", "Z"):
        # The gallery width uses the tree-level electroweak mass relation.
        physical_width = physical_width.replace(mw, cw * mz)
        physical_reference = physical_reference.replace(mw, cw * mz)
    physical_width = physical_width.replace(mh, physical_mass).replace(
        mass, physical_mass * (1 - beta**2).sqrt() / 2
    )
    physical_reference = physical_reference.replace(
        mass, physical_mass * (1 - beta**2).sqrt() / 2
    )
    _residual = (
        (physical_width - physical_reference)
        .expand()
        .replace(cw, (1 - sw**2).sqrt())
        .together()
    )
    assert _residual == E("0"), (particle_names, _residual.format_plain())
    if yukawa is not None:
        assert physical_width.replace(beta, E("1")).together() == E("0")
    higgs_decay_width = physical_width.replace(cw, (1 - sw**2).sqrt()).together()
    return (higgs_decay_width,)


@app.cell(hide_code=True)
def _(higgs_decay_width, mo):
    mo.vstack(
        [
            mo.md(
                r"**Total decay width** in terms of $M=m_H$ and $\beta$, with $m^2=M^2(1-\beta^2)/4$."
            ),
            higgs_decay_width,
            mo.callout(
                "The generated massive squared amplitude and integrated decay width agree with the FeynCalc reference.",
                kind="success",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
