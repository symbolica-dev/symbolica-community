import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Massive polarized spin states")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Massive polarized spin states

    [Browse all notebooks](/) · [QED cross section](/?file=hep/qed_cross_section.py) ·
    [Polarization sums in D dimensions](/?file=hep/polarization_sums.py)

    `Particle.spin_sum(p, i, j, spin_vector=s)` selects one physical spin state
    of a massive Dirac particle. Omitting `spin_vector` sums both states.
    The selected state's dimensionless spin vector obeys
    $p\cdot s=0$ and $s^2=-1$ in the $(+,-,-,-)$ metric.

    The shared density matrix is
    $\rho_s=(\not p\pm m)(1+\gamma^5\not s)/2$.
    The mass sign distinguishes particle and antiparticle; the spin-projector
    sign is the same for both. Do not reverse $s$ merely because the particle
    is an antiparticle. Selecting a state already supplies the projector's
    factor $1/2$, so `average=True` is rejected with `spin_vector`.

    These are massive spin states with a chosen rest-frame direction; no
    helicity convention is assumed. This example uses the tau, whose mass
    is nonzero in the stored model. `Particle.sum_spins(..., spin_vector=s)`
    uses the same density when replacing paired external wavefunctions.
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
    from symbolica.community import tensor as sp
    import math

    import marimo as mo
    from symbolica import E, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hep import (
        Boost,
        FourMomentum,
        Kinematics,
        Model,
        ThreeMomentum,
    )
    from symbolica.community.tensor import TensorExpression

    _set_namespace("polarized_spin")
    return (
        Boost,
        E,
        FourMomentum,
        Kinematics,
        Model,
        S,
        TensorExpression,
        ThreeMomentum,
        math,
        mo,
        sp,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell
def _(Model, mo):
    spin_model = Model.standard_model()
    species = mo.ui.dropdown(["Tau", "Antitau"], value="Tau", label="Physical particle")
    mo.vstack([species])
    return species, spin_model


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Choose the spin state
    """)
    return


@app.cell
def _(S, sp, species, spin_model):
    particle = spin_model.particle("ta-" if species.value == "Tau" else "ta+")
    sign = -1 if particle.is_antiparticle else 1
    spin_momentum, spin_vector = (
        sp.TensorName.vector("p").to_expression(),
        sp.TensorName.vector("s").to_expression(),
    )
    spin_mass = spin_model.particle("ta-").mass
    i, j, k, slot = S(
        "i",
        "j",
        "k",
        "slot_",
    )
    settings = dict(gamma=True, gamma_ordering="canonical")
    density = particle.spin_sum(spin_momentum, i, j, spin_vector=spin_vector)
    return (
        density,
        i,
        j,
        k,
        particle,
        settings,
        sign,
        slot,
        spin_mass,
        spin_momentum,
        spin_vector,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Recover the unpolarized sum
    """)
    return


@app.cell
def _(
    E,
    TensorExpression,
    density,
    i,
    j,
    particle,
    settings,
    slot,
    spin_momentum,
    spin_vector,
):
    _ordinary = particle.spin_sum(spin_momentum, i, j)
    _opposite = density.replace(spin_vector(slot), -spin_vector(slot))
    assert TensorExpression(
        (density + _opposite - _ordinary).expand()
    ).expand().simplify_algebra(
        contract="dots", **settings, epsilon=True
    ).expand().to_expression() == E("0")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Check the spinor trace
    """)
    return


@app.cell
def _(TensorExpression, density, i, j, settings, sign, spin_mass):
    density_trace = (
        TensorExpression(density.replace(j, i))
        .expand()
        .simplify_algebra(contract="dots", **settings, epsilon=True)
        .expand()
        .to_expression()
    )
    assert density_trace == sign * 2 * spin_mass
    return (density_trace,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Check the pure-state projector
    """)
    return


@app.cell
def _(
    E,
    Kinematics,
    TensorExpression,
    density,
    i,
    j,
    k,
    particle,
    settings,
    sign,
    spin_mass,
    spin_momentum,
    spin_vector,
):
    _kinematics = (
        Kinematics()
        .with_scalar_product(spin_momentum, spin_momentum, spin_mass**2)
        .with_scalar_product(spin_momentum, spin_vector, E("0"))
        .with_scalar_product(spin_vector, spin_vector, E("-1"))
    )
    _squared = particle.spin_sum(
        spin_momentum, i, k, spin_vector=spin_vector
    ) * particle.spin_sum(spin_momentum, k, j, spin_vector=spin_vector)
    _purity_residual = (
        TensorExpression((_squared - sign * 2 * spin_mass * density).expand())
        .expand()
        .simplify_algebra(contract="dots", **settings, epsilon=True)
        .expand()
        .to_expression()
    )
    assert _kinematics.apply(_purity_residual).expand() == E("0")
    return


@app.cell(hide_code=True)
def _(TensorExpression, density, density_trace, mo, settings):
    mo.vstack(
        [
            mo.md("**Selected spin density**"),
            TensorExpression(density)
            .expand()
            .simplify_algebra(contract="dots", **settings, epsilon=True),
            mo.md("**Spinor trace**"),
            density_trace,
            mo.md(
                "Verified: $\\rho_s+\\rho_{-s}=\\not p\\pm m$, $\\operatorname{tr}\\rho_s=\\pm2m$, and $\\rho_s^2=(\\pm2m)\\rho_s$. The last identity expresses the rank-one character of a pure spin state."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Boost a rest-frame spin direction

    Start with $p/m=(1,0,0,0)$ and $s=(0,\sin\theta,0,\cos\theta)$.
    The same `Boost` transforms both four-vectors. The controls show their
    components after a boost along $z$ and check the invariant constraints.
    Momenta below are measured in units of the particle mass.
    """)
    return


@app.cell
def _(mo):
    spin_speed = mo.ui.slider(
        start=-0.9, stop=0.9, step=0.1, value=0.5, label="Boost velocity along z"
    )
    spin_angle = mo.ui.slider(
        start=0, stop=180, step=5, value=45, label="Rest-frame spin angle (degrees)"
    )
    mo.vstack([spin_speed, spin_angle])
    return spin_angle, spin_speed


@app.cell
def _(Boost, FourMomentum, ThreeMomentum, math, mo, spin_angle, spin_speed):
    _angle = math.radians(spin_angle.value)
    _boost = Boost(ThreeMomentum(0.0, 0.0, spin_speed.value))
    boosted_momentum = _boost.apply(FourMomentum(1.0, 0.0, 0.0, 0.0))
    boosted_spin = _boost.apply(
        FourMomentum(0.0, math.sin(_angle), 0.0, math.cos(_angle))
    )
    spin_constraints = (
        boosted_momentum.dot(boosted_momentum),
        boosted_momentum.dot(boosted_spin),
        boosted_spin.dot(boosted_spin),
    )
    assert all(
        abs(_actual - _expected) < 1e-12
        for _actual, _expected in zip(spin_constraints, (1.0, 0.0, -1.0), strict=True)
    )
    mo.vstack(
        [
            mo.ui.table(
                [
                    {
                        "Vector": _label,
                        **dict(
                            zip(
                                ("time", "x", "y", "z"),
                                _vector.components(),
                                strict=True,
                            )
                        ),
                    }
                    for _label, _vector in (
                        ("p / m", boosted_momentum),
                        ("s", boosted_spin),
                    )
                ],
                selection=None,
            ),
            mo.md(r"The checks pass: $(p/m)^2=1$, $(p/m)\cdot s=0$, $s^2=-1$."),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
