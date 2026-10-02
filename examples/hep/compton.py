import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Massive Compton scattering")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Compton scattering

    [Browse all notebooks](/) · [Two-photon production](/?file=hep/diphoton.py)

    Generate $e^-\gamma\to e^-\gamma$, add both diagrams coherently, and average
    only the initial spins. The electron mass is retained throughout. An exact
    massive reference checks the result before taking the massless limit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Setup and notebook helpers

    The folded cell contains imports only. This example uses `hep.Amplitude` to
    handle diagram weights, external ports, and conjugation without a Python helper layer.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from symbolica import E, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community import hep, tensor

    _set_namespace("compton")
    return E, S, hep, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Generate the coherent amplitude

    Select the electron–photon vertex and let the process construct both exchange
    diagrams. `generate_amplitude()` retains their relative signs and weights.
    """)
    return


@app.cell
def _(hep):
    qed_model = hep.Model.standard_model()
    process = qed_model.process(["e-", "a"], ["e-", "a"], vertex_allow=["V_98"])
    amplitude = process.generate_amplitude(max_vertices=2, progress=None)
    assert len(amplitude.diagrams) == 2
    amplitude
    return amplitude, qed_model


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Declare the external kinematics

    External labels follow the process order: incoming electron, incoming photon,
    outgoing electron, outgoing photon. Thus $s+t+u=2m_e^2$.
    """)
    return


@app.cell
def _(E, S, hep, qed_model):
    P = hep.Kinematics.external_momentum
    s, t, u = S("s", "t", "u")
    electron = qed_model.particle("e-")
    mass, charge = electron.mass, -electron.electric_charge
    kinematics = hep.Kinematics.mandelstam(
        [P(i) for i in range(4)], [mass**2, E("0"), mass**2, E("0")], [s, t, u]
    )
    return charge, kinematics, mass, s, t, u


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Sum spins and reduce the tensor expression

    `squared()` includes the interference between the two diagrams. The spin sum
    includes the final states and averages the two incoming states. Enable Dirac
    and epsilon identities explicitly; no color identities are needed.
    """)
    return


@app.cell
def _(amplitude, kinematics, mass, s, t, u):
    spin_summed = amplitude.squared().sum_spins(average_initial=True)
    settings = dict(gamma=True, epsilon=True)
    scalar = spin_summed.expression().simplify_algebra(contract="dots", **settings)
    assert scalar.is_scalar
    compton_squared = (
        kinematics.apply(scalar)
        .to_expression()
        .replace(t, 2 * mass**2 - s - u)
        .together()
    )
    compton_squared
    return (compton_squared,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Check the massive reference

    The independent closed formula below agrees with the
    [FeynCalc example](https://feyncalc.github.io/FeynCalcExamples/QED/Tree/ElGa-ElGa).
    The final display also gives the massless limit.
    """)
    return


@app.cell
def _(E, charge, compton_squared, mass, s, u):
    expected = (
        2
        * charge**4
        * (
            -(mass**4) * (3 * s**2 + 14 * s * u + 3 * u**2)
            + mass**2 * (s**3 + 7 * s**2 * u + 7 * s * u**2 + u**3)
            + 6 * mass**8
            - s * u * (s**2 + u**2)
        )
        / ((s - mass**2) ** 2 * (u - mass**2) ** 2)
    )
    assert (compton_squared - expected).together() == E("0")
    massless = compton_squared.replace(mass, E("0")).together()
    massless
    return


if __name__ == "__main__":
    app.run()
