import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Electron anomalous magnetic moment",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Electron anomalous magnetic moment

    [Browse all notebooks](/) ·
    [Pauli form factor across threshold](/?file=hep/pauli_form_factor.py) ·
    [Massive electron self-energy](/?file=hep/electron_self_energy.py) ·
    [Unequal-mass bubble](/?file=hep/ibp_bubble.py) ·
    [Two-loop φ⁴](/?file=hep/ibp_phi4.py) · [Higgs to gluons](/?file=hep/higgs_gluons.py)

    Generate the QED vertex and extract its Pauli form factor, reproducing
    [FeynCalc's one-loop g−2 example](https://feyncalc.github.io/FeynCalcExamples/QED/OneLoop/El-GaEl).
    Both electrons are on shell, $p^2=p'^2=m^2$, while the photon probes a
    spacelike momentum transfer $q=p-p'$, $t=q^2<0$.

    Between the external spinors the vertex has the form
    $(F_1+F_2)\gamma^\mu-F_2(p+p')^\mu/(2m)$.
    Shared spin sums close the Dirac traces; a two-by-two Symbolica system
    extracts the coefficient of $(p+p')^\mu/(2m)$.
    We project at generic $t$ and take $t\to0$ only after integration because
    those two projectors become degenerate at zero momentum transfer.

    All generator weights are retained. Normalizing to the generated tree
    vertex fixes the coupling and external-fermion phase convention. The
    common loop measure is $i/(16\pi^2)$ in the OneLOop normalization.
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
    from symbolica import E, Matrix, Replacement, S, Symbol
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hepkit import IBPFamily, Kinematics, Model, oneloop
    from symbolica.community.tensor import TensorExpression

    _set_namespace("gminus2")
    return (
        E,
        IBPFamily,
        Kinematics,
        Matrix,
        Model,
        Replacement,
        S,
        Symbol,
        TensorExpression,
        hep,
        mo,
        np,
        oneloop,
        sp,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Specify the vertex kinematics

    Keep the Lorentz dimension symbolic and put the two external electrons on shell.
    """)
    return


@app.cell
def _(Kinematics, Model, S, hep, sp):
    model = Model.standard_model()
    dimension, transfer = S("D", "t")
    _electron = model.particle("e-")
    mass, charge = _electron.mass, -_electron.electric_charge
    mu = S("mu")
    K, P, mink, bis, metric, gamma = (
        hep.Kinematics.loop_momentum,
        hep.Kinematics.external_momentum,
        sp.Representation.mink,
        sp.Representation.bis,
        sp.TensorName.g().to_expression(),
        sp.TensorName.dirac_gamma().to_expression(),
    )
    a, b, c, d, index, wave, slot = S(
        "a",
        "b",
        "c",
        "d",
        "index_",
        "wave_",
        "slot_",
    )
    kinematics = (
        Kinematics(dimension, momenta=[K(0), P(0), P(1)])
        .with_scalar_product(P(0), P(0), mass**2)
        .with_scalar_product(P(1), P(1), transfer)
        .with_scalar_product(P(0), P(1), transfer / 2)
    )
    return (
        P,
        a,
        b,
        bis,
        c,
        charge,
        d,
        dimension,
        gamma,
        index,
        kinematics,
        mass,
        metric,
        mink,
        model,
        mu,
        slot,
        transfer,
        wave,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate the tree and one-loop vertices

    Align the generated external slots with the same projector labels before comparing the vertices.
    """)
    return


@app.cell
def _(a, b, bis, dimension, index, mink, model, mu, sp, wave):
    operators, diagrams = [], []
    for _loops in (0, 1):
        _generated = model.process(
            ["e-"], ["a", "e-"], vertex_allow=["V_98"]
        ).generate_diagrams(
            loops=_loops,
            max_vertices=1 + 2 * _loops,
            maximum_bridges=0,
            numerator_grouping=None,
            progress=None,
        )
        assert len(_generated.diagrams) == 1
        _diagram = _generated.diagrams[0]
        _operator = (
            model.expand_couplings(
                _diagram.numerator_expression(in_lmb=True).to_expression()
            )
            * _diagram.overall_factor_expression(evaluate=True)
            * _diagram.numerator_prefactor_expression()
        )
        for _edge in _diagram.external_edges:
            _representation = mink if _edge.external_index == 1 else bis
            _matches = list(
                _diagram.projector_expression().match(
                    wave(_edge.id, sp.PortPattern.exact(_representation(4), index)),
                    max_level=0,
                )
            )
            assert len(_matches) == 1
            _port = dict(_matches[0])[index]
            _operator = _operator.replace(
                sp.PortPattern.exact(_representation(4), _port),
                sp.PortPattern.exact(
                    _representation(4), {0: b, 1: mu, 2: a}[_edge.external_index]
                ),
            )
        operators.append(
            _operator.replace(
                sp.PortPattern.exact(sp.Representation.mink(4), index),
                sp.PortPattern.exact(sp.Representation.mink(dimension), index),
            )
        )
        diagrams.append(_diagram)
    return diagrams, operators


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Fix the overall normalization

    Use the generated tree coupling, including the fermion ordering sign.
    """)
    return


@app.cell
def _(E, Symbol, a, b, charge, dimension, gamma, mu, operators, sp):
    # Normalize to the generated tree vertex, including its external fermion sign.
    # A scalar loop contributes i/(16*pi^2) in the OneLOop convention below.
    tree_coupling = (
        operators[0]
        / gamma(
            sp.PortPattern.exact(sp.Representation.bis(4), a),
            sp.PortPattern.exact(sp.Representation.bis(4), b),
            sp.PortPattern.exact(sp.Representation.mink(dimension), mu),
        )
    ).together()
    assert (tree_coupling - Symbol.I * charge).expand() == E("0")
    vertex = (Symbol.I * operators[1] / (tree_coupling * charge**2)).expand()
    return (vertex,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Insert the external spin sums
    """)
    return


@app.cell
def _(P, a, b, c, d, dimension, index, model, slot, sp):
    particle = model.particle("e-")
    incoming = particle.spin_sum(P(0), b, c)
    outgoing = particle.spin_sum(P(2), d, a).replace(
        P(2, slot), P(0, slot) - P(1, slot)
    )
    incoming = incoming.replace(
        sp.PortPattern.exact(sp.Representation.mink(4), index),
        sp.PortPattern.exact(sp.Representation.mink(dimension), index),
    ).replace(
        sp.PortPattern.exact(sp.Representation.mink(4)),
        sp.PortPattern.exact(sp.Representation.mink(dimension)),
    )
    outgoing = outgoing.replace(
        sp.PortPattern.exact(sp.Representation.mink(4), index),
        sp.PortPattern.exact(sp.Representation.mink(dimension), index),
    ).replace(
        sp.PortPattern.exact(sp.Representation.mink(4)),
        sp.PortPattern.exact(sp.Representation.mink(dimension)),
    )
    return incoming, outgoing


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define the two form-factor structures

    The basis is $\gamma^\mu$ and $(p_{in}+p_{out})^\mu/(2m)$. The probes close their spinor indices.
    """)
    return


@app.cell
def _(P, a, b, c, d, dimension, gamma, mass, metric, mu, sp):
    vector_sum = 2 * P(
        0, sp.PortPattern.exact(sp.Representation.mink(dimension), mu)
    ) - P(1, sp.PortPattern.exact(sp.Representation.mink(dimension), mu))
    basis = [
        gamma(
            sp.PortPattern.exact(sp.Representation.bis(4), a),
            sp.PortPattern.exact(sp.Representation.bis(4), b),
            sp.PortPattern.exact(sp.Representation.mink(dimension), mu),
        ),
        vector_sum
        * metric(
            sp.PortPattern.exact(sp.Representation.bis(4), a),
            sp.PortPattern.exact(sp.Representation.bis(4), b),
        )
        / (2 * mass),
    ]
    probes = [
        gamma(
            sp.PortPattern.exact(sp.Representation.bis(4), c),
            sp.PortPattern.exact(sp.Representation.bis(4), d),
            sp.PortPattern.exact(sp.Representation.mink(dimension), mu),
        ),
        vector_sum
        * metric(
            sp.PortPattern.exact(sp.Representation.bis(4), c),
            sp.PortPattern.exact(sp.Representation.bis(4), d),
        )
        / (2 * mass),
    ]
    return basis, probes


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Evaluate the Dirac traces
    """)
    return


@app.cell
def _(TensorExpression, basis, incoming, kinematics, outgoing, probes, vertex):
    traces = []
    for _operator in basis + [vertex]:
        for _probe in probes:
            _trace = (
                TensorExpression((_operator * incoming * _probe * outgoing).expand())
                .simplify_algebra(contract="dots", gamma=True, epsilon=True)
                .expand()
                .to_expression()
            )
            traces.append(kinematics.apply(_trace).expand())
    return (traces,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Solve for the Pauli coefficient

    An independently specified Gram matrix checks the trace and basis normalization.
    """)
    return


@app.cell
def _(E, Matrix, dimension, mass, traces, transfer):
    # These independently specified traces also fix the normalization of the
    # projector basis gamma^mu and (p_in+p_out)^mu/(2m).
    expected_gram = [
        2 * (dimension - 2) * transfer + 8 * mass**2,
        8 * mass**2 - 2 * transfer,
        8 * mass**2 - 2 * transfer,
        (4 * mass**2 - transfer) ** 2 / (2 * mass**2),
    ]
    assert all(
        (actual - expected).expand() == E("0")
        for actual, expected in zip(traces[:4], expected_gram, strict=True)
    )
    projectors = Matrix.from_linear(2, 2, traces[:4])
    pauli_numerator = projectors.solve(Matrix.vec(traces[4:]))[1, 0].to_expression()
    return pauli_numerator, projectors


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Express the numerator in the integral family
    """)
    return


@app.cell
def _(S, diagrams, kinematics, pauli_numerator):
    family = diagrams[1].integral_family(kinematics=kinematics)
    denominators = S("d0", "d1", "d2")
    polynomial = family.rewrite_numerator(pauli_numerator, denominators).expand()
    terms = []
    for monomial, coefficient in polynomial.coefficient_list(*denominators):
        powers = [
            1 - monomial.to_polynomial(vars=denominators).degree(label)
            for label in denominators
        ]
        terms.append((powers, coefficient))
    return family, terms


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Reduce to scalar masters

    Identify equivalent tadpoles by their loop shifts before inserting the one-loop master values.
    """)
    return


@app.cell
def _(E, IBPFamily, Replacement, S, dimension, family, mass, terms, transfer):
    solution = IBPFamily(family, name="electron_vertex").reduce_laporta(
        [powers for powers, coefficient in terms], max_depth=2
    )
    assert {tuple(powers) for powers in solution.residuals} == {
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 0),
    }
    integral, tadpole, bubble = S("I", "A0", "B0")
    reduction = sum(
        (
            coefficient * solution.reduce(powers, integral=integral)
            for powers, coefficient in terms
        ),
        E("0"),
    )
    # Both one-line pinches are massive tadpoles; their loop shifts have unit Jacobian.
    # The two-line pinch carries q^2=t and equal masses m^2.
    reduction = reduction.replace_multiple(
        [
            Replacement(integral(0, 1, 0), tadpole),
            Replacement(integral(1, 0, 0), tadpole),
            Replacement(integral(1, 1, 0), bubble),
        ]
    ).together()
    expected = (
        2
        * (dimension - 5)
        * (-(dimension - 2) * tadpole + 2 * mass**2 * (dimension - 3) * bubble)
        / ((dimension - 3) * (transfer - 4 * mass**2))
    )
    assert (reduction - expected).together() == E("0")
    return bubble, reduction, solution, tadpole


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check UV finiteness

    Keep $D=4-2\epsilon$ through the Laurent expansion so dimension-dependent terms contribute correctly.
    """)
    return


@app.cell
def _(
    E,
    Replacement,
    S,
    bubble,
    dimension,
    mass,
    reduction,
    tadpole,
    transfer,
):
    epsilon, finite_a, finite_b, logarithm = S("eps", "Af", "Bf", "L")
    laurent = (
        reduction.replace(dimension, 4 - 2 * epsilon)
        .replace_multiple(
            [
                Replacement(tadpole, finite_a + mass**2 / epsilon),
                Replacement(bubble, finite_b + 1 / epsilon),
            ]
        )
        .series(epsilon, 0, 0)
        .to_expression()
        .expand()
    )
    assert laurent.coefficient(epsilon**-1).together() == E("0")
    finite = dict(laurent.coefficient_list(epsilon))[E("1")].together()
    assert (
        finite
        - 4 * (finite_a + mass**2 - mass**2 * finite_b) / (transfer - 4 * mass**2)
    ).together() == E("0")
    return finite, finite_a, finite_b, logarithm


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Take the static limit

    The Gordon decomposition converts the projected coefficient into $F_2(0)$.
    """)
    return


@app.cell
def _(
    E,
    Replacement,
    S,
    Symbol,
    finite,
    finite_a,
    finite_b,
    logarithm,
    mass,
    transfer,
):
    # Gordon decomposition gives F2=-b. Restoring e^2/(16*pi^2), the b=-2
    # limit gives F2(0)=e^2/(8*pi^2)=alpha/(2*pi), independently of m and mu.
    limit = (
        finite.replace(transfer, E("0"))
        .replace_multiple(
            [
                Replacement(finite_a, mass**2 * (1 - logarithm)),
                Replacement(finite_b, -logarithm),
            ]
        )
        .together()
    )
    assert limit == E("-2")

    alpha = S("alpha")
    anomalous_moment = (-alpha * limit / (4 * Symbol.PI)).together()
    return (anomalous_moment,)


@app.cell(hide_code=True)
def _(anomalous_moment, diagrams, family, mo, projectors, reduction, solution):
    mo.vstack(
        [
            mo.md("**Generated tree and one-loop vertex**"),
            mo.hstack(diagrams),
            mo.md("**Trace-projector matrix**"),
            projectors,
            mo.md("**Scalar integral family**"),
            family,
            mo.md("**Native IBP reduction of the projected coefficient**"),
            reduction.factor(),
            mo.ui.table([solution.stats], selection=None),
            mo.md(
                "The finite-depth residuals are two shifted massive tadpoles and "
                "the equal-mass bubble. Their identification follows directly from "
                "the displayed propagators; the solver does not certify a minimal basis."
            ),
            mo.md(r"""
            **Finite result:** with $A_f$ and $B_f$ the finite OneLOop coefficients,
            $b=4[A_f+m^2-m^2B_f]/(t-4m^2)$ and $F_2=-\alpha b/(4\pi)$.
            The $1/\epsilon$ pole cancels. Keeping $D=4-2\epsilon$ until after
            reduction retains finite terms from coefficients multiplying UV poles.

            At $t=0$, $A_f=m^2[1-\log(m^2/\mu^2)]$ and
            $B_f=-\log(m^2/\mu^2)$, so $b=-2$ independently of mass and scale.
            Since $a_e=(g-2)/2=F_2(0)$, the generated result is:
            """),
            anomalous_moment,
        ]
    )
    return


@app.cell
def _(mo):
    virtuality = mo.ui.slider(
        -3.0,
        2.0,
        step=0.25,
        value=0.0,
        label="log10(Q²/m²), where t = −Q²",
    )
    mo.vstack([virtuality])
    return (virtuality,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Evaluate at spacelike momentum
    """)
    return


@app.cell
def _(finite, finite_a, finite_b, mass, oneloop, transfer, virtuality):
    virtuality_ratio = 10.0**virtuality.value
    spacelike_transfer = -virtuality_ratio
    normalized_form_factor = (
        -complex(
            finite.evaluate(
                {
                    transfer: spacelike_transfer,
                    mass: 1.0,
                    finite_a: complex(oneloop.a0(1.0, 1.0)[0]),
                    finite_b: complex(oneloop.b0(spacelike_transfer, 1.0, 1.0, 1.0)[0]),
                }
            )
        )
        / 2
    )
    return normalized_form_factor, spacelike_transfer, virtuality_ratio


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Check an independent parameter integral
    """)
    return


@app.cell
def _(
    normalized_form_factor,
    np,
    oneloop,
    spacelike_transfer,
    virtuality_ratio,
):
    # Independent Feynman-parameter integral, without IBP coefficients or masters.
    _nodes, _weights = np.polynomial.legendre.leggauss(96)
    _x, _weights = (_nodes + 1) / 2, _weights / 2
    parameter_reference = float(
        np.dot(_weights, 1 / (1 + virtuality_ratio * _x * (1 - _x)))
    )
    _derivative = 1 + spacelike_transfer * complex(
        oneloop.db0(spacelike_transfer, 1.0, 1.0, 1.0)[0]
    )
    form_factor_error = abs(normalized_form_factor - parameter_reference)
    assert form_factor_error < 2e-12
    assert abs(normalized_form_factor - _derivative) < 2e-12
    return form_factor_error, parameter_reference


@app.cell(hide_code=True)
def _(
    form_factor_error,
    mo,
    normalized_form_factor,
    parameter_reference,
    virtuality_ratio,
):
    mo.vstack(
        [
            mo.md(r"""
            **Spacelike form factor beyond the zero-transfer limit**

            The independent parameter formula is
            $$\frac{F_2(t)}{\alpha/(2\pi)}
              =\int_0^1\!dx\,\frac{m^2}{m^2-tx(1-x)}
              =1+t\,\frac{dB_0(t,m^2,m^2)}{dt}.$$
            Move the slider to compare the generated reduction with quadrature
            and the shared OneLOop derivative. The normalized result approaches
            one as $Q^2/m^2\to0$.
            """),
            mo.ui.table(
                [
                    {
                        "Q²/m²": virtuality_ratio,
                        "Generated F₂ / (α/2π)": normalized_form_factor.real,
                        "Parameter integral": parameter_reference,
                        "Absolute error": form_factor_error,
                    }
                ],
                selection=None,
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
