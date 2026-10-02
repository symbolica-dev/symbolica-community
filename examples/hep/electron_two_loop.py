import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Two-loop electron self-energy")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Two-loop electron self-energy: Feynman gauge

    [Browse all notebooks](/) ·
    [Massive electron self-energy](/?file=hep/electron_self_energy.py) ·
    [Photon self-energy](/?file=hep/photon_self_energy.py) ·
    [Two-loop photon](/?file=hep/photon_two_loop.py) ·
    [Two-loop $\phi^4$](/?file=hep/ibp_phi4.py)

    Generate all three massless QED self-energy diagrams in **$\xi=1$**, apply
    the shared UV expansion, and run the native IBP solver. This reproduces the
    bare ultraviolet poles of the
    [FeynCalc example](https://feyncalc.github.io/FeynCalcExamples/QED/TwoLoops/Renormalization-LeAle-Massless).
    The photon-bubble diagram carries a symbolic flavor count $N_f$.

    With $d=4-2\epsilon$ and $a_4=e^2/(16\pi^2)$, the displayed pole coefficients
    multiply $i a_4^2\,\not p$. An auxiliary mass $M=m_{\rm UV}^2>0$ prevents
    infrared singularities while expanding through first order in $p$.
    The projector is $\operatorname{tr}(\not p\,\Gamma)/(4p^2)$; its temporary
    assumption $p^2\ne0$ drops out of the result.

    **Kernel convention.** Native scattering amplitudes include the external
    Wick-order factor for legs $[0,1]$. The conventional amputated kernel orders
    the fermion pair as $[\bar\psi,\psi]=[1,0]$. Remove only the named
    `ExternalFermionOrderingSign`; the closed-loop minus sign remains.
    Independently, the one-loop local rules give
    $-e^2\gamma_\mu\not k\gamma^\mu$ and hence
    $-e^2(2-d)B_0/2$, whose pole is $+e^2/\epsilon$ before the loop measure.

    Analytic vacuum poles and the one-loop counterterm insertion sum below are
    explicit reference inputs. This notebook does not generate a subtraction
    forest. Arbitrary $\xi$ and the finite off-shell self-energy remain outside
    this calculation.
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
    from symbolica import E, Replacement, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hepkit import (
        IBPFamily,
        IntegralFamily,
        Kinematics,
        Model,
        TensorReducer,
    )
    from symbolica.community.tensor import TensorExpression

    _set_namespace("electron_2l")
    return (
        E,
        IBPFamily,
        IntegralFamily,
        Kinematics,
        Model,
        Replacement,
        S,
        TensorExpression,
        TensorReducer,
        hep,
        mo,
        sp,
    )


@app.cell(hide_code=True)
def _():
    from symbolica.community.hepkit import Symbols

    return (Symbols,)


@app.cell(hide_code=True)
def _(
    IntegralFamily,
    TensorExpression,
    closed_loop,
    coordinates,
    d,
    denominator,
    dimension,
    edge_pattern,
    family,
    flavors,
    gamma,
    index,
    k,
    kinematics,
    mass,
    mass_pattern,
    mass_squared,
    momentum_pattern,
    mu,
    one,
    ordering,
    p,
    qed_model,
    quadratic_pattern,
    reducer,
    s,
    sp,
    uv_mass,
    vacuum_kinematics,
    value,
    wave,
    zero,
):
    def projected_vacuum_terms(diagram):
        """Project one UV-expanded self-energy and map its integrals to the common vacuum family."""
        raw_factor = diagram.overall_factor_expression()
        kernel_factor = raw_factor.replace(ordering(value), one)
        removed_ordering = (raw_factor / kernel_factor).replace(ordering(value), value)
        assert removed_ordering == -one
        # Evaluate every other native factor unchanged. In particular, a closed
        # fermion loop keeps its minus sign when external Wick ordering is removed.
        factor = diagram.overall_factor_expression(evaluate=True) / removed_ordering
        has_closed_loop = bool(raw_factor.matches(closed_loop(-1)))
        assert bool(kernel_factor.matches(closed_loop(-1))) == has_closed_loop
        assert factor == (-one if has_closed_loop else one)
        flavor_weight = flavors if has_closed_loop else one
        numerator = qed_model.expand_couplings(
            diagram.numerator_expression().to_expression()
        ).replace(mass, zero)
        uv = diagram.momentum_basis().route_expression(
            diagram.uv_expansion(uv_mass, numerator=numerator).to_expression()
        )
        uv = (
            uv.replace(
                sp.PortPattern.exact(sp.Representation.mink(dimension), index),
                sp.PortPattern.exact(sp.Representation.mink(d), index),
            )
            .replace(
                sp.PortPattern.exact(sp.Representation.mink(dimension)),
                sp.PortPattern.exact(sp.Representation.mink(d)),
            )
            .replace(uv_mass**2, mass_squared)
        )
        ports = {
            edge.external_index: dict(
                next(
                    diagram.projector_expression().match(
                        wave(
                            edge.id,
                            sp.PortPattern.exact(sp.Representation.bis(4), index),
                        ),
                        max_level=0,
                    )
                )
            )[index]
            for edge in diagram.external_edges
        }
        uv *= (
            gamma(
                sp.PortPattern.exact(sp.Representation.bis(4), ports[0]),
                sp.PortPattern.exact(sp.Representation.bis(4), ports[1]),
                sp.PortPattern.exact(sp.Representation.mink(d), mu),
            )
            * p(0, sp.PortPattern.exact(sp.Representation.mink(d), mu))
            / (4 * s)
        )
        pattern = denominator(
            edge_pattern, momentum_pattern, mass_pattern, quadratic_pattern
        )
        quadratics = {dict(match)[quadratic_pattern] for match in uv.match(pattern)}
        source_family = IntegralFamily(
            [k(0), k(1)], [], sorted(quadratics, key=str), kinematics=vacuum_kinematics
        )
        mapping = source_family.find_mapping(family)
        assert mapping is not None
        # Keep inverse denominators opaque during the polynomial tensor projection.
        for match in list(uv.match(pattern)):
            values = dict(match)
            formal = source_family.rewrite_numerator(
                values[quadratic_pattern], coordinates
            )
            assert formal in coordinates
            uv = uv.replace(
                denominator(
                    values[edge_pattern],
                    values[momentum_pattern],
                    values[mass_pattern],
                    values[quadratic_pattern],
                ),
                formal,
            )
        traced = (
            TensorExpression(uv.expand())
            .simplify_algebra(contract="dots", gamma=True, epsilon=True)
            .expand()
            .to_expression()
        )
        scalar = source_family.rewrite_numerator(
            kinematics.apply(reducer.reduce(traced)), coordinates
        )
        scalar *= factor * diagram.numerator_prefactor_expression()
        terms = []
        for monomial, coefficient in scalar.expand().coefficient_list(*coordinates):
            powers = [
                -int((monomial.derivative(den) * den / monomial).together())
                for den in coordinates
            ]
            assert (
                monomial
                == coordinates[0] ** -powers[0]
                * coordinates[1] ** -powers[1]
                * coordinates[2] ** -powers[2]
            )
            powers = mapping.map_powers(powers)

            terms.append((powers, coefficient))
        return terms, flavor_weight

    return (projected_vacuum_terms,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell
def _(Model):
    qed_model = Model.standard_model()
    return (qed_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Declare the regulator and tensor vocabulary
    """)
    return


@app.cell
def _(E, S, Symbols, hep, qed_model, sp):
    d, s, mass_squared, uv_mass, eps, log_mass, log_4pi, flavors, coupling = S(
        "D",
        "s",
        "M",
        "mUV",
        "eps",
        "Lm",
        "L4pi",
        "Nf",
        "a4",
    )
    k, p = hep.Kinematics.loop_momentum, hep.Kinematics.external_momentum
    _electron = qed_model.particle("e-")
    mass, charge = _electron.mass, _electron.electric_charge
    mink, bis, gamma, wave = (
        sp.Representation.mink,
        sp.Representation.bis,
        sp.TensorName.dirac_gamma().to_expression(),
        S("wave_"),
    )
    index, dimension, mu = S("index_", "dimension_", "mu")
    ordering, closed_loop, value = S(
        "feynkit_generator_factor::ExternalFermionOrderingSign",
        "feynkit_generator_factor::InternalFermionLoopSign",
        "value_",
    )
    denominator, edge_pattern, momentum_pattern, mass_pattern, quadratic_pattern = (
        Symbols.denominator,
        S("edge_"),
        S("momentum_"),
        S("mass_"),
        S("quadratic_"),
    )
    coordinates = list(S("d0", "d1", "d2"))
    tadpole_squared, sunset, integral = S("T2", "V", "I")
    zero, one, imaginary = E("0"), E("1"), E("1i")
    return (
        charge,
        closed_loop,
        coordinates,
        coupling,
        d,
        denominator,
        dimension,
        edge_pattern,
        eps,
        flavors,
        gamma,
        imaginary,
        index,
        integral,
        k,
        log_4pi,
        log_mass,
        mass,
        mass_pattern,
        mass_squared,
        momentum_pattern,
        mu,
        one,
        ordering,
        p,
        quadratic_pattern,
        s,
        sunset,
        tadpole_squared,
        uv_mass,
        value,
        wave,
        zero,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Build the common vacuum family

    The tensor reducer integrates both loop directions. All three diagrams will be mapped to the same scalar family.
    """)
    return


@app.cell
def _(IntegralFamily, Kinematics, TensorReducer, d, k, mass_squared, p, s, sp):
    kinematics = Kinematics(d, momenta=[k(0), k(1), p(0)]).with_scalar_product(
        p(0), p(0), s
    )
    vacuum_kinematics = Kinematics(d, momenta=[k(0), k(1)])
    family = IntegralFamily(
        [k(0), k(1)],
        [],
        [
            vacuum_kinematics.scalar_product(q, q) - mass_squared
            for q in (k(0), k(1), k(0) - k(1))
        ],
        kinematics=vacuum_kinematics,
    )
    reducer = TensorReducer(
        d,
        integrated=[
            k(0, sp.PortPattern.exact(sp.Representation.mink(d))),
            k(1, sp.PortPattern.exact(sp.Representation.mink(d))),
        ],
    )
    return family, kinematics, reducer, vacuum_kinematics


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## State the one-loop counterterm
    """)
    return


@app.cell
def _(eps, one):
    zpsi_one = -one / eps
    return (zpsi_one,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate the two-loop diagrams
    """)
    return


@app.cell
def _(qed_model):
    result = qed_model.process(["e-"], ["e-"], vertex_allow=["V_98"]).generate_diagrams(
        loops=2,
        max_vertices=4,
        maximum_bridges=0,
        self_energy=None,
        tadpoles=None,
        zero_snails=None,
        numerator_grouping=None,
        progress=None,
    )
    assert len(result.diagrams) == 3
    return (result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Project the UV expansion

    The folded routine performs the repeated per-diagram projection, preserves the closed-fermion-loop sign and returns exact integral powers.
    """)
    return


@app.cell
def _(flavors, projected_vacuum_terms, result):
    targets, diagram_integrals, flavor_weights = set(), [], []
    for _diagram in result.diagrams:
        _terms, _flavor_weight = projected_vacuum_terms(_diagram)
        diagram_integrals.append(_terms)
        flavor_weights.append(_flavor_weight)
        targets.update(tuple(powers) for powers, _ in _terms)
    assert flavor_weights.count(flavors) == 1
    assert len(targets) == 22
    return diagram_integrals, flavor_weights, targets


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Solve the shared IBP system
    """)
    return


@app.cell
def _(IBPFamily, family, k, targets):
    solution = IBPFamily(family, name="electron_two_loop").reduce_laporta(
        [list(target) for target in sorted(targets)], max_depth=2
    )
    assert solution.stats["rows"] > 0
    assert {tuple(powers) for powers in solution.residuals} == {
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    }
    # Verified unit-Jacobian maps relate the three disconnected tadpole products.
    for images in ([k(1), k(0)], [k(0), k(0) - k(1)]):
        assert family.mapping_to(family, images) is not None
    return (solution,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Identify the master basis
    """)
    return


@app.cell
def _(Replacement, integral, sunset, tadpole_squared):
    basis = {
        (0, 1, 1): tadpole_squared,
        (1, 0, 1): tadpole_squared,
        (1, 1, 0): tadpole_squared,
        (1, 1, 1): sunset,
    }
    basis_rules = [
        Replacement(integral(*powers), master) for powers, master in basis.items()
    ]
    # Explicit analytic inputs, not values inferred or certified by the IBP solver.
    return (basis_rules,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Supply the independent master poles
    """)
    return


@app.cell
def _(E, Replacement, eps, log_mass, mass_squared, sunset, tadpole_squared):
    master_poles = [
        Replacement(
            tadpole_squared, mass_squared**2 * (1 / eps**2 + 2 * (1 - log_mass) / eps)
        ),
        Replacement(
            sunset, mass_squared * (E("3/2") / eps**2 + (E("9/2") - 3 * log_mass) / eps)
        ),
    ]
    return (master_poles,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Assemble the bare UV pole
    """)
    return


@app.cell
def _(
    E,
    basis_rules,
    charge,
    d,
    diagram_integrals,
    eps,
    flavor_weights,
    flavors,
    imaginary,
    integral,
    log_4pi,
    log_mass,
    master_poles,
    one,
    solution,
    zero,
):
    bare_uv = zero
    bare_basis = zero
    for terms, flavor_weight in zip(diagram_integrals, flavor_weights, strict=True):
        reduced = sum(
            (
                coefficient * solution.reduce(powers, integral=integral)
                for powers, coefficient in terms
            ),
            zero,
        )
        reduced = reduced.replace_multiple(basis_rules).together()
        bare_basis += flavor_weight * reduced / (imaginary * charge**4)
        poles = reduced.replace_multiple(master_poles).replace(d, 4 - 2 * eps)
        # Each loop measure is i*(4pi)^(eps-2); divide by i*e^4/(16pi^2)^2.
        poles = (
            (-poles * (1 + 2 * eps * log_4pi) / (imaginary * charge**4))
            .series(eps, 0, -1)
            .to_expression()
        )
        bare_uv += flavor_weight * poles
    expected_bare = (
        one / (2 * eps**2)
        + (log_4pi - log_mass - E("17/12") - E("7/3") * flavors) / eps
    )
    assert (bare_uv - expected_bare).together() == zero

    # Reference one-loop CT insertion sum, in the same i*a4^2*slash(p) units.
    # Includes the auxiliary photon-mass CT: deltaZAm=-2*Nf/eps; the other
    # one-loop inputs are deltaZA=deltaZxi=-4*Nf/(3eps), deltaZe=2*Nf/(3eps),
    # deltaZpsi=-1/eps. Automatic forest/CT generation is not claimed here.
    return bare_basis, bare_uv


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Include subdivergences and verify the two-loop constant
    """)
    return


@app.cell
def _(
    E,
    bare_uv,
    coupling,
    eps,
    flavors,
    log_4pi,
    log_mass,
    one,
    zero,
    zpsi_one,
):
    counterterm_uv = (
        -one / eps**2 + (4 * flavors / 3 + E("2/3") - log_4pi + log_mass) / eps
    )
    zpsi_two = -(bare_uv + counterterm_uv).expand()
    assert zpsi_two.derivative(log_mass).expand() == zero
    assert zpsi_two.derivative(log_4pi).expand() == zero
    assert (
        zpsi_two - one / (2 * eps**2) - (4 * flavors + 3) / (4 * eps)
    ).together() == zero
    zpsi = 1 + coupling * zpsi_one + coupling**2 * zpsi_two
    assert (
        zpsi
        - 1
        + coupling / eps
        - coupling**2 * (one / (2 * eps**2) + (4 * flavors + 3) / (4 * eps))
    ).together() == zero
    return counterterm_uv, zpsi


@app.cell
def _(diagram_integrals, flavor_weights, flavors, mo, result):
    mo.vstack(
        [
            mo.md("## Generated diagrams"),
            mo.hstack(list(result.diagrams)),
            mo.ui.table(
                [
                    {
                        "Diagram": _diagram.name,
                        "Scalar terms": len(_terms),
                        "Flavor weight": str(_weight),
                        "Closed fermion loop": _weight == flavors,
                        "Native factor": str(_diagram.overall_factor_expression()),
                    }
                    for _diagram, _terms, _weight in zip(
                        result.diagrams, diagram_integrals, flavor_weights, strict=True
                    )
                ],
                selection=None,
            ),
        ]
    )
    return


@app.cell
def _(bare_basis, family, mo, solution, targets):
    mo.vstack(
        [
            mo.md("## Vacuum family and actual IBP reduction"),
            family,
            mo.md(
                f"The {len(targets)} scalar targets use a depth-two Laporta search "
                f"({solution.stats['rows']} rows from {solution.stats['seeds']} seeds). "
                "Its four residuals are **not certified masters**. Verified momentum "
                "maps identify three as the same disconnected tadpole product; "
                "the fourth is the equal-mass sunset."
            ),
            mo.ui.table(
                [{"Residual powers": str(_powers)} for _powers in solution.residuals],
                selection=None,
            ),
            mo.accordion(
                {
                    "Reduced scalar expression before loop measures (in i e⁴ units)": bare_basis.together()
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analytic inputs and normalization

    Define $L_M=\log M$ and $L_{4\pi}=\log(4\pi)$ in the reference's scale
    convention. For its normalized integrals, the supplied poles are
    $$T^2=M^2\left[\frac1{\epsilon^2}+\frac{2(1-L_M)}{\epsilon}\right],
    \qquad V=M\left[\frac{3}{2\epsilon^2}+\frac{9/2-3L_M}{\epsilon}\right].$$
    Each loop restores $i(4\pi)^{\epsilon-2}$. Keeping $d$ symbolic until after
    reduction retains the terms from $O(\epsilon)$ coefficients times double poles.

    The supplied one-loop insertion sum uses
    $\delta Z_\psi=-1/\epsilon$,
    $\delta Z_A=\delta Z_\xi=-4N_f/(3\epsilon)$,
    $\delta Z_e=2N_f/(3\epsilon)$ and the auxiliary photon-mass counterterm
    $\delta Z_{A m_{\rm UV}}=-2N_f/\epsilon$.
    In the same $i a_4^2\not p$ units it is
    $$-\frac1{\epsilon^2}+
    \frac{4N_f/3+2/3-L_{4\pi}+L_M}{\epsilon}.$$
    The auxiliary-mass and normalization logarithms cancel against the generated
    bare result.
    """)
    return


@app.cell
def _(bare_uv, counterterm_uv, mo, zpsi):
    mo.vstack(
        [
            mo.md("**Generated bare UV poles** (coefficient of $i a_4^2\\not p$)"),
            bare_uv.expand(),
            mo.md("**Reference one-loop counterterm insertion sum**"),
            counterterm_uv,
            mo.md("**Wave-function renormalization, ξ = 1**"),
            zpsi.expand(),
            mo.md(
                r"$$Z_\psi=1-\frac{a_4}{\epsilon} +a_4^2\left[\frac1{2\epsilon^2}+\frac{4N_f+3}{4\epsilon}\right].$$"
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
