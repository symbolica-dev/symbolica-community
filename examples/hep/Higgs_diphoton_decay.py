import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Higgs diphoton decay")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Higgs diphoton decay

    [Browse notebooks](/) · [Higgs to gluons](/?file=hep/higgs_gluons.py) ·
    [One-loop reduction](/?file=hep/oneloop_reduce.py)

    Compute the charged-fermion contribution to $H\to\gamma\gamma$ in two ways:
    first with symbolic tensor reduction and one-loop masters, then with a
    Monte Carlo integral over spatial loop momentum after the energy integration.
    Both routes end at the **same physical partial width**.

    We use the existing Standard Model top-quark vertices, with $y=m/v$ and
    $N_cQ^2=4/3$. This is the **top-loop contribution**, not the full Standard
    Model prediction including $W$ bosons. Setting the charge/color factor to one
    also gives a single unit-charge, color-singlet fermion with the same mass and
    Yukawa coupling. No model construction is needed.

    The numerical example stays below the pair threshold, $0<M<2m$, where the
    causal denominators have no threshold singularities. The inputs are illustrative
    leading-order parameters; the notebook does not perform a precision SM fit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup and notebook helpers

    The folded cells contain imports, repeated per-diagram preparation, energy
    substitution, and batch evaluation. Expand their code to inspect the details.
    The calculation below uses the native HEP operations explicitly.
    """)
    return


@app.cell(hide_code=True)
def _():
    import math
    import time
    import numpy as np
    import marimo as mo
    from symbolica import (
        E,
        S,
        Symbol,
        Replacement,
        NumericalIntegrator,
        PrintMode,
        AtomType,
    )
    from symbolica import set_namespace as _set_namespace
    from symbolica.community import hepkit as hep
    from symbolica.community import tensor
    from symbolica.community.hepkit import oneloop

    _set_namespace("Higgs_diphoton_decay")
    return (
        AtomType,
        E,
        NumericalIntegrator,
        PrintMode,
        Replacement,
        S,
        Symbol,
        hep,
        math,
        mo,
        np,
        oneloop,
        tensor,
        time,
    )


@app.cell(hide_code=True)
def _(E, Replacement, S, hep, oneloop, tensor):
    def prepared_numerator(diagram, model, dimension, indices, yukawa):
        """Retain native weights and aligned external ports; separate denominators."""
        amplitude = hep.Amplitude.from_diagram(diagram, dimension=dimension)
        edge, momentum, mass2, inverse = S("edge_", "momentum_", "mass2_", "inverse_")
        denominators = (
            diagram.denominator_expression(dimension=dimension, in_lmb=True)
            .to_expression()
            .replace(hep.Symbols.denominator(edge, momentum, mass2, inverse), inverse)
        )
        numerator = (amplitude.expression().to_expression() * denominators).together()
        numerator = numerator.replace(
            model.parameter("yt").symbol * E("1/2").sqrt(), yukawa
        )
        return tensor.TensorExpression(numerator).rename_indices(
            {leg.tensor_index: index for leg, index in zip(amplitude.legs[1:], indices)}
        )

    def scalar_contraction(expression, kinematics):
        """Contract a closed tensor, check its interface, and impose kinematics."""
        result = tensor.TensorExpression(expression).contract().to_dots()
        assert result.is_scalar
        return kinematics.apply(result).to_expression().together()

    def resolved_master_poles(reductions, invariant, mass):
        """Resolve only pole coefficients in the massive, subthreshold domain."""
        rules = []
        # Massive triangles are UV/IR finite; the logarithmic bubble residue is
        # independent of its masses and invariant. Specialize these pole-only calls
        # before formula construction to avoid opening unused triangle branches.
        assumptions = [Replacement(invariant, E("1")), Replacement(mass, E("2"))]
        for reduction in reductions:
            for _, master in reduction.terms:
                call = master.to_expression()
                coefficients = oneloop.master_coefficients(call)
                for position, power in ((1, -1), (2, -2)):
                    assert master.kind in ("bubble", "triangle")
                    residue = oneloop.get_expression(
                        call.replace_multiple(assumptions), coefficient=power
                    )
                    rules.append(Replacement(coefficients[position], residue))
        return rules

    def mass_shifted_reduction(family, four_numerator, dimension, shift):
        """Four-dimensional numerator coefficients with only propagator masses shifted."""
        shifted_family = hep.IntegralFamily(
            family.loop_momenta,
            family.external_momenta,
            [denominator + shift for denominator in family.denominators],
            kinematics=family.kinematics,
        )
        # The numerator was already generated and projected in four dimensions.
        # Embed its scalar dot products in the symbolic-D denominator family.
        embedded_numerator = four_numerator.replace(
            tensor.Representation.mink(4).to_expression(),
            tensor.Representation.mink(dimension).to_expression(),
        )
        return oneloop.reduce(shifted_family, [1, 1, 1], numerator=embedded_numerator)

    return (
        mass_shifted_reduction,
        prepared_numerator,
        resolved_master_poles,
        scalar_contraction,
    )


@app.cell(hide_code=True)
def _(E, Replacement, Symbol, math, np):
    def cff_triangle_density(cff, diagram, radius, cosine, higgs_mass, loop_mass):
        """Route generated CFF surfaces; return the OneLOop C0 density in d^3 k."""
        # Rest frame: P0=(M,0,0,0), P1=(M/2,0,0,M/2), P2=P0-P1.
        external = [
            (higgs_mass, E("0")),
            (higgs_mass / 2, higgs_mass / 2),
            (higgs_mass / 2, -higgs_mass / 2),
        ]
        energies, shifts = {}, {}
        for edge in diagram.edges:
            signature = edge.momentum_signature()
            shifts[edge.id] = sum(
                (c * p[0] for c, p in zip(signature.external, external)), E("0")
            )
            longitudinal = sum(
                (c * p[1] for c, p in zip(signature.external, external)), E("0")
            )
            loop_sign = signature.loops[0]
            energies[edge.id] = (
                loop_sign**2 * radius**2
                + 2 * loop_sign * radius * cosine * longitudinal
                + longitudinal**2
                + loop_mass**2
            ).sqrt()
        substitutions = []
        for surface in cff.surfaces:
            assert surface.kind in ("energy", "h")
            value = (
                sum((energies[i] for i in surface.positive_energies), E("0"))
                - sum((energies[i] for i in surface.negative_energies), E("0"))
                + sum((c * shifts[i] for i, c in surface.external_shift), E("0"))
            )
            substitutions.append(Replacement(E(surface.symbol_name), value))
        energy_product = math.prod(energies[e.id] for e in diagram.internal_edges)
        # Bare CFF denominators use positive energy sums. For three propagators,
        # the physical dk0/(2*pi) contour and spatial (2*pi)^-3 measure, divided
        # by i/(16*pi^2), give -1/(4*pi*E1*E2*E3). The vacuum check below fixes
        # this normalization independently of the Higgs amplitude.
        return -cff.to_expression().replace_multiple(substitutions) / (
            4 * Symbol.PI * energy_product
        )

    def batch_integrand(evaluator):
        """Adapt the Symbolica integrator's samples to its native batch evaluator."""

        def evaluate(samples):
            values = evaluator.evaluate(np.asarray([sample.c for sample in samples]))[
                :, 0
            ]
            assert np.isfinite(values).all()
            return values.tolist()

        return evaluate

    return batch_integrand, cff_triangle_density


@app.cell(hide_code=True)
def _(AtomType, PrintMode, math, mo, tensor):
    # Native tensor notation supplies indices, fractions, Greek letters and MathML.
    # Presentation aliases never enter the algebra or the numerical evaluators.
    def _print_equation(value, mode, **kwargs):
        if value.get_type() != AtomType.Fn:
            return None
        lhs, *parts = value
        if mode == PrintMode.Typst:
            body = [tensor.to_typst(part) for part in parts]
            rhs = body[0] if len(body) == 1 else f"frac({body[0]},{body[1]})"
            return tensor.to_typst(lhs) + " = " + rhs
        if mode == PrintMode.Latex:
            body = [part.to_latex().strip("$") for part in parts]
            rhs = body[0] if len(body) == 1 else rf"\frac{{{body[0]}}}{{{body[1]}}}"
            return lhs.to_latex().strip("$") + " = " + rhs
        return None

    # Symbol printers are immutable. Scope presentation aliases to this preamble
    # execution so rerunning it or editing a label cannot redefine a printer.
    # The registered callback stays alive, so its identity cannot be reused.
    _display_namespace = f"diphoton_notation_{id(_print_equation)}"
    _equation_head = tensor.TensorName(
        _display_namespace + "::equation", print=_print_equation
    ).to_expression()

    def paper_symbol(name, *, latex=None, typst=None):
        return tensor.TensorName(
            _display_namespace + "::" + name,
            print={"latex": latex or name, "typst": typst or name},
        ).to_expression()

    def equation(label, expression, *, rational=False):
        # The custom Symbolica equation printer consumes ordinary expressions.
        if isinstance(expression, tensor.TensorExpression):
            expression = expression.to_expression()
        # Keep integer coefficients in the denominator of rational scalars,
        # instead of displaying nested fractions such as (1/256 * ...)/(...).
        if rational:
            quotient = expression.to_rational_polynomial()
            parts = [
                quotient.numerator().to_expression(),
                quotient.denominator().to_expression(),
            ]
        else:
            parts = [expression]
        return tensor.formatted(
            _equation_head(
                paper_symbol(label) if isinstance(label, str) else label, *parts
            ),
            settings=tensor.DisplaySettings(index_style="raw", factor_gap="0.15em"),
        )

    def measurement(value, error):
        """Two significant uncertainty digits, with matching central-value precision."""
        decimals = max(0, 1 - math.floor(math.log10(error)))
        return rf"\left({value:.{decimals}f}\pm {error:.{decimals}f}\right)"

    mo.Html(tensor.load_math_font())
    return equation, measurement, paper_symbol


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate the two loop orientations

    Select the existing $t\bar tH$ and $t\bar t\gamma$ vertices by particle content.
    The generated graphs keep the fermion-loop sign, symmetry weights and routing.
    We retain both orientations and compare their reduced answers before adding them.
    """)
    return


@app.cell
def _(hep, mo):
    model = hep.Model.standard_model()
    top, higgs, photon = [model.particle(name) for name in ("t", "H", "a")]
    vertices = [
        vertex
        for vertex in model.vertex_rules
        if sorted(vertex.particles)
        in [sorted([top.name, top.antiname, boson.name]) for boson in (higgs, photon)]
    ]
    generated = model.process(
        [higgs], [photon, photon], vertex_allow=vertices
    ).generate_diagrams(
        loops=1,
        max_vertices=3,
        maximum_bridges=0,
        numerator_grouping=None,
        progress=None,
    )
    assert len(vertices) == len(generated.diagrams) == 2
    mo.hstack([mo.Html(d.to_html(momenta=True)) for d in generated.diagrams])
    return generated, model, photon, top


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Keep the loop dimension symbolic

    The Higgs momentum is $P_0=p+q$, with $p=P_1$, $p^2=q^2=0$ and
    $s=P_0^2=2p\cdot q$. Bispinor traces remain four-dimensional in size,
    $\operatorname{tr}\mathbf1=4$; Lorentz contractions use $D=4-2\epsilon$.
    The numerator is stripped of the common $N_cQ^2e^2ym$ only after its native
    Dirac and color algebra has been evaluated.
    """)
    return


@app.cell
def _(E, S, generated, hep, model, prepared_numerator, tensor, top):
    D, s, eps, y = S("D", "s", "eps", "y")
    mu, nu = S("mu", "nu")
    m = top.mass
    e = model.parameter("ee").symbol
    K, P = hep.Kinematics.loop_momentum, hep.Kinematics.external_momentum
    lorentz = tensor.Representation.mink(D)
    kinematics = (
        hep.Kinematics(D, momenta=[K(0), P(0), P(1)])
        .with_scalar_product(P(0), P(0), s)
        .with_scalar_product(P(1), P(1), E("0"))
        .with_scalar_product(P(0), P(1), s / 2)
    )
    numerators = [
        prepared_numerator(d, model, D, (mu, nu), y) for d in generated.diagrams
    ]
    algebra = dict(
        gamma=True, color=True, color_substitute_cof_dimension_invariants=True
    )
    traces = [n.simplify_algebra(contract="dots", **algebra) for n in numerators]
    normalization = 3 * top.charge**2 * e**2 * y * m
    return (
        D,
        K,
        P,
        algebra,
        eps,
        kinematics,
        lorentz,
        m,
        mu,
        normalization,
        nu,
        s,
        traces,
        y,
    )


@app.cell(hide_code=True)
def _(Replacement, S, m, mo, mu, normalization, nu, tensor, traces):
    # Use physical glyphs only in this presentation copy of the computed tensor.
    _numerator_display = (
        (traces[0] / normalization)
        .to_expression()
        .expand()
        .replace_multiple(
            [
                Replacement(mu, S("μ")),
                Replacement(nu, S("ν")),
                Replacement(m, S("m")),
            ]
        )
    )
    mo.accordion(
        {
            "Computed Dirac/color numerator · first orientation": tensor.formatted(
                _numerator_display,
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Project the physical form factor

    The transverse projector is

    $$
    P_\perp^{\mu\nu}=g^{\mu\nu}-\frac{p^\mu q^\nu+q^\mu p^\nu}{p\cdot q},
    \qquad P_{\perp\,\mu\nu}g^{\mu\nu}=D-2.
    $$

    Hence the coefficient of $g^{\mu\nu}$ seen by physical photons is
    $P_{\perp\,\mu\nu}N^{\mu\nu}/(D-2)$.

    A contraction with $g_{\mu\nu}/4$ is **not** this projection: longitudinal
    terms can vanish against physical polarizations and still have a nonzero metric
    trace. Likewise $k^\mu k^\nu\to g^{\mu\nu}k^2/D$ is only a vacuum average.
    Here `TensorReducer` must retain the two independent external vectors:

    $$
    k_\perp^2=k^2-\frac{4(k\cdot p)(k\cdot q)}s,\qquad
    P_{\perp\,\mu\nu}k^\mu k^\nu=k_\perp^2.
    $$

    In particular, the projected $4k^\mu k^\nu-g^{\mu\nu}k^2$ contributes
    $4k_\perp^2/(D-2)-k^2$, rather than zero.
    """)
    return


@app.cell
def _(
    D,
    K,
    P,
    hep,
    kinematics,
    lorentz,
    mu,
    normalization,
    nu,
    s,
    scalar_contraction,
    tensor,
    traces,
):
    _momentum = tensor.TensorName(P.get_name())
    mu_slot, nu_slot = lorentz(mu), lorentz(nu)
    g_metric = lorentz.g(mu, nu)
    p_mu, p_nu = _momentum(1, mu_slot), _momentum(1, nu_slot)
    q_mu, q_nu = _momentum(0, mu_slot) - p_mu, _momentum(0, nu_slot) - p_nu
    transverse_metric = g_metric - 2 * (p_mu * q_nu + q_mu * p_nu) / s
    projector = transverse_metric / (D - 2)
    assert scalar_contraction(transverse_metric * g_metric, kinematics) == D - 2
    assert scalar_contraction(projector * (p_mu * q_nu), kinematics) == 0
    assert scalar_contraction(g_metric * (p_mu * q_nu) / D, kinematics) == s / (2 * D)
    reducer = hep.TensorReducer(
        D,
        integrated=[K(0, tensor.PortPattern.exact(lorentz))],
        external=[
            P(0, tensor.PortPattern.exact(lorentz)),
            P(1, tensor.PortPattern.exact(lorentz)),
        ],
    )
    # TensorReducer currently accepts and returns ordinary Symbolica expressions.
    reduced_tensors = [
        tensor.TensorExpression(reducer.reduce(trace.to_expression()))
        for trace in traces
    ]
    projected_numerators = [
        (scalar_contraction(value * projector, kinematics) / normalization).together()
        for value in reduced_tensors
    ]
    return (
        g_metric,
        p_mu,
        p_nu,
        projected_numerators,
        q_mu,
        q_nu,
        reduced_tensors,
        transverse_metric,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reduce the generated integral families

    `integral_family()` reads the propagators and their routing from each graph.
    `oneloop.reduce()` converts the projected scalar numerators into master
    integrals while preserving their exact dimension dependence. No handwritten
    Passarino–Veltman or integration-by-parts rules are needed.
    """)
    return


@app.cell
def _(E, generated, kinematics, oneloop, projected_numerators):
    families = [
        diagram.integral_family(kinematics=kinematics) for diagram in generated.diagrams
    ]
    reductions = [
        oneloop.reduce(family, [1, 1, 1], numerator=numerator)
        for family, numerator in zip(families, projected_numerators)
    ]
    assert (
        reductions[0].to_expression() - reductions[1].to_expression()
    ).together() == 0
    coefficient_D = sum((r.to_expression() for r in reductions), E("0")).together()
    # Define K by the projected coefficient = -4 Nc Q^2 e^2 y m K.
    K_D = (-coefficient_D / 4).together()
    return K_D, families, reductions


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check gauge invariance before taking the limit

    The two independent contractions of each Ward vector vanish after integration.
    These tests use the generated numerator and masters, before restricting to
    physical photon polarizations or inserting a known form factor.
    """)
    return


@app.cell
def _(
    E,
    families,
    kinematics,
    normalization,
    oneloop,
    p_mu,
    p_nu,
    q_mu,
    q_nu,
    reduced_tensors,
    scalar_contraction,
):
    ward_residuals = []
    for _ward in (p_mu * p_nu, p_mu * q_nu, q_mu * q_nu):
        _sum = E("0")
        for _family, _value in zip(families, reduced_tensors):
            _numerator = scalar_contraction(_value * _ward, kinematics) / normalization
            _sum += oneloop.reduce(
                _family, [1, 1, 1], numerator=_numerator
            ).to_expression()
        ward_residuals.append(_sum.together())
    assert ward_residuals == [E("0")] * 3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Keep the finite term from dimensional reduction

    Abbreviate $B_0=B_0(s;m^2,m^2)$ and
    $C_0=C_0(0,s,0;m^2,m^2,m^2)$, with the native master normalization.
    The equations below are rendered from the calculated coefficients; the short
    master labels are presentation aliases only.

    The bubble coefficient is $2\epsilon+O(\epsilon^2)$ and its UV residue is one.
    Its finite contribution is **2**, even though the amplitude has no UV pole.
    Setting $D=4$ before combining master coefficients loses it.
    `reduction_coefficients()` retains this contribution automatically.
    """)
    return


@app.cell
def _(D, E, K_D, eps, m, oneloop, reductions, resolved_master_poles, s):
    B0 = oneloop.B0(s, m**2, m**2, 1)
    C0 = oneloop.C0(0, s, 0, m**2, m**2, m**2, 1)
    assert (
        K_D - 2 * (4 - D) / (D - 2) * B0 - (8 * m**2 / (D - 2) - s) * C0
    ).together() == 0
    bubble_linear = (
        K_D.expand()
        .coefficient(B0)
        .replace(D, 4 - 2 * eps)
        .series(eps, 0, 1)
        .to_expression()
        .expand()
        .coefficient(eps)
    )
    assert bubble_linear == 2
    pole_rules = resolved_master_poles(reductions, s, m)
    laurent_K = [
        (-sum((oneloop.reduction_coefficients(r)[i] for r in reductions), E("0")) / 4)
        .replace_multiple(pole_rules)
        .together()
        for i in range(3)
    ]
    K_finite, simple_pole, double_pole = laurent_K
    assert simple_pole == double_pole == 0
    C0_finite = oneloop.master_coefficients(C0)[0]
    triangle_weight = K_finite.expand().coefficient(C0_finite).together()
    rational_term = (K_finite - triangle_weight * C0_finite).together()
    assert (triangle_weight - (4 * m**2 - s)).expand() == 0 and rational_term == 2
    return B0, C0, C0_finite, K_finite, rational_term, triangle_weight


@app.cell(hide_code=True)
def _(
    B0,
    C0,
    C0_finite,
    D,
    E,
    K_D,
    K_finite,
    Replacement,
    S,
    equation,
    m,
    mo,
    paper_symbol,
):
    master_notation = [
        Replacement(m, S("m")),
        Replacement(C0_finite, paper_symbol("C_0")),
    ]
    mo.vstack(
        [
            equation(
                "K_D",
                sum(
                    (
                        (
                            K_D.expand().coefficient(master).factor()
                            if label == "B_0"
                            else K_D.expand().coefficient(master).apart(D)
                        )
                        * paper_symbol(label)
                        for master, label in [(B0, "B_0"), (C0, "C_0")]
                    ),
                    E("0"),
                ).replace(m, S("m")),
            ),
            equation(
                "K",
                K_finite.expand().collect(C0_finite).replace_multiple(master_notation),
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Rational terms: separate $R_2$ and $R_1$

    Use the definitions of [Ossola, Papadopoulos and Pittau,
    Eqs. (3)–(4), (11)–(12) and (16)–(18)](https://arxiv.org/abs/0802.1876).
    Their $n=4+\epsilon_{\rm OPP}$ is our $D=4-2\epsilon$.
    With four-dimensional external photon states, split

    $$
    \bar k=k+\widetilde k,\quad t=\widetilde k^2=-\mu^2,\quad
    \bar D_i=D_i+t,\quad \bar N(\bar k)=N(k)+\widetilde N(k,t).
    $$

    $R_2$ comes from $\widetilde N$; $R_1$ comes from reducing $N(k)$ using
    four-dimensional denominators while the integration denominators are
    $\bar D_i$. These labels refer to contributions to our stripped, projected
    $K$. They are not separately gauge-invariant tensor amplitudes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Split the representation, then the generated numerator.** Declare a Minkowski
    space of dimension 4 and an orthogonal Euclidean block of dimension $n_\epsilon$.
    Its positive norm is $\mu^2$; the Lorentzian embedding supplies the minus sign:

    $$
    g_D=g_4\oplus(-\delta_\epsilon),\qquad
    \bar k^2=k_4^2-\mu^2,\qquad n_\epsilon=D-4=-2\epsilon.
    $$

    The representation dimension stays symbolic until after contraction. The
    identity traces below are computed by `contract()`. External momenta and photon
    indices have only four-dimensional components. We impose this physical embedding
    explicitly; different dimension labels alone do not establish orthogonality.
    """)
    return


@app.cell
def _(
    D,
    E,
    K,
    P,
    Replacement,
    S,
    g_metric,
    hep,
    kinematics,
    lorentz,
    mu,
    nu,
    s,
    scalar_contraction,
    tensor,
    transverse_metric,
):
    n_epsilon = S("n_epsilon")
    four_space = tensor.Representation.mink(4)
    epsilon_space = tensor.Representation.euc(n_epsilon)
    four_dimension = four_space.id("a", "a").contract().to_expression()
    epsilon_dimension = epsilon_space.id("a", "a").contract().to_expression()
    epsilon_loop = tensor.TensorName.vector("ell_epsilon")(epsilon_space)
    mu_squared = tensor.dot(epsilon_loop, epsilon_loop).to_expression()
    four_kinematics = (
        hep.Kinematics(four_dimension, momenta=[K(0), P(0), P(1)])
        .with_scalar_product(P(0), P(0), s)
        .with_scalar_product(P(1), P(1), E("0"))
        .with_scalar_product(P(0), P(1), s / 2)
    )
    _index = S("physical_index_")
    physical_slots = [
        Replacement(
            lorentz(_index).to_expression(), four_space(_index).to_expression()
        ),
        Replacement(lorentz.to_expression(), four_space.to_expression()),
    ]
    _momentum = tensor.TensorName(P.get_name())
    _p_mu, _p_nu = _momentum(1, four_space(mu)), _momentum(1, four_space(nu))
    _q_mu = _momentum(0, four_space(mu)) - _p_mu
    _q_nu = _momentum(0, four_space(nu)) - _p_nu
    four_g = four_space.g(mu, nu)
    four_transverse_metric = four_g - 2 * (_p_mu * _q_nu + _q_mu * _p_nu) / s
    four_transverse_dimension = scalar_contraction(
        four_transverse_metric * four_g, four_kinematics
    )
    four_projector = four_transverse_metric / four_transverse_dimension
    full_transverse_dimension = scalar_contraction(
        transverse_metric * g_metric, kinematics
    )
    assert four_dimension == 4 and four_transverse_dimension == 2
    assert (full_transverse_dimension - four_transverse_dimension).replace(
        D, four_dimension + n_epsilon
    ).expand() == epsilon_dimension
    return (
        epsilon_dimension,
        four_dimension,
        four_g,
        four_kinematics,
        four_projector,
        four_space,
        full_transverse_dimension,
        mu_squared,
        n_epsilon,
        physical_slots,
    )


@app.cell(hide_code=True)
def _(D, epsilon_dimension, equation, four_dimension, mo, n_epsilon):
    mo.hstack(
        [
            equation("d_4", four_dimension),
            equation(
                "d_epsilon", epsilon_dimension.replace(n_epsilon, D - four_dimension)
            ),
        ],
        justify="center",
        gap=4,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Generate the same two numerators directly in the four-dimensional representation
    and run their gamma/color algebra again. In the already traced $D$-dimensional
    numerators, split $\bar k^2$ into the two component norms and restrict the
    remaining Lorentz ports to the physical block. Subtracting the independent
    four-dimensional calculation gives $\widetilde N$, with its tensor structure
    and coefficient determined by Symbolica.
    """)
    return


@app.cell
def _(
    K,
    algebra,
    equation,
    four_dimension,
    four_g,
    four_kinematics,
    generated,
    kinematics,
    model,
    mu,
    mu_squared,
    normalization,
    nu,
    paper_symbol,
    physical_slots,
    prepared_numerator,
    traces,
    y,
):
    four_traces = [
        prepared_numerator(
            diagram, model, four_dimension, (mu, nu), y
        ).simplify_algebra(contract="dots", **algebra)
        for diagram in generated.diagrams
    ]
    k2 = kinematics.scalar_product(K(0), K(0))
    four_k2 = four_kinematics.scalar_product(K(0), K(0))
    split_traces = [
        trace.to_expression()
        .replace(k2, four_k2 - mu_squared)
        .replace_multiple(physical_slots)
        for trace in traces
    ]
    epsilon_numerators = [
        ((split - four.to_expression()) / normalization).expand()
        for split, four in zip(split_traces, four_traces)
    ]
    # A tensor equality checks the complete split, not just one scalar projection.
    assert all(
        (value + 4 * mu_squared * four_g.to_expression()).expand() == 0
        for value in epsilon_numerators
    )
    equation(
        paper_symbol(
            "N_epsilon_ports", latex=r"N_{\epsilon}^{\mu\nu}", typst="N_epsilon^(mu nu)"
        ),
        epsilon_numerators[0].replace(
            mu_squared, paper_symbol("mu_squared", latex=r"\mu^2", typst="mu^2")
        ),
    )
    return epsilon_numerators, four_traces, k2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The displayed $N_\epsilon$ is for one orientation, stripped of
    $N_cQ^2e^2ym$; both orientations are retained below.

    **Compute the extra-dimensional integral.** Rotational invariance of the
    transverse integration gives the *integrated* identity

    $$
    \int\frac{\widetilde k^2}{\bar D_0\bar D_1\bar D_2}
    =\frac{\operatorname{tr}1_\epsilon}{\operatorname{tr}P_\perp}
    \int\frac{k_\perp^2}{\bar D_0\bar D_1\bar D_2}.
    $$

    The ratio is built from the contracted representations above. This is not a
    pointwise replacement. Reduce it before expanding in $\epsilon$, keeping its
    vanishing dimension factor until it multiplies the UV pole. We use the
    $C_0$ measure for $I_t=\int d^D\bar k/(i\pi^{D/2})\,
    \widetilde k^2/(\bar D_0\bar D_1\bar D_2)$, with scale factors approaching one.
    """)
    return


@app.cell
def _(
    D,
    E,
    K,
    P,
    epsilon_dimension,
    equation,
    families,
    four_dimension,
    full_transverse_dimension,
    k2,
    kinematics,
    m,
    n_epsilon,
    oneloop,
    resolved_master_poles,
    s,
):
    kp = kinematics.scalar_product(K(0), P(1))
    kq = kinematics.scalar_product(K(0), P(0)) - kp
    kperp2 = k2 - 4 * kp * kq / s
    dimensional_weight = (epsilon_dimension / full_transverse_dimension).replace(
        n_epsilon, D - four_dimension
    )
    tilde_reduction = oneloop.reduce(
        families[0], [1, 1, 1], numerator=dimensional_weight * kperp2
    )
    I_t = (
        oneloop.reduction_coefficients(tilde_reduction)[0]
        .replace_multiple(resolved_master_poles([tilde_reduction], s, m))
        .together()
    )
    assert I_t == -E("1/2")
    equation("I_t", I_t)
    return (I_t,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Calculate $R_2$.** Contract the computed $N_\epsilon^{\mu\nu}$ with the
    physical projector. Since $t=-\mu^2$, the coefficient of $t$ is minus the
    coefficient of the Euclidean norm. Multiply it by the computed $I_t$, sum the
    two orientations, and apply the same $-1/4$ normalization that defines $K$.
    """)
    return


@app.cell
def _(
    E,
    I_t,
    epsilon_numerators,
    equation,
    four_kinematics,
    four_projector,
    mu_squared,
    scalar_contraction,
):
    evanescent_coefficients = [
        (
            -scalar_contraction(value * four_projector, four_kinematics) / mu_squared
        ).together()
        for value in epsilon_numerators
    ]
    assert evanescent_coefficients == [E("4"), E("4")]
    R2 = (-sum(evanescent_coefficients, E("0")) * I_t / 4).together()
    equation("R_2", R2)
    return (R2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Calculate $R_1$ by shifting the denominator masses.** In the
    four-dimensional numerator reduction, hold the Yukawa and numerator mass
    fixed and use $m_{i,\rm prop}^2\to m^2-t$ in every propagator. This is
    $\bar D_i=D_i+t$, not a mass shift of the whole amplitude.

    `oneloop.reduce()` then finds the shifted triangle coefficient $c_0(t)$.
    Its linear coefficient multiplies $I_t$; all bubble and tadpole coefficients
    vanish in this example. First `TensorReducer` projects the independently
    generated four-dimensional numerators. The later substitution $D=4$ selects
    the four-dimensional master coefficients of the shifted family; it never
    changes the regulated result.
    """)
    return


@app.cell
def _(
    K,
    P,
    four_dimension,
    four_kinematics,
    four_projector,
    four_space,
    four_traces,
    hep,
    normalization,
    scalar_contraction,
    tensor,
):
    four_reducer = hep.TensorReducer(
        four_dimension,
        integrated=[K(0, tensor.PortPattern.exact(four_space))],
        external=[
            P(0, tensor.PortPattern.exact(four_space)),
            P(1, tensor.PortPattern.exact(four_space)),
        ],
    )
    four_projected_numerators = [
        (
            scalar_contraction(
                four_reducer.reduce(trace.to_expression()) * four_projector,
                four_kinematics,
            )
            / normalization
        ).together()
        for trace in four_traces
    ]
    return (four_projected_numerators,)


@app.cell
def _(
    D,
    E,
    I_t,
    R2,
    S,
    equation,
    families,
    four_projected_numerators,
    m,
    mass_shifted_reduction,
    mo,
    paper_symbol,
    rational_term,
):
    tilde_k2 = S("t")
    shifted_reductions = [
        mass_shifted_reduction(family, numerator, D, tilde_k2)
        for family, numerator in zip(families, four_projected_numerators)
    ]

    shifted_triangle_weight = (
        (
            -sum(
                (
                    coefficient
                    for reduction in shifted_reductions
                    for coefficient, master in reduction.terms
                    if master.kind == "triangle"
                ),
                E("0"),
            )
            / 4
        )
        .replace(D, 4)
        .together()
    )
    assert all(
        coefficient.replace(D, 4).together() == 0
        for reduction in shifted_reductions
        for coefficient, master in reduction.terms
        if master.kind != "triangle"
    )
    R1 = (shifted_triangle_weight.expand().coefficient(tilde_k2) * I_t).together()
    assert R1 == R2 == 1 and R1 + R2 == rational_term
    mo.vstack(
        [
            equation(
                paper_symbol("c_shifted", latex=r"c_0(t)", typst="c_0(t)"),
                shifted_triangle_weight.expand().replace(m, S("m")),
            ),
            equation("R_1", R1),
        ]
    )
    return (R1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two independent extractions give $R_1=1$, $R_2=1$, hence
    $K=(4m^2-s)C_0+R_1+R_2$. In particular, **the earlier finite bubble
    contribution 2 is their sum, not $R_2$ alone**. The explicit split checks
    both the tensor projection and the sign convention $t=-\mu^2$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Turn the amplitude into a decay width

    With $y=m/v$, write

    $$
    \mathcal M=-\frac{\alpha N_cQ^2}{2\pi v}A_{1/2}
     T^{\mu\nu}\epsilon^*_{1\mu}\epsilon^*_{2\nu},\quad
     T^{\mu\nu}=\frac{s}{2}g^{\mu\nu}-q^\mu p^\nu,\quad
     A_{1/2}=\frac{4m^2}{s}K.
    $$

    Use the native physical photon spin sums, with each photon as the other's null
    reference. The Higgs has no spin average. The identical-photon factor $1/2!$
    belongs in the final two-body phase space, once.
    """)
    return


@app.cell
def _(
    E,
    P,
    S,
    hep,
    mu,
    nu,
    photon,
    s,
    scalar_contraction,
    tensor,
):
    rho, sigma = S("rho", "sigma")
    physical_kinematics = (
        hep.Kinematics(momenta=[P(1), P(2)])
        .with_scalar_product(P(1), P(1), E("0"))
        .with_scalar_product(P(2), P(2), E("0"))
        .with_scalar_product(P(1), P(2), s / 2)
    )
    physical_lorentz = tensor.Representation.mink(4)
    physical_mu, physical_nu = physical_lorentz(mu), physical_lorentz(nu)
    _momentum = tensor.TensorName(P.get_name())
    physical_tensor = s / 2 * physical_lorentz.g(mu, nu) - _momentum(
        2, physical_mu
    ) * _momentum(1, physical_nu)
    conjugate_tensor = physical_tensor.rename_indices({mu: rho, nu: sigma})
    spin_sum = photon.spin_sum(P(1), mu, rho, reference=P(2)) * photon.spin_sum(
        P(2), nu, sigma, reference=P(1)
    )
    tensor_norm = scalar_contraction(
        physical_tensor * conjugate_tensor * spin_sum, physical_kinematics
    )
    assert tensor_norm == s**2 / 2
    return (
        physical_kinematics,
        physical_mu,
        physical_nu,
        physical_tensor,
        tensor_norm,
    )


@app.cell(hide_code=True)
def _(
    P,
    equation,
    paper_symbol,
    physical_mu,
    physical_nu,
    physical_tensor,
    tensor,
):
    _momentum = tensor.TensorName(P.get_name())
    _physical_notation = [
        tensor.TensorRule(_momentum(i, slot), tensor.TensorName.vector(name)(slot))
        for i, name in [(1, "p"), (2, "q")]
        for slot in [physical_mu, physical_nu]
    ]
    equation(
        paper_symbol("T_ports", latex=r"T^{\mu\nu}", typst="T^(mu nu)"),
        physical_tensor.replace(_physical_notation),
    )
    return


@app.cell
def _(P, S, Symbol, physical_kinematics, s, tensor_norm):
    M, v = S("M", "v", is_positive=True)
    alpha, charge_color, A, Abar = S("alpha", "NcQ2", "A", "Abar")
    decay_kinematics = physical_kinematics.with_scalar_product(P(0), P(0), M**2)
    phase_space = decay_kinematics.two_body_phase_space(P(1), P(2)).expand()
    flux = decay_kinematics.flux(P(0))
    squared_amplitude = (
        tensor_norm * (alpha * charge_color / (2 * Symbol.PI * v)) ** 2 * A * Abar
    )
    width = (
        (squared_amplitude * 4 * Symbol.PI * phase_space / flux / 2)
        .replace(s, M**2)
        .together()
    )
    assert (
        width
        - alpha**2 * charge_color**2 * M**3 * A * Abar / (256 * Symbol.PI**3 * v**2)
    ).together() == 0
    return A, Abar, M, alpha, charge_color, v, width


@app.cell(hide_code=True)
def _(A, Abar, alpha, charge_color, equation, paper_symbol, width):
    width_display = (
        width.replace(
            A * Abar,
            paper_symbol("abs_A2", latex=r"|A_{1/2}|^2", typst='abs(A_("1/2"))^2'),
        )
        .replace(charge_color, paper_symbol("N_c") * paper_symbol("Q") ** 2)
        .replace(alpha, paper_symbol("alpha", latex=r"\alpha"))
    )
    equation("Gamma", width_display, rational=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluate the analytic answer

    The form factor below is built from the generated numerator's **computed finite
    master reduction**. `compile_native()` evaluates those native masters directly,
    including their analytic continuation and normalization. No closed-form Higgs
    amplitude is supplied to the evaluator.

    The benchmark uses the fermion contribution and conventions of
    [Djouadi, section 2.3.1](https://arxiv.org/abs/hep-ph/0503172).
    Independent Feynman-parameter quadrature and explicit Dirac matrices in the
    regression tests check the normalization, mass dependence, and rational split.
    """)
    return


@app.cell
def _():
    parameters = {
        "higgs_mass": 125.0,
        "fermion_mass": 173.0,
        "vev": 246.22,
        "alpha": 1 / 137.035999084,
        "charge_color": 4 / 3,
        "seed": 2026,
        "iterations": 8,
        "samples_per_iteration": 20000,
    }
    return (parameters,)


@app.cell
def _(
    A,
    Abar,
    K_finite,
    M,
    alpha,
    charge_color,
    m,
    oneloop,
    parameters,
    s,
    v,
    width,
):
    assert 0 < parameters["higgs_mass"] < 2 * parameters["fermion_mass"]
    analytic_form_factor = 4 * m**2 / s * K_finite
    master_evaluator = oneloop.compile_native([analytic_form_factor], [s, m])
    point = {s: parameters["higgs_mass"] ** 2, m: parameters["fermion_mass"]}
    analytic_A = complex(
        master_evaluator.evaluate_complex([complex(point[s]), complex(point[m])])[0, 0]
    )
    _prefactor = complex(
        (width / (A * Abar))
        .together()
        .evaluate(
            {
                M: parameters["higgs_mass"],
                v: parameters["vev"],
                alpha: parameters["alpha"],
                charge_color: parameters["charge_color"],
            }
        )
    )
    assert abs(_prefactor.imag) < 1e-20
    width_prefactor = _prefactor.real
    analytic_width = width_prefactor * abs(analytic_A) ** 2
    return analytic_A, analytic_width, point, width_prefactor


@app.cell(hide_code=True)
def _(C0_finite, K_finite, S, equation, m, paper_symbol, s):
    equation(
        paper_symbol("A_half", latex=r"A_{1/2}", typst='A_("1/2")'),
        (4 * m**2 / s * K_finite.expand().collect(C0_finite))
        .replace(C0_finite, paper_symbol("C_0"))
        .replace(m, S("m")),
    )
    return


@app.cell(hide_code=True)
def _(analytic_A, analytic_width, mo, parameters):
    mo.md(rf"""
    $$
    \begin{{gathered}}
    M={parameters["higgs_mass"]:g}\,\mathrm{{GeV}},\qquad
    m={parameters["fermion_mass"]:g}\,\mathrm{{GeV}},\qquad
    v={parameters["vev"]:g}\,\mathrm{{GeV}},\\
    \alpha^{{-1}}={1 / parameters["alpha"]:.9f},\qquad
    N_cQ^2={parameters["charge_color"]:.7g},\\[5pt]
    A_{{1/2}}={analytic_A.real:.9f},\qquad
    \Gamma_{{\rm analytic}}={1e6 * analytic_width:.7f}\,\mathrm{{keV}}.
    \end{{gathered}}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate the causal momentum-space representation

    `build_cff()` performs the graph's loop-energy integration as a
    [Cross-Free Family representation](https://arxiv.org/abs/2211.09653). This is the causal organization of the loop-tree-dual
    residues: spurious differences of on-shell energies have already canceled.
    Only spatial momentum remains to integrate.

    We use the **generated** surfaces and edge momentum signatures, not a
    handwritten triangle formula. Their normalization is converted to the same
    $C_0$ convention as the analytic calculation,
    $\int d^4k/(2\pi)^4\,1/(D_0D_1D_2)=iC_0/(16\pi^2)$.
    Both graph orientations are averaged at the density level; their multiplicity
    is already included in $K$.
    """)
    return


@app.cell
def _(E, M, S, cff_triangle_density, generated, m):
    cff_results = [diagram.build_cff() for diagram in generated.diagrams]
    r, z = S("r", "z")
    triangle_density = (
        sum(
            (
                cff_triangle_density(cff, diagram, r, z, M, m)
                for cff, diagram in zip(cff_results, generated.diagrams)
            ),
            E("0"),
        )
        / 2
    )
    return cff_results, r, triangle_density, z


@app.cell(hide_code=True)
def _(E, Replacement, cff_results, equation, mo, paper_symbol):
    _surface_notation = [
        Replacement(E(surface.symbol_name), paper_symbol(f"eta_{i}"))
        for i, surface in enumerate(cff_results[0].surfaces)
    ]
    _surface_equations = [
        equation(
            f"eta_{i}",
            sum((paper_symbol(f"E_{j}") for j in surface.positive_energies), E("0"))
            - sum((paper_symbol(f"E_{j}") for j in surface.negative_energies), E("0"))
            + sum(
                (
                    coefficient
                    * paper_symbol(
                        f"shift_{j}", latex=rf"q_{{{j}}}^0", typst=f"q_{j}^0"
                    )
                    for j, coefficient in surface.external_shift
                ),
                E("0"),
            ),
        )
        for i, surface in enumerate(cff_results[0].surfaces)
    ]
    mo.vstack(
        [
            equation(
                "C", cff_results[0].to_expression().replace_multiple(_surface_notation)
            ),
            mo.accordion(
                {
                    "Causal surfaces in the generated edge routing": mo.vstack(
                        [
                            mo.md(r"""
                $E_i>0$ is the on-shell energy of edge $i$; $q_i^0$ is its external
                energy shift, fixed by the generated momentum signature. These
                definitions are read directly from the CFF surface objects.
                """),
                            *_surface_equations,
                        ]
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Match the four-dimensional Monte Carlo to the regulated amplitude

    Our Monte Carlo integrates the **reduced scalar-master density**. That density
    contains neither $R_1$ nor $R_2$; both must be matched before sampling. More
    samples cannot recover a missing finite term. A scheme that instead integrates
    the unreduced four-dimensional numerator with a regulator-consistent UV
    subtraction can already include $R_1$: it must not add that term a second time.

    Use a normalized massive vacuum density, obtained from the **same CFF graph**
    with zero external momenta. `get_expression()` evaluates its master analytically;
    we divide the CFF density by that computed integral to normalize it:

    $$
    C_0(0,0,0;m^2,m^2,m^2)=-\frac1{2m^2},\qquad
    \rho_R=-2m^2\rho_{\rm vac},\qquad \int d^3\mathbf k\,\rho_R=1.
    $$

    The complete spatial integral is therefore

    $$
    K=\int d^3\mathbf k\,
    \left[(4m^2-s)\rho_C(\mathbf k)+(R_1+R_2)\rho_R(\mathbf k)\right].
    $$

    This finite matching density represents the already integrated rational terms;
    it is not a four-dimensional realization of $\widetilde k^2$ itself.
    Its coefficients come from the two calculations above. They are not fitted or
    added to a scalar-triangle estimate at the end. Evaluating the densities on the
    same samples retains their correlation in the heavy-fermion cancellation.
    """)
    return


@app.cell
def _(
    E,
    M,
    R1,
    R2,
    Replacement,
    S,
    Symbol,
    cff_results,
    cff_triangle_density,
    equation,
    generated,
    m,
    mo,
    oneloop,
    r,
    s,
    triangle_density,
    triangle_weight,
    z,
):
    vacuum_density = cff_triangle_density(
        cff_results[0], generated.diagrams[0], r, z, E("0"), m
    )
    assert (
        vacuum_density + 3 / (8 * Symbol.PI * (r**2 + m**2) ** E("5/2"))
    ).together() == 0
    vacuum_master = oneloop.select_branch(
        oneloop.get_expression(oneloop.C0(0, 0, 0, m**2, m**2, m**2, 1), coefficient=0),
        [Replacement(m, E("1"))],  # select only the massive branch; retain symbolic m
    ).together()
    assert (vacuum_master + 1 / (2 * m**2)).together() == 0
    rational_density = (vacuum_density / vacuum_master).together()
    complete_density = (
        triangle_weight.replace(s, M**2) * triangle_density
        + (R1 + R2) * rational_density
    )
    mo.vstack(
        [
            equation("C_vac", vacuum_master.replace(m, S("m")), rational=True),
            equation(
                "rho_R", rational_density.factor().replace(m, S("m")), rational=True
            ),
        ]
    )
    return (complete_density,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compile the integrand and sample it

    In the Higgs rest frame the density is axially symmetric. Integrate the
    azimuth exactly and map the remaining variables to the unit square:

    $$
    r=\frac{mx}{1-x},\qquad \cos\theta=2u-1,\qquad
    J=\frac{4\pi r^2m}{(1-x)^2}.
    $$

    This is a spatial momentum-space integral, not a Feynman-parameter integral.

    Symbolica compiles the complete expression once and evaluates whole batches.
    Its adaptive Monte Carlo integrator supplies the sampling weights and error
    estimate. No failed or nonfinite evaluations are discarded.
    """)
    return


@app.cell
def _(E, M, Replacement, S, Symbol, complete_density, m, np, parameters, r, z):
    x, u = S("x", "u")
    radius = m * x / (1 - x)
    jacobian = 4 * Symbol.PI * radius**2 * m / (1 - x) ** 2
    unit_square_density = (
        complete_density.replace_multiple(
            [Replacement(r, radius), Replacement(z, 2 * u - 1)]
        )
        * jacobian
    )
    numerical_expression = unit_square_density.replace_multiple(
        [
            Replacement(M, E(str(parameters["higgs_mass"]))),
            Replacement(m, E(str(parameters["fermion_mass"]))),
        ]
    )
    evaluator = numerical_expression.evaluator([x, u])
    # Compile before timing the Monte Carlo computation.
    assert np.isfinite(evaluator.evaluate([[0.2, 0.3], [0.7, 0.8]])).all()
    return (evaluator,)


@app.cell
def _(
    NumericalIntegrator,
    analytic_width,
    batch_integrand,
    evaluator,
    m,
    parameters,
    point,
    s,
    time,
    width_prefactor,
):
    _started = time.perf_counter()
    mc_K, mc_K_error, mc_chi2 = NumericalIntegrator.continuous(2).integrate(
        batch_integrand(evaluator),
        max_n_iter=parameters["iterations"],
        n_samples_per_iter=parameters["samples_per_iteration"],
        min_error=0.0,
        seed=parameters["seed"],
        show_stats=False,
    )
    mc_seconds = time.perf_counter() - _started
    mc_A = 4 * point[m] ** 2 / point[s] * mc_K
    mc_A_error = 4 * point[m] ** 2 / point[s] * mc_K_error
    mc_width = width_prefactor * mc_A**2
    mc_width_error = 2 * width_prefactor * abs(mc_A) * mc_A_error
    width_pull = (mc_width - analytic_width) / mc_width_error
    assert abs(width_pull) < 5
    assert 0 < mc_width_error / mc_width < 0.005
    return (
        mc_A,
        mc_A_error,
        mc_chi2,
        mc_seconds,
        mc_width,
        mc_width_error,
        width_pull,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare the physical rates

    The error on the width is propagated from the sampled amplitude,
    $\sigma_\Gamma=2\Gamma\,\sigma_A/|A|$; this linear approximation is appropriate
    at the small relative errors shown here. The fixed seed makes this comparison
    reproducible. Increase the sample budget to reduce the uncertainty.

    The last two rows are deliberately incomplete analytic predictions for the
    four-dimensional scalar-master integral. Keeping only $R_2$ still misses $R_1$;
    dropping both is worse. Their disagreement is a physics error, not Monte Carlo noise.
    """)
    return


@app.cell
def _(K_finite, R1, m, point, rational_term, s):
    wrong_A = complex((4 * m**2 / s * (K_finite - rational_term)).evaluate(point))
    only_R2_A = complex((4 * m**2 / s * (K_finite - R1)).evaluate(point))
    return only_R2_A, wrong_A


@app.cell(hide_code=True)
def _(
    analytic_A,
    analytic_width,
    mc_A,
    mc_A_error,
    mc_chi2,
    mc_seconds,
    mc_width,
    mc_width_error,
    measurement,
    mo,
    only_R2_A,
    parameters,
    width_prefactor,
    width_pull,
    wrong_A,
):
    comparison = [
        {
            "Method": "Generated numerator → masters",
            "A₁/₂": analytic_A.real,
            "Γ [keV]": 1e6 * analytic_width,
            "MC error [keV]": None,
        },
        {
            "Method": "CFF Monte Carlo + R₁ + R₂",
            "A₁/₂": mc_A,
            "Γ [keV]": 1e6 * mc_width,
            "MC error [keV]": 1e6 * mc_width_error,
        },
        {
            "Method": "Incomplete: R₂ only (missing R₁)",
            "A₁/₂": only_R2_A.real,
            "Γ [keV]": 1e6 * width_prefactor * abs(only_R2_A) ** 2,
            "MC error [keV]": None,
        },
        {
            "Method": "Incomplete: neither R₁ nor R₂",
            "A₁/₂": wrong_A.real,
            "Γ [keV]": 1e6 * width_prefactor * abs(wrong_A) ** 2,
            "MC error [keV]": None,
        },
    ]
    mo.vstack(
        [
            mo.md(rf"""
    $$
    \begin{{aligned}}
    A_{{1/2}}^{{\rm analytic}} &= {analytic_A.real:.7f}, &
    A_{{1/2}}^{{\rm MC}} &= {measurement(mc_A, mc_A_error)},\\[4pt]
    \Gamma_{{\rm analytic}} &= {1e6 * analytic_width:.7f}\,\mathrm{{keV}}, &
    \Gamma_{{\rm MC}} &= {measurement(1e6 * mc_width, 1e6 * mc_width_error)}\,\mathrm{{keV}}.
    \end{{aligned}}
    $$
            """),
            mo.ui.table(
                comparison,
                selection=None,
                format_mapping={
                    "A₁/₂": "{:.7f}",
                    "Γ [keV]": "{:.6f}",
                    "MC error [keV]": lambda value: (
                        "—" if value is None else f"{value:.2g}"
                    ),
                },
            ),
            mo.md(
                f"Width difference: **{width_pull:+.2f}σ**. "
                f"{parameters['iterations'] * parameters['samples_per_iteration']:,} samples; "
                f"{mc_seconds:.2f} s excluding evaluator construction and compilation. "
                f"Integrator χ²: {mc_chi2:.2f}."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The analytic and numerical routes include both fermion-loop orientations,
    the dimensional rational term, physical photon polarizations, and the
    identical-particle phase-space factor. The numerical domain is explicitly
    subthreshold. Above threshold, a contour prescription or local threshold
    subtraction is needed; increasing this real-domain Monte Carlo budget alone
    would not provide it.
    """)
    return


if __name__ == "__main__":
    app.run()
