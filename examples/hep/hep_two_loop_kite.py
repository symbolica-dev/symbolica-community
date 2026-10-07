import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Two-loop kite: 6 zeta(3)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A two-loop master integral: where $6\zeta(3)$ comes from

    [Browse all notebooks](/)

    **How does a rational Feynman integrand produce a zeta value?**
    The massless two-loop propagator, or *kite*, gives a compact example.
    Dimensional analysis fixes its dependence on the external momentum to
    $I(Q^2)=C/Q^2$, but leaves the dimensionless coefficient $C$ undetermined.
    We will calculate it from the graph.

    Propagator integrals like this are building blocks of perturbative
    two-point functions. The central line couples the two loop momenta, so
    the answer is more interesting than a product of one-loop bubbles.
    This finite scalar integral is a useful first master-integral calculation;
    a complete amplitude also needs its numerators, couplings, and other diagrams.

    Generate the graph with HEPkit, derive its parameter representation, and
    integrate it exactly. Then **change the projective gauge, integration
    order, and momentum scale** and compare with numerical integration. Finally,
    compute an AMFlow starting value and use DiffExp series transport to move
    to other spacelike values of $p^2$.
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup and notebook helpers

    We use Symbolica with HEPkit for the symbolic calculation, and NumPy,
    SciPy, and Matplotlib for the numerical comparison. Expand the cells to
    explore the code or change the integration parameters.
    """)


@app.cell(hide_code=True)
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import qmc
    from symbolica import E, S, get_citations
    from symbolica.community import hepkit as hep
    from symbolica.community.hepkit import integration

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        }
    )

    def is_kite(candidate):
        pairs = [
            tuple(sorted((edge.source, edge.target)))
            for edge in candidate.internal_edges
        ]
        return (
            candidate.loop_count == 2
            and len(pairs) == len(set(pairs)) == 5
            and len({vertex for pair in pairs for vertex in pair}) == 4
        )

    flow_cache_root = Path.home() / ".cache/symbolica/kite/central-line-one-core-v3"
    return E, S, flow_cache_root, get_citations, hep, integration, is_kite, mo, np, plt, qmc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The two-loop kite

    We evaluate the **positive Euclidean integral**, with normalized loop measures:

    $$
    I(Q^2)=\int\frac{d^4 k_E}{\pi^2}\frac{d^4\ell_E}{\pi^2}\,
    \frac{1}{k_E^2\,(k_E-p_E)^2\,\ell_E^2\,(\ell_E-p_E)^2\,(k_E-\ell_E)^2},
    \qquad p_E^2=Q^2>0.
    $$

    All internal masses vanish and all five propagator powers are one. The normalization is exactly the measure above, with coupling constants and diagram symmetry factors stripped off. HEPkit uses Minkowski inverse propagators, so we set $p^2=-Q^2$ when constructing $F$; its resulting parametric integral is the Euclidean quantity defined above.

    The off-shell external momentum removes the on-shell infrared singularity. Both the full graph and its one-loop subgraphs are UV convergent. For the exact parameter calculation we can therefore work directly at $D=4$, without an epsilon expansion or subtraction prescription. Dimensional counting already predicts $I\propto(Q^2)^{-1}$; the nontrivial task is its coefficient.

    ### Generate the diagram

    Use the built-in `hep.Model.phi3()` model to generate the two-loop two-point diagrams. Its mass is symbolic; we explicitly take the massless limit when constructing the parametric integrand below. We select the kite as the graph with **four interaction vertices and five distinct internal vertex pairs**: the other generated topology contains parallel lines.

    The external momentum stays off shell: $p^2=-Q^2<0$. This virtuality supplies the single scale of the integral.
    """)


@app.cell
def _(hep, is_kite, mo):
    model = hep.Model.phi3()
    mass = model.particle("phi").mass
    process = model.process(["phi"], ["phi"])
    generated = process.generate_diagrams(
        loops=2,
        threads=1,
        max_vertices=4,
        allow_self_loops=False,
        maximum_bridges=0,
        progress=None,
    )

    (diagram,) = [candidate for candidate in generated if is_kite(candidate)]
    print(f"Generated {len(generated)} diagrams; selected the unique kite topology.")
    print(
        f"Loops: {diagram.loop_count}; internal propagators: {len(diagram.internal_edges)}"
    )

    mo.output.append(diagram)
    return diagram, mass


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Derive the integral family from the generated diagram

    An integral family describes the propagators with arbitrary integer powers. HEPkit derives it directly from the diagram and assigns the loop momenta. Here the five propagators form a complete independent basis: no additional irreducible scalar products are needed.

    The table associates each Feynman parameter with a propagator and its momentum. Equivalent choices of loop momenta give the same integral.

    The family initially has a common mass $m$. We obtain the massless integral by setting $m=0$ in its second Symanzik polynomial, then factor out the scale: $F=Q^2\widehat F$.
    """)


@app.cell
def _(E, S, diagram, hep, mass, mo):
    Q2 = S("kite::Q2")
    P = hep.Kinematics.external_momentum()
    p = P(0)
    kin = hep.Kinematics(E("4"), momenta=[p]).with_scalar_product(p, p, -Q2)
    family = diagram.integral_family(kinematics=kin)
    edges = sorted(diagram.internal_edges, key=lambda edge: edge.id)
    # A two-loop family with one external momentum has 2*3/2 + 2*1 = 5 scalar products.
    # The five physical propagators already span that space: no auxiliary ISPs are needed.
    mo.output.append(
        mo.vstack([mo.md("**Integral family before the massless limit:**"), family])
    )
    x = S(*[f"kite::x{i}" for i in range(1, len(edges) + 1)])
    routing = diagram.loop_momentum_basis.edge_signatures
    _rows = ["| Parameter | Diagram edge | Routed momentum |", "|---|---:|---|"]
    _rows += [
        f"| ${parameter}$ | {edge.id} | ${routing[edge.id].format_momentum()}$ |"
        for parameter, edge in zip(x, edges)
    ]
    mo.output.append(mo.md("\n".join(_rows)))
    U, F_massive = family.symanzik(x)
    F = F_massive.replace(mass, E("0"))
    Fhat = (F / Q2).cancel()
    # Symanzik parameters follow the physical propagators' edge order.
    mo.output.append(mo.vstack([mo.md("**First Symanzik polynomial $U$:**"), U]))
    mo.output.append(
        mo.vstack(
            [mo.md("**Scale-free second polynomial $\\widehat F=F/Q^2$:**"), Fhat]
        )
    )
    return Fhat, Q2, U, p, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From $U,F$ to a convergent projective integral

    For $L=2$, $N=\sum_i\nu_i=5$, and $D=4$, the scalar parameter formula gives

    $$
    \frac{\Gamma(N-LD/2)}{\prod_i\Gamma(\nu_i)}
    \int_{x_i\ge0}d^5x\;\delta\!\left(1-\sum_i x_i\right)
    \frac{U^{N-(L+1)D/2}}{F^{N-LD/2}}
    =\int d^5x\;\delta\!\left(1-\sum_i x_i\right)\frac{1}{UF}.
    $$

    Here the gamma prefactor is exactly $\Gamma(1)=1$. Since $U$ has degree two
    and $F$ degree three, the integrand has degree $-5$. The Cheng–Wu theorem
    lets us fix one parameter to one instead of imposing $\sum_i x_i=1$.
    For the default choice **$x_5=1$**, the remaining integral is

    $$C\equiv Q^2I(Q^2)=\int_0^\infty dx_1\,dx_2\,dx_3\,dx_4\;
    \frac{1}{U(x_1,x_2,x_3,x_4,1)\,\widehat F(x_1,x_2,x_3,x_4,1)}.$$

    Choose a different fixed parameter below. This changes the coordinates
    on parameter space, so $C$ should stay the same. This projective gauge
    choice is unrelated to gauge fixing of a quantum field.
    """)


@app.cell(hide_code=True)
def _(mo):
    gauge = mo.ui.dropdown(
        options={f"x{i + 1} = 1": i for i in range(5)},
        value="x5 = 1",
        label="Fixed parameter",
    )
    reverse_order = mo.ui.checkbox(value=False, label="Reverse integration order")
    mo.hstack([gauge, reverse_order], justify="start")
    return gauge, reverse_order


@app.cell
def _(E, Fhat, U, gauge, mo, reverse_order, x):
    integrand = (1 / (U * Fhat)).replace(x[gauge.value], E("1"))
    variables = [parameter for i, parameter in enumerate(x) if i != gauge.value]
    if reverse_order.value:
        variables = variables[::-1]
    mo.output.append(
        mo.vstack(
            [
                mo.md(
                    "**Integration order:** " + " → ".join(str(v) for v in variables)
                ),
                integrand,
            ]
        )
    )
    return integrand, variables


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exact integration

    `integrate` uses $[0,\infty)$ in the **supplied variable order**. It differs from Symbolica's `Expression.integrate(x)`, which computes an antiderivative. We integrate the four remaining Feynman parameters and restore the overall factor $1/Q^2$.
    """)


@app.cell
def _(E, integrand, integration, mo, variables):
    C = integration.integrate(integrand, variables)
    MZV = integration.mzv_symbol()
    # At depth one, a multiple zeta value is the Riemann zeta function.
    C = C.replace(MZV(3), E("3").zeta())
    mo.output.append(mo.vstack([mo.md("**Exact coefficient $C=Q^2 I(Q^2)$:**"), C]))
    return (C,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The coefficient is $6\zeta(3)$, with **transcendental weight three**.
    It agrees with the unit-power, four-dimensional limit of the
    [massless two-loop two-point integral](https://arxiv.org/abs/hep-ph/0308311).
    The zeta value comes from the parameter integrations; dimensional analysis
    determines only the factor $1/Q^2$.

    The family above also describes integrals with raised propagator powers
    and polynomial numerators. These are the objects related by
    integration-by-parts identities in an amplitude calculation. We evaluate
    the finite unit-power member here. In dimensional regularization,
    higher orders in $\epsilon$ can be needed when reduction coefficients
    have poles; the $D=4$ value alone does not supply those terms.
    """)


@app.cell(hide_code=True)
def _(mo):
    virtuality = mo.ui.slider(
        start=1,
        stop=10,
        step=1,
        value=4,
        label="Euclidean Q²",
        show_value=True,
    )
    mo.vstack(
        [
            mo.md(
                "### Restore the momentum scale\n\nChoose $Q^2$ in a fixed unit of momentum squared. The integral has the inverse unit."
            ),
            virtuality,
        ]
    )
    return (virtuality,)


@app.cell
def _(C, E, Q2, mo, virtuality):
    I = C / Q2
    coefficient_value = C.evaluate({}, decimal_digit_precision=40).real
    value_at_scale = I.replace(Q2, E(str(virtuality.value)))
    mo.output.append(
        mo.vstack(
            [
                mo.md("**Calculated integral $I(Q^2)$:**"),
                I,
                mo.md(f"At $Q^2={virtuality.value}$:"),
                value_at_scale,
                mo.md(
                    f"$I = {float(value_at_scale.evaluate({}).real):.12g}$; "
                    f"$Q^2 I = {float(coefficient_value):.12g}$ remains constant."
                ),
            ]
        )
    )
    return (coefficient_value,)


@app.cell
def _(E):
    # Keep the known value independent of the calculated coefficient.
    reference = (6 * E("3").zeta()).evaluate({}, decimal_digit_precision=40).real
    return (reference,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Numerical comparison

    For an independent comparison, we integrate the same four-parameter expression numerically using scrambled Sobol points.

    Map the unit hypercube to the projective domain with

    $$x_i=\left(\frac{u_i}{1-u_i}\right)^3,\qquad
    \prod_i dx_i=\prod_i\frac{3u_i^2}{(1-u_i)^4}\,du_i.$$

    The cubic map allocates more resolution near the boundaries than a linear-fractional map; these regions matter for this improper integral. Eight independently scrambled nets provide a mean and a **scatter-based standard-error estimate**, not a rigorous error bound. The estimates at different sizes share seeds and are correlated.
    """)


@app.cell
def _(integrand, mo, np, qmc, reference, variables):
    # Evaluate the integrand at a batch of integration points.
    numeric_kernel = integrand.evaluator(variables, n_cores=1, jit_compile=False)
    replicates = 8
    powers = [10, 12, 14, 16]  # points per scramble: 1024 ... 65536
    estimates, standard_errors = ([], [])
    for power in powers:
        replicate_means = []
        for replicate in range(replicates):
            u = qmc.Sobol(d=4, scramble=True, seed=20261005 + replicate).random_base2(
                power
            )
            points = (u / (1 - u)) ** 3
            jacobian = np.prod(3 * u**2 / (1 - u) ** 4, axis=1)
            values = numeric_kernel.evaluate(points).reshape(-1) * jacobian
            replicate_means.append(values.mean())
        estimates.append(float(np.mean(replicate_means)))
        standard_errors.append(
            float(np.std(replicate_means, ddof=1) / np.sqrt(replicates))
        )
    _rows = [
        "| Points / scramble | Estimate of C | Estimated standard error |",
        "|---:|---:|---:|",
    ]
    _rows += [
        f"| {2**power:,} | {mean:.8f} | {error:.8f} |"
        for power, mean, error in zip(powers, estimates, standard_errors)
    ]
    mo.output.append(mo.md("\n".join(_rows)))
    relative_error = abs(estimates[-1] / float(reference) - 1)
    print(f"Final relative difference from 6 ζ(3): {relative_error:.2e}")
    return estimates, powers, standard_errors


@app.cell(hide_code=True)
def _(coefficient_value, estimates, mo, reference, standard_errors):
    mo.md(rf"""
    | Determination of $C$ | Value |
    |:---|---:|
    | Exact integration, evaluated numerically | {float(coefficient_value):.12f} |
    | Known result $6\zeta(3)$ | {float(reference):.12f} |
    | Sobol integration | {estimates[-1]:.8f} ± {standard_errors[-1]:.8f} |
    """)


@app.cell
def _(estimates, mo, np, plt, powers, reference, standard_errors):
    points_per_scramble = 2 ** np.array(powers)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    left.errorbar(
        points_per_scramble,
        estimates,
        yerr=standard_errors,
        fmt="o-",
        capsize=4,
        color="#0e7490",
        label="Scrambled Sobol (±1 estimated SE)",
    )
    left.axhline(
        float(reference), color="#b45309", linestyle="--", label="Exact: 6 ζ(3)"
    )
    left.set(
        xscale="log",
        xlabel="Points per scramble",
        ylabel="Dimensionless coefficient C",
        title="Four-dimensional numerical cross-check",
    )
    left.legend(fontsize=8)
    left.grid(alpha=0.2)

    errors = np.abs(np.array(estimates) / float(reference) - 1)
    right.loglog(points_per_scramble, errors, "o-", color="#0e7490")
    right.set(
        xlabel="Points per scramble",
        ylabel="Relative difference from exact result",
        title="Convergence with fixed reproducible seeds",
    )
    right.grid(alpha=0.2, which="both")
    mo.output.append(fig)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluate the graph with AMFlow

    An independent route is to evaluate the dimensionally regulated family
    numerically at one point, then solve its differential equations at other
    points. We use HEPkit's implementations of the
    [AMFlow auxiliary-mass method](https://arxiv.org/abs/2201.11669) and
    [DiffExp series-transport method](https://arxiv.org/abs/2006.05510).

    The starting point is $p^2=-1$, or $Q^2=1$, in the same units as above.
    AMFlow introduces an auxiliary mass, computes a boundary where the integrals
    simplify, and flows back to the massless family. **The starting values are
    computed numerically; $6\zeta(3)$ is used only for comparison.**

    For this calculation use $D=4-2\epsilon$. Although the kite is finite,
    its IBP reduction contains $1/\epsilon^2$ coefficients. The master integrals
    must therefore be retained through $\epsilon^2$ to recover the finite term.

    The numerical evaluator uses $d^Dk/(i\pi^{D/2})$ per loop and Minkowski
    propagators $q^2+i0$. Wick rotation contributes $(-1)^5$ for the five unit
    propagators, so we negate the finite numerical coefficient to compare
    with the positive Euclidean integral defined above.

    **Compute the starting value once**, then change the destination below.
    The starting value is computed with one worker. Moving the slider reuses
    that value; opening this section does not start the boundary calculation.
    """)


@app.cell(hide_code=True)
def _():
    try:
        import symbolica.community.hep.integration as numerical_flow
    except ModuleNotFoundError as _error:
        if _error.name not in {
            "symbolica.community.hep",
            "symbolica.community.hep.integration",
            "symbolica.community.hep_integration_native",
        }:
            raise
        numerical_flow = None
    return (numerical_flow,)


@app.cell(hide_code=True)
def _(mo, numerical_flow):
    run_amflow = mo.ui.run_button(label="Compute AMFlow starting value at p² = −1")
    mo.vstack(
        [
            mo.md(
                "The numerical-flow sections require a Symbolica community installation with `hep.integration`."
            ),
            run_amflow,
        ]
    )
    return (run_amflow,)


@app.cell
def _(E, Q2, S, diagram, flow_cache_root, hep, mass, mo, numerical_flow, p, run_amflow):
    mo.stop(
        not run_amflow.value,
        mo.md("Press **Compute AMFlow starting value** to evaluate this graph."),
    )
    mo.stop(
        numerical_flow is None,
        mo.callout(
            "This installation does not include HEPkit's numerical integration API.",
            kind="info",
        ),
    )
    flow_dimension, flow_epsilon = S("kite_flow::D", "kite_flow::epsilon")
    flow_kinematics = hep.Kinematics(flow_dimension, momenta=[p]).with_scalar_product(
        p, p, -Q2
    )
    flow_diagram_family = diagram.integral_family(kinematics=flow_kinematics)
    # Keep the graph's routing and propagator order while taking the massless limit.
    flow_family = hep.IntegralFamily(
        flow_diagram_family.loop_momenta,
        flow_diagram_family.external_momenta,
        [
            denominator.replace(mass, E("0"))
            for denominator in flow_diagram_family.denominators
        ],
        kinematics=flow_kinematics,
    )
    flow_evaluator = numerical_flow.IntegralEvaluator(
        options=numerical_flow.EvaluationOptions(
            digits=8,
            guard_digits=24,
            series_order=70,
            workers=1,
            mass_mode="branch",
            recursion="auxiliary_mass",
            cache_directory=str(flow_cache_root / "reductions"),
            sample_cache_directory=str(flow_cache_root / "samples"),
        ),
    )
    prepared_flow = flow_evaluator.prepare(
        flow_family,
        [[1] * len(flow_family.denominators)],
        [Q2],
        flow_epsilon,
        branch_domain="Euclidean Q2 > 0; p squared = -Q2",
    )
    flow_cache = numerical_flow.BoundaryCache()
    destination_caches = {}
    _, seed_last = prepared_flow.required_master_range({Q2: E("1")}, 0, 0)
    with mo.status.spinner(title="Computing the AMFlow starting value"):
        amflow_boundary = prepared_flow.generate_boundary(
            flow_cache,
            {Q2: E("1")},
            last=seed_last,
            extra_digits=10,
        )
    (amflow_target,) = prepared_flow.project_targets(amflow_boundary, 0, 0, digits=8)
    wick_sign = (-1) ** len(flow_family.denominators)
    amflow_value = wick_sign * amflow_target.coefficients[0]
    amflow_error = amflow_target.absolute_errors[0]
    mo.output.append(
        mo.md(rf"""
    **AMFlow at $p^2=-1$: $I_E={float(amflow_value.real):.10f}$**

    Estimated absolute numerical uncertainty: {float(amflow_error):.2e}.
    The common master basis has {len(prepared_flow.basis)} members, retained
    through $\epsilon^{seed_last}$ for the finite kite projection.
    """)
    )
    return (
        amflow_boundary,
        amflow_target,
        amflow_value,
        destination_caches,
        flow_cache,
        prepared_flow,
        wick_sign,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## DiffExp transport in $p^2$

    The master integrals obey a system
    $$\frac{d\mathbf J}{dQ^2}=A(Q^2,\epsilon)\mathbf J.$$
    HEPkit derives this system from the same family using IBP identities.
    DiffExp transport solves it by local series expansions and carries the
    AMFlow starting values to new points. All paths stay on the positive
    $Q^2$ axis, so they do not cross the singular point $p^2=0$.

    For this single-scale massless example, dimensional analysis also predicts
    $I(Q^2,\epsilon)\propto(Q^2)^{-1-2\epsilon}$.
    Its finite term should satisfy $Q^2I(Q^2)=6\zeta(3)$.
    The table compares actual transported values with this prediction;
    the scaling formula is not used to generate the numerical destinations.
    """)


@app.cell(hide_code=True)
def _(mo):
    transport_destination = mo.ui.slider(
        start=1,
        stop=10,
        step=1,
        value=4,
        label="Destination −p² = Q²",
        show_value=True,
    )
    mo.output.append(transport_destination)
    return (transport_destination,)


@app.cell
def _(
    E,
    Q2,
    amflow_value,
    destination_caches,
    flow_cache,
    mo,
    numerical_flow,
    prepared_flow,
    reference,
    transport_destination,
    wick_sign,
):
    flow_destinations = sorted({1, 2, 4, 8, 10, transport_destination.value})
    flow_results = []
    with mo.status.spinner(title="Transporting the master integrals in p²"):
        for destination in flow_destinations:
            point = {Q2: E(str(destination))}
            leading, last = prepared_flow.required_master_range(point, 0, 0)
            # Start every path from the original AMFlow boundary so uncertainty
            # does not accumulate when the slider visits successive destinations.
            if destination not in destination_caches:
                _destination_cache = numerical_flow.BoundaryCache()
                _destination_cache.extend(flow_cache)
                destination_caches[destination] = _destination_cache
            transported = prepared_flow.transport(
                destination_caches[destination],
                point,
                leading,
                last,
                admit_straight_path=True,
            )
            (target,) = prepared_flow.project_targets(transported, 0, 0, digits=8)
            value = wick_sign * target.coefficients[0]
            flow_results.append((destination, value, target.absolute_errors[0]))
    _rows = [
        "| $p^2$ | AMFlow / DiffExp $I_E$ | Estimated absolute uncertainty | $6\\zeta(3)/(-p^2)$ |",
        "|---:|---:|---:|---:|",
    ]
    _rows += [
        f"| {-q2} | {float(value.real):.10f} | {float(error):.2e} | {float(reference) / q2:.10f} |"
        for q2, value, error in flow_results
    ]
    mo.output.append(
        mo.vstack(
            [
                mo.md("\n".join(_rows)),
                mo.md(
                    "The $p^2=-1$ entry is the AMFlow boundary; the other entries reuse it through differential-equation transport."
                ),
            ]
        )
    )
    return (flow_results,)


@app.cell(hide_code=True)
def _(flow_results, mo, np, plt, reference):
    _fig, (_value_axis, _scale_axis) = plt.subplots(
        1, 2, figsize=(10, 3.8), constrained_layout=True
    )
    _q2 = np.array([row[0] for row in flow_results], dtype=float)
    _values = np.array([float(row[1].real) for row in flow_results])
    _errors = np.array([float(row[2]) for row in flow_results])
    _curve = np.linspace(1, 10, 200)
    _value_axis.plot(-_curve, float(reference) / _curve, label=r"$6\zeta(3)/(-p^2)$")
    _value_axis.errorbar(
        -_q2, _values, yerr=_errors, fmt="o", capsize=3, label="AMFlow / DiffExp"
    )
    _value_axis.set(
        xlabel=r"$p^2$", ylabel=r"$I_E$", title="Transport to spacelike momenta"
    )
    _value_axis.legend(fontsize=8)
    _scale_axis.errorbar(-_q2, _q2 * _values, yerr=_q2 * _errors, fmt="o", capsize=3)
    _scale_axis.axhline(float(reference), linestyle="--", label=r"$6\zeta(3)$")
    _scale_axis.set(
        xlabel=r"$p^2$",
        ylabel=r"$(-p^2) I_E$",
        title="The scale-independent coefficient",
    )
    _scale_axis.legend(fontsize=8)
    mo.output.append(_fig)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explore further

    The result $I(Q^2)=6\zeta(3)/Q^2$ connects the two-loop graph to a transcendental constant. The numerical estimates approach the same value as the number of sample points increases.

    Try a different fixed parameter or reverse the integration order: the
    exact coefficient stays the same, while numerical convergence can change.
    Increase `powers` in the numerical cell to resolve the boundary regions better.

    Moving on to a massive kite introduces dimensionless mass ratios and can
    require elliptic functions. Taking $Q^2\to0$ is also not a substitution into
    this off-shell result: the exactly massless, zero-momentum integral is
    scaleless and is set to zero in dimensional regularization, where UV and
    IR singularities must be treated together.

    ### References

    - [SubTropica online](https://subtropi.ca/) and its [paper companion examples](https://github.com/SubTropica/SubTropica/blob/main/PaperChecks.wl): inspiration for the propagators → Symanzik polynomials → parameter integration workflow; the companion includes a generic massive kite.
    - I. Bierenbaum and S. Weinzierl, [*The massless two-loop two-point function*](https://arxiv.org/abs/hep-ph/0308311): analytic results and multiple-zeta-value structure for this family (the four-dimensional unit-power limit has coefficient $6\zeta(3)$).
    - X. Liu and Y.-Q. Ma, [*AMFlow*](https://arxiv.org/abs/2201.11669): auxiliary-mass boundary evaluation.
    - M. Hidding, [*DiffExp*](https://arxiv.org/abs/2006.05510): series solutions of differential equations in kinematics.
    - F. Brown, [*The massless higher-loop two-point function*](https://arxiv.org/abs/0804.1660): background on hyperlogarithmic evaluation and the appearance of multiple zeta values.
    """)


@app.cell(hide_code=True)
def _(C, estimates, get_citations, mo):
    mo.accordion(
        {
            "Software citations": mo.vstack(
                [
                    mo.md(
                        f"Software used for `{C}` and the {len(estimates)} numerical estimates."
                    ),
                    *get_citations(),
                ]
            )
        }
    )


if __name__ == "__main__":
    app.run()
