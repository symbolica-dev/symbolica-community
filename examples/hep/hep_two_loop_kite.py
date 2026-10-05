import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Two-loop kite: 6 zeta(3)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A two-loop master integral: where $6\zeta(3)$ comes from

    [Browse all notebooks](/)

    Compute the **massless two-loop propagator (kite) integral** with HEPkit: generate the diagram, derive its integral family and Symanzik polynomials, and perform the parameter integrations exactly.

    $$\boxed{I(Q^2)=\frac{6\zeta(3)}{Q^2}},\qquad Q^2>0.$$

    This is a finite, coupled two-loop example: the central propagator prevents the two loop integrations from factorizing. It illustrates the transcendental constants that appear in perturbative amplitudes.

    The graph-to-parameter-integral workflow is inspired by [SubTropica](https://subtropi.ca/) and its [worked examples](https://github.com/SubTropica/SubTropica/blob/main/PaperChecks.wl). We choose the **massless, Euclidean** kite; generic massive kites can require elliptic functions.

    **You will see:** the Feynman diagram, its integral family, the parameter representation, and an exact result compared with numerical integration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Setup

    We use Symbolica with HEPkit for the symbolic calculation, and NumPy,
    SciPy, and Matplotlib for the numerical comparison. Expand the cells to
    explore the code or change the integration parameters.
    """)
    return


@app.cell(hide_code=True)
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import qmc
    import marimo as mo

    from symbolica import E, S
    from symbolica.community import hepkit as hep
    from symbolica.community.hepkit import integration

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 120})
    return E, S, hep, integration, mo, np, plt, qmc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The two-loop kite

    We evaluate the **positive Euclidean integral**, with normalized loop measures:

    $$
    I(Q^2)=\int\frac{d^4 k_E}{\pi^2}\frac{d^4\ell_E}{\pi^2}\,
    \frac{1}{k_E^2\,(k_E-p_E)^2\,\ell_E^2\,(\ell_E-p_E)^2\,(k_E-\ell_E)^2},
    \qquad p_E^2=Q^2>0.
    $$

    All internal masses vanish and all five propagator powers are one. The normalization is exactly the measure above, with coupling constants and diagram symmetry factors stripped off. HEPkit uses Minkowski inverse propagators, so we set $p^2=-Q^2$ when constructing $F$; its resulting parametric integral is the Euclidean quantity defined above.

    The off-shell external momentum removes the on-shell infrared singularity. Both the full graph and its one-loop subgraphs are UV convergent. We therefore work directly at $D=4$, without an epsilon expansion or subtraction prescription. Dimensional counting already predicts $I\propto(Q^2)^{-1}$; the nontrivial task is its coefficient.

    ### Generate the diagram

    Use the built-in `hep.Model.phi3()` model to generate the two-loop two-point diagrams. Its mass is symbolic; we explicitly take the massless limit when constructing the parametric integrand below. We select the kite as the graph with **four interaction vertices and five distinct internal vertex pairs**: the other generated topology contains parallel lines.

    The external momentum stays off shell: $p^2=-Q^2<0$. This virtuality supplies the single scale of the integral.
    """)
    return


@app.cell
def _(hep, mo):
    model = hep.Model.phi3()
    mass = model.particle("phi").mass
    process = model.process(["phi"], ["phi"])
    generated = process.generate_diagrams(
        loops=2, threads=1, max_vertices=4, allow_self_loops=False,
        maximum_bridges=0, progress=None,
    )

    def is_kite(candidate):
        pairs = [tuple(sorted((edge.source, edge.target)))
                 for edge in candidate.internal_edges]
        return (candidate.loop_count == 2 and len(pairs) == len(set(pairs)) == 5
                and len({vertex for pair in pairs for vertex in pair}) == 4)

    (diagram,) = [candidate for candidate in generated if is_kite(candidate)]
    print(f"Generated {len(generated)} diagrams; selected the unique kite topology.")
    print(f"Loops: {diagram.loop_count}; internal propagators: {len(diagram.internal_edges)}")

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
    return


@app.cell
def _(E, S, diagram, hep, mass, mo):
    Q2 = S('kite::Q2')
    p = hep.Kinematics.external_momentum(0)
    kin = hep.Kinematics(E('4'), momenta=[p]).with_scalar_product(p, p, -Q2)
    family = diagram.integral_family(kinematics=kin)
    edges = sorted(diagram.internal_edges, key=lambda edge: edge.id)
    # A two-loop family with one external momentum has 2*3/2 + 2*1 = 5 scalar products.
    # The five physical propagators already span that space: no auxiliary ISPs are needed.
    mo.output.append(mo.vstack([mo.md('**Integral family before the massless limit:**'), family]))
    x = S(*[f'kite::x{i}' for i in range(1, len(edges) + 1)])
    x5 = x[4]
    routing = diagram.loop_momentum_basis.edge_signatures
    _rows = ['| Parameter | Diagram edge | Routed momentum |', '|---|---:|---|']
    _rows += [f'| ${parameter}$ | {edge.id} | ${routing[edge.id].format_momentum()}$ |' for parameter, edge in zip(x, edges)]
    mo.output.append(mo.md('\n'.join(_rows)))
    U, F_massive = family.symanzik(x)
    F = F_massive.replace(mass, E('0'))
    Fhat = (F / Q2).cancel()
    # Symanzik parameters follow the physical propagators' edge order.
    mo.output.append(mo.vstack([mo.md('**First Symanzik polynomial $U$:**'), U]))
    mo.output.append(mo.vstack([mo.md('**Scale-free second polynomial $\\widehat F=F/Q^2$:**'), Fhat]))
    return Fhat, Q2, U, x, x5


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. From $U,F$ to a convergent projective integral

    For $L=2$, $N=\sum_i\nu_i=5$, and $D=4$, the scalar parameter formula gives

    $$
    \frac{\Gamma(N-LD/2)}{\prod_i\Gamma(\nu_i)}
    \int_{x_i\ge0}d^5x\;\delta\!\left(1-\sum_i x_i\right)
    \frac{U^{N-(L+1)D/2}}{F^{N-LD/2}}
    =\int d^5x\;\delta\!\left(1-\sum_i x_i\right)\frac{1}{UF}.
    $$

    Here the gamma prefactor is exactly $\Gamma(1)=1$. Homogeneity permits the Cheng–Wu choice **$x_5=1$** instead of $\sum_i x_i=1$. This leaves four independent integrations over $[0,\infty)$:

    $$C\equiv Q^2I(Q^2)=\int_0^\infty dx_1\,dx_2\,dx_3\,dx_4\;
    \frac{1}{U(x_1,x_2,x_3,x_4,1)\,\widehat F(x_1,x_2,x_3,x_4,1)}.$$

    The choice $x_5=1$ is a convenient projective gauge. Any one of the five parameters can instead be fixed to one, with the other four integrated over the positive half-line.
    """)
    return


@app.cell
def _(E, Fhat, U, mo, x, x5):
    integrand = (1/(U*Fhat)).replace(x5, E("1"))
    variables = list(x[:4])
    mo.output.append(integrand)
    return integrand, variables


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Exact integration

    `integrate` uses $[0,\infty)$ in the **supplied variable order**. It differs from Symbolica's `Expression.integrate(x)`, which computes an antiderivative. We integrate the four remaining Feynman parameters and restore the overall factor $1/Q^2$.
    """)
    return


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
    The answer has **transcendental weight three**, because $\zeta(3)$ has weight three. It is a two-loop master integral that can appear after IBP reduction in massless two-point calculations. The remaining $1/Q^2$ dependence follows from dimensions, not an additional integration.

    In an amplitude calculation, integration-by-parts identities reduce many integrals to a small set of masters. This result supplies the value of one such master at $D=4$.
    """)
    return


@app.cell
def _(C, E, Q2, mo):
    I = C / Q2
    # Symbolica's built-in Riemann zeta supplies an independent numerical reference.
    reference = (6 * E("3").zeta()).evaluate({}, decimal_digit_precision=40).real
    print("Q² I(Q²) =", reference)
    print("At Q²=4, I =", reference/4)
    mo.output.append(I)
    return (reference,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Numerical comparison

    For an independent comparison, we integrate the same four-parameter expression numerically using scrambled Sobol points.

    Map the unit hypercube to the projective domain with

    $$x_i=\left(\frac{u_i}{1-u_i}\right)^3,\qquad
    \prod_i dx_i=\prod_i\frac{3u_i^2}{(1-u_i)^4}\,du_i.$$

    The cubic map allocates more resolution near the boundaries than a linear-fractional map; these regions matter for this improper integral. Eight independently scrambled nets provide a mean and a **scatter-based standard-error estimate**, not a rigorous error bound. The estimates at different sizes share seeds and are correlated.
    """)
    return


@app.cell
def _(integrand, mo, np, qmc, reference, variables):
    # Evaluate the integrand at a batch of integration points.
    numeric_kernel = integrand.evaluator(variables, n_cores=1, jit_compile=False)
    replicates = 8
    powers = [12, 14, 16, 18]  # points per scramble: 4096 ... 262144
    estimates, standard_errors = ([], [])
    for power in powers:
        replicate_means = []
        for replicate in range(replicates):
            u = qmc.Sobol(d=4, scramble=True, seed=20261005 + replicate).random_base2(power)
            points = (u / (1 - u)) ** 3
            jacobian = np.prod(3 * u ** 2 / (1 - u) ** 4, axis=1)
            values = numeric_kernel.evaluate(points).reshape(-1) * jacobian
            replicate_means.append(values.mean())
        estimates.append(float(np.mean(replicate_means)))
        standard_errors.append(float(np.std(replicate_means, ddof=1) / np.sqrt(replicates)))
    _rows = ['| Points / scramble | Estimate of C | Estimated standard error |', '|---:|---:|---:|']
    _rows += [f'| {2 ** power:,} | {mean:.8f} | {error:.8f} |' for power, mean, error in zip(powers, estimates, standard_errors)]
    mo.output.append(mo.md('\n'.join(_rows)))
    relative_error = abs(estimates[-1] / float(reference) - 1)
    print(f'Final relative difference from 6 ζ(3): {relative_error:.2e}')
    return estimates, powers, standard_errors


@app.cell
def _(estimates, mo, np, plt, powers, reference, standard_errors):
    points_per_scramble = 2**np.array(powers)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    left.errorbar(points_per_scramble, estimates, yerr=standard_errors,
                  fmt="o-", capsize=4, color="#0e7490", label="Scrambled Sobol (±1 estimated SE)")
    left.axhline(float(reference), color="#b45309", linestyle="--", label="Exact: 6 ζ(3)")
    left.set(xscale="log", xlabel="Points per scramble", ylabel="Dimensionless coefficient C",
             title="Four-dimensional numerical cross-check")
    left.legend(fontsize=8)
    left.grid(alpha=.2)

    errors = np.abs(np.array(estimates)/float(reference)-1)
    right.loglog(points_per_scramble, errors, "o-", color="#0e7490")
    right.set(xlabel="Points per scramble", ylabel="Relative difference from exact result",
              title="Convergence with fixed reproducible seeds")
    right.grid(alpha=.2, which="both")
    mo.output.append(fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explore further

    The result $I(Q^2)=6\zeta(3)/Q^2$ connects the two-loop graph to a transcendental constant. The numerical estimates approach the same value as the number of sample points increases.

    Try increasing `powers` or changing the scramble seeds to study the numerical convergence. You can also reverse the integration order or choose a different parameter to fix in the projective gauge. Changing masses or going on shell is a different analytic problem: a generic massive kite can be elliptic, while on-shell limits can require regulation. The finite Euclidean assumptions here are part of the computation.

    ### References

    - [SubTropica online](https://subtropi.ca/) and its [paper companion examples](https://github.com/SubTropica/SubTropica/blob/main/PaperChecks.wl): inspiration for the propagators → Symanzik polynomials → parameter integration workflow; the companion includes a generic massive kite.
    - I. Bierenbaum and S. Weinzierl, [*The massless two-loop two-point function*](https://arxiv.org/abs/hep-ph/0308311): analytic results and multiple-zeta-value structure for this family (the four-dimensional unit-power limit has coefficient $6\zeta(3)$).
    - F. Brown, [*The massless higher-loop two-point function*](https://arxiv.org/abs/0804.1660): background on hyperlogarithmic evaluation and the appearance of multiple zeta values.
    """)
    return


if __name__ == "__main__":
    app.run()
