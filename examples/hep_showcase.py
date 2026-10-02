import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="One-loop gluon propagator and UV expansion",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The gluon propagator at one loop

    Generate the QCD contributions to the gluon two-point function,
    $g^* \to g^*$, and inspect a diagram and its symbolic expressions.
    The external momentum is off shell.
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
    from symbolica.community.hepkit import Symbols

    return (Symbols,)


@app.cell(hide_code=True)
def _():
    from symbolica.community import tensor as sp
    import marimo as mo
    from copy import copy
    from typing import TypedDict
    from symbolica import S, E, Expression, AtomType
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.tensor import (
        Representation,
        TensorExpression,
        TensorName,
        TensorPattern,
        dot,
    )
    from symbolica.community.hepkit import Model, FeynmanDiagram

    _set_namespace("hep_gluon")

    contraction_settings = dict()

    class UVRow(TypedDict):
        loop: str
        transverse: TensorExpression
        longitudinal: TensorExpression

    return (
        AtomType,
        E,
        Expression,
        FeynmanDiagram,
        Model,
        Representation,
        S,
        TensorExpression,
        TensorName,
        TensorPattern,
        UVRow,
        contraction_settings,
        copy,
        dot,
        mo,
        sp,
    )


@app.cell(hide_code=True)
def _(D, FeynmanDiagram, TensorExpression, adjoint):
    color_settings = dict(color=True, color_substitute_cof_dimension_invariants=True)

    def project_color(diagram: FeynmanDiagram) -> TensorExpression:
        """Average external color, retaining the complete Lorentz/Dirac numerator."""
        numerator = diagram.numerator_expression(in_lmb=True).with_lorentz_dimension(D)
        a, b = [
            slot for slot in numerator.structure.slots if slot.representation == adjoint
        ]
        projector = TensorExpression.g(adjoint)(a, b) / adjoint.dimension
        projected = numerator * projector
        return projected.simplify_algebra(
            contract="selected", representations=[adjoint], **color_settings
        ).to_dots()

    return (project_color,)


@app.cell(hide_code=True)
def _(D, P, TensorExpression, contraction_settings, lorentz, p2, sp):
    def external_projector(
        numerator: TensorExpression, *, longitudinal=False
    ) -> TensorExpression:
        """Project the Lorentz trace after the separate color average.

        P_T = (g_mu_nu - p_mu p_nu/p^2) / (D-1).
        Thus P_T . [(p^2 g_mu_nu - p_mu p_nu) Pi] = p^2 Pi.
        """
        if numerator == 0:
            return numerator
        mu, nu = [
            slot for slot in numerator.structure.slots if slot.representation == lorentz
        ]
        momentum_pair = P(mu).outer(P(nu)) / p2
        if longitudinal:
            return momentum_pair
        lorentz_metric = TensorExpression.g(lorentz)(mu, nu)
        return (lorentz_metric - momentum_pair) / (D - 1)

    def apply_projector(
        numerator: TensorExpression, projector: TensorExpression
    ) -> TensorExpression:
        """Contract every named external slot with its matching projector slot."""
        if numerator == 0:
            return numerator
        if set(numerator.list_dangling()) != set(projector.list_dangling()):
            raise ValueError(
                "Numerator and projector must have matching external slots"
            )
        # Preserve the named Einstein indices when attaching the projector.
        # Contract connected factors first. Only distribute residual metric
        # structures; scalar coefficients and unrelated sums stay factored.
        return (
            (numerator * projector)
            .contract()
            .to_dots()
            .expand(sp.TensorName.g().to_expression())
            .contract(**contraction_settings)
            .to_dots()
        )

    return apply_projector, external_projector


@app.cell(hide_code=True)
def _(Expression, FeynmanDiagram):
    def diagram_weight(diagram: FeynmanDiagram) -> Expression:
        """Evaluate native graph factors, including the closed-ghost-loop sign once."""
        return (
            diagram.overall_factor_expression(evaluate=True)
            * diagram.numerator_prefactor_expression()
        )

    return (diagram_weight,)


@app.cell(hide_code=True)
def _(
    AtomType,
    E,
    FeynmanDiagram,
    K,
    MOMENTUM,
    P,
    TensorExpression,
    lorentz,
    prop,
):
    def graph_propagators(diagram: FeynmanDiagram) -> TensorExpression:
        """Keep the instantiated model denominators and native graph routing."""
        if len(diagram.internal_edges) != 2:
            raise ValueError("This example supports two-propagator bubbles only")
        product = E("1")
        for edge in diagram.internal_edges:
            quadratic = edge.propagator.denominator
            (model_momentum,) = [
                variable
                for variable in quadratic.get_all_indeterminates(enter_functions=False)
                if variable.get_type() == AtomType.Fn
            ]
            mass_squared = -quadratic.replace(model_momentum, 0)
            if (quadratic + mass_squared).expand() != model_momentum**2:
                raise ValueError(
                    "This example requires propagators of the form P^2 - M^2"
                )
            momentum = diagram.loop_momentum_basis.route_expression(
                MOMENTUM(edge.id, lorentz), loop_momenta=[K], external_momenta=[P]
            )
            product *= prop(momentum.to_expression(), mass_squared)
        return TensorExpression(product)

    return (graph_propagators,)


@app.cell(hide_code=True)
def _(Expression, TensorExpression, dot_pattern, k, mUV, mass2_, prop, q_, t):
    def evaluate_propagators(expression: Expression) -> TensorExpression:
        """Interpret prop(q, M2) as 1/(q.q-M2), retaining the model's mass term."""
        return TensorExpression(
            expression.to_expression().replace(
                prop(q_, mass2_), 1 / (dot_pattern(q_, q_) - mass2_)
            )
        )

    def uv_deform(expression: Expression) -> TensorExpression:
        """Scale the full integrand and shift normalized propagator masses."""
        scaled = expression.to_expression().replace(k, k / t)
        # Factoring t^2 from a rescaled propagator rescales its mass to t^2*m^2.
        # Apply prop(q,M2) -> prop(q,M2+(1-t^2)*mUV^2) in that normalization.
        return TensorExpression(
            scaled.replace(
                prop(q_, mass2_),
                t**2 * prop(t * q_, t**2 * mass2_ + (1 - t**2) * mUV**2),
            )
        )

    return evaluate_propagators, uv_deform


@app.cell(hide_code=True)
def _(
    Expression,
    FeynmanDiagram,
    MOMENTUM,
    TensorExpression,
    TensorPattern,
    dot,
    index_,
    k,
    k2,
    lorentz,
):
    def uv_tensor_input(
        expression: Expression, diagram: FeynmanDiagram
    ) -> TensorExpression:
        """Name the loop vector with its graph edge, retaining compact dot products."""
        (loop_edge,) = diagram.loop_momentum_basis.loop_edges
        # The reducer handles mixed dot powers and their independent contractions;
        # radial dot(Q,Q) factors remain scalar coefficients.
        return TensorExpression(
            expression.to_expression().replace(
                k, MOMENTUM(loop_edge, lorentz).to_expression()
            )
        )

    def uv_scalar_invariants(
        expression: Expression, diagram: FeynmanDiagram
    ) -> TensorExpression:
        """Restore the notebook's loop square after reduction."""
        (loop_edge,) = diagram.loop_momentum_basis.loop_edges
        loop_vector = MOMENTUM(loop_edge, lorentz)
        reduced = expression.to_expression().replace(
            dot(loop_vector, loop_vector).to_expression(), k2
        )
        if (
            reduced.replace(
                TensorPattern(MOMENTUM, args=[loop_edge], ports=[index_]), 0
            )
            != reduced
        ):
            raise ValueError("Tensor reduction left an unreduced loop momentum")
        return TensorExpression(reduced)

    return uv_scalar_invariants, uv_tensor_input


@app.cell(hide_code=True)
def _(D, TensorExpression, k, k2, uv_probe):
    def uv_pole_residue(expression: TensorExpression) -> TensorExpression:
        """Extract the logarithmic radial tail: its one-loop pole is i/(16*pi^2*eps)."""
        tail = expression.to_expression().replace(k, k / uv_probe)
        # Include the one-loop measure and keep its scale-independent term.
        logarithmic = (tail / uv_probe**4).series(uv_probe, 0, 0)[0]
        return TensorExpression((logarithmic * k2**2).replace(D, 4).expand(k2))

    return (uv_pole_residue,)


@app.cell(hide_code=True)
def _(
    D,
    FeynmanDiagram,
    TensorExpression,
    copy,
    evaluate_propagators,
    t,
    uv_deform,
    uv_pole_residue,
    uv_scalar_invariants,
    uv_tensor_input,
):
    def uv_expansion_data(
        expression: TensorExpression, diagram: FeynmanDiagram
    ) -> dict[str, TensorExpression]:
        """Expand an expression copy, with numerator and massive UV denominators together."""
        uv_copy = copy(expression)
        deformed = uv_deform(uv_copy)
        measure_factor = t ** (-4 * diagram.loop_count)
        with_measure = evaluate_propagators(deformed) * measure_factor
        # Retain every UV-divergent power, then remove the measure bookkeeping.
        series = TensorExpression(
            (
                with_measure.to_expression().series(t, 0, 0).to_expression()
                / measure_factor
            ).expand(t)
        )
        reduced = uv_scalar_invariants(
            diagram.tensor_reduce(D, expression=uv_tensor_input(series, diagram)),
            diagram,
        )
        counterterm = TensorExpression(-reduced.to_expression().replace(t, 1))
        return {
            "copy": uv_copy,
            "deformed": deformed,
            "series": series,
            "reduced": reduced,
            "counterterm": counterterm,
            "residue": -uv_pole_residue(counterterm),
        }

    return (uv_expansion_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell
def _(Representation, S, Symbols, TensorName, dot, sp):
    D, eps = S("D", "eps")
    lorentz = Representation.mink(D)
    adjoint = Representation.coad(8)
    P = TensorName.vector("p")
    loop_momenta = [
        TensorName.vector("k"),
        TensorName.vector("l"),
        TensorName.vector("m"),
        TensorName.vector("n"),
    ]
    K = loop_momenta[0]
    p, k = P(lorentz).to_expression(), K(lorentz).to_expression()
    index_ = S("index_")
    MOMENTUM = TensorName(Symbols.edge_momentum.get_name())
    t = S("t", is_scalar=True)  # a scale that can move out of dot products
    mUV = S("mUV", is_scalar=True)
    uv_probe = S("uv_probe", is_scalar=True)
    prop = S("prop", is_scalar=True)
    q_, mass2_ = S("q_", "mass2_")
    # Rewrite patterns contain wildcards rather than concrete rank-one tensors.
    dot_pattern = sp.TensorPattern.dot
    # Python aliases for actual dot expressions, not substitute scalar symbols.
    k2, p2 = dot(k, k).to_expression(), dot(p, p).to_expression()
    return (
        D,
        K,
        MOMENTUM,
        P,
        adjoint,
        dot_pattern,
        eps,
        index_,
        k,
        k2,
        loop_momenta,
        lorentz,
        mUV,
        mass2_,
        p2,
        prop,
        q_,
        t,
        uv_probe,
    )


@app.cell
def _(Model):
    model = Model.standard_model()
    g = model.particle("g")
    g
    return g, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate the diagrams

    Keep exactly two powers of the QCD coupling and no QED vertices.
    The current selection keeps the gluon, ghost, and bottom-quark bubbles.
    The zero-snail filter removes the massless tadpole.
    """)
    return


@app.cell
def _(g, model):
    loops = 1
    diagrams = model.process(
        [g], [g], particle_veto=["c", "t", "s", "u", "d"]
    ).generate_diagrams(loops=loops, coupling_orders={"QCD": 2 * loops, "QED": 0})
    return diagrams, loops


@app.cell(hide_code=True)
def _(diagrams, mo):
    mo.md(f"""
    **{len(diagrams)} diagrams** with the filters above.
    The bottom mass is kept symbolic. Remove the particle veto to include
    the other quark flavors.
    """)
    return


@app.cell(hide_code=True)
def _(diagrams, mo):
    _options = {
        f"{i}: {', '.join(e.particle_name for e in d.internal_edges)}": i
        for i, d in enumerate(diagrams)
    }
    _default = next(
        (
            label
            for label, index in _options.items()
            if len(diagrams[index].internal_edges) == 2
            and all(
                edge.particle_name == "g" for edge in diagrams[index].internal_edges
            )
        ),
        next(iter(_options)),
    )
    diagram_index = mo.ui.dropdown(
        options=_options,
        # Select the gluon bubble without assuming its generated diagram index.
        value=_default,
        label="Particles in the loop",
    )
    diagram_index
    return (diagram_index,)


@app.cell
def _(diagram_index, diagrams):
    diagram = diagrams[diagram_index.value]
    diagram
    return (diagram,)


@app.cell(hide_code=True)
def _(diagram, mo):
    _options = {
        "None": ("none", None),
        "Internal lines": ("internal", None),
        "External lines": ("external", None),
    }
    _options.update(
        {
            f"{name} lines": ("particle", name)
            for name in sorted({edge.particle_name for edge in diagram.edges})
        }
    )
    highlight_selection = mo.ui.dropdown(
        options=_options,
        value="Internal lines",
        label="Highlight subgraph",
    )
    highlight_selection
    return (highlight_selection,)


@app.cell
def _(diagram, highlight_selection):
    # Physics subgraphs retain the parent diagram and its particle drawing styles.
    _kind, _particle = highlight_selection.value
    if _kind == "internal":
        highlighted_subgraph = diagram.filter(
            edge=lambda edge: not edge.data.is_external
        )
    elif _kind == "external":
        highlighted_subgraph = diagram.filter(edge=lambda edge: edge.data.is_external)
    elif _kind == "particle":
        highlighted_subgraph = diagram.filter(
            edge=lambda edge: edge.data.particle_name == _particle
        )
    else:
        highlighted_subgraph = None
    highlighted_subgraph
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspect the numerator

    The Feynman rules return a `TensorExpression`. First project the external
    color indices with $\delta^{ab}/8$ and contract all color tensors.
    Their scalar factors remain part of the projected numerator.
    The remaining expressions contain only Lorentz and Dirac tensors, with
    Lorentz slots continued to symbolic $D$ before contraction. Edge momenta
    are routed into the graph's loop-momentum basis before these operations,
    allowing equivalent terms to cancel during local tensor expansions.
    """)
    return


@app.cell
def _(diagram):
    diagram.numerator_expression(in_lmb=True)
    return


@app.cell
def _(diagram):
    r = diagram.numerator_expression(in_lmb=False)
    r
    return (r,)


@app.cell
def _(r):
    r.structure
    return


@app.cell
def _(Representation, mo):
    _display_rep = Representation.mink(4)
    mo.hstack(
        [
            mo.as_html(_display_rep.name),
            mo.as_html(_display_rep),
            mo.as_html(_display_rep(1)),
        ],
        justify="start",
        align="start",
        gap=2,
        wrap=True,
    )
    return


@app.cell
def _(TensorExpression):
    TensorExpression.dirac_gamma(4)(1, 2, 1).structure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Project and contract color

    Contract the numerator with the normalized adjoint projector. Scalar
    color factors and the relative coefficients of different color structures
    remain in the resulting Lorentz tensor.
    """)
    return


@app.cell
def _(diagram, project_color):
    color_projected_numerator = project_color(diagram)
    color_projected_numerator
    return (color_projected_numerator,)


@app.cell
def _(color_projected_numerator):
    color_projected_numerator
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Collect bispinor chains

    Choose a diagram containing a bottom-quark line in **Particles in the loop**
    to see slash notation: $\not{q}=\gamma^\mu q_\mu$.
    This view collects the bispinor factors into ordered chains and traces,
    leaving the Dirac traces unevaluated.
    """)
    return


@app.cell
def _(color_projected_numerator):
    bispinor_numerator = color_projected_numerator.contract().to_dots()
    bispinor_numerator
    return


@app.cell
def _(color_projected_numerator, contraction_settings):
    # Edge momenta already use the same loop basis, so equivalent terms can
    # cancel at each contraction. Expand only connected tensor sums, keeping
    # unrelated scalar factors intact.
    contracted_numerator = color_projected_numerator.simplify_algebra(
        contract="dots", gamma=True, epsilon=True
    )
    contracted_numerator
    return (contracted_numerator,)


@app.cell
def _(contracted_numerator):
    dotted = contracted_numerator.to_dots()
    return (dotted,)


@app.cell
def _(dotted):
    dotted.to_expression().collect_factors().collect_num()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Project the external Lorentz indices

    Color has already been averaged. For off-shell $p^2\ne0$, use
    $$\mathcal P_{\mu\nu}=
    \frac{1}{D-1}
    \left(g_{\mu\nu}-\frac{p_\mu p_\nu}{p^2}\right).$$
    It extracts $p^2\Pi_T$ from
    $(p^2g^{\mu\nu}-p^\mu p^\nu)\Pi_T$.
    This replaces the default external polarization vectors; it is not an
    additional polarization sum. The helper reads the actual external slots.
    """)
    return


@app.cell
def _(color_projected_numerator, external_projector):
    projector = external_projector(color_projected_numerator)
    projector
    return (projector,)


@app.cell
def _(diagram):
    diagram.loop_momentum_basis
    return


@app.cell
def _(
    P,
    apply_projector,
    contracted_numerator,
    diagram,
    loop_momenta,
    loops,
    projector,
):
    routed_numerator = diagram.loop_momentum_basis.route_expression(
        contracted_numerator, loop_momenta=loop_momenta[:loops], external_momenta=[P]
    )
    projected_numerator = (
        apply_projector(routed_numerator, projector).contract().to_dots()
    )
    projected_numerator
    return (projected_numerator,)


@app.cell
def _(diagram, diagram_weight, model, projected_numerator):
    weighted_projected_numerator = model.expand_couplings(
        projected_numerator * diagram_weight(diagram)
    )
    weighted_projected_numerator
    return (weighted_projected_numerator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The last expression includes the symmetry factor and closed-loop signs,
    with the model couplings expressed through the model’s strong coupling. The global
    The color average is already included in this expression and the UV steps.
    The native graph factor already includes the Grassmann minus for each
    closed ghost loop; `diagram_weight` evaluates it without an extra sign.
    All algebra functions are defined in the notebook cells above.

    The cells below expand an expression copy, apply FeynKit's tensor reduction,
    and compute the integrated UV counterterm.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Build the energy denominators

    `build_cff()` constructs the Cross-Free Family representation of this
    diagram. Its output is a normal Symbolica expression. This step constructs
    the denominators; it does not evaluate the loop integral.
    """)
    return


@app.cell
def _(diagram):
    cff = diagram.build_cff().to_expression()
    cff
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## UV expansion on a copy of the expression

    Read each internal edge's mass from the model and use
    the graph's momentum routing. Here `prop(q, M2)` denotes $1/(q^2-M^2)$.
    Copy the projected integrand with $C_{\rm color}$ kept outside and scale
    $k\to k/t$ in numerator and propagators together.
    After extracting $t^2$ from each propagator, apply
    $$\operatorname{prop}(k+tp,t^2m^2)\longrightarrow
    \operatorname{prop}(k+tp,t^2m^2+(1-t^2)m_{\rm UV}^2).$$
    At $t=1$ this recovers the physical integrand; at $t=0$ the denominator is
    $k^2-m_{\rm UV}^2$. Include the loop-measure factor $t^{-4L}$ for
    four-dimensional power counting and expand through $t^0$. For these
    quadratically divergent bubbles, the series starts at $t^{-2}$.
    Remove the measure factor before tensor reduction to recover the
    counterterm integrand. This bookkeeping does not change the symbolic
    dimension $D$ used in the contractions.
    All momentum squares remain dot products, including $p\cdot p$.
    """)
    return


@app.cell
def _(diagram, graph_propagators, weighted_projected_numerator):
    graph_propagator_product = graph_propagators(diagram)
    if not weighted_projected_numerator.is_scalar:
        raise ValueError("Contract all external indices before taking the UV series")
    projected_integrand = weighted_projected_numerator * graph_propagator_product
    projected_integrand
    return (projected_integrand,)


@app.cell
def _(D, P, diagram, loop_momenta, loops):
    graph_denominator = diagram.loop_momentum_basis.route_expression(
        diagram.denominator_expression(dimension=D),
        loop_momenta=loop_momenta[:loops],
        external_momenta=[P],
    )
    graph_denominator
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same massive expansion is available directly on a diagram. It retains
    every UV-divergent order automatically and returns the additive local
    counterterm in edge momenta with tagged `denom` propagators. Diagram-wide
    weights and Lorentz projectors remain separate. Pass `numerator=` for a prepared
    numerator, or call `diagram.filter(...).uv_counterterm(mUV)` to expand a selected region.
    This is a single UV limit, before forest subtraction and pole integration.
    """)
    return


@app.cell
def _(color_projected_numerator, diagram, mUV):
    diagram_local_uv_counterterm = diagram.uv_counterterm(
        mUV, numerator=color_projected_numerator
    )
    diagram_local_uv_counterterm
    return


@app.cell
def _(copy, projected_integrand, uv_deform):
    uv_expression_copy = copy(projected_integrand)
    uv_deformed_expression = uv_deform(uv_expression_copy)
    uv_deformed_expression
    return (uv_deformed_expression,)


@app.cell
def _(diagram, evaluate_propagators, t, uv_deformed_expression):
    uv_scaled_expression = evaluate_propagators(uv_deformed_expression)
    uv_measure_factor = t ** (-4 * diagram.loop_count)
    uv_with_measure = uv_scaled_expression * uv_measure_factor
    uv_series = uv_with_measure.to_expression().series(t, 0, 0)
    uv_series
    return uv_measure_factor, uv_series


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tensor-reduce the UV expansion

    Each expanded denominator depends only on $k^2$.
    `diagram.tensor_reduce(D, expression=uv_routed_series)` identifies the
    internal momentum from the graph and keeps $p$ external.
    `uv_tensor_input` names $k$ with its graph edge while retaining
    compact dot products. FeynKit reduces the mixed powers $(k\cdot p)^n$
    directly, including the vanishing odd-rank terms.
    """)
    return


@app.cell
def _(
    TensorExpression,
    diagram,
    uv_measure_factor,
    uv_series,
    uv_tensor_input,
):
    uv_routed_series = uv_tensor_input(
        TensorExpression(uv_series.to_expression() / uv_measure_factor), diagram
    )
    return (uv_routed_series,)


@app.cell
def _(D, diagram, uv_routed_series):
    uv_tensor_reduced = diagram.tensor_reduce(D, expression=uv_routed_series)
    uv_tensor_reduced
    return (uv_tensor_reduced,)


@app.cell
def _(TensorExpression, diagram, t, uv_scalar_invariants, uv_tensor_reduced):
    uv_reduced_series = uv_scalar_invariants(uv_tensor_reduced, diagram)
    uv_reduced_expression = TensorExpression(
        uv_reduced_series.to_expression().replace(t, 1)
    )
    uv_reduced_expression
    return (uv_reduced_series,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## UV counterterm

    The UV expansion already has massive denominators
    $(k^2-m_{\rm UV}^2)^n$. Tensor-reduce the series, set $t=1$, and negate it
    to obtain the angular-averaged counterterm. No mass is inserted afterward.

    The integrated pole is determined by the coefficient of $1/(k^2)^2$
    in its large-$k$ radial expansion. With $D=4-2\epsilon$ and measure
    $d^Dk/(2\pi)^D$, multiply that coefficient by $i/(16\pi^2\epsilon)$.
    The auxiliary $m_{\rm UV}$ cancels from this coefficient. The color
    average is already included in the numerator.
    Finite parts are not evaluated.
    """)
    return


@app.cell
def _(TensorExpression, t, uv_reduced_series):
    uv_counterterm_integrand = TensorExpression(
        -uv_reduced_series.to_expression().replace(t, 1)
    )
    uv_counterterm_integrand
    return (uv_counterterm_integrand,)


@app.cell
def _(E, eps, uv_counterterm_integrand, uv_pole_residue):
    uv_residue = -uv_pole_residue(uv_counterterm_integrand)
    uv_counterterm = -E("1i") * uv_residue / (16 * E("pi") ** 2 * eps)
    uv_counterterm
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check the sum

    For the selected gluon, ghost, and bottom loops, the transverse pole coefficient
    is $13G^2p^2/3$. The longitudinal pole cancels. This is a gluon-field counterterm;
    it is not the QCD beta-function coefficient. Keep $p^2\ne0$.
    """)
    return


@app.cell
def _(
    K,
    P,
    UVRow,
    apply_projector,
    contraction_settings,
    diagram_weight,
    diagrams,
    external_projector,
    graph_propagators,
    model,
    project_color,
    uv_expansion_data,
):
    uv_rows = []
    for _d in diagrams:
        _propagators = graph_propagators(_d)
        _weight = diagram_weight(_d)
        _stripped = project_color(_d)
        _routed = _d.loop_momentum_basis.route_expression(
            _stripped.simplify_algebra(contract="dots", gamma=True, epsilon=True),
            loop_momenta=[K],
            external_momenta=[P],
        )
        _transverse = (
            apply_projector(_routed, external_projector(_routed)).contract().to_dots()
        )
        _longitudinal = (
            apply_projector(_routed, external_projector(_routed, longitudinal=True))
            .contract()
            .to_dots()
        )
        _transverse = model.expand_couplings(_transverse * _weight)
        _longitudinal = model.expand_couplings(_longitudinal * _weight)
        if not _transverse.is_scalar or not _longitudinal.is_scalar:
            raise ValueError(
                "Contract all external indices before taking the UV series"
            )
        uv_rows.append(
            UVRow(
                loop=", ".join((_e.particle_name for _e in _d.internal_edges)),
                transverse=uv_expansion_data(_transverse * _propagators, _d)["residue"],
                longitudinal=uv_expansion_data(_longitudinal * _propagators, _d)[
                    "residue"
                ],
            )
        )
    return (uv_rows,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the total ultraviolet pole
    """)
    return


@app.cell
def _(E, TensorExpression, uv_rows):
    total_uv_residue = sum(
        (_row["transverse"] for _row in uv_rows), TensorExpression(0)
    )
    longitudinal_uv_residue = (
        sum((_row["longitudinal"] for _row in uv_rows), TensorExpression(0))
        .to_expression()
        .together()
    )
    assert longitudinal_uv_residue == E("0")
    return (total_uv_residue,)


@app.cell(hide_code=True)
def _(mo, uv_rows):
    mo.ui.table(
        [
            {
                "Loop": _row["loop"],
                "Transverse residue": _row["transverse"],
                "Longitudinal residue": _row["longitudinal"],
            }
            for _row in uv_rows
        ],
        selection=None,
    )
    return


@app.cell
def _(E, eps, total_uv_residue):
    total_uv_counterterm = -E("1i") * total_uv_residue / (16 * E("pi") ** 2 * eps)
    total_uv_counterterm
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    [Browse the separate HEP examples](/)
    """)
    return


if __name__ == "__main__":
    app.run()
