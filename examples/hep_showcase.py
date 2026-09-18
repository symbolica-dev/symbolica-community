import marimo

__generated_with = "0.24.2"
app = marimo.App(
    width="medium",
    app_title="One-loop gluon propagator and UV expansion",
)


@app.cell
def _():
    import marimo as mo
    from copy import copy
    from symbolica import S, E, N, Expression
    from symbolica.community import hep as fk, idenso
    from symbolica.community.spenso import TensorExpression, as_tensor
    from symbolica.community.hep import Model, SnailFilterOptions, Particle, FeynmanDiagram

    return (
        E,
        Expression,
        FeynmanDiagram,
        Model,
        S,
        SnailFilterOptions,
        TensorExpression,
        as_tensor,
        copy,
        fk,
        idenso,
        mo,
    )


@app.cell
def _(S):
    D, eps, p2, ell2, x = S(
        'hep_gluon::D', 'hep_gluon::eps', 'hep_gluon::p2',
        'hep_gluon::ell2', 'hep_gluon::x',
    )
    P, K = S('hep_gluon::P', 'hep_gluon::K', tags=['spenso::tensor', 'spenso::rank1'])
    L = S('hep_gluon::L')
    RED_P = S('hep_gluon::reduction_P')
    index_ = S('hep_gluon::index_')
    mink, metric, dot = S('spenso::mink', 'spenso::g', 'spenso::dot')
    MOMENTUM = S('FeynKit::Momentum')
    t = S("hep_gluon::t", is_scalar=True) # a scale that can move out of dot products
    Muv2 = S("hep_gluon::Muv2")
    # Python aliases for actual dot expressions, not substitute scalar symbols.
    k2 = dot(K(mink(D)), K(mink(D)))
    kp = dot(K(mink(D)), P(mink(D)))
    UV_K, UV_P = S("hep_gluon::uv_K", "hep_gluon::uv_P", tags=['spenso::tensor', 'spenso::rank1'])
    return (
        D,
        K,
        MOMENTUM,
        Muv2,
        P,
        UV_K,
        UV_P,
        dot,
        eps,
        index_,
        k2,
        kp,
        metric,
        mink,
        p2,
        t,
    )


@app.cell
def _(D, FeynmanDiagram, TensorExpression, as_tensor, index_, mink):
    def numerator_in_d(diagram: FeynmanDiagram) -> TensorExpression:
        """Continue Lorentz slots to D, retaining the external tensor interface."""
        return as_tensor(diagram.numerator_expression().replace(
            mink(4, index_), mink(D, index_)
        ))

    return (numerator_in_d,)


@app.cell
def _(TensorExpression, as_tensor, idenso, qcd_color_factors):
    def contract_indices(expression: TensorExpression) -> TensorExpression:
        """Close internal Dirac, color, and Lorentz indices, retaining external ports."""
        raw = expression.to_expression()
        for _ in range(12):
            simplified = idenso.simplify_metrics(
                idenso.simplify_color(idenso.simplify_gamma(raw.expand()))
            ).expand()
            # This revision mistakes cas(..., coad(8)) for an unresolved tensor
            # port. Evaluate the SU(3) scalar factors before re-inferring ports.
            if simplified == raw:
                return as_tensor(qcd_color_factors(simplified))
            raw = simplified
        raise RuntimeError('Tensor contractions did not reach a fixed point')

    return (contract_indices,)


@app.cell
def _(
    D,
    FeynmanDiagram,
    P,
    TensorExpression,
    as_tensor,
    index_,
    metric,
    mink,
    p2,
):
    def external_projector(diagram: FeynmanDiagram, *, longitudinal=False) -> TensorExpression:
        """Color-average the transverse trace (or the longitudinal Ward check).

        P_T = delta_ab (g_mu_nu - p_mu p_nu/p^2) / [8 (D-1)].
        Thus P_T . [delta_ab (p^2 g_mu_nu - p_mu p_nu) Pi] = p^2 Pi.
        """
        slots = [slot.replace(mink(4, index_), mink(D, index_))
                 for slot in diagram.numerator_expression().list_dangling()]
        mu, nu = [slot for slot in slots if slot.get_name() == 'spenso::mink']
        a, b = [slot for slot in slots if slot.get_name() == 'spenso::coad']
        if longitudinal:
            return as_tensor(metric(a, b) * P(mu) * P(nu) / (8 * p2))
        return as_tensor(metric(a, b) * (metric(mu, nu) - P(mu) * P(nu) / p2) / (8 * (D - 1)))

    def apply_projector(numerator: TensorExpression, projector: TensorExpression) -> TensorExpression:
        """Contract every named external slot with its matching projector slot."""
        if set(numerator.list_dangling()) != set(projector.list_dangling()):
            raise ValueError('Numerator and projector must have matching external slots')
        # Multiple compatible ports make typed multiplication ambiguous.
        # The explicit Einstein indices specify all four contractions here.
        return as_tensor(numerator.to_expression() * projector.to_expression())

    return apply_projector, external_projector


@app.cell
def _(FeynmanDiagram):
    def routing_coefficients(diagram: FeynmanDiagram) -> dict[int, tuple[int, int]]:
        """Convert physical external momenta to source-to-target edge momenta."""
        if diagram.loop_count != 1 or len(diagram.external_edges) != 2:
            raise ValueError('This example requires a one-loop two-point graph')
        vertices = {v.id: v for v in diagram.vertices}
        result = {}
        for edge in diagram.edges:
            signature = diagram.loop_momentum_basis.edge_signatures[edge.id]
            if len(signature.loops) != 1 or any(signature.external[1:]):
                raise ValueError('Expected one independent external momentum')
            loop, external = signature.loops[0], signature.external[0]
            source, target = vertices[edge.source], vertices[edge.target]
            if source.is_external or target.is_external:
                vertex = source if source.is_external else target
                # The basis stores physical incoming/outgoing p, whereas the
                # UFO numerator uses Momentum(edge) along source -> target.
                physical_into_graph = vertex.external_state == 'incoming'
                edge_into_graph = source.is_external
                if physical_into_graph != edge_into_graph:
                    external = -external
            result[edge.id] = (loop, external)
        # Check conservation in the exact convention used by the numerator.
        for vertex in diagram.vertices:
            if vertex.is_external:
                continue
            for component in (0, 1):
                balance = sum(
                    ((edge.target == vertex.id) - (edge.source == vertex.id))
                    * result[edge.id][component] for edge in diagram.edges
                )
                if balance:
                    raise ValueError(f'Momentum is not conserved at vertex {vertex.id}')
        return result

    return (routing_coefficients,)


@app.cell
def _(
    FeynmanDiagram,
    K,
    MOMENTUM,
    P,
    TensorExpression,
    as_tensor,
    index_,
    numerator_in_d,
    routing_coefficients,
):
    def route_numerator(diagram: FeynmanDiagram) -> TensorExpression:
        expression = numerator_in_d(diagram).to_expression()
        for edge_id, (loop, external) in routing_coefficients(diagram).items():
            expression = expression.replace(
                MOMENTUM(edge_id, index_), loop * K(index_) + external * P(index_)
            )
        return as_tensor(expression.expand())

    return (route_numerator,)


@app.cell
def _(
    D,
    P,
    TensorExpression,
    as_tensor,
    contract_indices,
    dot,
    idenso,
    mink,
    p2,
):
    def scalar_products(expression: TensorExpression) -> TensorExpression:
        """Keep compact scalar products inside the structured tensor expression."""
        return as_tensor(idenso.to_dots(contract_indices(expression).to_expression()).replace(
            dot(P(mink(D)), P(mink(D))), p2
        ).expand())

    return (scalar_products,)


@app.cell
def _(Expression, FeynmanDiagram, Model, S, index_):
    def diagram_weight(diagram: FeynmanDiagram, model: Model) -> Expression:
        """Evaluate provenance factors and supply the closed-ghost-loop sign.

        At pinned upstream 0ea21405, closed_fermion_loop_count only counts
        is_fermion() particles (UFO spin 2), excluding spin -1 FP ghosts.
        This example therefore includes their Grassmann minus explicitly.
        """
        factor = diagram.overall_factor_expression() * diagram.numerator_prefactor_expression()
        for name in ('AutG', 'CouplingsMultiplicity', 'ExternalFermionOrderingSign',
                     'InternalFermionLoopSign', 'AntiFermionSpinSumSign'):
            factor = factor.replace(S('feynkit_generator_factor::' + name)(index_), index_)
        if all(model.particle(edge.particle_name).spin == -1 for edge in diagram.internal_edges):
            factor = -factor
        return factor

    return (diagram_weight,)


@app.cell
def _(E, Expression, Model, S, TensorExpression, as_tensor):
    def qcd_color_factors(expression: Expression) -> Expression:
        """Evaluate the scalar SU(3) invariants used by this example."""
        return expression.replace(S('spenso::cas')(2, S('spenso::coad')(8)), 3).replace(
            S('spenso::idx')(2, S('spenso::cof')(3)), E('1/2')
        )

    def resolve_qcd(expression: TensorExpression, model: Model) -> TensorExpression:
        """Resolve QCD couplings and scalar SU(3) factors, preserving tensor type."""
        result = as_tensor(expression).to_expression()
        for coupling in model.couplings:
            if coupling.name in ('GC_10', 'GC_11'):
                result = result.replace(S('UFO::' + coupling.name), coupling.expression)
        return as_tensor(qcd_color_factors(result).expand())

    return qcd_color_factors, resolve_qcd


@app.cell
def _(D, Expression, P, TensorExpression, dot, mink, p2):
    def invariants(expression: TensorExpression) -> Expression:
        """Drop the scalar tensor interface, retaining the loop-momentum dots."""
        if not expression.is_scalar:
            raise ValueError('Contract all external indices before taking the UV series')
        return expression.to_expression().replace(
            dot(P(mink(D)), P(mink(D))), p2
        ).expand()

    return (invariants,)


@app.cell
def _(
    E,
    Expression,
    FeynmanDiagram,
    Model,
    S,
    k2,
    kp,
    p2,
    routing_coefficients,
):
    def bubble_denominator(diagram: FeynmanDiagram, model: Model) -> Expression:
        if len(diagram.internal_edges) != 2:
            raise ValueError('This example supports two-propagator bubbles only')
        denominator = E('1')
        routing = routing_coefficients(diagram)
        for edge in diagram.internal_edges:
            a, b = routing[edge.id]
            particle = model.particle(edge.particle_name)
            mass2 = E('0') if particle.is_massless else S('UFO::' + particle.mass_parameter)**2
            denominator *= a*a*k2 + 2*a*b*kp + b*b*p2 - mass2
        return denominator

    return (bubble_denominator,)


@app.cell
def _(
    D,
    E,
    Expression,
    S,
    TensorExpression,
    UV_K,
    UV_P,
    as_tensor,
    dot,
    index_,
    k2,
    kp,
    mink,
    p2,
):
    def uv_tensor_input(expression: Expression) -> TensorExpression:
        """Convert mixed dot powers to the indexed input FeynKit accepts.

        This only changes notation; no angular averaging is performed here.
        Each contraction receives independent dummy indices.
        Radial dot(K,K) factors remain scalar coefficients.
        """
        indexed = E('0')
        for power, coefficient in expression.expand().coefficient_list(kp):
            rank = 0 if power == E('1') else power.to_polynomial().degree(kp)
            tensor = E('1')
            for n in range(rank):
                mu = S(f'hep_gluon::uv_mu{n}')
                tensor *= UV_K(mink(D, mu)) * UV_P(mink(D, mu))
            indexed += coefficient * tensor
        return as_tensor(indexed)

    def uv_scalar_invariants(expression: Expression) -> Expression:
        """Restore the original loop-vector name and external invariant p2."""
        reduced = expression.replace(dot(UV_K(mink(D)), UV_K(mink(D))), k2)
        reduced = reduced.replace(dot(UV_P(mink(D)), UV_P(mink(D))), p2).expand()
        if reduced.replace(UV_K(index_), 0).replace(UV_P(index_), 0) != reduced:
            raise ValueError('Tensor reduction left an indexed momentum')
        return reduced

    return uv_scalar_invariants, uv_tensor_input


@app.cell
def _(D, UV_K, fk, mink):
    tensor_reducer = fk.TensorReducer(D).with_integrated_vector(UV_K(mink(D)))
    return (tensor_reducer,)


@app.cell
def _(
    D,
    E,
    Expression,
    K,
    copy,
    k2,
    mink,
    t,
    tensor_reducer,
    uv_scalar_invariants,
    uv_tensor_input,
):
    def uv_expansion_data(expression: Expression):
        """Expand a copy for k -> k/t, through t^4, then reduce the vacuum tensors."""
        uv_copy = copy(expression)
        scaled = uv_copy.replace(K(mink(D)), K(mink(D))/t)
        series = scaled.series(t, 0, 4).to_expression().expand()
        reduced = uv_scalar_invariants(tensor_reducer.reduce(uv_tensor_input(series)))
        a = (reduced.coefficient(t**2) * k2).expand()
        b = (reduced.coefficient(t**4) * k2**2).expand()
        remainder = (reduced - a*t**2/k2 - b*t**4/k2**2).expand()
        if remainder != E('0') or a.contains(k2) or b.contains(k2):
            raise ValueError('Expected UV terms A/k2 + B/k2^2 after angular averaging')
        return {'copy': uv_copy, 'series': series, 'reduced': reduced,
                'quadratic': a, 'logarithmic': b, 'residue': b.replace(D, 4).expand()}

    return (uv_expansion_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The gluon propagator at one loop

    Generate the QCD contributions to the gluon two-point function,
    $g^* \to g^*$, and inspect a diagram and its symbolic expressions.
    The external momentum is off shell.
    """)
    return


@app.cell
def _(Model, mo):
    model = Model(mo.notebook_location() / "hep_sm.json")
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
def _(SnailFilterOptions, g, model):
    diagrams = model.generate_diagrams(
        [g], [g], loops=1,
        coupling_orders={"QCD": 2, "QED": 0},
        particle_veto=["c", "t", "s", "u", "d"],
        zero_snails=SnailFilterOptions(),
    )
    return (diagrams,)


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
    diagram_index = mo.ui.dropdown(
        options={
            f"{i}: {', '.join(e.particle_name for e in d.internal_edges)}": i
            for i, d in enumerate(diagrams)
        },
        value="1: g, g",
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
def _(mo):
    mo.md("""
    ## Inspect the numerator

    The Feynman rules return a `TensorExpression`. Below we contract internal
    Dirac, color, and Lorentz indices, keeping the two external gluon indices.
    Lorentz slots are continued to symbolic $D$ before contraction.
    """)
    return


@app.cell
def _(diagram):
    r = diagram.numerator_expression()
    r
    return


@app.cell
def _(contract_indices, diagram, numerator_in_d):
    contracted_numerator = contract_indices(numerator_in_d(diagram))
    contracted_numerator
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Project the external indices

    For off-shell $p^2\ne0$, use the transverse, color-averaged projector
    $$\mathcal P^{ab}_{\mu\nu}=
    \frac{\delta^{ab}}{8(D-1)}
    \left(g_{\mu\nu}-\frac{p_\mu p_\nu}{p^2}\right).$$
    It extracts $p^2\Pi_T$ from
    $\delta^{ab}(p^2g^{\mu\nu}-p^\mu p^\nu)\Pi_T$.
    This replaces the default external polarization vectors; it is not an
    additional polarization sum. The helper reads the actual external slots.
    """)
    return


@app.cell
def _(diagram, external_projector):
    projector = external_projector(diagram)
    projector
    return (projector,)


@app.cell
def _(diagram):
    diagram.loop_momentum_basis
    return


@app.cell
def _(apply_projector, diagram, projector, route_numerator, scalar_products):
    routed_numerator = route_numerator(diagram)
    projected_numerator = scalar_products(apply_projector(routed_numerator, projector))
    projected_numerator
    return (projected_numerator,)


@app.cell
def _(
    as_tensor,
    diagram,
    diagram_weight,
    model,
    projected_numerator,
    resolve_qcd,
):
    weighted_projected_numerator = resolve_qcd(
        as_tensor(projected_numerator * diagram_weight(diagram, model)), model
    )
    weighted_projected_numerator
    return (weighted_projected_numerator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The last expression includes the symmetry factor and closed-loop signs,
    with the model couplings expressed through `UFO::G`.
    At FeynKit revision `0ea21405`, the generator's loop-sign counter excludes
    spin −1 ghosts: `diagram_weight` supplies their Grassmann minus explicitly.
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

    Include the two propagator denominators, then copy the projected expression.
    Declare `t` with `is_scalar=True` and set $k\to k/t$ directly inside the dots.
    Their normalization extracts $1/t^2$ from $k\cdot k$ and $1/t$ from $k\cdot p$;
    no scalar placeholder for either loop-momentum dot is needed. Expand at
    $t=0$ through $t^4$. These are the quadratically, linearly, and logarithmically
    UV-divergent terms in four dimensions; the integration measure is not included
    in the series. External momenta and physical masses are held fixed.
    """)
    return


@app.cell
def _(
    as_tensor,
    bubble_denominator,
    diagram,
    invariants,
    model,
    weighted_projected_numerator,
):
    projected_integrand = invariants(weighted_projected_numerator) / bubble_denominator(diagram, model)
    as_tensor(projected_integrand)
    return (projected_integrand,)


@app.cell
def _(D, K, copy, mink, projected_integrand, t):
    uv_expression_copy = copy(projected_integrand)
    uv_scaled_expression = uv_expression_copy.replace(K(mink(D)), K(mink(D)) / t)
    uv_series = uv_scaled_expression.series(t, 0, 4)
    uv_series
    return uv_expression_copy, uv_series


@app.cell
def _(projected_integrand, t, uv_expression_copy, uv_series):
    # The original expression is still available, with its full denominators.
    assert uv_expression_copy == projected_integrand
    uv_expanded_expression = uv_series.to_expression().replace(t, 1).expand()
    uv_expanded_expression
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tensor-reduce the UV expansion

    Each expanded denominator depends only on $k^2$. The reducer is
    `fk.TensorReducer(D).with_integrated_vector(UV_K(mink(D)))`, selecting
    $k$ and keeping $p$ external. `uv_tensor_input` only restores explicit
    indices for the mixed dots $(k\cdot p)^n$. FeynKit computes the
    tensor reduction, including the vanishing odd-rank terms.
    """)
    return


@app.cell
def _(uv_series, uv_tensor_input):
    uv_indexed_series = uv_tensor_input(uv_series.to_expression())
    return (uv_indexed_series,)


@app.cell
def _(as_tensor, tensor_reducer, uv_indexed_series):
    uv_tensor_reduced = tensor_reducer.reduce(uv_indexed_series)
    as_tensor(uv_tensor_reduced)
    return (uv_tensor_reduced,)


@app.cell
def _(t, uv_scalar_invariants, uv_tensor_reduced):
    uv_reduced_series = uv_scalar_invariants(uv_tensor_reduced)
    uv_reduced_expression = uv_reduced_series.replace(t, 1).expand()
    uv_reduced_expression
    return (uv_reduced_series,)


@app.cell
def _(E, k2, t, uv_reduced_series):
    uv_quadratic_coefficient = (uv_reduced_series.coefficient(t**2) * k2).expand()
    uv_log_coefficient = (uv_reduced_series.coefficient(t**4) * k2**2).expand()
    assert uv_reduced_series.coefficient(t**3) == E('0')
    uv_log_coefficient
    return uv_log_coefficient, uv_quadratic_coefficient


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## UV counterterm

    The angular-averaged asymptotic integrand is $A/k^2+B/(k^2)^2$.
    To give its subtraction an infrared-safe denominator, introduce an auxiliary
    $M_{\rm UV}^2$ and use
    $$C_{\rm UV}(k)=-\frac{A}{k^2-M_{\rm UV}^2}
    -\frac{B-A M_{\rm UV}^2}{(k^2-M_{\rm UV}^2)^2}.$$
    Its large-$k$ expansion cancels $A/k^2+B/(k^2)^2$. The compensating term
    $-A M_{\rm UV}^2$ makes the integrated UV pole independent of the auxiliary mass.
    This is an **angular-averaged** subtraction, not a pointwise tensor subtraction.

    With $D=4-2\epsilon$ and measure $d^Dk/(2\pi)^D$, the integrated MS counterterm
    is $-i B|_{D=4}/(16\pi^2\epsilon)$. The unregulated scaleless UV terms above
    must not be integrated as zero: their UV and infrared poles would cancel.
    """)
    return


@app.cell
def _(Muv2, k2, uv_log_coefficient, uv_quadratic_coefficient):
    uv_counterterm_integrand = -uv_quadratic_coefficient / (k2 - Muv2) - (
        uv_log_coefficient - uv_quadratic_coefficient * Muv2
    ) / (k2 - Muv2)**2
    uv_counterterm_integrand
    return


@app.cell
def _(D, E, eps, uv_log_coefficient):
    uv_residue = uv_log_coefficient.replace(D, 4).expand()
    uv_counterterm = -E('1i') * uv_residue / (16 * E('pi')**2 * eps)
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
    E,
    apply_projector,
    as_tensor,
    bubble_denominator,
    diagram_weight,
    diagrams,
    external_projector,
    invariants,
    mo,
    model,
    resolve_qcd,
    route_numerator,
    scalar_products,
    uv_expansion_data,
):
    _uv_rows = []
    for _d in diagrams:
        _denominator = bubble_denominator(_d, model)
        _weight = diagram_weight(_d, model)
        _transverse = resolve_qcd(as_tensor(scalar_products(apply_projector(route_numerator(_d), external_projector(_d))) * _weight), model)
        _longitudinal = resolve_qcd(as_tensor(scalar_products(apply_projector(route_numerator(_d), external_projector(_d, longitudinal=True))) * _weight), model)
        _uv_rows.append({
            'loop': ', '.join(_e.particle_name for _e in _d.internal_edges),
            'transverse': uv_expansion_data(invariants(_transverse) / _denominator)['residue'],
            'longitudinal': uv_expansion_data(invariants(_longitudinal) / _denominator)['residue'],
        })
    total_uv_residue = sum((_row['transverse'] for _row in _uv_rows), E('0')).expand()
    longitudinal_uv_residue = sum((_row['longitudinal'] for _row in _uv_rows), E('0')).expand()
    assert longitudinal_uv_residue == E('0')
    mo.ui.table([
        {'Loop': _row['loop'],
         'Transverse residue': _row['transverse'].format(color_top_level_sum=False, color_builtin_symbols=False),
         'Longitudinal residue': _row['longitudinal'].format(color_top_level_sum=False, color_builtin_symbols=False)}
        for _row in _uv_rows
    ], selection=None)
    return (total_uv_residue,)


@app.cell
def _(E, eps, total_uv_residue):
    total_uv_counterterm = -E('1i') * total_uv_residue / (16 * E('pi')**2 * eps)
    total_uv_counterterm
    return


if __name__ == "__main__":
    app.run()
