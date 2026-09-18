import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="Feynman-parameter UV cross-check")


@app.cell
def _():
    import marimo as mo
    from copy import copy
    from symbolica import S, E, N, Expression
    from symbolica.community import idenso
    from symbolica.community.spenso import TensorExpression, as_tensor
    from symbolica.community.hep import Model, SnailFilterOptions, Particle, FeynmanDiagram, TensorReducer

    return (
        E,
        Expression,
        FeynmanDiagram,
        SnailFilterOptions,
        Model,
        S,
        TensorExpression,
        as_tensor,
        TensorReducer,
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
    k2, kp, t, Muv2 = S("hep_gluon::k2", "hep_gluon::kp", "hep_gluon::t", "hep_gluon::Muv2")
    UV_K, UV_P = S("hep_gluon::uv_K", "hep_gluon::uv_P")
    return (
        D,
        K,
        L,
        MOMENTUM,
        P,
        RED_P,
        dot,
        ell2,
        eps,
        index_,
        metric,
        mink,
        p2,
        x,
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
def _(D, FeynmanDiagram, P, TensorExpression, as_tensor, index_, metric, mink, p2):
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
def _(FeynmanDiagram, K, MOMENTUM, P, TensorExpression, as_tensor, index_, numerator_in_d, routing_coefficients):
    def route_numerator(diagram: FeynmanDiagram) -> TensorExpression:
        expression = numerator_in_d(diagram).to_expression()
        for edge_id, (loop, external) in routing_coefficients(diagram).items():
            expression = expression.replace(
                MOMENTUM(edge_id, index_), loop * K(index_) + external * P(index_)
            )
        return as_tensor(expression.expand())

    return (route_numerator,)


@app.cell
def _(D, P, TensorExpression, as_tensor, contract_indices, dot, idenso, mink, p2):
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
def _(
    D,
    E,
    FeynmanDiagram,
    K,
    L,
    Model,
    P,
    RED_P,
    S,
    TensorReducer,
    as_tensor,
    apply_projector,
    diagram_weight,
    dot,
    ell2,
    external_projector,
    idenso,
    index_,
    mink,
    p2,
    resolve_qcd,
    route_numerator,
    routing_coefficients,
    x,
):
    def bubble_uv_data(diagram: FeynmanDiagram, model: Model, *, longitudinal=False):
        """Feynman-parameterize, shift, tensor-reduce and extract a bubble UV pole.

        Nothing mutates the original graph. The returned coefficient excludes
        1/eps and i/(16*pi^2), and includes diagram weights and model couplings.
        """
        copy = FeynmanDiagram.from_json(model, diagram.to_json())
        if len(copy.internal_edges) != 2:
            raise ValueError('UV example supports two-propagator bubbles only')
        routing = routing_coefficients(copy)
        chord = copy.loop_momentum_basis.loop_edges[0]
        first = next(edge for edge in copy.internal_edges if edge.id == chord)
        second = next(edge for edge in copy.internal_edges if edge.id != chord)
        if routing[chord] != (1, 0):
            raise ValueError('The first propagator must carry k')
        loop, external = routing[second.id]
        if abs(loop) != 1 or abs(external) != 1:
            raise ValueError('Expected the second propagator to carry +/- (k +/- p)')
        masses = [model.particle(edge.particle_name).mass_parameter for edge in (first, second)]
        if masses[0] != masses[1]:
            raise ValueError('This example requires equal propagator masses')
        mass2 = E('0') if model.particle(first.particle_name).is_massless else S('UFO::' + masses[0])**2
        delta = mass2 - x * (1 - x) * p2
        shift_sign = external // loop
        projector = external_projector(copy, longitudinal=longitudinal)
        shifted = apply_projector(route_numerator(copy), projector).replace(
            K(index_), L(index_) - shift_sign * x * P(index_)
        ).replace(P(index_), RED_P(index_)).expand()
        # Sew gamma chains before reduction; retain explicit loop-vector indices.
        for _ in range(12):
            new = idenso.simplify_gamma(idenso.simplify_color(shifted)).expand()
            if new == shifted:
                break
            shifted = new
        else:
            raise RuntimeError('Dirac/color algebra did not converge')
        reducer = TensorReducer(D).with_integrated_vector(L(mink(D)))
        reduced = reducer.reduce(shifted)
        reduced = reduced.replace(dot(RED_P(mink(D)), RED_P(mink(D))), p2)
        reduced = reduced.replace(dot(L(mink(D)), L(mink(D))), ell2).expand()
        # Mixed loop/external dots would indicate that reduction was incomplete.
        if reduced.replace(L(index_), 0) != reduced:
            raise ValueError('Unreduced loop momentum remains')
        a = reduced.coefficient(ell2)
        b = reduced.replace(ell2, 0)
        if (reduced - a * ell2 - b).expand() != E('0'):
            raise ValueError('Expected a numerator linear in ell^2 after reduction')
        # For d^D ell/(i*pi^(D/2)): pole[J_2] = 1/eps and
        # pole[ell^2 J_2] = 2*Delta/eps. Take D -> 4 only for the residue.
        residue_x = (2 * delta * a + b).replace(D, 4).expand()
        primitive = residue_x.integrate(x)
        residue = (primitive.replace(x, 1) - primitive.replace(x, 0)).expand()
        residue = resolve_qcd(as_tensor(residue * diagram_weight(copy, model)), model).to_expression()
        return dict(diagram=copy, delta=delta, shifted=shifted, reducer=reducer,
                    reduced=reduced, residue_x=residue_x, residue=residue)

    return (bubble_uv_data,)


@app.cell
def _(mo):
    mo.md(r"""
    # The gluon propagator: UV counterterm

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


@app.cell
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


@app.cell
def _(diagrams, mo):
    mo.md(f"""
    **{len(diagrams)} diagrams** with the filters above.
    The bottom mass is kept symbolic. Remove the particle veto to include
    the other quark flavors.
    """)
    return


@app.cell
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


@app.cell
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


@app.cell
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
def _(as_tensor, diagram, diagram_weight, model, projected_numerator, resolve_qcd):
    weighted_projected_numerator = resolve_qcd(
        as_tensor(projected_numerator * diagram_weight(diagram, model)), model
    )
    weighted_projected_numerator
    return


@app.cell
def _(mo):
    mo.md("""
    The last expression includes the symmetry factor and closed-loop signs,
    with the model couplings expressed through `UFO::G`.
    At FeynKit revision `0ea21405`, the generator's loop-sign counter excludes
    spin −1 ghosts: `diagram_weight` supplies their Grassmann minus explicitly.
    All algebra functions are defined in the notebook cells above.

    This optional reference notebook uses Feynman parameters to check the result.
    """)
    return


@app.cell
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


@app.cell
def _(mo):
    mo.md(r"""
    ## UV copy: combine denominators and shift the loop momentum

    Work on a JSON round-trip copy of the selected diagram. For these
    equal-mass bubbles, the propagators are $k^2-m^2$ and $(k+r p)^2-m^2$,
    with $r=\pm1$ read from the routing. Feynman parametrization gives
    $$\frac1{AB}=\int_0^1\!dx\,
      \frac1{[\ell^2-\Delta+i0]^2},\qquad
      \ell=k+r x p,\quad \Delta=m^2-x(1-x)p^2.$$
    Keep $p^2\ne0$ (spacelike for the pole calculation), so the massless bubbles
    have no infrared pole. The selected bottom loop retains its mass.

    We use conventional dimensional regularization, $D=4-2\epsilon$, with
    $\mathrm{tr}(1)=4$. Only the UV residue is computed, not the finite part.
    """)
    return


@app.cell
def _(bubble_uv_data, diagram, model):
    uv_data = bubble_uv_data(diagram, model)
    uv_diagram_copy = uv_data["diagram"]
    uv_diagram_copy
    return (uv_data,)


@app.cell
def _(uv_data):
    delta = uv_data["delta"]
    delta
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Tensor reduction in the shifted vacuum denominator

    Select **only** $\ell$ as integrated; $p$ is fixed. The reducer implements
    identities such as
    $$\ell^\mu\ell^\nu\longmapsto
       \frac{\ell^2}{D}g^{\mu\nu},\qquad
       (\ell\cdot p)^2\longmapsto\frac{\ell^2p^2}{D}.$$
    The shifted denominator depends only on $\ell^2$, which makes these
    rotational averages valid. Applying them to the unshifted bubble would
    give a wrong result.
    """)
    return


@app.cell
def _(D, L, TensorReducer, mink, uv_data):
    tensor_reducer = TensorReducer(D).with_integrated_vector(L(mink(D)))
    tensor_reduced = tensor_reducer.reduce(uv_data['shifted'])
    tensor_reduced
    return


@app.cell
def _(uv_data):
    reduced_scalar_numerator = uv_data["reduced"]
    reduced_scalar_numerator
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Integrate the UV pole and subtract it

    With the intermediate measure $d^D\ell/(i\pi^{D/2})$,
    $$\operatorname{Pole}\!\int\!\frac{1}{(\ell^2-\Delta)^2}
       =\frac1\epsilon,\qquad
      \operatorname{Pole}\!\int\!\frac{\ell^2}{(\ell^2-\Delta)^2}
       =\frac{2\Delta}{\epsilon}.$$
    For a reduced numerator $a(x,D)\ell^2+b(x,D)$, integrate
    $[2\Delta a+b]_{D=4}$ over $x\in[0,1]$ to obtain the residue $R$.
    Setting $D=4$ at this final step loses only finite terms.

    Restoring the usual $d^D\ell/(2\pi)^D$ measure, the projected loop pole is
    $iR/(16\pi^2\epsilon)$ and its MS counterterm insertion is its **negative**.
    Symmetry factors, loop signs, and the model couplings are included in $R$.
    """)
    return


@app.cell
def _(uv_data):
    uv_residue = uv_data["residue"]
    uv_residue
    return (uv_residue,)


@app.cell
def _(eps, uv_residue):
    from symbolica import E as parse

    loop_uv_pole = parse("1i") * uv_residue / (16 * parse("pi")**2 * eps)
    uv_counterterm = -loop_uv_pole
    uv_counterterm
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Sum the selected diagrams and check the Ward identity

    The longitudinal projector $\delta^{ab}p_\mu p_\nu/(8p^2)$ checks that
    the total UV pole is transverse. The gluon and ghost longitudinal terms
    cancel; each quark loop is separately transverse and its pole is mass independent.

    In Feynman gauge the transverse coefficient is
    $G^2(5C_A/3-4T_F n_f/3)$, with $C_A=3$ and $T_F=1/2$.
    The current veto leaves $n_f=1$, giving $13G^2/3$.
    See [the one-loop field counterterm](https://cds.cern.ch/record/319039/files/9701375.pdf).
    This is the gluon **field** counterterm, not the QCD beta-function coefficient.
    """)
    return


@app.cell
def _(bubble_uv_data, diagrams, mo, model):
    from symbolica import E as zero

    uv_results = [bubble_uv_data(_d, model) for _d in diagrams]
    ward_results = [bubble_uv_data(_d, model, longitudinal=True) for _d in diagrams]
    total_residue = sum((_row["residue"] for _row in uv_results), zero("0")).expand()
    longitudinal_residue = sum(
        (_row["residue"] for _row in ward_results), zero("0")
    ).expand()
    assert longitudinal_residue == zero("0"), "The UV pole is not transverse"
    mo.ui.table([
        {
            "Loop": ", ".join(_edge.particle_name for _edge in _d.internal_edges),
            "Transverse residue": _uv["residue"].format(color_top_level_sum=False, color_builtin_symbols=False),
            "Longitudinal residue": _ward["residue"].format(color_top_level_sum=False, color_builtin_symbols=False),
        }
        for _d, _uv, _ward in zip(diagrams, uv_results, ward_results)
    ], selection=None)
    return (total_residue,)


@app.cell
def _(eps, p2, total_residue):
    from symbolica import E as expression

    # Amplitude counterterm = -i (p^2 g - p p) delta_ab delta_Z3.
    delta_Z3_MS = (total_residue / p2).expand() / (
        16 * expression("pi")**2 * eps
    )
    delta_Z3_MS
    return


if __name__ == "__main__":
    app.run()
