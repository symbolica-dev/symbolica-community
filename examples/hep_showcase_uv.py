import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Feynman-parameter UV cross-check")


@app.cell
def _(mo):
    mo.md(r"""
    # The gluon propagator: UV counterterm

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
    from symbolica.community import hepkit as hep
    from symbolica.community import tensor as sp
    import marimo as mo
    from symbolica import S, E, Expression
    from symbolica.community.tensor import TensorExpression, as_tensor
    from symbolica.community.hepkit import (
        Model,
        SnailFilterOptions,
        FeynmanDiagram,
        TensorReducer,
    )

    contraction_settings = dict()
    return (
        E,
        Expression,
        FeynmanDiagram,
        Model,
        S,
        SnailFilterOptions,
        TensorExpression,
        TensorReducer,
        as_tensor,
        contraction_settings,
        hep,
        mo,
        sp,
    )


@app.cell(hide_code=True)
def _(D, FeynmanDiagram, TensorExpression):
    def numerator_in_d(diagram: FeynmanDiagram) -> TensorExpression:
        """Route into the loop basis and continue Lorentz slots to D before contraction."""
        return diagram.numerator_expression(in_lmb=True).with_lorentz_dimension(D)

    return (numerator_in_d,)


@app.cell(hide_code=True)
def _(
    D,
    FeynmanDiagram,
    P,
    TensorExpression,
    as_tensor,
    color_settings,
    index_,
    metric,
    p2,
    sp,
):
    def external_projector(
        diagram: FeynmanDiagram, *, longitudinal=False
    ) -> TensorExpression:
        """Color-average the transverse trace (or the longitudinal Ward check).

        P_T = delta_ab (g_mu_nu - p_mu p_nu/p^2) / [8 (D-1)].
        Thus P_T . [delta_ab (p^2 g_mu_nu - p_mu p_nu) Pi] = p^2 Pi.
        """
        slots = [
            slot.replace(
                sp.PortPattern.exact(sp.Representation.mink(4), index_),
                sp.PortPattern.exact(sp.Representation.mink(D), index_),
            )
            for slot in diagram.numerator_expression().list_dangling()
        ]
        mu, nu = [
            slot
            for slot in slots
            if slot.get_name() == sp.Representation.mink(4).name.name
        ]
        a, b = [
            slot
            for slot in slots
            if slot.get_name() == sp.Representation.coad(4).name.name
        ]
        if longitudinal:
            return as_tensor(metric(a, b) * P(mu) * P(nu) / (8 * p2))
        return as_tensor(
            metric(a, b) * (metric(mu, nu) - P(mu) * P(nu) / p2) / (8 * (D - 1))
        )

    def apply_projector(
        numerator: TensorExpression, projector: TensorExpression
    ) -> TensorExpression:
        """Contract every named external slot with its matching projector slot."""
        if set(numerator.list_dangling()) != set(projector.list_dangling()):
            raise ValueError(
                "Numerator and projector must have matching external slots"
            )
        # The explicit Einstein indices specify all four contractions here.
        # Native fixed-point simplifiers keep color and gamma expansion local.
        # The network contracts connected factors before residual metric expansion.
        return (
            (numerator * projector)
            .simplify_algebra(
                contract="dots", **color_settings, gamma=True, epsilon=True
            )
            .expand(metric)
            .contract()
            .to_dots()
        )

    return apply_projector, external_projector


@app.cell(hide_code=True)
def _(FeynmanDiagram):
    def routing_coefficients(diagram: FeynmanDiagram) -> dict[int, tuple[int, int]]:
        """Read the signed coefficients from the native one-loop momentum basis."""
        if diagram.loop_count != 1 or len(diagram.external_edges) != 2:
            raise ValueError("This example requires a one-loop two-point graph")
        result = {}
        # The graph's native basis owns momentum routing and conservation.
        for edge_id, signature in diagram.loop_momentum_basis.edge_signatures.items():
            if len(signature.loops) != 1 or any(signature.external[1:]):
                raise ValueError("Expected one independent external momentum")
            result[edge_id] = (signature.loops[0], signature.external[0])
        return result

    return (routing_coefficients,)


@app.cell(hide_code=True)
def _(
    FeynmanDiagram,
    K,
    P,
    TensorExpression,
    as_tensor,
    hep,
    index_,
    numerator_in_d,
):
    def route_numerator(diagram: FeynmanDiagram) -> TensorExpression:
        expression = numerator_in_d(diagram).to_expression()
        expression = expression.replace(
            hep.Kinematics.loop_momentum(0, index_), K(index_)
        )
        expression = expression.replace(
            hep.Kinematics.external_momentum(0, index_), P(index_)
        )
        return as_tensor(expression)

    return (route_numerator,)


@app.cell(hide_code=True)
def _(D, P, TensorExpression, dot, p2, sp):
    def scalar_products(expression: TensorExpression) -> TensorExpression:
        """Keep compact scalar products inside the structured tensor expression."""
        return TensorExpression(
            expression.contract()
            .to_dots()
            .to_expression()
            .replace(
                dot(
                    P(sp.PortPattern.exact(sp.Representation.mink(D))),
                    P(sp.PortPattern.exact(sp.Representation.mink(D))),
                ),
                p2,
            )
        )

    return (scalar_products,)


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
def _(E, Model, TensorExpression, sp):
    def qcd_color_factors(expression: TensorExpression) -> TensorExpression:
        """Evaluate the scalar SU(3) invariants used by this example."""
        return TensorExpression(
            expression.to_expression()
            .replace(
                sp.TensorPattern.casimir(
                    2, sp.PortPattern.exact(sp.Representation.coad(8))
                ),
                3,
            )
            .replace(
                sp.TensorPattern.dynkin_index(2, sp.Representation.cof(3)), E("1/2")
            )
        )

    def resolve_qcd(expression: TensorExpression, model: Model) -> TensorExpression:
        """Resolve QCD couplings and scalar SU(3) factors, preserving tensor type."""
        return qcd_color_factors(model.expand_couplings(expression))

    return (resolve_qcd,)


@app.cell(hide_code=True)
def _(
    D,
    E,
    FeynmanDiagram,
    K,
    L,
    Model,
    P,
    RED_P,
    TensorReducer,
    apply_projector,
    as_tensor,
    diagram_weight,
    dot,
    ell2,
    external_projector,
    index_,
    p2,
    resolve_qcd,
    route_numerator,
    routing_coefficients,
    sp,
    x,
):
    def bubble_uv_data(diagram: FeynmanDiagram, model: Model, *, longitudinal=False):
        """Feynman-parameterize, shift, tensor-reduce and extract a bubble UV pole.

        Nothing mutates the original graph. The returned coefficient excludes
        1/eps and i/(16*pi^2), and includes diagram weights and model couplings.
        """
        copy = FeynmanDiagram.from_json(model, diagram.to_json())
        if len(copy.internal_edges) != 2:
            raise ValueError("UV example supports two-propagator bubbles only")
        routing = routing_coefficients(copy)
        chord = copy.loop_momentum_basis.loop_edges[0]
        first = next(edge for edge in copy.internal_edges if edge.id == chord)
        second = next(edge for edge in copy.internal_edges if edge.id != chord)
        if routing[chord] != (1, 0):
            raise ValueError("The first propagator must carry k")
        loop, external = routing[second.id]
        if abs(loop) != 1 or abs(external) != 1:
            raise ValueError("Expected the second propagator to carry +/- (k +/- p)")
        masses = [
            model.particle(edge.particle_name).mass_parameter
            for edge in (first, second)
        ]
        if masses[0] != masses[1]:
            raise ValueError("This example requires equal propagator masses")
        mass2 = model.particle(first.particle_name).mass ** 2
        delta = mass2 - x * (1 - x) * p2
        shift_sign = external // loop
        projector = external_projector(copy, longitudinal=longitudinal)
        shifted = (
            apply_projector(route_numerator(copy), projector)
            .to_expression()
            .replace(K(index_), L(index_) - shift_sign * x * P(index_))
            .replace(P(index_), RED_P(index_))
        )
        # Projector application already contracts color, Dirac, and Lorentz
        # indices internally; only the selected loop vector is integrated below.
        reducer = TensorReducer(
            D, integrated=[L(sp.PortPattern.exact(sp.Representation.mink(D)))]
        )
        reduced = reducer.reduce(shifted)
        reduced = reduced.replace(
            dot(
                RED_P(sp.PortPattern.exact(sp.Representation.mink(D))),
                RED_P(sp.PortPattern.exact(sp.Representation.mink(D))),
            ),
            p2,
        )
        reduced = reduced.replace(
            dot(
                L(sp.PortPattern.exact(sp.Representation.mink(D))),
                L(sp.PortPattern.exact(sp.Representation.mink(D))),
            ),
            ell2,
        ).expand(ell2)
        # Mixed loop/external dots would indicate that reduction was incomplete.
        if reduced.replace(L(index_), 0) != reduced:
            raise ValueError("Unreduced loop momentum remains")
        a = reduced.coefficient(ell2)
        b = reduced.replace(ell2, 0)
        if any(
            power not in (E("1"), ell2) for power, _ in reduced.coefficient_list(ell2)
        ):
            raise ValueError("Expected a numerator linear in ell^2 after reduction")
        # For d^D ell/(i*pi^(D/2)): pole[J_2] = 1/eps and
        # pole[ell^2 J_2] = 2*Delta/eps. Take D -> 4 only for the residue.
        residue_x = (2 * delta * a + b).replace(D, 4).expand(x)
        primitive = residue_x.integrate(x)
        residue = primitive.replace(x, 1) - primitive.replace(x, 0)
        residue = resolve_qcd(as_tensor(residue * diagram_weight(copy)), model)
        return dict(
            diagram=copy,
            delta=delta,
            shifted=shifted,
            reducer=reducer,
            reduced=reduced,
            residue_x=residue_x,
            residue=residue,
        )

    return (bubble_uv_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell
def _(S, hep, sp):
    D, eps, p2, ell2 = S(
        "hep_gluon::D",
        "hep_gluon::eps",
        "hep_gluon::p2",
        "hep_gluon::ell2",
    )
    x = S("hep_gluon::x", is_scalar=True)
    P, K = S(
        "hep_gluon::P", "hep_gluon::K", tags=hep.Kinematics.external_momentum.get_tags()
    )
    L = sp.TensorName.vector("hep_gluon::L").to_expression()
    RED_P = sp.TensorName.vector("hep_gluon::reduction_P").to_expression()
    index_ = S("hep_gluon::index_")
    mink, metric, dot = (
        sp.Representation.mink,
        sp.TensorName.g().to_expression(),
        sp.TensorPattern.dot,
    )
    k2, kp, t, Muv2 = S(
        "hep_gluon::k2", "hep_gluon::kp", "hep_gluon::t", "hep_gluon::Muv2"
    )
    UV_K, UV_P = S("hep_gluon::uv_K", "hep_gluon::uv_P")
    return D, K, L, P, RED_P, dot, ell2, eps, index_, metric, p2, x


@app.cell
def _():
    # Resolve scalar color invariants within the native color simplifier.
    color_settings = dict(color=True, color_substitute_cof_dimension_invariants=True)
    return (color_settings,)


@app.cell
def _(Model):
    model = Model.standard_model()
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
    diagrams = model.process(
        [g], [g], particle_veto=["c", "t", "s", "u", "d"]
    ).generate_diagrams(
        loops=1, coupling_orders={"QCD": 2, "QED": 0}, zero_snails=SnailFilterOptions()
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
def _(color_settings, contraction_settings, diagram, numerator_in_d):
    contracted_numerator = numerator_in_d(diagram).simplify_algebra(
        contract="dots", **color_settings, gamma=True, epsilon=True
    )
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
def _(
    as_tensor,
    diagram,
    diagram_weight,
    model,
    projected_numerator,
    resolve_qcd,
):
    weighted_projected_numerator = resolve_qcd(
        as_tensor(projected_numerator * diagram_weight(diagram)), model
    )
    weighted_projected_numerator
    return


@app.cell
def _(mo):
    mo.md("""
    The last expression includes the symmetry factor and closed-loop signs,
    with the model couplings expressed through the model’s strong coupling.
    The native graph factor already includes the Grassmann minus for each
    closed ghost loop; `diagram_weight` evaluates it without an extra sign.
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
def _(D, L, TensorReducer, sp, uv_data):
    tensor_reducer = TensorReducer(
        D, integrated=[L(sp.PortPattern.exact(sp.Representation.mink(D)))]
    )
    tensor_reduced = tensor_reducer.reduce(uv_data["shifted"])
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
def _(E, eps, uv_residue):
    loop_uv_pole = E("1i") * uv_residue / (16 * E("pi") ** 2 * eps)
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
def _(E, bubble_uv_data, diagrams, mo, model):
    uv_results = [bubble_uv_data(_d, model) for _d in diagrams]
    ward_results = [bubble_uv_data(_d, model, longitudinal=True) for _d in diagrams]
    total_residue = sum(
        (_row["residue"].to_expression() for _row in uv_results), E("0")
    )
    longitudinal_residue = sum(
        (_row["residue"].to_expression() for _row in ward_results), E("0")
    ).together()
    assert longitudinal_residue == E("0"), "The UV pole is not transverse"
    mo.ui.table(
        [
            {
                "Loop": ", ".join(_edge.particle_name for _edge in _d.internal_edges),
                "Transverse residue": _uv["residue"]
                .to_expression()
                .format(color_top_level_sum=False, color_builtin_symbols=False),
                "Longitudinal residue": _ward["residue"]
                .to_expression()
                .format(color_top_level_sum=False, color_builtin_symbols=False),
            }
            for _d, _uv, _ward in zip(diagrams, uv_results, ward_results)
        ],
        selection=None,
    )
    return (total_residue,)


@app.cell
def _(E, eps, p2, total_residue):
    # Amplitude counterterm = -i (p^2 g - p p) delta_ab delta_Z3.
    delta_Z3_MS = (total_residue / p2) / (16 * E("pi") ** 2 * eps)
    delta_Z3_MS
    return


if __name__ == "__main__":
    app.run()
