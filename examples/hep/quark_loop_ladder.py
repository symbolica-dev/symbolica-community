import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Three gluon rungs inside a quark loop",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Three gluon rungs inside a quark loop

    [Browse notebooks](/) · [Companion ladder](/?file=hep/three_gluon_rung_ladder.py)

    Generate a **four-loop propagator ladder with three gluon rungs**,
    eight interaction vertices, eleven internal lines and two external gluons.
    The outer ring is one massless bottom-quark loop; only the three rungs
    are gluons. This represents one quark flavor, with no extra flavor factor.

    Work in Feynman gauge, SU(3), and symbolic Lorentz dimension $D$.
    The external contraction is $g_{\mu\nu}\delta_{ab}/8$: a metric trace
    and color average, with no spin average or on-shell condition on $p$.
    This single graph is not a complete, gauge-invariant self-energy.
    We reduce its numerator to dot products; no loop integration is performed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Setup and notebook helpers

    Imports and shared topology/projection helpers are folded below.
    The graph template only selects the topology: the generator supplies all
    Feynman rules, fermion arrows, symmetry factors and momentum routing.
    We require completion before expanding or routing the result.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from symbolica import E, Replacement, S
    from symbolica.community import hepkit as hep
    from symbolica.community import tensor as sp
    from generated_ladder_helpers import (
        dot_coordinates,
        ladder_filter,
        project_and_split,
        reduce_color,
    )

    return (
        E,
        Replacement,
        S,
        dot_coordinates,
        hep,
        ladder_filter,
        mo,
        project_and_split,
        reduce_color,
        sp,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate the ladder

    Restrict the model to quark–gluon vertices and match the completed topology,
    independently of the generator's vertex numbering.
    """)
    return


@app.cell
def _(S, hep, ladder_filter):
    model = hep.Model.standard_model()
    D = S("quark_loop_ladder::D")
    generated = model.process(["g"], ["g"], vertex_allow=["V_76"]).generate_diagrams(
        loops=4,
        max_vertices=8,
        allow_self_loops=False,
        self_energy=None,
        tadpoles=None,
        zero_snails=None,
        numerator_grouping=None,
        progress=None,
        filter=ladder_filter(outer_quark=True),
    )
    assert len(generated.diagrams) == 1
    diagram = generated.diagrams[0]
    diagram.validate()
    assert diagram.loop_count == 4
    assert len(diagram.vertices) == 8
    assert len(diagram.internal_edges) == 11
    graph_weight = diagram.overall_factor_expression(evaluate=True)
    return D, diagram, graph_weight, model


@app.cell(hide_code=True)
def _(diagram, graph_weight, mo):
    try:
        _drawing = diagram.render()
    except ImportError:
        _lines = ["graph LR"]
        for _edge in diagram.edges:
            _source = (
                f"v{_edge.source}" if _edge.source is not None else f"ext{_edge.id}"
            )
            _target = (
                f"v{_edge.target}" if _edge.target is not None else f"ext{_edge.id}"
            )
            _arrow = "-->" if _edge.particle_name in ("b", "b~") else "---"
            _lines.append(
                f'{_source} {_arrow}|"{_edge.particle_name}, e{_edge.id}"| {_target}'
            )
        _drawing = mo.mermaid("\n".join(_lines))
    mo.vstack(
        [
            _drawing,
            mo.md(
                "**Generated graph weight** (including the closed-fermion-loop sign):"
            ),
            graph_weight,
            mo.ui.table(
                [
                    {
                        "edge": edge.id,
                        "particle": edge.particle_name,
                        "source": edge.source,
                        "target": edge.target,
                    }
                    for edge in diagram.internal_edges
                ]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Project the numerator and reduce color

    Reduce the Dirac trace in edge momenta first, then use the generated
    routing to express the answer in four loop vectors and one external vector.
    The color factor below precedes the external $1/8$ average, which remains
    in the spacetime numerator. Couplings and propagator phases are retained.
    The Clifford algebra is D-dimensional with tr(1)=4; the quark mass is set to zero before reduction.
    """)
    return


@app.cell
def _(D, E, diagram, model, project_and_split, reduce_color, sp):
    raw_expression = model.expand_couplings(
        diagram.numerator_expression(in_lmb=False).to_expression()
    )
    raw_expression = raw_expression.replace(model.particle("b").mass, E("0"))
    numerator = sp.TensorExpression(raw_expression).with_lorentz_dimension(D)
    projector, color_tensor, spacetime_tensor = project_and_split(numerator, D)
    color_factor = reduce_color(color_tensor)
    color_factor
    return color_factor, spacetime_tensor


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Contract to dot products

    Apply Dirac trace identities and Lorentz contractions, then explicitly expand
    the scalar result. The full polynomial is available for download below.
    """)
    return


@app.cell
def _(color_factor, diagram, sp, spacetime_tensor):
    reduced = spacetime_tensor.simplify_algebra(
        color=False, gamma=True, contract="dots"
    )
    assert reduced.reduction_status == sp.ReductionStatus.Complete
    routed = diagram.momentum_basis().route_expression(reduced.expand())
    loop_dots = routed.contract().to_dots().expand().to_expression()
    dot_numerator = color_factor * loop_dots
    return (dot_numerator,)


@app.cell
def _(
    D,
    Replacement,
    S,
    diagram,
    dot_coordinates,
    dot_numerator,
    graph_weight,
    model,
):
    polynomial, scalar_products, dot_table = dot_coordinates(
        dot_numerator, D, "quark_loop_ladder"
    )
    gs = model.parameter("G").symbol
    # No color/Dirac tensors, free indices, edge vectors or hidden traces remain.
    assert set(polynomial.get_all_symbols()) <= set(scalar_products + [D, gs])
    assert polynomial != 0
    scale = S("quark_loop_ladder::scale")
    scaled = polynomial.replace_multiple(
        [Replacement(s, scale * s) for s in scalar_products]
    )
    assert (scaled - scale**4 * polynomial).expand(via_poly=True) == 0
    weighted_numerator = graph_weight * polynomial
    denominators = diagram.denominator_expression(dimension=D, in_lmb=True)
    denominators = denominators.to_expression().replace(model.particle("b").mass, 0)
    return denominators, dot_table, polynomial, weighted_numerator


@app.cell(hide_code=True)
def _(denominators, dot_table, mo, polynomial, weighted_numerator):
    _terms = list(polynomial.terms())
    mo.vstack(
        [
            mo.md(
                f"**Result:** {len(_terms):,} terms in 15 scalar products. "
                "Every term has momentum degree eight (four dot products)."
            ),
            mo.ui.table(dot_table),
            mo.md("**First five terms of the numerator, before the graph weight:**"),
            sum(_terms[:5]),
            mo.download(
                data=polynomial.to_canonical_string().encode(),
                filename="quark_loop_ladder_dot_numerator.txt",
                label="Download full dot-product numerator",
            ),
            mo.download(
                data=weighted_numerator.to_canonical_string().encode(),
                filename="quark_loop_ladder_weighted_numerator.txt",
                label="Download numerator including graph weight",
            ),
            mo.accordion({"Generated propagator denominators": denominators}),
            mo.md(
                "The numerator and denominators are separate. No integration measure, "
                "loop integral, counterterm or additional diagram is included."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
