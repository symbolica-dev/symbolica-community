import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Three-rung ladders")

with app.setup(hide_code=True):
    import marimo as mo

    from symbolica import Graph, Replacement, S
    from symbolica.community.hep import Model
    from symbolica.community.tensor import TensorExpression, Representation


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Three-rung ladders

    [Browse notebooks](/)

    Choose a **four-loop propagator ladder with three gluon rungs**,
    eight interaction vertices, eleven internal lines and two external gluons.
    The outer ring contains either gluons or one massless bottom-quark loop.
    The fermionic case represents one flavor, with no extra flavor factor.
    Four-gluon vertices and ghost loops are excluded.

    Work in Feynman gauge, SU(3), and symbolic Lorentz dimension $D$.
    The external contraction is $g_{\mu\nu}\delta_{ab}/8$: a metric trace
    and color average, with no spin average or on-shell condition on $p$.
    This single graph is not a complete, gauge-invariant self-energy.
    We reduce its numerator to dot products; no loop integration is performed.
    The fermionic Clifford algebra is $D$-dimensional with $\operatorname{tr}(1)=4$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Setup and notebook helpers

    Imports, topology selection, and scalar-product coordinates are defined below.
    This notebook needs no companion helper module.
    The graph template only selects the topology: the generator supplies all
    Feynman rules, fermion arrows, symmetry factors and momentum routing.
    """)
    return


@app.function(hide_code=True)
def ladder_filter(outer_quark=False):
    """Select a ring of eight vertices with three uncrossed gluon rungs.

    This graph is only an isomorphism target. Feynman diagrams, weights, rules,
    directed fermion flow and momentum routing all come from the generator.
    """
    target = Graph()
    for _ in range(8):
        target.add_node(0)
    # Generator labels external legs by their signed, one-based global index.
    target.add_node(-1)
    target.add_node(2)
    for i in range(8):
        target.add_edge(i, (i + 1) % 8, data=5 if outer_quark else 21)
    for i, j in ((1, 7), (2, 6), (3, 5), (0, 8), (4, 9)):
        target.add_edge(i, j, data=21)
    target = target.canonize()[0]

    def accept(graph, completed):
        # An incomplete branch can still grow into the requested ladder.
        if completed < len(graph):
            return True
        if len(graph) != 10 or graph.num_edges() != 13:
            return False
        # Ignore orientation only in the callback snapshot. The generator keeps
        # the physical quark arrows and their signs in the returned diagram.
        for edge in range(graph.num_edges()):
            graph.set_directed(edge, False)
        return graph.canonize()[0] == target

    return accept


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Generate and select a ladder

    Generate both graphs once, using three-gluon or bottom-quark–gluon vertices
    and the same topology filter. The selector updates the displayed graph and
    every subsequent numerator calculation.
    Generation is timed separately from the diagram display below: the first
    Typst render can take several seconds even when generation is fast.
    """)
    return


@app.cell
def _():
    model = Model.qcd()
    ladders = {}
    for _name, _quark, _vertex in (
        ("Gluonic", False, "V_36"),
        ("Fermionic", True, "V_76"),
    ):
        _generated = model.process(
            ["g"], ["g"], vertex_allow=[_vertex]
        ).generate_diagrams(
            loops=4,
            max_vertices=8,
            filter=ladder_filter(outer_quark=_quark),
        )
        ladders[_name] = _generated[0]
    return ladders, model


@app.cell(hide_code=True)
def _(ladders):
    ladder_choice = mo.ui.radio(
        options=list(ladders), value="Gluonic", inline=True, label="Ladder"
    )
    ladder_choice
    return (ladder_choice,)


@app.cell
def _(ladder_choice, ladders):
    diagram = ladders[ladder_choice.value]
    graph_weight = diagram.overall_factor_expression(evaluate=True)
    diagram
    return (diagram,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Build the numerator and projector

    Use the generated loop-momentum routing before contraction, keeping the
    momentum differences compact until the final expansion. Set the bottom-quark
    mass to zero before applying Dirac identities.
    Apply $g_{\mu\nu}\delta_{ab}/8$ directly to the generated external ports.
    Color and Lorentz factors stay together; couplings and propagator phases
    are retained.
    """)
    return


@app.cell
def _(diagram, model):
    D = S("D")

    _massless = model.expand_couplings(
        diagram.numerator_expression(in_lmb=True).to_expression()
    ).replace(model.particle("b").mass, 0)
    numerator = TensorExpression(_massless).with_lorentz_dimension(D)
    numerator
    return D, numerator


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Just substitute colour:
    """)
    return


@app.cell
def _(numerator, projector):
    (projector * numerator).simplify_algebra(gamma=False,
        contract="minimal", color_substitute_cof_dimension_invariants=True
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Create a projector
    """)
    return


@app.cell
def _(numerator):
    numerator.structure
    return


@app.cell
def _():
    return


@app.cell
def _(D, numerator):
    mink = Representation.mink(D)
    adjoint = Representation.coad(8)
    projector = (
        TensorExpression.g(mink) * TensorExpression.g(adjoint) / 8
    ).index(*(slot.dual() for slot in numerator.structure.slots))
    projector
    return (projector,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Contract to dot products

    Reduce the projected tensor in one call, including color and Dirac identities
    and Lorentz contractions.

    Graph-informed tensor simplfication resutls in fast execution.
    """)
    return


@app.cell
def _(numerator, projector):
    reduced = (projector * numerator).simplify_algebra(
        contract="dots", color_substitute_cof_dimension_invariants=True
    )
    reduced
    return (reduced,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Number of terms in the expression:
    """)
    return


@app.cell
def _(reduced):
    len(reduced.to_expression().expand())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
