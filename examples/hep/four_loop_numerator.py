import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Four-loop gluon propagator numerators",
)

with app.setup(hide_code=True):
    from collections import Counter

    import marimo as mo
    from symbolica import S
    from symbolica.community.hep import Model
    from symbolica.community.tensor import TensorExpression


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Four-loop gluon propagator numerators

    Generate all **four-loop 1PI gluon propagator graphs** in QCD with
    **one massless quark flavor**, represented by the down quark.
    Include three- and four-gluon vertices, quark loops, and ghost loops.
    Keep self-energy insertions and exclude tadpoles and reducible graphs.
    Each dropdown entry selects a topology with its particle assignment.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Setup and notebook helpers

    The hidden setup cell imports the graph, tensor and notebook primitives.
    The calculation below uses these APIs directly.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Generate the graphs

    The generator supplies the Feynman rules, fermion and ghost arrows,
    symmetry factors, and momentum routing. Generate the diagrams once, then
    select a topology to inspect its numerator and displayed graph.
    """)
    return


@app.cell
def _():
    model = Model.qcd()

    process = model.process(["g"], ["g"], particle_veto=["u", "c", "s", "t", "b"])

    generation_result = process.generate_diagrams(loops=4)
    return generation_result, model, process


@app.cell
def _(process):
    process
    return


@app.cell(hide_code=True)
def _(generation_result):
    mo.stop(
        not generation_result.report.completed,
        mo.md("Generation was interrupted. Rerun the generation cell to continue."),
    )
    _diagrams = generation_result.diagrams
    mo.stop(
        not _diagrams, mo.md("No four-loop gluon propagator graphs were generated.")
    )
    _options = {}
    for _index, _diagram in enumerate(_diagrams):
        _counts = Counter(edge.particle_name for edge in _diagram.internal_edges)
        _content = ", ".join(
            f"{count} {particle}" for particle, count in sorted(_counts.items())
        )
        _options[f"{_index + 1}: {_diagram.name} — {_content}"] = _index
    topology_choice = mo.ui.dropdown(
        options=_options,
        value=next(iter(_options)),
        allow_select_none=False,
        searchable=True,
        full_width=True,
        label="Topology",
    )
    mo.vstack(
        [
            mo.md(
                f"Generated **{len(_diagrams):,} graphs**. "
                "Search by graph name or internal particle content."
            ),
            topology_choice,
        ]
    )
    return (topology_choice,)


@app.cell
def _(generation_result, topology_choice):
    diagram = generation_result[topology_choice.value]
    diagram
    return (diagram,)


@app.cell
def _(diagram):
    diagram.tensor_reduce(S("D"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Build the numerator and projector

    Use the generated loop-momentum routing before contraction, keeping momentum
    differences compact. Set the quark mass to zero before applying Dirac identities.
    Apply $g_{\mu\nu}\delta_{ab}/8$ directly to the generated external ports.
    Color and Lorentz factors stay together; couplings and propagator phases
    are retained. The graph weight is applied separately below.
    """)
    return


@app.cell
def _(diagram, model):
    graph_weight = diagram.overall_factor_expression(evaluate=True)
    _massless = model.expand_couplings(
        diagram.numerator_expression(in_lmb=True) * graph_weight
    )
    numerator = _massless.with_lorentz_dimension(S("D")).collect_factors()
    numerator.structure
    return (numerator,)


@app.cell
def _(numerator):
    slots = numerator.structure.slots()
    projector = (
        TensorExpression.g(slots[0], slots[1])
        * TensorExpression.g(slots[2], slots[3])
        / 8
    )
    projector
    return (projector,)


@app.cell
def _(numerator, projector):
    projected_numerator = projector * numerator
    projected_numerator
    return (projected_numerator,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Perform simple contractions that do not generate new terms
    """)
    return


@app.cell
def _(projected_numerator):
    projected_numerator.contract().to_dots()
    return


@app.cell
def _(projected_numerator):
    projected_numerator.simplify_algebra(
        contract="minimal", color=True, gamma=False
    ).to_dots()
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Contract to dot products

    Reduce the projected tensor in one call, including color and Dirac identities
    and Lorentz contractions. The order of contractions is optimized.
    """)
    return


@app.cell
def _(projected_numerator):
    reduced = projected_numerator.simplify_algebra(contract="dots")
    reduced
    return (reduced,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    Number of terms after expanding the numerator:
    """)
    return


@app.cell
def _(reduced):
    len(reduced.expand().to_expression())
    return


if __name__ == "__main__":
    app.run()
