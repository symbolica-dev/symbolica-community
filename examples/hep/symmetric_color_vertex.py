import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Symmetric UFO color tensor")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Symmetric UFO color tensor

    [Browse all notebooks](/) · [Color algebra](/?file=hep/color_algebra.py)

    Generate a cubic interaction of adjoint scalar fields with color tensor
    $d^{abc}$. The model below reuses the embedded Standard Model scalar mass and coupling
    records, restricts the interactions to one cubic vertex, and gives the
    scalar an adjoint color index. The label `H` refers here to this toy field.

    The UFO color rule `d(1,2,3)` lowers to Idenso's normalized symmetric
    generator trace with the factor required by
    $d^{abc}=2\operatorname{Tr}(\{T^a,T^b\}T^c)$.
    [FeynCalc's reference](https://feyncalc.github.io/FeynCalcBook/SUND.html)
    gives $d^{abc}d^{abc}=40/3$ for SU(3), with $T_R=1/2$.
    The norm displayed below has the scalar coupling removed.
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
    from symbolica.community import tensor as sp
    import json

    import marimo as mo
    from symbolica import E, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hepkit import Model
    from symbolica.community.tensor import TensorExpression

    _set_namespace("symmetric_color_vertex")
    return E, Model, S, TensorExpression, json, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell
def _(Model, json):
    _source = json.loads(Model.standard_model().to_json())
    _source["name"] = "adjoint_scalar_example"
    for _particle in _source["particles"]:
        if _particle["name"] == "H":
            _particle["color"] = 8
    _source["vertex_rules"] = [
        _vertex for _vertex in _source["vertex_rules"] if _vertex["name"] == "V_9"
    ]
    _source["vertex_rules"][0]["color_structures"] = ["d(1,2,3)"]
    adjoint_model = Model.from_json(json.dumps(_source))
    return (adjoint_model,)


@app.cell
def _(adjoint_model, mo):
    symmetric_generated = adjoint_model.process(["H", "H"], ["H"]).generate_diagrams(
        loops=0,
        max_vertices=1,
        maximum_bridges=None,
        numerator_grouping=None,
        progress=None,
    )
    assert len(symmetric_generated.diagrams) == 1
    mo.vstack(
        [
            mo.md("**Generated cubic vertex**"),
            symmetric_generated.diagrams[0],
        ]
    )
    return (symmetric_generated,)


@app.cell
def _(E, S, TensorExpression, adjoint_model, mo, symmetric_generated):
    symmetric_color = TensorExpression(
        symmetric_generated.diagrams[0]
        .numerator_expression()
        .to_expression()
        .replace(adjoint_model.coupling("GC_69").symbol, E("1"))
    )
    assert len(symmetric_color.structure.slots) == 3
    assert (
        symmetric_color.dirac_adjoint().to_expression()
        == symmetric_color.to_expression()
    )
    # The tensor product contracts all three matching explicit adjoint indices.
    symmetric_color_norm = (
        symmetric_color * symmetric_color.dirac_adjoint()
    ).simplify_algebra(
        contract="dots",
        gamma=False,
        color=True,
        color_substitute_cof_dimension_invariants=True,
    )
    assert symmetric_color_norm.is_scalar
    assert symmetric_color_norm.to_expression() == E("40/3")
    # Initial-state color averaging is independently supplied by the particle API.
    _particle = adjoint_model.particle("H")
    assert TensorExpression(_particle.color_sum(S("x"), S("x"), average=True)).contract(
        rank_one=False, collect_chains=False, collect_traces=False
    ).to_dots().to_expression() == E("1")
    mo.vstack(
        [
            mo.md("**Shared representation of the generated color tensor**"),
            symmetric_color,
            mo.md("**Summed color norm**"),
            symmetric_color_norm,
            mo.md(
                "Reality, three adjoint ports, and the exact SU(3) normalization all pass."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
