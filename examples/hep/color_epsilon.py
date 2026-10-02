import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Antisymmetric UFO color tensors")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Antisymmetric UFO color tensors

    [Browse all notebooks](/) · [Color algebra](/?file=hep/color_algebra.py) ·
    [Symmetric color tensor](/?file=hep/symmetric_color_vertex.py)

    The UFO rules `Epsilon(1,2,3)` and `EpsilonBar(1,2,3)` describe the
    antisymmetric invariants of three SU(3) triplets and antitriplets.
    This toy model couples three **distinct complex scalar species**,
    $X_1,X_2,X_3$. Using the same bosonic field three times would make
    this antisymmetric interaction vanish.

    We reuse scalar parameter and Lorentz records from the embedded Standard Model,
    add the three new species, and retain only the two conjugate cubic
    vertices. These are toy color interactions, not Standard Model processes.
    The calculation below removes the scalar coupling and compares their
    color factors; it does not compute a decay rate.
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

    _set_namespace("color_epsilon")
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
    _source["name"] = "triplet_scalar_epsilon_example"
    _epsilon_base = max(abs(_p["pdg_code"]) for _p in _source["particles"]) + 10
    _scalar = next(_p for _p in _source["particles"] if _p["name"] == "G+")
    for _flavor in range(1, 4):
        for _sign in (1, -1):
            _name = f"X{_flavor}" + ("~" if _sign < 0 else "")
            _antiname = f"X{_flavor}" + ("~" if _sign > 0 else "")
            _source["particles"].append(
                dict(
                    _scalar,
                    name=_name,
                    antiname=_antiname,
                    texname=_name,
                    antitexname=_antiname,
                    pdg_code=_sign * (_epsilon_base + _flavor),
                    color=_sign * 3,
                    charge=0.0,
                )
            )
    _vertex = next(_v for _v in _source["vertex_rules"] if _v["name"] == "V_9")
    _source["vertex_rules"] = [
        dict(
            _vertex,
            name="epsilon" + ("_bar" if _conjugate else ""),
            particles=[
                f"X{_flavor}" + ("~" if _conjugate else "") for _flavor in range(1, 4)
            ],
            color_structures=[("EpsilonBar" if _conjugate else "Epsilon") + "(1,2,3)"],
        )
        for _conjugate in (False, True)
    ]
    epsilon_model = Model.from_json(json.dumps(_source))
    return (epsilon_model,)


@app.cell
def _(epsilon_model, mo):
    epsilon_diagrams = []
    for _incoming, _outgoing in (
        (["X1"], ["X2~", "X3~"]),
        (["X1~"], ["X2", "X3"]),
    ):
        _generated = epsilon_model.process(_incoming, _outgoing).generate_diagrams(
            loops=0,
            max_vertices=1,
            maximum_bridges=None,
            numerator_grouping=None,
            progress=None,
        )
        assert len(_generated.diagrams) == 1
        epsilon_diagrams.append(_generated.diagrams[0])
    mo.vstack(
        [
            mo.md("**Generated conjugate cubic vertices**"),
            mo.hstack(epsilon_diagrams, justify="space-around"),
        ]
    )
    return (epsilon_diagrams,)


@app.cell
def _(E, TensorExpression, epsilon_diagrams, epsilon_model, mo):
    epsilon_colors = [
        TensorExpression(
            _diagram.numerator_expression()
            .to_expression()
            .replace(epsilon_model.coupling("GC_69").symbol, E("1"))
        )
        for _diagram in epsilon_diagrams
    ]
    epsilon_norms = []
    for _color in epsilon_colors:
        assert len(_color.structure.slots) == 3
        _conjugate = _color.dirac_adjoint()
        assert _conjugate.dirac_adjoint().to_expression() == _color.to_expression()
        # Explicit matching indices contract the tensor with its dual conjugate.
        _norm = (_color * _conjugate).simplify_algebra(
            contract="dots", gamma=False, color=False, epsilon=True
        )
        assert _norm.is_scalar
        assert _norm.to_expression() == E("6")
        epsilon_norms.append(_norm)
    mo.vstack(
        [
            mo.md("**Typed color tensors**"),
            *epsilon_colors,
            mo.md(
                r"Conjugation exchanges triplet and antitriplet ports. Applying it "
                r"twice restores each tensor. The shared Idenso epsilon contraction "
                r"gives $\epsilon_{ijk}\bar\epsilon^{ijk}=3!=6$ for both vertices."
            ),
            *epsilon_norms,
        ]
    )
    return (epsilon_norms,)


@app.cell
def _(E, S, TensorExpression, epsilon_model, epsilon_norms):
    averaged_epsilon_norms = []
    for _name, _norm in zip(("X1", "X1~"), epsilon_norms):
        _particle = epsilon_model.particle(_name)
        # The trace of the shared completeness tensor counts initial color states.
        _states = (
            TensorExpression(_particle.color_sum(S("i"), S("i")))
            .contract(rank_one=False, collect_chains=False, collect_traces=False)
            .to_dots()
            .to_expression()
        )
        _averaged_trace = (
            TensorExpression(_particle.color_sum(S("i"), S("i"), average=True))
            .contract(rank_one=False, collect_chains=False, collect_traces=False)
            .to_dots()
            .to_expression()
        )
        assert _states == E("3")
        assert _averaged_trace == E("1")
        _averaged_norm = _norm.to_expression() / _states
        assert _averaged_norm == E("2")
        averaged_epsilon_norms.append(_averaged_norm)
    return (averaged_epsilon_norms,)


@app.cell(hide_code=True)
def _(averaged_epsilon_norms, mo):
    mo.vstack(
        [
            mo.md(
                r"**One incoming color average**"
                "\n\n"
                r"`Particle.color_sum` counts three incoming color states, and its "
                r"averaged identity has unit trace. Summing the two outgoing colors "
                r"and averaging the incoming color gives $6/3=2$ for either vertex. "
                r"The outgoing species are distinct, so no identical-particle "
                r"factor is required."
            ),
            *averaged_epsilon_norms,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
