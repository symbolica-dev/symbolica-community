import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Color Fierz contractions")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Color Fierz contractions

    [Browse all notebooks](/) · [Color conjugation](/?file=hep/color_algebra.py)

    A shared adjoint index can connect an open color chain to a closed trace,
    or two traces to each other. The fundamental Fierz identity contracts it:

    $$\sum_a (T^a)_{ij}\operatorname{Tr}(T^bT^aT^c)
      =\frac12(T^cT^b)_{ij}-\frac{1}{4N}\delta_{ij}\delta^{bc}.$$

    The order $T^cT^b$ follows by cutting the cyclic trace at $T^a$.
    `simplify_algebra(color=True)` applies this identity in shared Idenso algebra.
    We use $N=3$ and $\operatorname{Tr}(T^aT^b)=\delta^{ab}/2$ here.
    [FeynCalc reference](https://feyncalc.github.io/FeynCalcBook/SUNSimplify.html).
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
    import marimo as mo
    from symbolica import E
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.tensor import Representation, TensorExpression

    _set_namespace("color_fierz")
    return E, Representation, TensorExpression, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell
def _(Representation, TensorExpression):
    generator = TensorExpression.color_t(8, 3)
    fundamental = Representation.cof(3)
    adjoint = Representation.coad(8)
    _loop = (
        generator("b", "k", "l") * generator("a", "l", "m") * generator("c", "m", "k")
    )
    color_trace = _loop.contract(
        representations=[fundamental], metrics=False, rank_one=False
    ).to_dots()
    return adjoint, color_trace, fundamental, generator


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Connect the generator to the unevaluated trace
    """)
    return


@app.cell
def _(color_trace, generator):
    mixed_color = generator("a", "i", "j") * color_trace
    return (mixed_color,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Apply the Fierz identity
    """)
    return


@app.cell
def _(mixed_color):
    fierz_result = mixed_color.simplify_algebra(
        contract="dots",
        gamma=False,
        color=True,
        color_substitute_cof_dimension_invariants=True,
    )
    return (fierz_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Verify the open-chain formula
    """)
    return


@app.cell
def _(TensorExpression, adjoint, fierz_result, fundamental, generator):
    # An identity pairs a fundamental slot with a dual slot.
    _identity = TensorExpression.g(fundamental, fundamental.dual())("i", "j")
    expected = (
        generator("c", "i", "k") * generator("b", "k", "j") / 2
        - _identity * adjoint.g("b", "c") / 12
    )
    assert fierz_result.to_expression().expand() == (
        expected.simplify_algebra(contract="dots", gamma=False, color=True)
        .expand()
        .to_expression()
    )
    return (expected,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Check explicit SU(3) components

    This numerical oracle contracts the original generators independently of the symbolic Fierz kernel.
    """)
    return


@app.cell
def _(expected, mixed_color):
    _difference = (mixed_color - expected).undo_trace().undo_chain().to_network()
    _difference.execute()
    _components = _difference.result_tensor()[:]
    assert len(_components) == 576
    numeric_fierz_error = max(abs(complex(_value)) for _value in _components)
    assert numeric_fierz_error < 1e-12
    return (numeric_fierz_error,)


@app.cell(hide_code=True)
def _(fierz_result, mixed_color, mo, numeric_fierz_error):
    mo.vstack(
        [
            mo.md("**Generator contracted with a trace**"),
            mixed_color,
            mo.md("**Reduced open chain and identities**"),
            fierz_result,
            mo.md(
                f"Independent SU(3) matrix check: maximum component error {numeric_fierz_error:.2e}."
            ),
        ]
    )
    return


@app.cell
def _(fierz_result, mixed_color, mo):
    _settings = dict(
        gamma=False, color=True, color_evaluate_traces=False, color_expand_fierz=False
    )
    preserved_color = mixed_color.simplify_algebra(contract="dots", **_settings)
    assert preserved_color.to_expression() != fierz_result.to_expression()
    _reduced = preserved_color.simplify_algebra(
        gamma=False, color=True, color_substitute_cof_dimension_invariants=True
    )
    assert (_reduced.to_expression() - fierz_result.to_expression()).expand() == 0
    mo.vstack(
        [
            mo.md("**Keeping the trace and separate color lines**"),
            preserved_color,
            mo.md(
                "Disable both trace evaluation and Fierz expansion to retain this form. "
                "Re-enabling simplification gives the same reduced result."
            ),
        ]
    )
    return


@app.cell
def _(E, color_trace, mo):
    _closed = color_trace * color_trace.dirac_adjoint()
    trace_norm = _closed.simplify_algebra(
        contract="dots",
        gamma=False,
        color=True,
        color_substitute_cof_dimension_invariants=True,
    )
    assert trace_norm.is_scalar
    assert trace_norm.to_expression() == E("7/3")
    _network = _closed.undo_trace().undo_chain().to_network()
    _network.execute()
    numeric_trace_norm = _network.result_scalar()
    assert abs(complex(numeric_trace_norm) - 7 / 3) < 1e-12
    mo.vstack(
        [
            mo.md(r"""
            **Two contracted traces**

            $$\sum_{a,b,c}|\operatorname{Tr}(T^aT^bT^c)|^2
              =\frac{(N^2-1)(N^2-2)}{8N}=\frac73\quad(N=3).$$

            The exact symbolic result agrees with an independent numerical
            contraction of explicit SU(3) matrices from Spenso's tensor library.
            """),
            trace_norm,
            numeric_trace_norm,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
