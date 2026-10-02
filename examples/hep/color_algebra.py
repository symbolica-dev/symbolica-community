import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Color chains and conjugation")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Setup and notebook helpers

    Imports and supporting routines are folded below. Expand a cell’s code to
    inspect or edit it; the calculation that follows shows the HEP operations.
    """)
    return


@app.cell
def _():
    from symbolica.community import tensor as sp
    import marimo as mo
    from symbolica import E, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.tensor import AUTO, Representation, TensorExpression, chain

    _set_namespace("color_algebra")
    return AUTO, E, Representation, S, TensorExpression, chain, mo, sp


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
    explicit_word = generator("a", "i", "k") * generator("b", "k", "j")
    color_word = explicit_word.contract(
        representations=[fundamental], metrics=False, rank_one=False
    )
    color_conjugate = color_word.dirac_adjoint()
    _expected = (generator("b", "j", "k") * generator("a", "k", "i")).contract(
        representations=[fundamental], metrics=False, rank_one=False
    )
    assert color_conjugate.to_expression() == _expected.to_expression()
    assert color_conjugate.dirac_adjoint().to_expression() == color_word.to_expression()
    _settings = dict(gamma=False, color=True)
    assert color_conjugate.simplify_algebra(**_settings).to_expression() == (
        explicit_word.dirac_adjoint().simplify_algebra(**_settings).to_expression()
    )
    return color_conjugate, color_word, explicit_word, fundamental, generator


@app.cell
def _(color_conjugate, color_word, explicit_word, mo):
    mo.vstack(
        [
            mo.md("**Explicit indexed network**"),
            explicit_word,
            mo.md("**Collected chain**"),
            color_word,
            mo.md("**Complex conjugate**"),
            color_conjugate,
        ]
    )
    return


@app.cell
def _(E, color_conjugate, color_word, mo):
    color_norm = (color_word * color_conjugate).simplify_algebra(
        gamma=False, color=True, color_substitute_cof_dimension_invariants=True
    )
    assert color_norm.is_scalar
    assert color_norm.to_expression() == E("16/3")
    mo.vstack(
        [
            mo.md(
                r"**Summed norm:** $\sum_{a,b,i,j}|(T^aT^b)_{ij}|^2=16/3$ for $T_R=1/2$."
            ),
            color_norm,
        ]
    )
    return


@app.cell
def _(fundamental, generator, mo):
    _explicit_loop = (
        generator("a", "i", "j") * generator("b", "j", "k") * generator("c", "k", "i")
    )
    color_trace = _explicit_loop.contract(
        representations=[fundamental], metrics=False, rank_one=False
    )
    color_trace_conjugate = color_trace.dirac_adjoint()
    assert color_trace_conjugate.to_expression() != color_trace.to_expression()
    assert (
        color_trace_conjugate.dirac_adjoint().to_expression()
        == color_trace.to_expression()
    )
    _settings = dict(gamma=False, color=True)
    assert color_trace_conjugate.simplify_algebra(**_settings).to_expression() == (
        _explicit_loop.dirac_adjoint().simplify_algebra(**_settings).to_expression()
    )
    mo.vstack(
        [
            mo.md("**Three-generator trace**"),
            color_trace,
            mo.md("**Conjugate trace**"),
            color_trace_conjugate,
            mo.md(
                "Compact and expanded forms agree, and conjugating twice restores each original expression."
            ),
        ]
    )
    return color_trace, color_trace_conjugate


@app.cell
def _(
    AUTO,
    E,
    color_trace,
    color_trace_conjugate,
    fundamental,
    generator,
    mo,
    sp,
):
    _factors = [generator(a, AUTO, AUTO) for a in ("a", "b", "c")]
    symmetric_color_trace = sp.trace(
        fundamental, sp.FactorProjector.symmetric(*_factors)
    )
    antisymmetric_color_trace = sp.trace(
        fundamental, sp.FactorProjector.antisymmetric(*_factors)
    )
    assert (
        symmetric_color_trace.dirac_adjoint().to_expression()
        == symmetric_color_trace.to_expression()
    )
    assert (
        antisymmetric_color_trace.dirac_adjoint() + antisymmetric_color_trace
    ).simplify_algebra(contract="dots", gamma=False, color=True).to_expression() == E(
        "0"
    )
    _settings = dict(gamma=False, color=True)
    for _projected, _reference in (
        (symmetric_color_trace, (color_trace + color_trace_conjugate) / 2),
        (antisymmetric_color_trace, (color_trace - color_trace_conjugate) / 2),
    ):
        _residual = (_projected - _reference).simplify_algebra(**_settings)
        assert _residual.expand().to_expression() == E("0")
    mo.vstack(
        [
            mo.md(
                "\n            **Symmetric and antisymmetric generator groups**\n\n            Cyclic invariance reduces the six permutations of three generators\n            to two orientations. Their half-sum is the symmetric, real trace;\n            their half-difference is the antisymmetric, purely imaginary trace.\n            Conjugation preserves the compact projectors, including their signs.\n            "
            ),
            symmetric_color_trace,
            antisymmetric_color_trace,
            mo.md("**Conjugate antisymmetric trace**"),
            antisymmetric_color_trace.dirac_adjoint(),
        ]
    )
    return


@app.cell
def _(AUTO, E, S, TensorExpression, chain, fundamental, generator, mo):
    _z = S("color_weight_z")
    weighted_color_word = chain(
        fundamental("i"),
        fundamental.dual()("j"),
        _z**2 * generator("a", AUTO, AUTO),
        generator("b", AUTO, AUTO),
    )
    weighted_color_conjugate = weighted_color_word.dirac_adjoint()
    assert (
        weighted_color_conjugate.dirac_adjoint().to_expression()
        == weighted_color_word.to_expression()
    )
    _norm = weighted_color_word * weighted_color_conjugate
    _network = (
        TensorExpression(_norm.to_expression().replace(_z, E("1+2𝑖")))
        .undo_chain()
        .to_network()
    )
    _network.execute()
    weighted_numeric_norm = _network.result_scalar()
    assert abs(complex(weighted_numeric_norm) - 400 / 3) < 1e-12
    mo.vstack(
        [
            mo.md(r"""
            **Scalar weights inside an ordered color chain**

            `chain(...)` accepts weighted generators with unresolved fundamental
            ports. Powers, inverse factors and scalar functions conjugate using
            Symbolica's existing rules; the generator order and endpoints reverse.
            A scalar symbol is not assumed real.

            Here the word is $z^2(T^aT^b)_{ij}$. At $z=1+2i$, its summed norm is
            $|z^2|^2\,16/3=400/3$, checked independently with explicit SU(3) matrices.
            """),
            weighted_color_word,
            weighted_color_conjugate,
            weighted_numeric_norm,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
