import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Covariant tensor reduction")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Covariant tensor reduction

    [Browse all notebooks](/) · [Integral families](/?file=hep/integral_families.py)

    A rotationally invariant loop integral has no preferred direction. The
    angular average of $(k\cdot p)^2$ is $k^2p^2/D$, and odd moments vanish.
    `hep.TensorReducer` constructs that covariant average in symbolic dimension
    $D$. This is an integration identity, distinct from structural `contract()`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Setup and notebook helpers

    Expand the folded import cell to inspect the dependencies. This example
    uses the public tensor and HEP primitives directly; no helper layer is needed.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from symbolica import E, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community import hep, tensor


    _set_namespace("angular_demo")
    return S, hep, mo, tensor


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Declare the integration variable

    Both vectors inhabit the same Lorentz space. Only `k` is integrated;
    `p` is an external vector and remains in the answer.
    """)
    return


@app.cell
def _(S, hep, tensor):
    D = S("D")
    lorentz = tensor.Representation.mink(D)
    k, p = tensor.TensorName.vector("k"), tensor.TensorName.vector("p")
    reducer = hep.TensorReducer(D, integrated=[k])
    return k, lorentz, p, reducer


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Reduce a compact scalar product

    Pass the numerator's expression to the reducer. No arithmetic expansion or
    sequence of tensor simplification passes is required.
    """)
    return


@app.cell
def _(k, lorentz, p):
    numerator = (k(1, lorentz) * p(lorentz))**2 * (k(2, lorentz) * p(2, lorentz))**2
    numerator
    return (numerator,)


@app.cell
def _(numerator, reducer, tensor):
    reduced = reducer.reduce(numerator.to_expression())
    tensor.TensorExpression(reduced)
    return


if __name__ == "__main__":
    app.run()
