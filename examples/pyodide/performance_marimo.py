import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
async def _(mo):
    import sys

    if sys.platform == "emscripten":
        import micropip

        # The exported notebook and wheel are served from the same directory.
        _wheel = mo.notebook_location() / "symbolica-3.0.1-cp314-abi3-pyemscripten_2026_0_wasm32.whl"
        await micropip.install(str(_wheel))
    from symbolica import E, S
    from symbolica.community import hep
    return E, S, hep


@app.cell
def _(E, S, hep, mo):
    _expanded = E("(x+1)^8").expand()
    _integral = E("1/(1+x^2)").integrate(S("x"))
    assert _expanded == E("x^8+8*x^7+28*x^6+56*x^5+70*x^4+56*x^3+28*x^2+8*x+1")
    assert _integral == E("atan(x)")
    assert hep.ThreeMomentum(3.0, 4.0, 0.0).on_shell().components() == (5.0, 3.0, 4.0, 0.0)
    mo.md(f"""
    # Symbolica Community performance build

    **WASM checks passed:** algebra, symbolic integration, and HEP kinematics.

    `(x+1)^8 = {_expanded}`

    `integral(1/(1+x^2), x) = {_integral}`
    """)
    return


if __name__ == "__main__":
    app.run()
