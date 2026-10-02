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

        from compression import zstd
        from pathlib import Path
        from pyodide.http import pyfetch
        import time

        # Fetch a .zst file as ordinary bytes (no HTTP Content-Encoding).
        # Pyodide's Python 3.14 includes zstd, so no decoder package is needed.
        _name = "symbolica-3.0.1-cp314-abi3-pyemscripten_2026_0_wasm32.whl"
        _url = mo.notebook_location() / (_name + ".zst")
        _response = await pyfetch(str(_url))
        if _response.status != 200:
            raise RuntimeError(f"Package download failed: HTTP {_response.status}")
        _compressed = await _response.bytes()
        _start = time.perf_counter()
        _local = Path("/tmp") / _name
        _local.write_bytes(zstd.decompress(_compressed, options={
            zstd.DecompressionParameter.window_log_max: 27,
        }))
        print(f"ZSTD_ARCHIVE downloaded={len(_compressed)} decode_and_write_seconds={time.perf_counter()-_start:.3f}")
        del _compressed
        await micropip.install("emfs:" + str(_local))
        _local.unlink()
    from symbolica import E, S
    from symbolica.community import hepkit as hep
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
