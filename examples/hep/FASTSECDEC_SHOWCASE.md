# FastSecDec notebooks

Both notebooks are available in the community example browser:

```sh
python -m marimo edit examples/
```

Or open one directly from the repository root:

```sh
python -m marimo edit examples/hep/fastsecdec_showcase.py
python -m marimo edit examples/hep/gghh.py
```

[The interactive showcase](fastsecdec_showcase.py) starts with a massive scalar
triangle and offers box and two-loop numerator examples, plus an optional longer
gg → HH calculation. Select **Generate** and **Integrate** to start the respective
calculations. Live status, sector inspection, and QMC/Havana controls are included.
Its `showcase/` helpers and `fixtures/fastsecdec/` and `fixtures/gghh/` inputs are
included beside the notebook; no FastSecDec source checkout is required to run it.

[The standalone gg → HH notebook](gghh.py) contains the full walkthrough in one
file: `Model.standard_model()`, inline masses and helicities, diagram generation,
numerator contraction, `diagram.sector_decompose(...)`, and numerical integration.
Use marimo's editor controls to enable and run the expensive cells, which start
disabled. Both notebooks finish with `get_citations()` and a BibTeX download.

Use marimo 0.24.2 or newer and a standard community installation containing
`symbolica.community.hepkit.sector_decomposition`. Browser execution also needs
the explicit Wasm wheel and asset manifest produced by the
[FastSecDec browser exporter](https://github.com/alphal00p/fastSecDec/blob/main/examples/hepkit/export.py);
copying a notebook alone does not package a browser wheel. The exporter supports
both notebooks (`--notebook dashboard` or `--notebook gghh`).

These are copies of the notebooks and example assets in
[FastSecDec examples/hepkit](https://github.com/alphal00p/fastSecDec/tree/main/examples/hepkit).
The computational implementation and substantive bindings remain in FastSecDec.
See its [full walkthrough](https://github.com/alphal00p/fastSecDec/blob/main/examples/hepkit/README.md)
and [build guide](https://github.com/alphal00p/fastSecDec/blob/main/examples/hepkit/BUILD.md).
