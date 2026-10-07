# FastSecDec notebooks

The notebooks are available in the community example browser:

```sh
python -m marimo edit examples/
```

Or open one directly from the repository root:

```sh
python -m marimo edit examples/hep/fastsecdec_showcase.py
python -m marimo edit examples/hep/gghh_complete.py
```

[The self-contained gg → HH study](gghh_complete.py) includes all of its input,
presentation and lifecycle helpers in folded notebook cells. Copy this one file
to another directory and run it with marimo and a current community installation;
it needs no neighboring Python modules, graph fixtures or parameter cards.
It offers the current native one- and two-loop Standard Model catalogue, selecting
a one-loop diagram by default, and keeps generation, inspection, QMC/Havana,
pause/resume, runtime parameters and citations in the notebook. Scientific work
starts only through explicit buttons. Browser export uses
`--notebook gghh_complete` with the FastSecDec exporter and packages only the
community wheel and its manifest.

The default generation mode is `symbolic`, with Taylor subtraction and
native eager evaluators on one caller-owned worker. Preparation uses HEPKit's `contract="dots"`
and `to_dots()` to resolve closed tensor networks before scalar parametrization.
Incoming gluon self-products are exact zero before sector discovery; other
declared products and model inputs remain runtime parameters. The **Kinematic symbols** panel explains the
`dot_i_j` momentum and polarization products, whose values are supplied at
integration. Failed generation retains its error and never represents a zero
integral. The corrected native expression layout is embedded in the community
extension: updating its wheel requires a fresh Python kernel.

[The legacy copied showcase](fastsecdec_showcase.py) starts with a massive scalar
triangle and offers box and two-loop numerator examples, plus an optional longer
gg → HH calculation. Select **Generate** and **Integrate** to start the respective
calculations. Live status, sector inspection, and QMC/Havana controls are included.
Its `showcase/` helpers and `fixtures/fastsecdec/` and `fixtures/gghh/` inputs are
included beside the notebook; no FastSecDec source checkout is required to run it.

The [current modular gg → HH notebook](https://github.com/alphal00p/fastSecDec/blob/main/examples/hepkit/gghh.py)
is maintained in FastSecDec alongside its helper modules and native tests.
The older local [ggHH walkthrough](gghh.py) is retained as a legacy example;
its disabled-cell controls are not the current workflow. Use `gghh_complete.py`
above for the current single-file version. The copied scalar showcase and its
helper tree also retain their earlier workflow; the canonical
[scalar showcase](https://github.com/alphal00p/fastSecDec/blob/main/examples/hepkit/fastsecdec_showcase.py)
lives in FastSecDec.

Use marimo 0.24.2 or newer and a standard community installation containing
`symbolica.community.hepkit.sector_decomposition`. Browser execution also needs
the explicit Wasm wheel and asset manifest produced by the
[FastSecDec browser exporter](https://github.com/alphal00p/fastSecDec/blob/main/examples/hepkit/export.py);
copying a notebook alone does not package a browser wheel. The exporter supports
the current notebooks (`--notebook dashboard`, `--notebook gghh`, or
`--notebook gghh_complete`). This PR does not rebuild or validate a browser wheel.

These are copies of the notebooks and example assets in
[FastSecDec examples/hepkit](https://github.com/alphal00p/fastSecDec/tree/main/examples/hepkit).
The computational implementation and substantive bindings remain in FastSecDec.
See its [full walkthrough](https://github.com/alphal00p/fastSecDec/blob/main/examples/hepkit/README.md)
and [build guide](https://github.com/alphal00p/fastSecDec/blob/main/examples/hepkit/BUILD.md).
