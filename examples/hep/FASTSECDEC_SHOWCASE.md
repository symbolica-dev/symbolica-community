# FastSecDec showcase

Open `examples/fastsecdec_showcase.py` with **marimo 0.24.2** and the community
wheel built with the native FastSecDec bridge:

```sh
python -m marimo run examples/fastsecdec_showcase.py
```

The wheel requires the opt-in `experimental-fastsecdec` feature. Follow the
[build prerequisites and dependency setup](FASTSECDEC_BUILD.md) before building
or installing it.

The notebook imports the existing native HEPKit diagram, model, kinematics and
Symbolica owners. `fastsecdec_inputs.py` owns the four example builders;
`fastsecdec_views.py` only schedules bounded caller steps and presents native
snapshots. No numerical outputs are bundled. Use the existing Symbolica license
configuration of your environment; never put a license in the notebook or export.

1. Choose the massive triangle, massless box, rank-two box or coupled sunset.
   Edit the mass and scalar invariants and select **Apply inputs**. Only the
   triangle uses `m`; only boxes use `t`. Every case requires negative `s`, and
   boxes require negative `t`.
2. Inspect the native graph and weighted scalar numerator. Graph display uses
   HEPKit's renderer, with a visible error if that renderer is unavailable.
3. Select **Run**. Generation and compilation are synchronous and emit genuine
   typed phase observations. During these phases, use marimo's interrupt control;
   an in-flight algebra operation may delay the next safe boundary.
4. Once kernels are ready, each refresh invokes at most one native QMC package on
   the same Python thread. The refresh interval controls caller scheduling, not
   a worker pool or a guaranteed repaint interval. Disable refresh to pause
   scheduling or click refresh once to advance one package. The displayed
   integration wall time includes refresh waits and excludes caller-cancelled
   intervals between Cancel and Resume. Native worker time is separate.
5. **Cancel** stops future packages and saves the native accepted checkpoint.
   It does not add an extra package just to obtain a native cancellation label.
   The caller message and native stopping reason are shown separately.
   **Resume checkpoint** restores native coverage, numerical replay state and
   diagnostics. A numerical error retains its stage and accepted prefix.
6. Read the entire signed Laurent vector and full covariance. Missing estimates
   remain missing. The chart records the highest signed requested epsilon order
   whenever the native session provides a valid estimate; successive observations
   share samples. The 0.1% selected-order target requires complete production;
   native all-vector convergence is reported separately. No allocation is enlarged
   automatically.

Changing or reapplying the form does not change the run already displayed. Cancel
an active run before starting a different one. Download native kernel bytes and
checkpoint bytes after stopping or completion; these are the bridge's own codecs,
not a second notebook persistence format.

## Browser export with an explicit asset mount

A native wheel cannot run in Pyodide. First obtain a tested community wheel for
Python 3.14 compatible with marimo 0.24.2's Pyodide 314.0.0 runtime, containing
FastSecDec's portable interpreted backend.
The export helper packages that **existing** wheel; it does not compile it or
certify responsiveness:

```sh
python examples/hep/export_fastsecdec_showcase.py \
  --wheel /absolute/path/symbolica-3.0.0-cp314-abi3-pyemscripten_2026_0_wasm32.whl \
  --output /absolute/path/new-fastsecdec-site
python -m http.server --directory /absolute/path/new-fastsecdec-site 8000
```

The exported notebook fetches `public/fastsecdec/manifest.json`, checks the SHA-256
of the wheel and asset archive, installs that explicit wheel, and mounts the
helpers and native fixture files at `/fastsecdec-showcase` before importing them.
The archive contains the scalar model, four DOT inputs, builder, display helper
and fixture attribution. No repository `Path` is presumed to exist in a browser.
The installed wheel determines the real backend label (`native_o2` or
`portable_interpreted`). Serve the whole export directory over HTTP. Marimo's
runtime assets may still require internet access; this helper does not claim an
offline export.

The native run, browser import, rendered controls, cancellation/resume and bounded
numerical checks are separate validation gates. A successful export alone passes
none of the numerical gates. The gg → HH example remains outside this notebook
until ordinary native feasibility is demonstrated.

The control and output APIs were checked against installed marimo 0.24.2. Related
upstream documentation: [refresh](https://docs.marimo.io/api/inputs/refresh/),
[outputs](https://docs.marimo.io/api/outputs/), and
[WebAssembly data files](https://docs.marimo.io/guides/wasm/).

The tested configuration used a wheel built with Pyodide 314.0.7 in the actual
marimo 314.0.0 browser runtime. Asset mounting, native graph rendering and the
triangle's Run/Cancel/Resume lifecycle passed. Separate portable bridge tests
generated and compiled all four examples; the three non-triangle cases accepted
one 32-point package each, without claiming complete estimates or convergence.
These gates do not establish uniform performance across browsers or inputs.
