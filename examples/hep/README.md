# HEP bindings

The GammaLoop `feynkit` branch is bundled through `feynkit-py`. Its public API
is flat under `symbolica.community.hepkit`, sharing the same Symbolica kernel as
Spenso and Idenso:

```python
from symbolica.community.hepkit import FeynmanDiagram, Model, TensorReducer
```

| Component | Examples of public classes |
| --- | --- |
| `feynkit-graph` | `FeynmanDiagram`, `DiagramEdge`, `DiagramVertex`, `LoopMomentumBasis` |
| `feynkit-generator` | `Process`, `GenerationResult`, `GenerationProgress` |
| `feynkit-cff` | `CffGenerator`, `CffResult`, `CffSurface`, `CffOrientation` |
| `feynkit-tensor` | `TensorReducer`; reduction methods also live on `FeynmanDiagram` |
| `feynkit-py` | All of the above, plus models, UFO loading, kinematics, and jet clustering |

Class identities and the exception hierarchy are preserved; their Python
`__module__` and the bundled stubs name `symbolica.community.hepkit`.

Numerical external states use the shared GammaLoop/MadGraph conventions:

```python
from symbolica.community import hepkit as hep

p = hep.FourMomentum(150.0, 0.0, 0.0, 150.0)
epsilon = p.wavefunction("epsilon", hep.Helicity.PLUS)
components = epsilon.components
assert epsilon.bar().bar() == epsilon
```

`FourMomentum.wavefunction` accepts `scalar`, `epsilon`, `epsilon_bar`, `u`,
`u_bar`, `v` and `v_bar`. Scalars use zero helicity; spinors use plus or minus;
massive vectors also admit zero helicity. The inherited longitudinal convention
is undefined at rest or zero mass and raises `KinematicsError`. Vector components
follow `(E,x,y,z)` with metric `+---`; spinors use the chiral gamma basis and
`bar()` takes the Dirac adjoint. These numerical external states have four
components (one for scalars), independently of the symbolic internal dimension.
They include no couplings, helicity sum or averaging.

Start with [gamma algebra](gamma_simplification.py), then
[color algebra](color_algebra.py) and [tensor reduction](tensor_reduction.py).
[Higgs diphoton decay](Higgs_diphoton_decay.py) follows a generated fermion loop
through symbolic reduction, native master evaluation, and a complete CFF Monte
Carlo decay-width calculation, with separate numerator and mass-shift
derivations of the R₂ and R₁ rational terms and typeset physical results.
[Tensor benchmarks](tensor_benchmarks.py) keep timing experiments, component
oracles, and optional FORM runs separate from the introductory examples.

Two generated four-loop propagator examples reduce three-rung ladders to dot
products: the [all-gluon ladder](three_gluon_rung_ladder.py) and the
[outer quark-loop ladder](quark_loop_ladder.py). Both select their topology with
the graph generator, retain its Feynman rules and weights, and project with
`g_mu_nu delta_ab/8` in Feynman gauge with symbolic Lorentz dimension. The quark
example uses one massless flavor. Each notebook displays the generated graph,
the 15 scalar-product coordinates, and downloadable full numerators, keeping
propagator denominators separate; these examples do not integrate the loops.

The tensor examples use `symbolica.community.tensor` consistently:

1. Construct typed tensors and declare their representations and dimensions.
2. Use `simplify_algebra(AlgebraSettings(...))` for explicitly enabled identity
   families, such as `gamma=GammaSimplifySettings()` or
   `color=ColorSimplifySettings()`.
3. Use `contract(ContractSettings(...))` for structural contractions and
   requested chain or trace notation. Collecting a trace does not evaluate it.
4. Use `to_dots()` only to normalize existing scalar-product notation. Convert
   to a scalar `Expression` before scalar substitutions, series, or integration.
5. Check the result against an exact identity or an independent reference.

Reduction returns an alias-owning result; its `to_expression()` materializes a
`TensorExpression`, whose own `to_expression()` exposes the underlying scalar
expression once all tensor ports have been contracted. Arithmetic expansion is
an explicit step in examples that need polynomial coefficients. Neither settings
type implicitly enables unrelated algebraic identities.

The Marimo directory server (`marimo edit examples/` from the repository root)
provides separate worked IBP notebooks:

- [One-loop reduction and master evaluation](oneloop_reduce.py): pass a shared
  `hepkit.IntegralFamily` and explicit powers to `hepkit.oneloop.reduce` for primitive
  `oneloopmaster::` symbols, exported as `oneloop.A0`, `B0`, `C0`, and `D0`
  and evaluated through their native Rust hooks.
  Interactive examples cover a triangle numerator, an exact dilogarithmic C0
  expression with `get_expression` and `select_branch`, and a squared tadpole
  whose finite term includes epsilon-dependent prefactors.
- [Unequal-mass bubble](ibp_bubble.py): raised powers and numerical master evaluation.
- [Bubble differential equations](ibp_differential_equations.py): compatible equations
  in the invariant and masses, with numerical transport from a boundary value.
- [Massless form-factor triangle](ibp_triangle.py): raised powers and infrared poles.
- [Massive two-loop 2- and 3-point functions](ibp_two_loop_masses.py): symbolic-index
  recurrences with independent masses, application to raised powers and linear
  combinations, and exact checks of the defining IBP identities.
- [Two-loop phi4 self-energy](ibp_phi4.py) and [vertex](phi4_two_loop_vertex.py):
  generated diagrams, reductions and counterterms.
- [Four-loop vacuum IBP laboratory](four_loop_reduction.py): standard
  DOT graphs for H, X, BMW and FG become routed HEPKit families, then run a
  native single-worker RustRed candidate search with streamed progress and
  lazy rule, guard and terminal inspection. It generates the candidates live;
  it does not load a prepared rule catalog or assert full-family closure.

All these examples use the shared `hepkit.IntegralFamily` frontend with
`hepkit.Kinematics` and a symbolic dimension. The one-loop example chooses
`hepkit.oneloop.reduce` for reduction to OneLoopMaster symbols; `hepkit.IBPFamily`
provides native RustRed reductions of the same families. Links inside each
notebook stay on the same server.

## Numerical transport and supplied reductions

Native numerical loop evaluation lives under
`symbolica.community.hep.integration`. The [Higgs-jet notebook](gg_hg.py) uses
these classes with the same HEPKit model, families and exact kinematics as the
other examples. The native empty-cache boundary calculation and independently
refined observables are recorded in the
[validation report](https://github.com/alphal00p/RustFlow/blob/main/docs/python-notebook-status.md).
Opening the notebook starts no boundary calculation.

`IntegralEvaluator(reductions=tables)` accepts a `ReductionTables` collection.
Use `with_family` to add exact rules and declared residual masters for a native
`hepkit.IntegralFamily`; it returns a new collection and rejects duplicate scopes
or uncovered rule leaves. Rules use the family's original epsilon symbol with
the dimension convention already substituted. Propagator order, routing,
parameter namespaces, dimension and numerator-slot roles belong to the scope.
Equivalent rational coefficient forms share a scope, while original denominator
restrictions remain distinct. Preserve uncancelled expressions when admitting
rules, or pass their exclusions explicitly as `nonzero_conditions`.

Auxiliary deformations, specialized kinematics and recursive boundary families
need their own admitted tables. Supplied identities remain the caller's
mathematical contract; admission checks scope and consistency of the reduction
graph. `PreparedIntegralFamily` exposes the retained basis, target reductions
and nonzero conditions for inspection. `BoundaryCache` separates entries for
different ordered bases, normalizations and conditions, including after binary
reload. A condition omitted from a simplified differential matrix still
restricts endpoint and path admissibility.

Focused native-object examples and regressions are executable without the
two-loop notebook:

```sh
python -m pytest tests/test_loop_integration_reductions.py -q
```

### Higgs-plus-jet calculation and acceptance

The [gg→Hg notebook](gg_hg.py) separates native boundary generation, physical
transport and coherent amplitude assembly. All three stages use one loaded
HEPKit model. HEPKit supplies diagrams, state sums and model-parameter
expansion; Spenso and Idenso supply tensor and color contractions. The physical
form-factor projections depend on the actual kinematics, including nearby
evaluations. Numerical references never supply boundary or amplitude inputs.

For a live demonstration, select **Load supplied boundaries**. The notebook
ships all sixteen independently refined, native-generated starting values in
`data/gg_hg/boundaries.json.gz` (1,054,943 bytes). Each coefficient keeps its exact
binary rational, working precision, comparison error, and 40-digit input cap.
The manifest records its checksum and origin. Import verifies the equations,
ordered bases, normalization, coordinates and root sheets, then admits the
values through the current build's supplied-boundary API. This portable data
does not bypass binary cache compatibility checks. Destination integrals and
amplitudes are still computed live. **Recompute boundaries (native)** remains
available for the full calculation.

The controller uses one worker by default. On Emscripten it creates no worker
thread and yields to the event loop after each checkpointed configuration;
`await session.wait_async()` is its headless asynchronous entry point.
Cancellation is observed between configurations, while each native solver or
amplitude call is synchronous. A build without automatic boundary generation
disables that control and reports insufficient supplied accuracy without
attempting a native recomputation. Full browser solver timing requires the
transport-enabled WASM build; native timings are not browser measurements.
Browser cache files live in the current page's virtual filesystem. Binary
reload works within that session; page reload does not persist the files.

The [single-core transport comparison](https://github.com/alphal00p/RustFlow/blob/main/reports/performance/2026-10-06-physical-transport-order16/README.md)
measured all sixteen configurations with the same supplied seeds and growing
cache. At requested 20-digit accuracy, initial guard 20/order 16 took 68.1 s,
versus 91.5 s with guard 30/order 32 and 270.2 s with guard 60/order 96.
Independent exact-arithmetic checks cover all 4,360 final coefficients,
8,720 inserted coefficients, eight form factors and three observables.
The notebook uses guard 20/order 16, retaining adaptive checks; automatic
boundary generation keeps its previous settings. A guard-10 trial took 64.4 s
but reduced the master accuracy estimate from 28 to 20 digits; guard 20 retains
the additional margin. These are native solver component measurements from one
run of each profile, excluding browser startup, UI and checkpoint I/O.

To prepare a browser demonstration, use a community Pyodide wheel containing
the physical transport API. Supplied-boundary transport, exact binary restart,
nearby reuse and cancellation passed the
[actual Pyodide smoke gate](../../reports/2026-10-06-browser-loop-transport/report.json)
on one core without license credentials. Full gg→Hg scientific and browser UI
acceptance remains separate. The exporter requires a successful
`test_pyodide.mjs` transport smoke report
matching the exact wheel checksum. A Python namespace alone is insufficient.
Marimo's exporter requires `uv` in the build environment.

```sh
python scripts/export_gg_hg_wasm.py /path/to/symbolica-*-pyemscripten_2026_0_wasm32.whl /path/to/new-export
python scripts/serve_wasm_bundle.py /path/to/new-export
```

The export includes the wheel, controller, exact model, supplied starting
values and comparison-only observable data. The notebook fetches its inputs
and wheel relative to its own URL, verifies their checksums, and installs the
verified wheel bytes from Pyodide's filesystem.
Ordinary HTML export does not package these external assets automatically.

The load-or-compute controls preserve accumulated intermediate points. Forced
boundary recomputation archives both verified-boundary banks and all completed
numerical samples under `numerical-history/` before starting, while retaining
exact reduction data. A durable generation record lets a restarted session
reuse only new work, even at unchanged precision with identical sample keys.
If preparation is interrupted between directory moves, ordinary stages fail
closed; create or reuse a session and select **Recompute boundaries** again to
recover. Prior numerical files remain in the history directories. Forced
transport resets its bank to the saved seed-only bank. Cancellation can leave
completed samples or complete configurations for restart; it never admits a
partial set of transported configurations to amplitude assembly. Forced
amplitude assembly clears derived values before projection, reuses its exact
contraction kernel, and publishes new form factors and observables only after
successful evaluation at the current kinematics. Native
cancellation checks occur between algebra/reducer operations, so stopping a
large operation may take time. Use one live session per cache directory.

Each transport stage first persists its current bank, including valid in-memory
points retained after a previous failed save. Newly computed points are then
checkpointed after each configuration. Exact cache hits do not rewrite the
unchanged bank individually, so an exact repeated 16-configuration query needs
one checkpoint rather than sixteen.

The lightweight controller and notebook checks start no two-loop evaluation:

```sh
python -m pytest tests/test_hep_gg_hg.py tests/test_hep_notebooks.py -k gg_hg -q
```

The long runners require the same release-built native extension. First run the
independent Euclidean-anchor check with an empty output directory:

```sh
python examples/hep/gg_hg_anchor_acceptance.py --directory /path/to/euclidean-anchors --workers 16
```

This computes the planar point `(-1/10,-1/25,-1/50)` and nonplanar point
`(-1/10,-1/5,-1)` with principal roots. These exact inputs are independent of
the comparison files. Both native calculations must finish before either
archived numerical reference is opened. All 240 planar and 305 nonplanar
coefficients through epsilon power four are checked against their recorded
40-digit allowances and native uncertainty estimates, with at least 20 digits
of independently checked native accuracy. No Mathematica or plugin runtime is
required. The output is `/path/to/euclidean-anchors/anchor-acceptance.json`;
individual run reports remain under `runs/`.

Use `--resume` to continue completed configurations and samples. `--force`
archives both old numerical banks and sample directories before either family
starts, retaining exact reductions. An interrupted forced run can then resume
only its new numerical work. `--cancel-file PATH` requests cooperative native
cancellation when that file exists. Verified banks and completed samples are
stored separately. If interruption occurs while the old directories are being
archived, restart with `--force`; a persisted marker refuses ordinary resume
until both families have been safely prepared.

Pass the anchor report to the full physical run to require matching native
source identity, installed extension and steering sources before computation.
Use an empty directory for a cold physical run:

```sh
python examples/hep/gg_hg_acceptance.py --directory /path/to/empty-run \
  --anchor-report /path/to/euclidean-anchors/anchor-acceptance.json
python examples/hep/gg_hg_acceptance.py --directory /path/to/run --resume \
  --anchor-report /path/to/euclidean-anchors/anchor-acceptance.json
```

Configuration concurrency is optional and defaults to one. To run four
independent boundary configurations with a total budget of 64 finite-epsilon
sample workers, use:

```sh
python examples/hep/gg_hg_acceptance.py --directory /path/to/empty-run \
  --boundary-workers 4 --workers 64 --interrupt-after-samples 1 \
  --anchor-report /path/to/euclidean-anchors/anchor-acceptance.json
```

The controller caps configuration concurrency at the total worker budget and
divides that budget evenly (rounding down), giving 16 sample workers per
configuration in this example. Each configuration uses a private native cache
initialized from the saved seed bank; a single coordinator merges and
atomically saves successful results as they finish. The returned results keep
configuration order. A failure cancels siblings through their shared native
cancellation token, and completed configurations remain available for restart.
The report includes configuration timings and the actual worker allocation.
One native model is shared by the session throughout.

Add `--interrupt-after-samples 1` to exercise typed cancellation during the
forced higher-precision boundary run, after a complete sample checkpoint and
before its first verified configuration. The initial cold run is completed
without this interruption. After cancellation, the runner reloads and resumes
the new generation's completed samples; its previously cleared boundary banks
remain empty. Archived pre-force samples and active new samples are counted
separately, with retained-payload checks for both. Progress and timing records
include the numerical generation. Both interrupted work and cancellation latency are
reported. With the defaults, the cold run starts with 30-digit seeds and forced
refinement requests ten extra digits (40 unless observable error propagation
already required stronger seeds). These are different precision workloads, so
their timings are not a same-precision speedup measurement.

Once all sixteen native physical seeds are generated, the runner checks 4,360
transport coefficients and the coherent EW/HEFT/interference observables. It
then checks binary reload with exact repeated hits and retained provenance,
forced recomputation at increased seed precision, and nearby evaluation from
accumulated physical points. Its JSON report distinguishes cold, resumed, warm,
interrupted forced-refinement, resumed refinement and nearby timings, retaining
the requested seed precision on every stage. Reference access begins only after the
first native amplitude exists, and reference precision caps remain in the
comparison report. The first report is written before expensive work starts.
While a stage is running, `acceptance.json` is replaced atomically every five
seconds with the stage status, elapsed time, recent native events, completed
configuration timings and completed-sample file count. It is also updated at
stage completion; warm cache hits return immediately without waiting for the
polling interval. Observing progress neither changes numerical results nor
admits new boundary evidence. A failure or external interrupt is recorded
before waiting for native workers to stop.

Independent native planar and nonplanar Euclidean-anchor validation is the
separate prerequisite described above. Supply its report with `--anchor-report`
to associate the full physical run with that matching-source evidence. Success
of the lightweight tests alone must not be reported as completion of the
native-boundary acceptance. Recorded cold native results, source versions and
uncertainty limits are available in the validation report linked above; a new
runtime or dependency graph needs its own acceptance evidence.

## Live RustRed generation and artifact exploration

The four-loop laboratory needs a native community build containing the
`rustred-feynkit/campaign-api` feature, enabled in this checkout. It uses the
host's existing Symbolica kernel; do not install a separate RustRed extension
to provide the notebook's native objects. From an activated virtual environment
with the project's native build dependencies and a suitable Symbolica license:

```sh
pip install 'maturin>=1.13.2,<2' 'marimo>=0.24.2,<0.25'
RUSTFLOW_WORKSPACE_FEATURES=pyo3/extension-module RUSTFLOW_WORKSPACE_NO_DEFAULT_FEATURES=0 \
  maturin develop --release --locked --extras notebook-display
marimo edit examples/hep/four_loop_reduction.py
```

Select **Generate H → X → BMW → FG** explicitly. The notebook's collapsed setup and reusable
`rustred_campaign_support.py` handle presentation and polling; the RustRed
engine owns generation, events, cancellation and artifact access. Each family
runs its physical positive-sector downset with auxiliary ISP coordinates
nonpositive, exact sparse arithmetic and numerical search depth two. The
visible configuration cell controls the worker count and search settings.
Native session elapsed time includes preparation, solving and bundle assembly;
the native timing report separates these phases. Artifact writing and explicit
view-call timings are recorded separately.

Sessions do not call back into Python from native worker threads. Polling
returns bounded batches and a current aggregate snapshot; a slow consumer can
lose intermediate events, with their count reported explicitly. **Cancel** is
cooperative, not a promise to interrupt the current algebra operation. Keep
refresh enabled to move to the next family. These sessions and lazy views must
not be inherited across `fork`; start a new process using `spawn` instead.

Generated candidate artifacts can be reopened with
`hepkit.rustred.CandidateArtifact.open_file(path, bundle_max_entries=10_000_000)`.
The example uses the same explicit collection-entry allowance for saving and
reopening; this changes transport capacity, not the search or integral scope.
Sector/rule/terminal pages
and metadata do not decode all coefficient polynomials. Table search filters
only the current fetched page (up to 25 rows), not the full artifact; use the
page offsets to browse other pages. Rule structure is capped at 64 KiB by
default, with explicit larger allowances. RHS and guard previews contain ten
rows, not the entire selected rule; their raw JSON is bounded to the same page.
Changing a coefficient ID does not decode it: click **Render** explicitly for
an 8 KiB native printer preview, with larger budgets available on request.
Rendering is disabled during generation; reopen the artifact after it drains.
The shared Symbolica state is imported once and decoded coefficients are cached.
Encoded artifact bytes and structural records still occupy memory; lazy browsing
avoids eager coefficient decoding and large initial HTML, not all file loading.
Displays are bounded previews, not an alternate serialization
format; the binary artifact remains authoritative. Open only trusted generated
artifacts. A finite list of residual terminals or a completed generation session
is not a certificate of closure, termination or master minimality.

After generation drains, **Normalize completed terminal sets** explicitly calls
the existing native family-local normalization. It reports raw records, distinct
keys, unit aliases and weighted outputs separately, with paged relations and
lazy coefficient views. Unsupported shapes remain outputs. The discussion
compares this convention with FMFT's 19 symbolic representatives, without
equating those representatives to raw residuals or numerical Laurent constants.
Neither a favorable count nor this normalization proves master independence.

The optional **Evaluate H numerator once** action is a separate post-generation
calculation. It uses the same H graph with a new symbolic-mass family, native
FeynKit tensor reduction and Vakint's shipped RustRed assets, not the candidate
files just generated. It compares five computed Laurent coefficients with the
existing 32-digit H rank-four reference at the stated scales. FORM is not run;
the numerical master inputs are supplied by Vakint, not newly computed here.
This single integral test does not establish arbitrary-index family closure.
The initial state never evaluates it; an explicit click is required after
generation drains, and the result is cached against reactive UI reruns.

`vakint.integral_from_diagram(...)` is a native binding to Vakint's Rust
`VakintExpression::from_diagram`, not a Python graph/algebra adapter. It consumes
FeynKit's stored routing and validates the family without rematching the graph.
The standalone Rust entry is available through Vakint's `feynkit-ingress` feature.
RustRed use also registers the SpideR strategy paper (arXiv:2604.25916) with
Symbolica's process-wide `get_citations()`; importing the module alone does not.
The native-ingress follow-up passes 136 installed-host tests, including the
unchanged 32-digit H reference with an invalid FORM path, simultaneous scalar
substitutions, general mappings and selected-view validation. This regression
check does not repeat the generation workload reported below.

### Measured notebook validation

On 5 October 2026, one explicit **Generate** click completed a fresh H → X →
BMW → FG run with one native worker and the settings above. No checkpoints or
precomputed candidates were reused, and no sector failed.

| Family | Sectors | Native generation (s) | Rules | Raw residuals | Normalized outputs |
| --- | ---: | ---: | ---: | ---: | ---: |
| H | 314 | 91.753 | 21,318 | 386 | 22 |
| X | 328 | 281.629 | 19,907 | 445 | 19 |
| BMW | 134 | 110.898 | 9,018 | 179 | 17 |
| FG | 124 | 57.284 | 9,266 | 145 | 16 |

The native sessions totaled 541.563 seconds (9.03 minutes); the notebook
controller took 544.258 seconds including artifact collection, writing and
polling. Compilation is separate and excluded. These are shared-host
observations, not runtime guarantees or reduction/closure benchmarks.

Live browsing, page-local search and paging passed. Each artifact also reopened
in a fresh process in 0.108–0.511 seconds: structural browsing decoded zero
recurrence coefficients, and one explicit render decoded exactly one. These
are fresh-process observations, not cold operating-system-cache timings.
Normalization took 0.098–0.315 seconds per family. The resulting integer key
sets exactly matched the corresponding shipped Vakint normalized output sets;
this does not establish byte-identical programs, coefficient-by-coefficient
equivalence or automatic installation of the new candidates.

The separate H numerator evaluation matched all five 32-digit reference
coefficients at relative tolerance `1e-30`, with an invalid FORM executable
path. Native symbolic evaluation took 8.941 seconds and numerical substitution
0.002 seconds. It used Vakint's shipped assets, not the freshly generated
programs. **Generation and terminal normalization are not closure proofs or
proofs of an independent master basis.**

## Notebook tensor displays

Install the optional live-display dependencies alongside the community package:

```sh
pip install 'symbolica[notebook-display]'
```

Large `TensorExpression` outputs and tensor-backed `formatted()` results page
automatically in marimo and Jupyter. To choose the initial maximum page size:

```python
expression.paged(page_size=25, settings=None, notation_source=None)
```

Previous/Next replace the displayed page. The selector offers 25, 100, 250, or
500 terms, defaulting to 25; complex expressions may show fewer. Ellipses mark omitted portions.
When an inner sum is paged, a compact expression outline keeps its surrounding
factors and parentheses visible. Numbered ellipses identify the selected sum
and other omitted subexpressions; the selected sum's terms appear separately
below the outline. Open-subexpression buttons and Parent navigation inspect
other large parts without expanding the expression or introducing scalar-product
aliases. Wrapping adds no inner vertical scroll area.

Each request selects at most 2,000 expression nodes, depth 32, and 64 KiB of
expression data. Rendering limits its output to 256 KiB and 10,000 MathML
elements, and compilation to ten seconds in a separate process. The status line
explains when a budget reduces the selected maximum. The Horizontal scroll
checkbox switches between wrapped terms and a scrollable line (the default). A viewer caches at most three pages. Oversized individual terms become
bounded subexpression previews. These limits are independent of marimo's
8 MB output guard, which remains enabled.

Static exports and environments without Anywidget show a bounded preview.
Navigation requires a live notebook kernel. `to_html()` and `to_svg()` remain
explicit full-export methods; ordinary Symbolica styled text keeps its existing
truncation. Existing notebook kernels need to load the rebuilt package before
using the new display hooks.

After installing a community wheel built from this checkout:

```sh
python examples/hep/diagrams.py
```

The small `scalar_phi3.json` model is a one-particle, cubic-interaction subset
of GammaLoop's `scalars_2p_3p.json` test fixture (MIT / Apache-2.0). It requires
no external model downloads or integral backend.

The bindings are included in both native and Pyodide builds. Native-only
dependencies such as Vakint remain excluded from WASM. Diagram SVG/HTML
rendering may additionally require the upstream optional `typst-py` renderer;
generation, DOT/JSON export, and CFF algebra do not require it.

Regenerate only the HEP stubs, without rewriting the core or other modules:

```sh
cargo run --bin stub_gen --no-default-features --features python_stubgen -- --hep-only
```
