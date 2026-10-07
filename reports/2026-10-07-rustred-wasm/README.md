# RustRed / HEPKit WebAssembly acceptance

Date: 2026-10-07. These are functional **development-profile** measurements,
not optimized performance comparisons. No prior rules were injected into the
browser notebook. Four-loop browser closure is not claimed.

## Build and provenance

The complete Community extension was built with one shared Symbolica kernel,
Cargo's development profile, debug assertions enabled and debug symbols
disabled. The first build took 4m26s in the Rust compilation phase; the final
Python-wrapper alignment fixes took an additional 2m03s incremental compilation.
Packaging and validation are separate from those compile times.

- RustRed: `00bf379338193441ba69de6ab19c26885eaf93f7`.
- FeynKit (`alphal00p/gammaloop`, branch `feynkit`):
  `71552e8942236817185bf71e42d0583d93a7bc08`.
- Symbolica: `ed2374f1d880d52c3a7ca48cd7c22f4baad5c020`.
- Rust toolchain: 1.98.1; Emscripten: 5.0.3.
- Wheel: `symbolica-3.0.0-cp314-abi3-pyemscripten_2026_0_wasm32.whl`.
- Wheel SHA-256:
  `83bd852596eac7ec8fe5b239525cd7f49a34e9a0e8eaa52fdf9daf400ec2d60d`.
- Wheel size: 68,138,746 bytes; uncompressed WASM module: 320,007,215 bytes.

## Completed checks

The tracked `test_pyodide.mjs --rustred-only` gate completed successfully using
Pyodide 314.0.4. It includes the shared kernel, offline rendering, HEPKit graph
and tensor APIs, retained Python-wrapper allocation checks, and the export
allowlist (280 exports, of which only three are functions).

Fresh K6 generation, exact certification, cold loading and reductions passed:
38 terminal master keys, homogeneity restoration, numerator powers, pinches and
master-only output. Candidate generation took 3.073s; the complete K6 check
took 79.261s. These boundaries are not interchangeable.

An optional native64-to-WASM32 canary loaded the unchanged native artifact
bytes (SHA-256 `d656f5792e2daf7ece63596f6cfac6d3dd095bf71e67a1ab424963f36a58d854`),
matched all 38 master keys and reproduced all 11 exact reductions. Coefficients
were compared algebraically, not by printer strings. An independent Pyodide
run also passed the fresh K6 and native-artifact checks.

The actual static Marimo notebook was served over HTTP and executed in a
Chromium Pyodide worker. The browser downloaded the wheel and WASM; there was
no native notebook WebSocket or computation server. Recorded results:

| Browser action | Result | Observed wall time |
| --- | --- | ---: |
| Load notebook / runtime | Ready; no automatic generation | 15.28s |
| Explicit Generate | 623 candidate rules; 38 sectors | 4.41s |
| Explicit Certify | 38 finite terminal masters | 20.84s |
| Lazy coefficient rendering | Passed | Not timed separately |
| Dotted reduction | 30 master terms; exact check passed | 7.12s |
| Numerator reduction | One master term; exact check passed | Not timed separately |

There were no browser page errors. Screenshots were inspected. UI timings
include notebook scheduling and rendering and differ from the solver-only
boundaries. This is single-worker synchronous execution, not browser
multithreading or live event streaming during a running calculation.
A second fresh browser run reproduced all assertions (startup 14.29s,
generation 3.50s, certification 18.18s, dotted reduction 6.13s).

Native adapter regressions passed (16 tests), as did the focused Community
notebook/API tests (61), JS scope/export-policy tests (13), and all six checked
dependency ownership feature graphs. Native artifact persistence (81) and
solver/application execution controls (17 and 8) also passed during this port.

## Explicit limitation: unrelated full-suite failure

The **default full Community debug gate is not green** at these pins. After
RustRed passes, RustFlow `9599e358` aborts in
`KinematicTransport.add_boundary`: a Python wrapper contains an inline `u128`
whose required alignment exceeds Pyodide's Python-object allocation alignment.
RustRed and FeynKit's analogous wrappers were fixed upstream by retaining their
payloads in aligned Rust-owned allocations. RustFlow is not changed here.

The full gate remains enabled by default. The explicit focused gate reports
`scope: rustred-only` and never emits a successful loop-transport receipt.
Vakint and native compiled evaluators remain native-only in this integration.

## Reproduction

Use the repository's configured Pyodide cross-build environment, a suitable
Symbolica license in the environment and a fresh output directory:

```sh
CARGO_PROFILE_DEV_DEBUG=0 WASM_RUST_PROFILE=dev WASM_OPT_LEVEL=-O0 \
  WASM_SKIP_TESTS=1 bash scripts/build_wasm_performance.sh dist/wasm-dev
PYODIDE_DIST_DIR="$(pyodide config get dist_dir)" \
  node .github/scripts/test_pyodide.mjs dist/wasm-dev --rustred-only
python scripts/export_rustred_wasm.py \
  dist/wasm-dev/symbolica-3.0.0-cp314-abi3-pyemscripten_2026_0_wasm32.whl \
  dist/three-loop-browser
python -m http.server --directory dist/three-loop-browser 8000
```

Open the page, then click Generate, Certify generated rules and Reduce to
certified masters. The exporter requires the matching real runtime receipt.
The [HEP notebook guide](../../examples/hep/README.md#rustred-in-pyodide) describes
the additional native-artifact canary. Build outputs and browser evidence are
not committed; this report records their identities and measured scope.

## Follow-up: name the terminal values

The 38 certificate entries are raw labelled keys, not 38 independent masters.
The existing native `IBPFamily.normalize_candidate_terminals` API was checked
on a freshly generated graph-family candidate with exactly the same 38 raw
keys: it returned 33 unit-coefficient aliases, five canonical representatives
and no skipped shapes. Their multiplicities are 16 three-tadpole products,
12 sunset × tadpole products, three basketballs, six five-line connected
vacua, and one Mercedes. The notebook now names these types explicitly and
retains the original certificate and raw reductions. See the
[complete named census](https://github.com/alphal00p/rustred/blob/main/docs/k6_terminal_names.md).

The updated notebook passed the same real-browser generation, certification
and reduction controls using the existing validated wheel. Dependency updates
following that wheel only document the required aligned wrapper allocations
and the terminal census; no solver or ABI behavior changed. The wheel hashes
and source revisions recorded above remain its actual build provenance, not
the subsequently updated lockfile's identity.

The follow-up adds a native regression for all 38 aliases and their five
representatives. The focused native routing/normalization checks passed, as
did the three-loop notebook checks and the six dependency-ownership graphs.
A broader HEP-notebook sweep in the pre-existing native test environment was
not clean (193 passed, 105 failed, one skipped); it includes unrelated missing
optional packages such as NumPy and older installed API mismatches. It is not
reported as full native-suite acceptance, and those tests were not weakened.
