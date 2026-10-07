# Automatic loop evaluation in Pyodide

The complete Community development-profile Pyodide gate passed with Community
`99ccc95a1ed3391acbdbe82d266ef28f45a950b3` and RustFlow
`b34ff6b3261bcc48f867d704e1d560e1bc4a4fec`. This is an actual installation of the
complete wheel in Pyodide, with one shared Symbolica kernel and no extension
replacement. All dependencies were resolved from the committed Git pins.

## Validated behavior

- Exact tadpole IBP target reductions and equivalent scoped supplied reductions,
  using native HEPKit families, expressions and arbitrary-precision values.
- Fresh finite-epsilon tadpole and massless-bubble evaluation, compared against
  independent gamma-function values at 20 decimal digits.
- A 20-digit tadpole Laurent expansion and a freshly generated boundary with
  30 verified digits, followed by physical mass transport and binary restart.
- Supplied-boundary transport, intermediate-point reuse, exact repeated cache
  hits, retained uncertainty/provenance, pre-cancellation and queued progress.
- Typed rejection of multiple workers and retained-wrapper alignment checks.
- Shared graph/tensor operations, rendering, the Standard Model Higgs-jet helper,
  fresh RustRed K6 generation/certification, exact reductions and the integration
  contract/Symanzik checks. The default full-Community gate was used.
- All six dependency ownership/feature graphs and 17 focused graph/runner tests.

The build took 41.29 seconds with an existing private dependency cache. The
fresh full runtime gate took **574.09 seconds**.
This unoptimized development profile exercises debug alignment assertions;
these timings are correctness checks, not release-performance measurements.
No new complete Higgs-jet boundary/amplitude or Chromium notebook acceptance is
claimed here. Browser support refers to Pyodide/Emscripten; this report does not
establish bare `wasm32-unknown-unknown` runtime support.

## Artifact and evidence

- Wheel: `symbolica-3.0.0-cp314-abi3-pyemscripten_2026_0_wasm32.whl`, 69,191,265 bytes.
- Wheel SHA-256: `ca73c9304e5314b3ff0d99454fc8bfa858640bb83936a660fc60e0173cd65785`.
- WebAssembly module SHA-256: `de8baf43e5f89df66e80e591818b2754dce641d099209e7d50098b84b9366cec`.
- Evidence archive SHA-256: `419ff1f36975426c536cc130a16f785f57bbe8047c5ee4aa241a7d7c12cf74e7`.

[report.json](report.json) records exact owners, source hashes and runtime
receipts. [evidence.tar.gz](evidence.tar.gz) contains build/runtime/check logs,
the capability receipts and the checked source snapshots. No binaries or
license credentials are included. The earlier working-source diagnostic also
passed (544.05 seconds); the evidence here certifies the immutable published
pin, independently of that diagnostic.

The default `.github/scripts/test_pyodide.mjs` gate includes automatic checks.
It writes `automatic_boundary_generation` and numerical evidence into the
wheel-hash-bound `loop-transport-validation.json` only after all gates succeed.
`--rustred-only` retains its explicitly narrower scope.

## Browser execution

Evaluation is synchronous and restricted to one worker. Progress can be polled
after a call returns; pre-cancelled controls stop a subsequent call. A callback
in the same worker cannot interrupt synchronous work already in progress.
Cache files use Pyodide's virtual filesystem and require explicit export for
persistence across browser sessions. Large multiloop boundary calculations may
still be impractical in a live notebook.

## Reproduction

With the configured Pyodide cross-build environment and an appropriate Symbolica
license in the environment, use a private Cargo target and fresh output folder:

```sh
CARGO_TARGET_DIR=/path/to/private-target \
  CARGO_PROFILE_DEV_DEBUG=0 WASM_RUST_PROFILE=dev WASM_OPT_LEVEL=-O0 \
  WASM_SKIP_TESTS=1 bash scripts/build_wasm_performance.sh dist/wasm-dev
PYODIDE_DIST_DIR="$(pyodide config get dist_dir)" \
  node .github/scripts/test_pyodide.mjs dist/wasm-dev
```
