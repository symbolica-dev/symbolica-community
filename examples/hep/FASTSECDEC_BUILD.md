# Building the FastSecDec bridge

The bridge is enabled by the opt-in `experimental-fastsecdec` Cargo feature and
consumes the exact published FastSecDec revision in `Cargo.toml`.
Its native HEPKit, Symbolica and Numerica dependencies currently need the small
reviewed source patches distributed by that FastSecDec checkout. Apply those
patches through an explicit Cargo config so every native Python object retains
one Rust owner. The setup helper clones pinned public revisions into a new
directory outside this community checkout and refuses to overwrite existing
output or local reference trees. Keeping the dependency workspace outside this
checkout avoids Cargo's automatic membership of nested path dependencies.

**Current source-build prerequisite:** ordinary resolution without this owner
config is blocked even with the feature disabled. Cargo resolves optional
dependencies for the lockfile, and published OneLOop requires SymJIT 2.26.0 while
FastSecDec requires 2.26.4. The reviewed local OneLOop patch aligns that pin and
its evaluator-cache identity. The feature controls compiled module availability;
it does not remove this resolver prerequisite. Standard release workflows have
not been changed or certified for this experimental bridge.

Use Rust 1.98.1, Python 3.12, Git, a C/C++ toolchain, `m4`, `pkg-config` and
`sha256sum` for the validated native configuration. On macOS, install `coreutils`
to provide `sha256sum`. Create a Python environment and install
`maturin==1.15.0`, `pytest` and `marimo==0.24.2`, then run from this checkout:

```sh
bash scripts/prepare_fastsecdec_dependencies.sh /absolute/path/new-dependencies
cargo --config /absolute/path/new-dependencies/overlay-community-git.toml \
  metadata --locked --features experimental-fastsecdec --format-version 1 > /tmp/community-metadata.json
python .github/scripts/check_fastsecdec_dependencies.py /tmp/community-metadata.json native
CARGO_HOME=/absolute/path/new-dependencies/cargo-home \
  maturin develop --locked --features experimental-fastsecdec
pytest -q tests/test_hep_fastsecdec.py tests/test_hep_fastsecdec_inputs.py \
  tests/test_hep_wavefunctions.py
python -m marimo run examples/fastsecdec_showcase.py
```

The helper reads the Git revision from Cargo's manifest output. Its temporary
FastSecDec clone supplies bootstrap instructions and dependency patches;
`overlay-community-git.toml` deliberately contains no FastSecDec path override.
Cargo loads the bridge dependency directly from the published Git revision.
The generated config contains absolute paths only within the new output
directory. Keep it available for subsequent builds; do not commit it or copy it
over an existing Cargo config. For maturin and Pyodide, use the generated
command-scoped `cargo-home`: maturin 1.15 does not forward its `--config` option
to its metadata subprocess. This home starts with only the owner config and
uses separate caches and locks; it copies no credentials, global config or
shared-cache symlinks. The first build downloads its own Cargo dependencies.
The command-scoped environment reaches both metadata and compilation.

Provide the existing Symbolica license through your environment when required.
The tests accept `SYMBOLICA_LICENSE` or `SYMBOLICA_LICENSE_KEY`; the Pyodide
runners accept `SYMBOLICA_LICENSE_KEY`. No license belongs in source files,
exported notebook assets or the generated dependency config.

## Pyodide

Use the project's existing Pyodide toolchain setup: Python 3.14,
`pyodide-build==0.39.0`, `maturin==1.15.0`, Pyodide 314.0.7 and Rust 1.98.0 with
`wasm32-unknown-emscripten`. Pass the same explicit owner config into the existing
build script:

```sh
export WASM_FASTSECDEC=1
CARGO_HOME=/absolute/path/new-dependencies/cargo-home \
  bash scripts/build_wasm_performance.sh /absolute/path/new-wasm-wheel
export PYODIDE_DIST_DIR="$(pyodide config get dist_dir)"
node .github/scripts/test_fastsecdec_pyodide.mjs /absolute/path/new-wasm-wheel
```

The build script already runs the existing Pyodide community smoke test unless
`WASM_SKIP_TESTS=1`. The additional runner executes the same 39 bridge, input and
shared-wavefunction controls used by the native gate. The bridge is compiled
with FastSecDec's portable interpreted evaluator; native and portable features
must not be enabled together.

See [the showcase guide](FASTSECDEC_SHOWCASE.md) for the explicit asset export and
the separate browser lifecycle checks. The tested Pyodide 314.0.7 wheel also
passed the actual marimo 0.24.2 browser runtime, which uses Pyodide 314.0.0.
These results do not certify Windows builds, all browsers or convergence of all
four examples.
