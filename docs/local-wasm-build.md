# Building the WASM wheel locally

WASM wheels are built locally rather than in GitHub Actions. The commands below
build either the core package or the complete Community package for Pyodide
314.x (Python 3.14), including the matching resource wheels used by validation.

## Prerequisites

Use a Linux machine with a native C compiler, an archiver, Rustup, uv, and Node.js
24 or newer available on `PATH`. A release build needs substantial memory and
disk space; reduce `CARGO_BUILD_JOBS` if several compiler processes exhaust RAM.

From the repository root, create a Python 3.14 environment and install the
cross-compilation tools:

```bash
uv venv --python 3.14 .venv
source .venv/bin/activate
uv pip install pyodide-build==0.39.0 maturin==1.15.0

export PYODIDE_VERSION=314.0.7
export PYODIDE_XBUILDENV_PATH="$HOME/.cache/symbolica-community/pyodide-xbuildenv"
export RUSTUP_TOOLCHAIN=1.98.0

rustup toolchain install "$RUSTUP_TOOLCHAIN" --profile minimal \
  --target wasm32-unknown-emscripten
pyodide xbuildenv install "$PYODIDE_VERSION"
```

Use an existing Python 3.14 environment instead if one is already configured.
The cross-build environment contains the matching Emscripten SDK and Pyodide
runtime. Rust 1.98 is used because Pyodide's default Rust version predates this
package's minimum Rust version.

## Build and validate

Choose a fresh output directory for each build. The build script captures the
compiler output there and rejects a directory that already has that capture.
Keep `WASM_OUTDIR` set for the remaining commands.

```bash
WASM_OUTDIR="dist/wasm-local-$(date +%Y%m%d-%H%M%S)"
export CARGO_BUILD_JOBS=2

WASM_RUST_PROFILE=release-performance \
WASM_FEATURES=wasm \
WASM_OPT_LEVEL=-O0 \
WASM_OPT_DEDUPLICATE_FUNCTIONS=1 \
WASM_SKIP_TESTS=1 \
  bash scripts/build_wasm_performance.sh "$WASM_OUTDIR"

python scripts/build_data_bundle.py --out "$WASM_OUTDIR" --check-catalog
```

This uses the committed lockfile, Rust optimization level 3 with ThinLTO,
minimal Binaryen processing, and duplicate-function elimination. Tests are
deferred until the resource wheels and final compression are ready.

For a core-only build, use a separate fresh output directory and set
`WASM_FEATURES=wasm-core` in the build command. This compiles the kernel and
symbolic integration without Community extensions, then removes Community
Python packages and examples from the wheel and regenerates its `RECORD`.
Core stubs, the resource loader, and resource requirements are preserved.
The two variants have the same package name, version, and wheel filename;
keep them in separate directories and install one variant per interpreter.

Preserve the Deflate wheel before recompressing the direct-download copy:

```bash
mkdir -p "$WASM_OUTDIR/pypi"
cp "$WASM_OUTDIR"/*-pyemscripten_2026_0_wasm32.whl "$WASM_OUTDIR/pypi/"

python scripts/compress_wheel_zstd.py \
  "$WASM_OUTDIR"/*-pyemscripten_2026_0_wasm32.whl

export PYODIDE_DIST_DIR
PYODIDE_DIST_DIR=$(pyodide config get dist_dir)
node .github/scripts/test_pyodide.mjs "$WASM_OUTDIR"

python scripts/check_distribution_size.py "$WASM_OUTDIR"/*.whl
```

The validation installs the final Zstd wheel and both data wheels with
`micropip`, then exercises algebra, integration, the Community modules, and the
WASM export list. If configured, it uses `SYMBOLICA_LICENSE_KEY` or
`SYMBOLICA_LICENSE` from the environment. Validation receipts are written into
the output directory.

When validating a core-only wheel, prefix each Node command with
`SYMBOLICA_EXPECT_COMMUNITY=0`. The core checks exercise algebra and symbolic
integration and require Community packages to be absent from both the archive
and the installed interpreter. Build the resource wheels in the core output
directory too, as above; the validation runner uses the same resource setup.

To validate the preserved Deflate wheel too, copy the resource wheels beside it
and run the same suite:

```bash
cp "$WASM_OUTDIR"/symbolica_fonts-*.whl \
  "$WASM_OUTDIR"/symbolica_vakint_data-*.whl "$WASM_OUTDIR/pypi/"
node .github/scripts/test_pyodide.mjs "$WASM_OUTDIR/pypi"
```

## Distribution

The top-level WASM wheel uses **ZIP Zstandard level 22** and can be hosted for
direct installation in the matching Pyodide runtime. Keep the matching data
wheels available to users as well.

The copy in `pypi/` uses **Deflate**. PyPI's
[upload validator](https://github.com/pypi/warehouse/blob/main/warehouse/forklift/legacy.py)
accepts stored and Deflate ZIP entries and rejects Zstandard entries, regardless
of the upload client. The native release CI does not upload these local WASM
artifacts.
