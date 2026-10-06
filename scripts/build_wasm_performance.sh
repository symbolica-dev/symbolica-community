#!/usr/bin/env bash
# Requires pyodide-build 0.39.0, maturin 1.15.0, Python 3.14, and Rust >=1.96
# with wasm32-unknown-emscripten installed. Validated with Rust 1.98.0.
set -euo pipefail
cd "$(dirname "$0")/.."
outdir="${1:-dist/wasm-performance}"
# Keep Binaryen postprocessing minimal for an experiment baseline. The private
# wrapper captures LLVM's output before applying the requested Binaryen level.
wasm_opt_level="${WASM_OPT_LEVEL:--O0}"
rust_profile="${WASM_RUST_PROFILE:-release-performance}"
wasm_features="${WASM_FEATURES:-wasm}"
case "$wasm_features" in
  wasm|wasm-core) ;;
  *) echo "Invalid WASM_FEATURES: $wasm_features (expected wasm or wasm-core)" >&2; exit 2 ;;
esac
case "$wasm_opt_level" in
  -O0|-O1|-O2|-O3|-Os|-Oz) ;;
  *) echo "Invalid WASM_OPT_LEVEL: $wasm_opt_level" >&2; exit 2 ;;
esac
case "$rust_profile" in
  release-performance|release-small) ;;
  *) echo "Invalid WASM_RUST_PROFILE: $rust_profile" >&2; exit 2 ;;
esac
if [[ -e "$outdir/compiler-output" ]]; then
  echo "Choose a fresh output directory; compiler-output already exists in $outdir" >&2
  exit 2
fi
mkdir -p "$outdir"
outdir="$(cd "$outdir" && pwd)"
binaryen_sdk="$(pyodide config get emsdk_dir)/upstream"
capture_tools="$(mktemp -d)"
trap 'rm -rf "$capture_tools"' EXIT
mkdir "$capture_tools/bin"
for binary in "$binaryen_sdk"/bin/*; do
  [[ "${binary##*/}" == wasm-opt ]] && continue
  ln -s "$binary" "$capture_tools/bin/${binary##*/}"
done
cp scripts/capture_wasm_opt.py "$capture_tools/bin/wasm-opt"
chmod +x "$capture_tools/bin/wasm-opt"
export EM_BINARYEN_ROOT="$capture_tools"
export REAL_WASM_OPT="$binaryen_sdk/bin/wasm-opt"
export WASM_OPT_CAPTURE_DIR="$outdir/compiler-output"
export WASM_OPT_CAPTURE_INPUT_NAME=symbolica_community.wasm
export WASM_OPT_FORCE_LEVEL="$wasm_opt_level"
# Fingerprint the same host feature graph that maturin builds below. Native
# reducers remain outside this target; portable transport uses supplied values.
export RUSTFLOW_WORKSPACE_MANIFEST="$PWD/Cargo.toml"
export RUSTFLOW_WORKSPACE_FEATURES="$wasm_features,pyo3/extension-module"
export RUSTFLOW_WORKSPACE_NO_DEFAULT_FEATURES=1
# Pyodide sets CC/AR for target code. Cargo build dependencies (including the
# fingerprint hash) execute on the host and must retain native compilers.
export HOST_CC="${HOST_CC:-$(command -v cc)}"
export HOST_AR="${HOST_AR:-$(command -v ar)}"
# Use the same Emscripten linker settings at every Binaryen level; changing the
# linker's -O flag can also change its defaults outside wasm-opt.
# Bind internal references locally so ThinLTO does not expose Rust symbols
# through self-imports in the side module's global offset table.
pyodide build . --outdir "$outdir" --no-isolation \
  -C "maturin.build-args=--locked --profile $rust_profile --no-default-features --features $wasm_features -- -C link-arg=-sEXPORTED_FUNCTIONS=_PyInit_core -C link-arg=-Wl,-Bsymbolic -C link-arg=-O3"
export SYMBOLICA_EXPECT_COMMUNITY=1
if [[ "$wasm_features" == wasm-core ]]; then
  python scripts/prepare_core_wheel.py "$outdir"/*-pyemscripten_2026_0_wasm32.whl
  export SYMBOLICA_EXPECT_COMMUNITY=0
fi
# CI tests separately after recompressing the wheel for distribution.
if [[ "${WASM_SKIP_TESTS:-0}" != 1 ]]; then
  export PYODIDE_DIST_DIR
  PYODIDE_DIST_DIR="$(pyodide config get dist_dir)"
  node .github/scripts/test_pyodide.mjs "$outdir"
fi
