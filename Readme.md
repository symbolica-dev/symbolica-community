<h1 align="center">
  <br>
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://symbolica.io/logo_dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://symbolica.io/logo.svg">
  <img src="https://symbolica.io/logo.svg" alt="logo" width="200">
</picture>
  <br>
</h1>

<p align="center">
<a href="https://symbolica.io"><img alt="Symbolica website" src="https://img.shields.io/static/v1?label=symbolica&message=website&color=orange&style=flat-square"></a>
  <a href="https://reform.zulipchat.com"><img alt="Zulip Chat" src="https://img.shields.io/static/v1?label=zulip&message=discussions&color=blue&style=flat-square"></a>
    <a href="https://github.com/benruijl/symbolica_community"><img alt="Symbolica website" src="https://img.shields.io/static/v1?label=github&message=development&color=green&style=flat-square&logo=github"></a>
</p>

# Community-enhanced Symbolica 

This repository contains the [Symbolica](https://github.com/benruijl/symbolica) library, bundled with additional community contributions.

Version 3.0 ships core Symbolica and symbolic integration via
`symbolica-integrate` 2.0, plus Idenso, Spenso, FeynKit HEP tools, Vakint, and the
example extension. The GammaLoop extensions track its `feynkit` branch, with
the tested revision pinned in `Cargo.lock`. PyEmscripten wheels include Idenso,
Spenso, HEP, and the example extension. Vakint and Spenso's compiled evaluators
require a native installation.

The integrator enables `compressed-step-metadata`, preserving integration steps
while storing their rule sources and descriptions in a Brotli-compressed catalog.


## Usage

To use core Symbolica features, simply write:
```python
from symbolica import *
```
See the [documentation](https://symbolica.io/docs) for further help.

FeynKit's diagram, generation, CFF, tensor-reduction, model, and kinematics
classes share one flat namespace:

```python
from symbolica.community.hepkit import FeynmanDiagram, Model, Generator, TensorReducer
```

See the [HEP example](examples/hep/README.md) for a complete one-loop calculation.

Native numerical loop integration is available in a separate namespace, using
the same HEPKit families, diagrams, kinematics, and Symbolica expressions:

```python
from symbolica.community.hep.integration import (
    IntegralEvaluator, KinematicTransport, BoundaryCache, EvaluationOptions,
)
```

The bindings live in the `symbolica-amflow` dependency and register in this
shared extension. They preserve arbitrary precision, supply typed errors and
cancellation, and retain reusable intermediate points in binary boundary caches.
The [gg → Hg Marimo notebook](examples/hep/gg_hg.py) stages native boundary
generation, physical transport, and coherent EW/HEFT amplitude assembly. Opening
it starts no two-loop evaluation. Its complete empty-cache boundary acceptance
is still pending; archived numerical seeds are never an evaluation fallback.
Run it with `marimo edit examples/hep/gg_hg.py`. Long numerical acceptance lives
in `examples/hep/gg_hg_acceptance.py`, separately from lightweight smoke tests.

Builds currently require the native owner patches and Cargo overrides documented
in [the integration dependency guide](https://github.com/alphal00p/RustFlow/blob/main/docs/dependency-embedding.md).
Generate these hints with `stub_gen --hepkit-only`; the public package and stubs
are under `python/symbolica/community/hep/integration/`. This module is excluded
from browser builds, and existing `hepkit` and Hyperbolica imports are preserved.

One-loop reduction from [one-loop-reduce](https://github.com/ecavan/one-loop-reduce)
is available in `hep.oneloop`. Native builds also expose OneLoopMaster's scalar
integral evaluation there, sharing the same Symbolica kernel:

```python
from symbolica.community.hepkit import FourMomentum, oneloop

p = FourMomentum(3.0, 1.0, 0.0, 0.0)
finite, pole, double_pole = oneloop.b0(p.mass_squared, 4.0, 4.0)
```

The uppercase exports `oneloop.A0`, `B0`, `dB0`, `C0`, and `D0` are primitive
Symbolica symbols. Construct a symbolic master directly, for example
`oneloop.B0(s, m0_squared, m1_squared, mu_squared)`. Lowercase `a0`, `b0`,
`db0`, `c0`, and `d0` numerically return all three Laurent coefficients and
accept the precision and backend options.

Tagged master calls containing only numeric arguments evaluate through the
native Rust hooks during construction when at least one argument is inexact,
for example `oneloop.A0(0, Float("2", decimal_digits=50), 1)` after importing
`Float` from Symbolica. Untagged calls and calls containing only exact arguments
remain symbolic; `Expression.evaluate` also evaluates tagged calls explicitly.

Reduce a family to OneLoopMaster's symbols, then evaluate its Laurent
coefficients through their registered native Rust hooks:

```python
from symbolica import E, S
from symbolica.community import hepkit as hep
from symbolica.community.hepkit import oneloop

k, D, mass_squared = S("example::k", "example::D", "example::mass_squared")
kinematics = hep.Kinematics(D, momenta=[k])
family = hep.IntegralFamily(
    [k], [], [kinematics.scalar_product(k, k) - mass_squared], kinematics=kinematics,
)
reduction = oneloop.reduce(family, [2]).simplify()
coefficients = oneloop.reduction_coefficients(reduction, mu_squared=E("1"))
print([c.evaluate({mass_squared: 2 + 0j}) for c in coefficients])  # [-log(2), 1, 0]
```

`oneloop.reduce` takes Feynkit's existing `hep.IntegralFamily` and one signed
power per ordered denominator. The same family supports `hep.IBPFamily`,
momentum mappings, completion, and diagram extraction. Zero powers omit a
denominator; negative powers contribute to the numerator. Optional scalar
numerators use the family's `Kinematics.scalar_product` notation. Feynkit's
model-backed `hep.Propagator` remains the shared particle-propagator type.

Use a symbolic kinematic dimension such as `D` for dimensional regularization.
`Reduction.dimension` retains that symbol. `reduction_coefficients` expands it
at `D = 4 - 2*eps` through the order needed for the finite term and
returns `[finite, simple_pole, double_pole]`. It rejects coefficients with a pole
at `D=4`, since those require positive-order master coefficients that
OneLoopMaster does not supply. The masses, squared invariants, and squared scale
must be independent of `D`. A family with fixed integer dimension is rejected
instead of losing epsilon-dependent contributions to the finite term.

`Reduction.to_expression(mu_squared=...)` uses the canonical
`oneloopmaster::{A0,B0,C0,D0}` heads, appends the squared scale (default `1`), and
retains exact dimension dependence. A single `master.to_expression()` from
`reduction.terms` feeds directly into `oneloop.master_coefficients()` for tagged
calls or `oneloop.get_expression()` for exact formulas. The leading tags
`0`, `-1`, and `-2` select Laurent coefficients for numerical evaluation;
untagged calls describe the whole master for symbolic inspection. No conversion
between master namespaces is needed.

The native hooks use OneLoopMaster's generated Rust arithmetic, without building
an expression evaluator or loading its evaluator caches. For inspecting an
analytical region, use `oneloop.select_branch(oneloop.get_expression(master),
replacement_rules)`: the probe selects conditional branches while preserving
symbolic kinematics. The [marimo example](examples/hep/oneloop_reduce.py) shows
this for a nontrivial triangle C0.

Master arguments use the AVH/OneLOop ordering. Invariants, masses, and momentum
shifts are extracted from the shared family's actual inverse denominators and
kinematics; users do not supply a second family or a separate invariant list.
Numerical master evaluation requires a native build.

The host fetches OneLoopMaster from `alphal00p/oneloopmaster` and the reducer
from the public `lcnbr/one-loop-reduce` fork, which CI can access. All modules
share Symbolica 3.0.1 through the kernel's community-branch patch. The reducer includes
the canonical master symbols, scale-aware expressions, and feature selection needed
by this host. Run
`.venv-feynkit/bin/python examples/oneloop_smoke.py` to check numeric evaluation,
SymJIT, arbitrary precision, and expressions shared with FeynKit. This example
requires NumPy for Symbolica's expression evaluator. After rebuilding the host,
restart any running Python or notebook kernel to load the updated extension.
Run `python examples/oneloop_reduce.py` for reduction followed by master
evaluation, and `python -m pytest tests/test_oneloop_reduce.py` for the integration
checks. To regenerate the combined one-loop type hints, run
`cargo run --no-default-features --features python_stubgen --bin stub_gen -- --oneloop-only`;
native evaluator declarations are maintained in `stubs/oneloop.pyi`.

The public Python packages and their type hints use matching directories under
`python/symbolica/community/tensor/` and `python/symbolica/community/hepkit/`,
including the `oneloop/`, `ibp/` and `vakint/` subpackages. Import them with
`from symbolica.community import tensor` and
`from symbolica.community.hepkit.vakint import Vakint`.
The older `community.spenso` and `community.vakint` paths re-export those APIs.
Regenerate tensor hints with
`cargo run --no-default-features --features python_stubgen --bin stub_gen -- --tensor-only`.
Use `--vakint-only` with the same command to regenerate the Vakint hints.
CI checks the packaged stub layout and type-checks these public imports.

`python/symbolica/core.pyi` retains the canonical Symbolica 3.0.1 declarations
from the kernel's `symbolica.pyi`, plus the community
`get_citations() -> list[Citation]` function. Community stub generation preserves
this file. To compare source stubs or wheels with the kernel stub, run
`python .github/scripts/check_core_stub.py --canonical /path/to/symbolica.pyi python/symbolica/core.pyi dist/*.whl`.

#### Installation 

This package can be installed for Python 3.9 or newer using `pip`:

```sh
pip install symbolica
```

or can be manually built using `maturin`:

```bash
cargo run --features "python_stubgen" --no-default-features # generate type hints
maturin build --release
```


## For developers

### Pyodide releases

The `PyPi wheel generation` workflow includes a Pyodide 314.0.7 build for
Python 3.14 (`pyemscripten_2026_0_wasm32`). It installs the wheel with `micropip`
and checks basic algebra and package contents before publishing it to the
same PyPI project. Run the workflow with `publish` disabled to test a
build without uploading it to PyPI.

The WebAssembly build uses `--no-default-features --features wasm`, selecting
Symbolica's Rust numeric backends and disabling native code generation.
Native builds use GMP, MPFR, and native code generation with the system allocator.
Symbolica's optional mimalloc allocator is disabled because it can crash when
Python imports the extension on a worker thread and later uses another thread.
Native releases and PyEmscripten releases have separate publication jobs.

The `release-small` Cargo profile optimizes for size (`opt-level = "z"`), uses
fat LTO and one codegen unit, and strips symbols. The PyEmscripten job uses this
profile and limits exports to `PyInit_core`, runtime helpers, and the inventory
registration globals for Symbolica and its extensions. CI checks this export list.
It retains Rust panic unwinding so PyO3 can turn panics into Python exceptions.

Install the published wheel in Pyodide 314.x (Python 3.14) with:

```python
import micropip
await micropip.install("symbolica")
```

To build the same wheel locally after setting up the toolchain from the workflow:

```sh
pyodide build . --no-isolation -C maturin.build-args="--locked --profile release-small --no-default-features --features wasm -- -C link-arg=-sEXPORTED_FUNCTIONS=_PyInit_core"
```

For a speed-oriented release, use `release-performance` (`opt-level = 3`, fat
LTO, one codegen unit, stripped symbols, and panic unwinding). Symbolica 3.0.1
requires Rust 1.96 or newer; Rust 1.98.0 was used with the Pyodide 314.0.7
cross-build environment and Emscripten 5.0.3:

```sh
rustup toolchain install 1.98.0 --profile minimal --target wasm32-unknown-emscripten
export RUSTUP_TOOLCHAIN=1.98.0
bash scripts/build_wasm_performance.sh
```

The script builds into `dist/wasm-performance`, preserves the linked input in
`compiler-output`, and runs the Pyodide smoke tests. It keeps LLVM's `-O3`
optimization and uses Binaryen `-O0` postprocessing as a baseline for `wasm-opt`
experiments. Set `WASM_OPT_LEVEL=-O3` (or `-O2`, `-Os`,
`-Oz`) to choose additional Binaryen optimization. These passes can take a long
time on this large module. This flag does not change Rust's optimization level.
To compare Rust size optimization, set `WASM_RUST_PROFILE=release-small` and pass
a separate output directory to the script. Existing compiler captures are never
overwritten. A private optimizer wrapper keeps Emscripten linker settings fixed
while selecting Binaryen optimization; it does not modify the installed SDK.
Install the resulting wheel by URL with `await micropip.install(wheel_url)` in
Pyodide 314.x or a marimo WebAssembly notebook using Python 3.14 and the
`pyemscripten_2026_0_wasm32` platform. Serve the wheel with CORS headers when it
is hosted on a different origin. `examples/pyodide/performance_marimo.py` is a
marimo example; export it with `marimo export html-wasm` and place the wheel
beside the exported `index.html`. Native-only Vakint, OneLOop, the RustRed IBP
bridge, and native code generation are unavailable in this build.

For browser delivery, a stored ZIP wheel compressed with HTTP Brotli can be
substantially smaller than a conventional deflated wheel. Prepare it with
`python scripts/prepare_browser_wheel.py WHEEL OUTPUT_DIRECTORY` (requires the
Python `brotli` package). Serve the normal `.whl` URL with the adjacent `.whl.br`
file as its body and `Content-Encoding: br`; install the `.whl` URL with
`micropip`. `scripts/serve_wasm_bundle.py DIRECTORY` provides a local test server.
The stored wheel is large without HTTP compression. This changes only delivery,
not the package contents or execution speed.

`scripts/measure_wheel_zstd.py STORED_WHEEL` (Python 3.14+) measures and saves
zstd levels 3, 9, 19, and 22 using an 8 MiB window for HTTP delivery. Serve a
chosen representation as `.whl.zst` beside the stored wheel; the test server
negotiates `Content-Encoding: zstd` or Brotli based on client support and size.
The HTTP window limit is specified in [RFC 9659](https://www.rfc-editor.org/rfc/rfc9659.html).

For a smaller download, `--levels 22 --window-log 27` permits a larger zstd window.
Serve that archive as ordinary `.zst` bytes without `Content-Encoding: zstd`.
`examples/pyodide/performance_zstd_marimo.py` downloads it, decodes it using
Python 3.14's built-in `compression.zstd`, and installs the wheel from Pyodide's
filesystem. This path needs no additional decoder package. It adds a one-time
decompression step while preserving the compiled code and execution speed.

For a local browser playground using the built wheel, see
[the Pyodide example](examples/pyodide/README.md).

### Adding extensions

These instructions apply once the extensions have been ported to Symbolica 3.0.

If you are developing a Python package that uses Symbolica, your users can simply `import symbolica`.
If you are developing a Rust crate, your crate can be added to `symbolica-community`, which allows you to write Python functions that use Symbolica classes and types, while sharing the same state/engine as the other included packages. The process is straightforward:

- Make sure your crate has a struct called `CommunityModule` that implements `SymbolicaCommunityModule`
- Create the folder `example` in `python/symbolica/community` and write a `__init__.py` that contains a description of your module
- Add your crate `example` to `Cargo.toml`:
  - Extend the feature list: `python_stubgen = ["symbolica/python_stubgen", "example/python_stubgen"]`
  - Extend the dependencies: `example = { git = "..." }`
- Register your crate as a submodule in `lib.rs` by extending the `core` function:
  - `register_extension::<example::CommunityModule>(m)?;`


## Note for macOS users using GNU gcc installed with MacPorts

Some ports of the `GNU gcc` compiler from MacPorts miss the `libgcc_s.1.dylib` library, which contains symbols required by the `mpfr` dependency of Symbolica. 

If you encounter an error like this:

```
Undefined symbols for architecture arm64:
  "___emutls_get_address", referenced from:
      _mpfr_check_range in libgmp_mpfr_sys-e988bd27f251f250.rlib[90](exceptions.o)
  [...]
```

Then try recompiling with the following rust flag:

```bash
RUSTFLAGS="-L/opt/local/lib/libgcc -l dylib=gcc_s"
```


### Exact definite integration

```python
from symbolica import S
from symbolica.community.hepkit import integration
x = S("x")
assert integration.integrate(1/(x+1)**2, [x]) == 1
```

This shared-kernel API is backed by Hyperbolica. `integrate` uses `[0,+Infinity)`
in the supplied variable order; `integrate_over` accepts directed intervals.
`prepare` reuses lowered inputs, and the detailed variants return expression
and algebraic-letter metadata. `Expression.integrate(x)` remains the separate
antiderivative API. Use `from symbolica.community.hepkit import ibp` for the
existing native IBP tools. Reduction and master evaluation remain separate.

The same expression API is available in Pyodide, where execution is serial
regardless of `IntegrationOptions.parallel`. IBP retains native-only availability.
See `examples/hep_integration.py` for explicit integration of HEPkit Symanzik
polynomials with stated normalization and projective gauge.

Integration types are generated from Hyperbolica's binding metadata by
`stub_gen --hepkit-only`; no separate Symbolica expression class is declared.
The standalone Hyperbolica wheel has been retired. Import expression
constructors from `symbolica` and catch `integration.IntegrationError`.
