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
`symbolica-integrate` 2.0, plus Idenso, Spenso, FeynKit HEP tools and Vakint.
The HEP extensions share the public owner revision pinned below and in
`Cargo.lock`. PyEmscripten wheels include Idenso, Spenso, HEP tools and numerical
loop evaluation, automatic boundary generation and transport with supplied
boundary values. Browser calculations use portable arithmetic on one worker.
Vakint and Spenso's compiled evaluators require a native installation.

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

Numerical loop transport is available in a separate namespace, using
the same HEPKit families, diagrams, kinematics, and Symbolica expressions:

```python
from symbolica.community.hep.integration import (
    KinematicTransport, BoundaryCache, EvaluationOptions,
)
```

The bindings live in the `symbolica-amflow` dependency and register in this
shared extension. They preserve arbitrary precision, supply typed errors and
cancellation, and retain reusable intermediate points in binary boundary caches.
Native and browser builds expose `IntegralEvaluator`, `PreparedIntegralFamily`
and `ReductionTables` for automatic evaluation and boundary generation. Check
`integration.automatic_boundary_generation_available` before offering those
operations in custom feature builds. Browser calls run synchronously: progress
can be read after a call returns and a pre-cancelled control stops the next
call. Another browser callback cannot interrupt an active synchronous call.
Cache files live in Pyodide's virtual filesystem; download them for persistence
across browser sessions.
The [automatic browser evaluation status](reports/2026-10-07-automatic-integrals-wasm/README.md)
records the complete Pyodide gate, including automatic evaluation and verified
30-digit boundary generation.
The [gg → Hg Marimo notebook](examples/hep/gg_hg.py) shows the native transport,
cache, form-factor projection and coherent EW/HEFT amplitude API calls directly.
It loads supplied starting boundaries, then computes transport and the amplitude
live on one core. The earlier publication wheel passed
[complete empty-cache boundary and amplitude acceptance](https://github.com/alphal00p/RustFlow/blob/92cfc9d2babfd95af8205b1193b4ec303bf8b610/reports/validation/2026-10-06-gg-hg-publication-complete/summary.json);
that report identifies its tested runtime. Full boundary regeneration remains
in the separate acceptance runner. Comparison-only references never serve as an implicit
evaluation fallback.
Run it with `marimo edit examples/hep/gg_hg.py`. Long numerical acceptance lives
in `examples/hep/gg_hg_acceptance.py`, separately from lightweight smoke tests.

The checked-in manifest and lock fetch the shared dependencies from public Git
sources. Build this checkout with the locked native installation command below;
no sibling checkout, local path override or manual owner patch is required.
The host owns the complete shared dependency graph, as described in the
[integration dependency guide](https://github.com/alphal00p/RustFlow/blob/b3a4843e8327835d1ec3ada5a6f32f1841bab2c2/docs/dependency-embedding.md).
Generate these hints with `stub_gen --hepkit-only`; the public package and stubs
are under `python/symbolica/community/hep/integration/`. Browser builds expose
automatic evaluation, supplied-boundary solvers, transport, cache and amplitude APIs. Existing
`hepkit` and Hyperbolica imports are preserved. The
[actual Pyodide smoke report](reports/2026-10-06-browser-loop-transport/report.json)
records exact restart, nearby reuse and cancellation on one core without license
credentials. The [complete Pyodide calculation](reports/2026-10-06-pyodide-gg-hg/README.md)
and [visible notebook gate](reports/2026-10-06-visible-higgs-api/README.md) cover
the scientific outputs and actual Chromium rendering separately.

The Git dependency selects RustFlow
[`b3a4843e8327835d1ec3ada5a6f32f1841bab2c2`](https://github.com/alphal00p/RustFlow/commit/b3a4843e8327835d1ec3ada5a6f32f1841bab2c2).
HEPKit, Linnet, Spenso, Idenso and rendering share the official GammaLoop
`feynkit` branch at
[`69a6b97e6cd81ecba4f2000d68a4ca97245dab6c`](https://github.com/alphal00p/gammaloop/commit/69a6b97e6cd81ecba4f2000d68a4ca97245dab6c).
Vakint retains its separate implementation at
[`6203c6cbba6ae5e90329ba5081fad55319e678db`](https://github.com/ValentinHirschi/gammaloop/commit/6203c6cbba6ae5e90329ba5081fad55319e678db).
Hyperbolica is pinned to
[`31292085504b794dc444a006eba1d3013ed30944`](https://github.com/benruijl/hyperbolica/commit/31292085504b794dc444a006eba1d3013ed30944).
RustRed uses official main
[`f5237ce1c8725b9cb0e18bade464806df607806f`](https://github.com/alphal00p/rustred/commit/f5237ce1c8725b9cb0e18bade464806df607806f),
with `campaign-api` enabled and experimental reconstruction disabled.
These revisions keep over-aligned tensor, diagram-group and exact-coefficient
payloads behind Rust-owned pointers at the WASM Python allocation boundary.

Symbolica uses the native evaluator-composition prerequisite at
[`1deccb8538ccb91dc2c1e58fc0a2e900d2276bf4`](https://github.com/ValentinHirschi/symbolica/commit/1deccb8538ccb91dc2c1e58fc0a2e900d2276bf4),
published in [Symbolica PR #54](https://github.com/symbolica-dev/symbolica/pull/54)
against `community`. That prerequisite remains a separate open PR. It lets
FastSecDec compose numerical sector maps and endpoint dual jets through the
native evaluator owner. Numerica and Graphica retain official community commit
[`ed2374f1d880d52c3a7ca48cd7c22f4baad5c020`](https://github.com/symbolica-dev/symbolica/commit/ed2374f1d880d52c3a7ca48cd7c22f4baad5c020),
including the negative-integer serialization correction. `Cargo.lock` selects
one owner of each crate; the newer FeynKit and numerical dependencies are preserved.
FastSecDec is pinned to
[`dea66bfd116356092d3a68f19fdb77d2a6f55b25`](https://github.com/alphal00p/fastSecDec/commit/dea66bfd116356092d3a68f19fdb77d2a6f55b25).
It exposes symbolic and numerical-dual generation, retained eager sessions,
Taylor/IBP subtraction and measured shared-formula preparation. The
[current ggHH notebook](examples/hep/gghh_complete.py) defaults to symbolic
generation with native `to_dots` simplification and generation-time gluon mass
shells. The [original triple box](https://github.com/alphal00p/fastSecDec/tree/dea66bfd116356092d3a68f19fdb77d2a6f55b25/examples/gghh_triple_box)
and [outer-loop triple box](https://github.com/alphal00p/fastSecDec/tree/dea66bfd116356092d3a68f19fdb77d2a6f55b25/examples/gghh_triple_box_bis)
are maintained in FastSecDec; publishing their cards does not claim completed
three-loop generation or integration.
The native graph wheel used in CI is built from the same HEPKit owner selected
by this manifest and lock; it is a separate test dependency requiring Python
3.10 or newer. The community package retains its Python 3.9 minimum.

Changing these dependencies changes the numerical cache source identity. Earlier
snapshots and benchmark or notebook acceptance reports retain their original
runtime provenance; their acceptance does not certify this updated wheel.
Fresh numerical work must use the updated runtime's own cache identity.

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
share Symbolica 3.0.1 through the kernel patch described above. The reducer includes
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

on WASM with

```python
import micropip
await micropip.install("symbolica")
```

or can be manually built using `maturin`. Community builds include
`hepkit.sector_decomposition`:

```bash
RUSTFLOW_WORKSPACE_FEATURES=python_stubgen RUSTFLOW_WORKSPACE_NO_DEFAULT_FEATURES=1 \
  cargo run --locked --features python_stubgen --no-default-features --bin stub_gen
RUSTFLOW_WORKSPACE_FEATURES=pyo3/extension-module RUSTFLOW_WORKSPACE_NO_DEFAULT_FEATURES=0 \
  maturin build --release --locked
```

The native build forwards the `pyo3/extension-module` feature selected by
`pyproject.toml` so numerical-cache fingerprints describe the actual host
dependency graph. Stub generation uses its separate feature selection.

Release builds use ThinLTO, one code generation unit, and symbol stripping.
Community bindings, RustRed, symbolic integration rules, and Typst rendering use Rust's size-oriented
`opt-level = "s"` (`-Os`); the numerical kernel and other physics engines retain
`-O3`. PyPI builds use Deflate level 9 and check that every uploaded distribution
is smaller than 100,000,000 bytes. Check local artifacts with
`python scripts/check_distribution_size.py dist/*.whl dist/*.tar.gz`.


For a browser build, use `--no-default-features --features wasm` (the
`scripts/build_wasm_performance.sh` default). The explicit `wasm-core` feature
keeps only the Symbolica kernel and integration, without community modules.

For a faster functional WASM build, select Cargo's development profile and
omit debug sections:

```bash
CARGO_PROFILE_DEV_DEBUG=0 WASM_RUST_PROFILE=dev WASM_OPT_LEVEL=-O0 \
  bash scripts/build_wasm_performance.sh dist/wasm-dev
```

Use a fresh output directory. The script also sets the development link
optimization to `-O0`; release profiles retain `-O3`. Development wheels are
for correctness and compatibility checks, with no release-performance or
download-size claims. Set `WASM_SKIP_TESTS=1` only when running the Pyodide
acceptance gate separately against the resulting wheel.

## For developers

### Adding extensions

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
regardless of `IntegrationOptions.parallel`. RustRed's exact IBP API is also
included in the `wasm` Community build: `hepkit.IBPFamily` and
`hepkit.rustred` share the browser's Symbolica kernel. Generation executes
synchronously with one worker; `hepkit.rustred.execution_capabilities()` reports
that live polling and in-flight cancellation are unavailable. The
[three-loop notebook](examples/hep/three_loop_reduction.py) generates its own
rules, certifies them, and reduces to their finite terminal basis in this mode.
See the [browser export instructions](examples/hep/README.md#rustred-in-pyodide).
The Pyodide gate defaults to the full Community suite; `--rustred-only` retains
Symbolica, HEPKit graph/tensor/layout, exact RustRed and WASM-export checks while
skipping the separate loop-transport and integration-contract gates. Its receipt
states the selected scope. The current development-profile full-suite run passes
RustRed and its native-artifact canary but aborts later in RustFlow's
`KinematicTransport.add_boundary` due to a separate PyO3 alignment issue;
focused RustRed validation is not a full-Community debug-build success claim.
See `examples/hep_integration.py` for explicit integration of HEPkit Symanzik
polynomials with stated normalization and projective gauge.

Integration types are generated from Hyperbolica's binding metadata by
`stub_gen --hepkit-only`; no separate Symbolica expression class is declared.
The standalone Hyperbolica wheel has been retired. Import expression
constructors from `symbolica` and catch `integration.IntegrationError`.
