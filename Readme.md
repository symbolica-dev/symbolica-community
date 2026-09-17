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
from symbolica.community.hep import FeynmanDiagram, Model, Generator, TensorReducer
```

See the [HEP example](examples/hep/README.md) for a complete one-loop calculation.

#### Installation 

This package can be installed for Python 3.7 or newer using `pip`:

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
Native builds retain Symbolica's default features, including GMP and MPFR.
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
