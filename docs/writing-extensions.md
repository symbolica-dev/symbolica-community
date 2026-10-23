# Writing a Symbolica Community extension

This guide is for agents implementing a new extension that consumes hepkit and
other Community components. Develop the algorithms, PyO3 bindings, Python
classes, tests, and packaging in the extension's **own repository**. Release and
test that repository independently; submit a small integration PR to Symbolica
Community when it is ready. Routine extension development should not require
editing Community first.

## Architecture: one library, two hosts

Expose your extension as a Rust library with a reusable registration entry point.
Use that same library in two builds:

| Build | Native host | Python imports |
| --- | --- | --- |
| Standalone development | Your repository builds one extension containing your bindings and the upstream components they need. | `from my_extension import *` |
| Community integration | Community links your library and the other components into its existing `symbolica.core` extension. | Imports from `symbolica`, `symbolica.community.hepkit`, and your Community module. |

Both builds must use the **same upstream Rust/PyO3 types** in arguments and return
values. A hepkit diagram should remain a hepkit diagram when it passes through
your API, and a tensor or Symbolica expression should remain its original type.
Do not recreate those classes, wrap them in lookalike classes, or make JSON and
strings the normal interoperability boundary.

Re-exporting a class makes another name for that class in the current native
build. It does not create an ABI for exchanging Rust-backed objects between
independently compiled Python extensions. Build all participating components into
one native library so they share Python class identities and Symbolica's kernel
state. In particular, do not test by importing hepkit from an installed Community
wheel and passing its objects into a separately compiled standalone wheel.

## Start from the actual component APIs

Inspect the intended Community revision's [manifest](../Cargo.toml),
[lockfile](../Cargo.lock), [module registration](../src/lib.rs), and
[hepkit registration](../src/hepkit.rs). Consult its generated Python stubs and
the corresponding dependency sources before writing bindings. `hepkit` is the
public Python namespace; its base bindings come from the `feynkit-py` crate.
Community adds further components to that namespace.

Examples of existing types are:

| Python type | Rust type to reuse |
| --- | --- |
| `symbolica.Expression` | `symbolica::api::python::PythonExpression` |
| `hepkit.Model` | `feynkit_py::PyModel` |
| `hepkit.FeynmanDiagram` | `feynkit_py::PyFeynmanDiagram` |
| `hepkit.IntegralFamily` | `feynkit_py::PyIntegralFamily` |
| `hepkit.Kinematics` | `feynkit_py::PyKinematics` |
| `tensor.TensorExpression` | `spynso3::expression::TensorExpression` |
| `tensor.Tensor` | `spynso3::Spensor` |
| `graph.RenderSettings` | `linnet_render_py::PyRenderSettings` |

For example, accept `PyRef<'_, PyFeynmanDiagram>` for a borrowed input, or
`Py<PyFeynmanDiagram>` when retaining the Python object. Return an upstream
wrapper or `Py<UpstreamType>` as appropriate. `Py<T>` can be stored inside a
`#[pyclass]`; `clone_ref(py)` retains the same Python object rather than copying
the underlying physics data. See the [PyO3 ownership reference](https://docs.rs/pyo3/0.28.3/pyo3/struct.Py.html).

Keep computation in ordinary Rust APIs where practical. A method exposed by
`#[pymethods]` is not necessarily public to Rust callers: inspect the upstream
wrapper's public accessors and underlying crate before trying to call it directly.
If an essential accessor is missing, contribute it to the component's repository.
For example, `PyFeynmanDiagram::as_diagram()`, `PyModel::as_model()`, and
`PyIntegralFamily::as_family()` expose their native values. Preserve their
validation: `as_diagram()` rejects incomplete selected regions. Prefer direct
Rust constructors and helper calls in new bindings; avoid hardcoded Python
imports to access another component when its Rust API can provide the operation.

## Repository and dependencies

A small extension can keep its Rust library and standalone host in one crate:

```text
my-extension/
  Cargo.toml
  Cargo.lock
  pyproject.toml
  src/lib.rs          # Algorithms, PyO3 API, CommunityModule
  src/standalone.rs   # Optional standalone Python entry point
  tests/             # Rust and Python behavior tests
```

For a larger project, separate the algorithms, bindings, and standalone host into
workspace crates in the same repository. The bindings must remain linkable as an
`rlib`; gate the standalone `#[pymodule]` so Community can consume the library
without also enabling a second host.

This is a manifest starting point for the examples below. **Complete the root
patch tables and resolve the lockfile against the Community revision you target
before building.** The dependency versions shown match the inspected API and
are not a promise of compatibility with every future Community release.

```toml
[package]
name = "my-extension"
version = "0.1.0"
edition = "2024"

[lib]
name = "my_extension"
crate-type = ["rlib", "cdylib"]

[features]
default = []
standalone = []
native = ["symbolica/integer-gmp", "symbolica/float-mpfr", "symbolica/native_code_generation", "feynkit-py/native", "spynso3/native", "linnet-render-py/native"]

[dependencies]
pyo3 = { version = "0.28", features = ["multiple-pymethods"] }
symbolica = { version = "3.0.1", default-features = false, features = ["python_export"] }
feynkit-py = { git = "https://github.com/alphal00p/gammaloop", branch = "feynkit", default-features = false, features = ["ufo"] }
spynso3 = { git = "https://github.com/alphal00p/gammaloop", branch = "feynkit", default-features = false }
linnet-render-py = { git = "https://github.com/alphal00p/gammaloop", branch = "feynkit", default-features = false }
```

Use `https://github.com/alphal00p/gammaloop` with `branch = "feynkit"` for
Feynkit and companion workspace crates, including applicable registry patches.
Commit their exact resolved revision in `Cargo.lock` and keep them on one source
and revision. Vakint has a separately maintained pin; do not force it onto that
Feynkit revision.

Align Symbolica, Numerica, Graphica, PyO3, and any other shared types with the
target host. Copy the relevant root patches from the target Community manifest
instead of independently choosing versions. Cargo reads patches from the
workspace root, so a dependency's own patches will not configure your standalone
build. Likewise, your library's lockfile does not control Community's resolution.
See [Cargo's patch rules](https://doc.rust-lang.org/cargo/reference/overriding-dependencies.html).

Check the resolved graph, including enabled features:

```sh
cargo tree --locked --no-default-features --features native -d
cargo tree --locked --no-default-features --features native -i symbolica
cargo tree --locked --no-default-features --features native -i feynkit-py
cargo tree --locked --no-default-features --features native -e features -i pyo3
```

Resolve duplicate owners of shared types. Identical version numbers from
different Git sources are not sufficient. Use local path overrides only for
development and keep machine-specific paths out of committed files.

If integration requires a Feynkit update, follow [AGENTS.md](../AGENTS.md): fetch
`origin/feynkit`, consolidate needed fixes onto that latest upstream head with
JJ, resolve conflicts, check and push the `feynkit` bookmark to
`alphal00p/gammaloop`, then update Community's lockfile and verify its packaged
integration. Remove obsolete fork patches. Do not publish a dependency on a
temporary branch or fork to bypass this step.

## Define your API with PyO3

The following deliberately small `src/lib.rs` demonstrates a new Python struct
that takes **both** a hepkit diagram and a tensor component and returns those
same upstream types. Replace this holder with your algorithm's useful API; it
does not implement a new physics calculation.

```rust
use feynkit_py::PyFeynmanDiagram;
use pyo3::prelude::*;
use spynso3::expression::TensorExpression;
use symbolica::api::python::SymbolicaCommunityModule;

#[pyclass(module = "symbolica.community.my_extension")]
pub struct DiagramInput {
    diagram: Py<PyFeynmanDiagram>,
    numerator: Py<TensorExpression>,
}

#[pymethods]
impl DiagramInput {
    #[new]
    fn new(diagram: Py<PyFeynmanDiagram>, numerator: Py<TensorExpression>) -> Self {
        Self { diagram, numerator }
    }

    #[getter]
    fn diagram(&self, py: Python<'_>) -> Py<PyFeynmanDiagram> {
        self.diagram.clone_ref(py)
    }

    #[getter]
    fn numerator(&self, py: Python<'_>) -> Py<TensorExpression> {
        self.numerator.clone_ref(py)
    }
}

pub struct CommunityModule;

impl SymbolicaCommunityModule for CommunityModule {
    fn get_name() -> String {
        "my_extension".into()
    }

    fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<DiagramInput>()?;
        Ok(())
    }
}

#[cfg(feature = "standalone")]
mod standalone;
```

Use your intended final Community namespace in `#[pyclass(module = ...)]` and
stub metadata. Re-exporting upstream types must preserve their original
`__module__` and identity. This metadata alone does not make a Python import
path exist; the host supplies the packages and module registration.

`register_module` adds only your classes and functions. It must not initialize
Symbolica symbols or recreate dependency modules. Put symbol registration and
other runtime setup in the trait's `initialize(py)` hook; make repeated
initialization safe. Implement `get_citations()` when your methods require
scientific attribution. Return meaningful Python exceptions for invalid input,
and document whether outputs alias, copy, or mutate their inputs.

## Build a standalone host and re-export its inputs

In `src/standalone.rs`, register the dependencies and your API into the **same
native library**. Separate internal modules avoid accidental name collisions.
Re-export the public names needed by your tests, including upstream classes and
constructors. An explicit `__all__` keeps star imports predictable.

```rust
use pyo3::prelude::*;
use symbolica::api::python::{create_symbolica_module, SymbolicaCommunityModule};

#[pymodule]
fn my_extension(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let core = PyModule::new(py, "_core")?;
    create_symbolica_module(&core)?;
    let graph = PyModule::new(py, "_graph")?;
    linnet_render_py::GraphModule::register_module(&graph)?;
    let tensor = PyModule::new(py, "_tensor")?;
    spynso3::SpensoModule::register_module(&tensor)?;
    let hepkit = PyModule::new(py, "_hepkit")?;
    feynkit_py::FeynkitModule::register_module(&hepkit)?;
    crate::CommunityModule::register_module(m)?;

    for name in ["E", "S", "Expression"] {
        m.add(name, core.getattr(name)?)?;
    }
    for name in ["Model", "FeynmanDiagram"] {
        m.add(name, hepkit.getattr(name)?)?;
    }
    m.add("TensorExpression", tensor.getattr("TensorExpression")?)?;
    m.add("RenderSettings", graph.getattr("RenderSettings")?)?;
    for child in [&core, &graph, &tensor, &hepkit] {
        m.add_submodule(child)?;
    }

    linnet_render_py::GraphModule::initialize(py)?;
    spynso3::SpensoModule::initialize(py)?;
    feynkit_py::FeynkitModule::initialize(py)?;
    crate::CommunityModule::initialize(py)?;
    m.add(
        "__all__",
        ["DiagramInput", "E", "S", "Expression", "Model",
         "FeynmanDiagram", "TensorExpression", "RenderSettings"],
    )?;
    Ok(())
}
```

This is a minimal host for the example below, using the base Feynkit, tensor, and
graph bindings. It does not install canonical `symbolica.*` packages or reproduce
every feature Community adds to hepkit: IBP, one-loop reduction, numerical
integration, sector decomposition, and Vakint have additional crates and setup.
If your API depends on them, include their bindings and initialization too.

Some existing **Rust PyO3 binding helpers** call back through Python. In the
inspected Feynkit revision, `LoopMomentumBasis.momentum_replacements()` uses
`py.import("symbolica.core")` to obtain `Replacement`; rich displays import
`symbolica.community.tensor` to call formatting functions. These are imports
issued by the bindings, not a requirement of the native physics algorithms.
For those operations, supply the canonical facades or module aliases and parent
package attributes, all backed by this same native build. The upstream
[combined-host test](https://github.com/alphal00p/gammaloop/blob/5acf8481fd08d68fb3071656b2e59f559906eaf9/crates/feynkit-py/tests/combined_community.rs)
demonstrates that setup. Replacing these lookups with public Rust helpers upstream
would remove that import-path dependency.

Include required Python helper files and data setup for the features you use.
Do not satisfy an internal `import symbolica` by loading a second installed native
kernel. The minimal host also does not install Community's symbolic integration
callbacks. Test every feature you actually use, including serialization or
notebook display if you promise support for them.

For this single-crate layout, use a `pyproject.toml` such as:

```toml
[build-system]
requires = ["maturin>=1.13.2,<2.0"]
build-backend = "maturin"

[project]
name = "my-extension"
version = "0.1.0"
requires-python = ">=3.9"

[tool.maturin]
module-name = "my_extension"
features = ["standalone", "native"]
```

Build and test in a private environment after preparing your lockfile:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install 'maturin>=1.13.2,<2.0' pytest
cargo test --locked --no-default-features --features native
maturin develop --locked
python -m pytest tests
```

Modern maturin handles PyO3's extension linking mode. Keep that mode out of
ordinary Rust test builds; it differs from linking a Rust test executable to
Python. Enable an `abi3` feature only if you intend to support and test that ABI.
See [PyO3's build guide](https://pyo3.rs/v0.28.0/building-and-distribution.html).

## Keep test bodies and user code unchanged

During standalone development, import the upstream types and your API together:

```python
from my_extension import *
```

After Community integration, replace only that import block:

```python
from symbolica import E, S, Expression
from symbolica.community.hepkit import Model, FeynmanDiagram
from symbolica.community.tensor import TensorExpression
from symbolica.community.graph import RenderSettings
from symbolica.community.my_extension import DiagramInput
```

The rest of a test or user example stays the same. For the binding example above:

```python
def test_shared_inputs_and_outputs():
    model = Model.phi3()
    diagram = FeynmanDiagram.from_dot(model, '''digraph bubble {
        ext [style=invis];
        ext -> a [particle="phi"];
        a -> b [particle="phi", lmb_id=0];
        a -> b [particle="phi"];
        b -> ext [particle="phi"];
    }''')
    numerator = diagram.numerator_expression()
    value = DiagramInput(diagram, numerator)

    assert isinstance(value.diagram, FeynmanDiagram)
    assert isinstance(value.numerator, TensorExpression)
    assert value.diagram is diagram
    assert value.numerator is numerator
    assert isinstance(value.numerator.to_expression(), Expression)
    assert value.numerator.to_expression() == E("1")

    # Feed an output into another hepkit operation.
    restored = FeynmanDiagram.from_dot(model, value.diagram.to_dot())
    assert restored.loop_count == 1
```

For CI, put the selected import block in one test helper and select it per build;
run the same assertions in separate standalone and Community processes. Do not
use a catch-all `ImportError` fallback that could silently test the wrong host.
Add real algorithm checks, invalid-input cases, and interoperability tests that
feed outputs into downstream components. Preserve all assertions when migrating
the tests. Identity assertions apply to this holder example; a transforming
algorithm should instead check its documented result semantics.

## Release independently, then integrate

Before release, generate and check `.pyi` stubs alongside the bindings, including
upstream input and return types and the standalone re-exports. Keep the optional
`python_stubgen` feature aligned with the component crates and avoid enabling
Community's `module`/ABI mode for stub generation. Test a built wheel in a fresh
environment as well as the development install. State the supported Python,
platform, and Community/dependency revisions; only claim Pyodide support if its
feature combination and wheel have been tested.

Publish the extension's source release and, if desired, its standalone wheel from
its own repository. Its test host can operate before Community includes it.
The standalone wheel is not a dynamically discovered Community plugin.

Once ready, submit a Community PR containing the integration work:

1. Add your Rust library dependency with a reviewed release pin and update
   `Cargo.lock`, preserving one source for every shared type.
2. Connect its features to the appropriate Community features. Add its
   `CommunityModule` to [native registration](../src/lib.rs), or register it in
   [hepkit](../src/hepkit.rs) if that is the agreed public namespace. Include the
   initialization and citation hooks, not just the class registration.
3. Add the Python facade, for example
   `python/symbolica/community/my_extension/__init__.py`, following the existing
   facades: import the native exports, call `initialize_module()`, and remove
   that helper from the public namespace. Wire in stub generation and docs.
4. Build the complete Community extension in a private environment. Run your
   unchanged behavioral tests with the Community imports, relevant existing
   tests, and stub/type checks. Verify a packaged wheel, not only a source-tree
   import. Do not copy individual native extensions into a shared installation.
5. Follow the [contribution guidelines](../Contributing.md). Include the upstream
   release, dependency alignment, supported platforms, and verification results
   in the PR. Keep ownership and maintenance of the extension in its own repo.

For work in this repository, follow [AGENTS.md](../AGENTS.md): use an isolated JJ
workspace, establish the responsible person's name and email before committing,
and set the commit author explicitly with JJ. Preserve other workspaces and
running notebooks. Use existing license configuration without committing or
printing keys. Publishing source and deploying a shared notebook server are
separate actions.
