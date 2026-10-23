# Writing a Symbolica Community extension

Develop your extension's algorithms, Rust library, PyO3 classes and bindings,
tests, and packaging in **its own repository**. Test and release it independently,
then submit a small integration PR to Symbolica Community. Routine development
should not require changing Community first.

## Design and shared types

Name public APIs for their function: `sector_decompose`, rather than `pysecdec`.
Particle-physics APIs belong under `symbolica.community.hepkit`, optionally in a
submodule such as `hepkit.diagram_tools`. Reuse existing components: there should
be one graph generator, not a separate implementation in each extension. Submit
enhancements and missing Rust accessors to the component's owning repository.
Consider adding a method to an existing class before introducing a new wrapper.

Use the actual upstream PyO3 types for inputs and outputs:

| Python type | Rust type |
| --- | --- |
| `hepkit.FeynmanDiagram` | `feynkit_py::PyFeynmanDiagram` |
| `hepkit.Model` | `feynkit_py::PyModel` |
| `hepkit.IntegralFamily` | `feynkit_py::PyIntegralFamily` |
| `tensor.TensorExpression` | `spynso3::expression::TensorExpression` |
| `symbolica.Expression` | `symbolica::api::python::PythonExpression` |

Accept `PyRef<'_, T>` for borrowed Python inputs or `Py<T>` to retain an object.
Return the upstream wrapper or `Py<T>`; do not use lookalike classes or serialize
objects merely to pass them between components. Inspect the generated stubs and
public Rust accessors, such as `as_diagram()`, and preserve their validation.

Build all participating components into **one native library**, sharing Python
class identities and one Symbolica kernel. Your standalone host links your
library and its dependencies; Community later links that same library into
`symbolica.core`. Re-exports are aliases within that build, not an ABI between
separately compiled extensions. Do not mix an installed Community wheel's objects
with a separately compiled standalone wheel.

Align shared dependencies and features with Community's [manifest](../Cargo.toml)
and [lockfile](../Cargo.lock), including the workspace root's Cargo patches.
Source Feynkit and its companion crates from
`https://github.com/alphal00p/gammaloop`, `branch = "feynkit"`, at one locked
revision. Keep Symbolica, Numerica, Graphica and PyO3 consistent across components;
check the resolved graph with `cargo tree -d`.

Rust bindings must call other components' Rust APIs directly. Do not import
`symbolica.core`, `symbolica.community.tensor`, or other canonical Python modules
to find constructors or functions. Expose a missing Rust API upstream. Optional
downstream implementations can use the registration mechanism below.

## Define your library with PyO3

Keep bindings linkable as an `rlib`; a standalone host also builds a `cdylib`.
Gate its `#[pymodule]` behind a `standalone` feature. Enable the shared components'
Python bindings and PyO3's `multiple-pymethods` where required.

This minimal `src/lib.rs` accepts a HEPKit diagram and a tensor component and
returns those same objects. Replace the holder with your useful algorithm:

```rust
use feynkit_py::PyFeynmanDiagram;
use pyo3::prelude::*;
use spynso3::expression::TensorExpression;
use symbolica::api::python::SymbolicaCommunityModule;

#[pyclass(module = "symbolica.community.hepkit.diagram_tools")]
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
    fn get_name() -> String { "hepkit_diagram_tools".into() }
    fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<DiagramInput>()
    }
}

#[cfg(feature = "standalone")]
mod standalone;
```

`register_module` adds classes and functions. Put symbol setup and optional
implementation registration in `initialize(py)`; make repeated initialization
safe. Add `get_citations()` when appropriate. Document validation and whether
outputs alias, copy, or mutate their inputs.

The internal registration name above produces
`symbolica.community.hepkit_diagram_tools_native` in Community. A Python facade
provides the public `symbolica.community.hepkit.diagram_tools` package;
`get_name()` does not create nested packages. Preserve upstream classes'
`__module__` metadata when re-exporting them.

## Standalone host and unchanged tests

In `src/standalone.rs`, register dependencies in the same native host and re-export
the names your tests need. This example registers Symbolica directly on the host:

```rust
use pyo3::prelude::*;
use symbolica::api::python::{create_symbolica_module, SymbolicaCommunityModule};

#[pymodule]
fn my_extension(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    create_symbolica_module(m)?;
    let graph = PyModule::new(py, "_graph")?;
    linnet_render_py::GraphModule::register_module(&graph)?;
    let tensor = PyModule::new(py, "_tensor")?;
    spynso3::SpensoModule::register_module(&tensor)?;
    let hep = PyModule::new(py, "_hepkit")?;
    feynkit_py::FeynkitModule::register_module(&hep)?;
    crate::CommunityModule::register_module(m)?;
    for name in ["Model", "FeynmanDiagram"] {
        m.add(name, hep.getattr(name)?)?;
    }
    m.add("TensorExpression", tensor.getattr("TensorExpression")?)?;
    for child in [&graph, &tensor, &hep] { m.add_submodule(child)?; }
    linnet_render_py::GraphModule::initialize(py)?;
    spynso3::SpensoModule::initialize(py)?;
    feynkit_py::FeynkitModule::initialize(py)?;
    crate::CommunityModule::initialize(py)?;
    m.add("__all__", ["DiagramInput", "E", "S", "Expression",
        "Model", "FeynmanDiagram", "TensorExpression"])?;
    Ok(())
}
```

Configure maturin for this module and the required standalone/native features.
Build in a private environment. Include additional components, initialization,
Python helpers and data when your API needs them; the minimal host does not
install every Community feature. Test promised serialization and display support.

Standalone tests import their inputs and your API together:

```python
from my_extension import *
```

After Community integration, change only the import block:

```python
from symbolica import E, S, Expression
from symbolica.community.hepkit import Model, FeynmanDiagram
from symbolica.community.tensor import TensorExpression
from symbolica.community.hepkit.diagram_tools import DiagramInput
```

The test body stays the same:

```python
def test_shared_types():
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
    assert value.diagram is diagram
    assert value.numerator is numerator
    assert isinstance(value.diagram, FeynmanDiagram)
    assert isinstance(value.numerator, TensorExpression)
    assert isinstance(value.numerator.to_expression(), Expression)
```

Run the same assertions in separate standalone and Community processes. Add
algorithm, invalid-input and downstream interoperability checks. Exercise native
bindings with canonical `symbolica.*` imports blocked and aliases absent.

## Add methods through the class owner

The class owner defines the `#[pymethods]` forwarder, documentation and `.pyi`
signature. A backend registers its implementation behind that method; registration
does not install methods on another crate's class or generate stubs.

Symbolica's `ExtensionPoint`, `ExtensionImplementation`, `PythonExtensionFunction`
and `PYTHON_INTEGRATION` APIs merged into `community` in
[Symbolica PR #125](https://github.com/symbolica-dev/symbolica/pull/125). Enable
`python_export` and select a compatible revision or release containing that change.
A point stores a typed Rust function or function table, without a Python module
lookup or a reverse dependency on its provider.

For example, **propose** this method and point to the owner of `FeynmanDiagram`.
`tensor_reduce()` is not an existing method; first inspect the current
`TensorReducer` API and reuse its implementation:

```rust
use crate::PyFeynmanDiagram;
use pyo3::prelude::*;
use spynso3::expression::TensorExpression;
use symbolica::api::python::ExtensionPoint;

pub type DiagramReducer =
    for<'py> fn(Python<'py>, &PyFeynmanDiagram) -> PyResult<Py<TensorExpression>>;
pub static TENSOR_REDUCE: ExtensionPoint<DiagramReducer> =
    ExtensionPoint::new("hepkit.FeynmanDiagram.tensor_reduce");

#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyFeynmanDiagram {
    /// Reduce this diagram using the registered tensor-reduction implementation.
    fn tensor_reduce(&self, py: Python<'_>) -> PyResult<Py<TensorExpression>> {
        TENSOR_REDUCE.get()?(py, self)
    }
}
```

The provider's library registers its adapter to the existing reducer. Assuming
the owner accepted and exported the proposed interface, this initialization
helper is called from the provider's `CommunityModule::initialize`:

```rust
use feynkit_py::{DiagramReducer, TENSOR_REDUCE};
use pyo3::prelude::*;
use symbolica::api::python::ExtensionImplementation;
use crate::reduce_diagram; // Your adapter with the DiagramReducer signature.

static REDUCER: ExtensionImplementation<DiagramReducer> =
    ExtensionImplementation::new("my-extension", reduce_diagram);

pub fn initialize_reducer() -> PyResult<()> {
    TENSOR_REDUCE.register(&REDUCER)?;
    Ok(())
}
```

Return `Py<TensorExpression>` through the tensor component's public constructors,
which initialize its `Expression` base class too. Generate the matching owner
stub using its existing machinery, including method documentation:

```python
from symbolica.community.tensor import TensorExpression

class FeynmanDiagram:
    def tensor_reduce(self) -> TensorExpression:
        """Reduce this diagram using the registered tensor-reduction implementation."""
        ...
```

Keep stub arguments, defaults and return types accurate even for a Rust forwarder
accepting opaque `**kwargs`; Feynkit's sector methods use manual stub inventory
entries for this. Imports in `.pyi` files describe types, not runtime lookups.

Declare actual `static` points and descriptors, not `const` values or temporaries.
Re-registering the same descriptor is safe; a different descriptor conflicts,
even with the same provider name. Missing providers fail explicitly; errors
convert to `PyErr` for `?`. There is no replace or unregister operation. Register
before use in both hosts, and test missing, repeated and conflicting registration.

For existing PyO3 argument parsing, `PythonExtensionFunction` can wrap a directly
referenced `#[pyfunction]` with `wrap_pyfunction!`, preserving defaults and
exceptions without global Python objects. The owner forwards through
`POINT.call(py, (slf,), kwargs)`. The separate
[Feynkit #208](https://github.com/alphal00p/gammaloop/pull/208) and
[FastSecDec #3](https://github.com/alphal00p/fastSecDec/pull/3) integrations provide
`SECTOR_DECOMPOSE` and its provider; those consumer changes remain pending.
Symbolica's merge alone does not supply them. For atomic function tables, see
[the integration point](https://github.com/symbolica-dev/symbolica/blob/4f18c5e75374cbdeb229d6cac6345ca7f6eb9238/src/api/python/symbolic_integration.rs)
and [Community's host](../src/lib.rs).

## Release independently, then integrate

Generate and check stubs for your bindings and standalone re-exports. Run focused
Rust/Python tests and test a built wheel in a fresh environment. Record supported
platforms and dependency revisions; use reviewed upstream sources for releases.
Publish your repository's release and optional standalone wheel independently.

Then submit a small Community PR to add the library dependency and lockfile,
connect features and [registration](../src/lib.rs) or [hepkit setup](../src/hepkit.rs),
and add the public hepkit facade, stubs and documentation. The facade imports its
native exports and calls their initialization hook. Build the complete Community
wheel and rerun the same behavior tests with Community imports. The extension's
implementation and maintenance stay in its own repository.
