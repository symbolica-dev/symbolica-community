//! Expose the complete FeynKit Python API in one community namespace.

use pyo3::{
    Bound, PyResult, Python,
    types::{PyAnyMethods, PyDictMethods, PyModule, PyModuleMethods, PyType},
};
use symbolica::api::python::SymbolicaCommunityModule;

pub struct HepModule;

impl SymbolicaCommunityModule for HepModule {
    fn get_name() -> String {
        "hep".to_owned()
    }

    fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
        feynkit_py::initialize_feynkit(module)?;
        // Reuse the upstream classes themselves, including their methods and
        // exception hierarchy, while making introspection point to our public API.
        for value in module.dict().values() {
            if value.is_instance_of::<PyType>()
                && value.getattr("__module__")?.extract::<String>()?
                    == "symbolica.community.feynkit"
            {
                value.setattr("__module__", "symbolica.community.hep")?;
            }
        }
        Ok(())
    }

    fn initialize(py: Python<'_>) -> PyResult<()> {
        feynkit_py::FeynkitModule::initialize(py)
    }
}
