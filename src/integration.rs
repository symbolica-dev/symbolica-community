//! Register Hyperbolica's bindings in the existing shared Symbolica extension.
use pyo3::{
    Bound, PyResult,
    types::{PyAnyMethods, PyModule, PyModuleMethods},
};
use symbolica::api::python::SymbolicaCommunityModule;

pub fn register(hep: &Bound<'_, PyModule>) -> PyResult<()> {
    let name = "symbolica.community.hepkit_integration_native";
    let module = PyModule::new(hep.py(), name)?;
    hyperbolica::python::CommunityModule::register_module(&module)?;
    hep.add("_integration_native", &module)?;
    hep.py()
        .import("sys")?
        .getattr("modules")?
        .set_item(name, &module)?;
    Ok(())
}
