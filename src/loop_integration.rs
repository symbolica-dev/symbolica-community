//! Numerical loop integration shares the host's Symbolica and HEPKit objects.
use pyo3::{
    Bound, PyResult,
    types::{PyAnyMethods, PyModule, PyModuleMethods},
};

pub fn get_citations() -> Vec<symbolica::api::python::Citation> {
    symbolica_amflow::python::get_citations()
}

pub fn register(hep: &Bound<'_, PyModule>) -> PyResult<()> {
    let name = "symbolica.community.hep_integration_native";
    let module = PyModule::new(hep.py(), name)?;
    symbolica_amflow::python::register(&module)?;
    hep.add("_loop_integration_native", &module)?;
    hep.py()
        .import("sys")?
        .getattr("modules")?
        .set_item(name, &module)?;
    Ok(())
}
