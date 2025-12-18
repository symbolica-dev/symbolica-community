use pyo3::{
    pymodule,
    types::{PyAnyMethods, PyModule, PyModuleMethods},
    Bound, PyResult,
};
use symbolica::api::python::{create_symbolica_module, SymbolicaCommunityModule};

#[cfg(feature = "python_stubgen")]
use pyo3_stub_gen::define_stub_info_gatherer;

fn register_extension<T: SymbolicaCommunityModule>(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let native_name = format!("{}_native", T::get_name());
    let child_module = PyModule::new(m.py(), &native_name)?;
    T::register_module(&child_module)?;
    m.add_submodule(&child_module)?;

    m.py().import("sys")?.getattr("modules")?.set_item(
        format!("symbolica.community.{}", native_name),
        &child_module,
    )?;
    Ok(())
}

#[pymodule]
fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    create_symbolica_module(m)?;

    register_extension::<idenso::python::IdensoModule>(m)?;
    register_extension::<spynso3::SpensoModule>(m)?;
    register_extension::<vakint::symbolica_community_module::VakintWrapper>(m)?;
    register_extension::<example_extension::CommunityModule>(m)?;

    Ok(())
}

#[cfg(feature = "python_stubgen")]
define_stub_info_gatherer!(stub_info);
