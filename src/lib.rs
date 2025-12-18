use pyo3::{
    pyfunction, pymodule,
    types::{PyAnyMethods, PyModule, PyModuleMethods},
    wrap_pyfunction, Bound, PyResult, Python,
};
use symbolica::api::python::{create_symbolica_module, SymbolicaCommunityModule};

#[cfg(feature = "python_stubgen")]
use pyo3_stub_gen::define_stub_info_gatherer;

macro_rules! register_module {
    ($m:expr, $module_type:ty) => {{
        let native_name = format!("{}_native", <$module_type>::get_name());

        #[pyfunction]
        fn initialize_module(py: Python) -> PyResult<()> {
            <$module_type>::initialize(py)
        }

        let child_module = PyModule::new($m.py(), &native_name)?;
        child_module.add_function(wrap_pyfunction!(initialize_module, &child_module)?)?;

        <$module_type>::register_module(&child_module)?;
        $m.add_submodule(&child_module)?;

        $m.py().import("sys")?.getattr("modules")?.set_item(
            format!("symbolica.community.{}", native_name),
            &child_module,
        )?;
    }};
}

#[pymodule]
fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    create_symbolica_module(m)?;

    register_module!(m, idenso::python::IdensoModule);
    register_module!(m, spynso3::SpensoModule);
    register_module!(m, vakint::symbolica_community_module::VakintWrapper);
    register_module!(m, example_extension::CommunityModule);

    Ok(())
}

#[cfg(feature = "python_stubgen")]
define_stub_info_gatherer!(stub_info);
