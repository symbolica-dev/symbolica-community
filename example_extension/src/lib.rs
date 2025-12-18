use pyo3::{
    Bound, PyResult, Python, pyfunction,
    types::{PyModule, PyModuleMethods},
    wrap_pyfunction,
};

#[cfg(feature = "python_stubgen")]
use pyo3_stub_gen::{define_stub_info_gatherer, derive::gen_stub_pyfunction};
use symbolica::api::python::{PythonExpression, SymbolicaCommunityModule};

pub struct CommunityModule;

impl SymbolicaCommunityModule for CommunityModule {
    fn get_name() -> String {
        "example_extension".to_string()
    }

    fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(add_two, m)?)?;
        Ok(())
    }

    fn initialize(_py: Python) -> PyResult<()> {
        println!("Initializing example_extension community module");
        Ok(())
    }
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.community.example_extension")
)]
#[pyfunction]
/// Add two to an atom.
pub fn add_two(atom: &PythonExpression) -> PythonExpression {
    (&atom.expr + 2).into()
}

#[cfg(feature = "python_stubgen")]
define_stub_info_gatherer!(stub_info);
