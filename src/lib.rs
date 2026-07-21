use pyo3::{
    Bound, PyResult, Python,
    exceptions::PyRuntimeError,
    pyfunction, pymodule,
    types::{PyAnyMethods, PyModule, PyModuleMethods},
    wrap_pyfunction,
};
use symbolica::{
    api::python::{
        PythonIntegrationFunctions, PythonIntegrationStep, SymbolicaCommunityModule,
        create_symbolica_module, set_python_integration_functions,
    },
    atom::{Atom, Symbol},
};
use symbolica_integrate::Integrate;

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

fn integrate(expression: &Atom, variable: Symbol) -> Result<Atom, Atom> {
    expression.integrate(variable)
}

fn integrate_with_steps(
    expression: &Atom,
    variable: Symbol,
) -> (Result<Atom, Atom>, String, Vec<PythonIntegrationStep>) {
    let explanation = expression.integrate_with_steps(variable);
    let overview = format!("{}", explanation);
    let steps = explanation
        .steps
        .into_iter()
        .map(|step| {
            PythonIntegrationStep::new(
                step.rule,
                step.depth,
                step.description.to_owned(),
                step.references
                    .iter()
                    .map(|reference| (*reference).to_owned())
                    .collect(),
                step.source.to_owned(),
                step.input,
                step.output,
            )
        })
        .collect();

    (explanation.result, overview, steps)
}

#[pymodule]
fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    set_python_integration_functions(PythonIntegrationFunctions {
        integrate,
        integrate_with_steps,
    })
    .map_err(PyRuntimeError::new_err)?;

    create_symbolica_module(m)?;

    register_module!(m, idenso::python::IdensoModule);
    register_module!(m, spynso3::SpensoModule);
    register_module!(m, vakint::symbolica_community_module::VakintWrapper);
    register_module!(m, example_extension::CommunityModule);

    Ok(())
}

#[cfg(feature = "python_stubgen")]
define_stub_info_gatherer!(stub_info);
