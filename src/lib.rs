use pyo3::{
    Bound, PyResult, pymodule,
    types::{PyModule, PyModuleMethods},
    wrap_pyfunction,
};
use symbolica::{
    api::python::{
        Citation, ExtensionImplementation, PYTHON_INTEGRATION, PythonIntegrationFunctions,
        PythonIntegrationStep, create_symbolica_module,
    },
    atom::{Atom, Symbol},
};
use symbolica_integrate::Integrate;

mod citations;
mod data;
#[cfg(feature = "community")]
mod hepkit;
#[cfg(feature = "community")]
mod integration;
#[cfg(feature = "community")]
mod loop_integration;
#[cfg(feature = "community")]
mod oneloop;

#[cfg(feature = "community")]
use pyo3::{Python, pyfunction, types::PyAnyMethods};
#[cfg(feature = "community")]
use symbolica::api::python::SymbolicaCommunityModule;

#[cfg(feature = "python_stubgen")]
use pyo3_stub_gen::define_stub_info_gatherer;

#[cfg(feature = "community")]
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
    record_integration_usage();
    expression.integrate(variable)
}

fn integrate_with_steps(
    expression: &Atom,
    variable: Symbol,
) -> (Result<Atom, Atom>, String, Vec<PythonIntegrationStep>) {
    record_integration_usage();
    let explanation = expression.integrate_with_steps(variable);
    let overview = explanation.to_string();
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

// Both integration operations are installed together, preserving one provider.
static INTEGRATION: ExtensionImplementation<PythonIntegrationFunctions> =
    ExtensionImplementation::new(
        "symbolica-integrate",
        PythonIntegrationFunctions {
            integrate,
            integrate_with_steps,
        },
    );

#[pymodule]
fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    PYTHON_INTEGRATION.register(&INTEGRATION)?;
    create_symbolica_module(m)?;
    m.add_function(wrap_pyfunction!(data::_set_data_directories, m)?)?;
    m.add_function(wrap_pyfunction!(data::_data_manifest, m)?)?;
    m.add_function(wrap_pyfunction!(data::_install_data_asset, m)?)?;
    m.add_function(wrap_pyfunction!(data::_read_data_asset, m)?)?;
    m.add_function(wrap_pyfunction!(citations::get_citations, m)?)?;
    #[cfg(feature = "community")]
    {
        register_module!(m, linnet_render_py::GraphModule);
        register_module!(m, spynso3::SpensoModule);
        register_module!(m, hepkit::HepKitModule);
        #[cfg(not(target_arch = "wasm32"))]
        register_module!(m, vakint::symbolica_community_module::VakintWrapper);
    }
    Ok(())
}

#[cfg(feature = "python_stubgen")]
define_stub_info_gatherer!(stub_info);

static INTEGRATION_USED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[inline]
fn record_integration_usage() {
    use std::sync::atomic::Ordering;
    if !INTEGRATION_USED.load(Ordering::Relaxed) {
        INTEGRATION_USED.store(true, Ordering::Relaxed);
    }
}

fn integration_citations() -> Vec<Citation> {
    if !INTEGRATION_USED.load(std::sync::atomic::Ordering::Relaxed) {
        return Vec::new();
    }
    vec![
        Citation {
            id: "https://github.com/symbolica-dev/symbolica-integrate".into(),
            url: "https://github.com/symbolica-dev/symbolica-integrate".into(),
            reference: "Ben Ruijl. Symbolica-integrate (2026).".into(),
            bibtex: r#"@software{symbolica_integrate,
  author = {Ruijl, Ben},
  title = {Symbolica-integrate},
  year = {2026},
  url = {https://github.com/symbolica-dev/symbolica-integrate}
}"#
            .into(),
            reasons: vec!["Symbolic integration.".into()],
            description: String::new(),
            relevance: None,
        },
        Citation {
            id: "https://rulebasedintegration.org".into(),
            url: "https://rulebasedintegration.org".into(),
            reference: "Albert D. Rich, Patrick Scheibe and contributors. Rubi.".into(),
            bibtex: r#"@software{rubi,
  author = {Rich, Albert D. and Scheibe, Patrick and {Rubi contributors}},
  title = {Rubi},
  url = {https://rulebasedintegration.org}
}"#
            .into(),
            reasons: vec!["Integration rules used by Symbolica-integrate.".into()],
            description: "Credits in the Symbolica-integrate README.".into(),
            relevance: None,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::{
        Python,
        types::{PyDict, PyDictMethods},
    };
    use std::ffi::CString;

    #[test]
    fn integration_extension_point_works_in_a_standalone_host() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let module = PyModule::new(py, "standalone_core")?;
            core(&module)?;
            PYTHON_INTEGRATION.register(&INTEGRATION)?;
            let locals = PyDict::new(py);
            locals.set_item("core", module)?;
            let source = CString::new(
                r#"
import builtins
original_import = builtins.__import__
def reject_symbolica(name, *args, **kwargs):
    if name == "symbolica" or name.startswith("symbolica."):
        raise AssertionError("integration imported " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = reject_symbolica
try:
    x = core.S("extension_integration_x")
    expression = x**2
    result = expression.integrate(x)
    assert isinstance(result, core.Expression)
    assert result.derivative(x) == expression
    explained, overview, steps = expression.integrate_with_steps(x)
    assert isinstance(explained, core.Expression)
    assert explained.derivative(x) == expression
    assert isinstance(overview, str) and overview
    assert steps
    assert all(isinstance(step, core.IntegrationStep) for step in steps)
    assert "https://github.com/symbolica-dev/symbolica-integrate" in {
        citation.id for citation in core.get_citations()
    }
finally:
    builtins.__import__ = original_import
"#,
            )
            .unwrap();
            py.run(&source, Some(&locals), Some(&locals))
        })
        .unwrap();
    }
}
