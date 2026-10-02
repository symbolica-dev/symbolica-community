//! Expose FeynKit, one-loop reduction, and native master evaluation together.

use pyo3::{Bound, PyResult, Python, types::PyModule};
use symbolica::api::python::{Citation, SymbolicaCommunityModule};

pub struct HepKitModule;

impl SymbolicaCommunityModule for HepKitModule {
    fn get_citations() -> Vec<Citation> {
        let mut citations = feynkit_py::FeynkitModule::get_citations();
        citations.extend(crate::oneloop::get_citations());
        #[cfg(not(target_arch = "wasm32"))]
        if rustred_feynkit::was_used() {
            citations.push(Citation {
                id: "https://github.com/alphal00p/rustred".into(),
                reference: "Gregor Kälin and Valentin Hirschi. RustRed (2026).".into(),
                bibtex: r#"@software{rustred,
  author = {Kälin, Gregor and Hirschi, Valentin},
  title = {RustRed},
  year = {2026},
  url = {https://github.com/alphal00p/rustred}
}"#
                .into(),
                reasons: vec!["Provides the native HEP IBP solver.".into()],
                description: String::new(),
                relevance: None,
            });
        }
        citations
    }

    fn get_name() -> String {
        "hepkit".to_owned()
    }

    fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
        feynkit_py::initialize_feynkit(module)?;
        crate::oneloop::register(module)?;
        #[cfg(not(target_arch = "wasm32"))]
        rustred_feynkit::register_hep_module(module)?;
        Ok(())
    }

    fn initialize(py: Python<'_>) -> PyResult<()> {
        feynkit_py::FeynkitModule::initialize(py)?;
        oneloopreduce_python::CommunityModule::initialize(py)?;
        #[cfg(not(target_arch = "wasm32"))]
        oneloop_native::CommunityModule::initialize(py)?;
        Ok(())
    }
}
