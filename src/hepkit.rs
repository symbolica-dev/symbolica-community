//! Expose FeynKit, one-loop reduction, and native master evaluation together.

use pyo3::{Bound, PyResult, Python, types::PyModule};
use symbolica::api::python::{Citation, SymbolicaCommunityModule};

pub struct HepKitModule;

impl SymbolicaCommunityModule for HepKitModule {
    fn get_citations() -> Vec<Citation> {
        let mut citations = feynkit_py::FeynkitModule::get_citations();
        citations.extend(crate::oneloop::get_citations());
        citations.extend(fastsecdec_python::get_citations());
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
            citations.push(Citation {
                id: "arXiv:2604.25916".into(),
                reference: "Christoph Dlapa, Gregor Kälin, Zhengwen Liu and Rafael A. Porto. Nonlocal-in-time tail effects in gravitational scattering to fifth post-Minkowskian and tenth self-force orders. Phys. Rev. D 114, 024029 (2026). doi:10.1103/wkp4-vy6g.".into(),
                bibtex: r#"@article{Dlapa:2026oyq,
  author = {Dlapa, Christoph and K{\"a}lin, Gregor and Liu, Zhengwen and Porto, Rafael A.},
  title = {Nonlocal-in-time tail effects in gravitational scattering to fifth post-Minkowskian and tenth self-force orders},
  eprint = {2604.25916},
  archivePrefix = {arXiv},
  primaryClass = {hep-th},
  reportNumber = {DESY 26-055},
  doi = {10.1103/wkp4-vy6g},
  journal = {Phys. Rev. D},
  volume = {114},
  number = {2},
  pages = {024029},
  year = {2026}
}"#
                .into(),
                reasons: vec!["Describes the sparse parametric IBP strategy underlying RustRed's SpIReD-inspired solver.".into()],
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
        fastsecdec_python::register(module)?;
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
