use symbolica::api::python::{Citation, SymbolicaCommunityModule};

/// Return cumulative citations for the Symbolica features used in this process.
/// Importing a community module does not count as using it. Calls never reset
/// usage, and duplicate references merge their reasons under one stable ID.
#[cfg_attr(
    feature = "python_stubgen",
    pyo3_stub_gen::derive::gen_stub_pyfunction(module = "symbolica.core")
)]
#[pyo3::pyfunction]
pub fn get_citations() -> Vec<Citation> {
    let mut citations = vec![Citation {
        id: "doi:10.5281/zenodo.17054381".into(),
        reference: "Ben Ruijl. Symbolica (2025). doi:10.5281/zenodo.17054381.".into(),
        bibtex: r#"@software{ruijl_symbolica_2025,
  author = {Ruijl, Ben},
  title = {Symbolica},
  year = {2025},
  version = {0.18.0},
  doi = {10.5281/zenodo.17054381},
  url = {https://zenodo.org/records/17054381}
}"#
        .into(),
        reasons: vec!["Symbolic and numerical computation with Symbolica.".into()],
        description: String::new(),
        relevance: None,
    }];
    let mut community = Vec::new();
    community.extend(crate::hep::HepModule::get_citations());
    community.extend(spynso3::SpensoModule::get_citations());
    #[cfg(not(target_arch = "wasm32"))]
    community.extend(vakint::symbolica_community_module::VakintWrapper::get_citations());
    community.extend(crate::integration_citations());
    for citation in community {
        if let Some(existing) = citations.iter_mut().find(|entry| entry.id == citation.id) {
            for reason in citation.reasons {
                if !existing.reasons.contains(&reason) {
                    existing.reasons.push(reason);
                }
            }
        } else {
            citations.push(citation);
        }
    }
    citations
}
