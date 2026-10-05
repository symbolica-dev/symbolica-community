//! Thin HEPKit ownership and caller-stepped execution boundary.
mod error;
mod generation;
mod input;
mod kernels;
mod session;
mod status;

use pyo3::{prelude::*, types::PyModule};

pub fn register(hep: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = PyModule::new(hep.py(), "symbolica.community.hepkit_fastsecdec_native")?;
    module.add_class::<input::PyIntegral>()?;
    module.add_class::<generation::PyGeneratedIntegral>()?;
    module.add_class::<kernels::PyKernels>()?;
    module.add_class::<session::PyQmcSettings>()?;
    module.add_class::<session::PyQmcSession>()?;
    error::register(&module)?;
    status::register(&module)?;
    hep.add("_fastsecdec_native", &module)?;
    hep.py()
        .import("sys")?
        .getattr("modules")?
        .set_item("symbolica.community.hepkit_fastsecdec_native", &module)?;
    Ok(())
}
