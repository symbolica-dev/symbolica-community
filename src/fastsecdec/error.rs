use std::fmt::Display;

use pyo3::{create_exception, exceptions::PyRuntimeError, prelude::*, types::PyModule};

create_exception!(fastsecdec, FastSecDecError, PyRuntimeError);
create_exception!(fastsecdec, CancelledError, FastSecDecError);

pub(crate) fn native(py: Python<'_>, stage: &'static str, cause: impl Display) -> PyErr {
    with_stage(py, stage, FastSecDecError::new_err(cause.to_string()))
}

pub(crate) fn cancelled(py: Python<'_>, stage: &'static str) -> PyErr {
    with_stage(
        py,
        stage,
        CancelledError::new_err(format!("{stage} cancelled")),
    )
}

fn with_stage(py: Python<'_>, stage: &'static str, error: PyErr) -> PyErr {
    // Exception attributes carry machine-readable stages independently of text.
    if let Err(attribute_error) = error.value(py).setattr("stage", stage) {
        return attribute_error;
    }
    error
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module
        .py()
        .get_type::<FastSecDecError>()
        .setattr("__module__", "symbolica.community.hepkit.fastsecdec")?;
    module
        .py()
        .get_type::<CancelledError>()
        .setattr("__module__", "symbolica.community.hepkit.fastsecdec")?;
    module.add("FastSecDecError", module.py().get_type::<FastSecDecError>())?;
    module.add("CancelledError", module.py().get_type::<CancelledError>())?;
    Ok(())
}
