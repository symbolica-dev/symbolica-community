use std::{rc::Rc, time::Instant};

use fastsecdec::{
    kernel::KernelSet,
    status::{GenerationSnapshot, GenerationStage, GenerationTimings},
};
use pyo3::{prelude::*, types::PyBytes};

use super::{
    error,
    session::{PyQmcSession, PyQmcSettings},
    status::PyGenerationSnapshot,
};

/// Compiled native evaluator owners. Each session owns one lazy execution context.
#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(
    name = "Kernels",
    module = "symbolica.community.hepkit.fastsecdec",
    unsendable
)]
pub(crate) struct PyKernels {
    pub(crate) inner: Rc<KernelSet>,
    pub(crate) status: GenerationSnapshot,
}

#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyKernels {
    /// Persist the library's native portable program/metadata codec, excluding machine code.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self
            .inner
            .artifact_bytes()
            .map_err(|e| error::native(py, "artifact", e))?;
        Ok(PyBytes::new(py, bytes))
    }

    /// Validate native programs and construct evaluators for the current host backend.
    #[staticmethod]
    fn from_bytes(py: Python<'_>, artifact: &Bound<'_, PyBytes>) -> PyResult<Self> {
        py.check_signals()?;
        let started = Instant::now();
        let inner = KernelSet::from_bytes(artifact.as_bytes())
            .map_err(|e| error::native(py, "artifact", e))?;
        py.check_signals()?;
        let count = inner.sectors().len();
        let elapsed = started.elapsed().as_secs_f64();
        let status = GenerationSnapshot {
            stage: GenerationStage::Complete,
            completed: count,
            total: Some(count),
            sectors: count,
            kernels: count,
            elapsed_seconds: elapsed,
            timings: GenerationTimings {
                compilation_seconds: elapsed,
                total_seconds: elapsed,
                ..GenerationTimings::default()
            },
            coefficient_expansion: None,
            detail: "Native artifact loaded; original generation timings unavailable".into(),
        };
        Ok(Self {
            inner: Rc::new(inner),
            status,
        })
    }

    #[getter]
    fn content_id(&self) -> &str {
        self.inner.content_id()
    }
    #[getter]
    fn orders(&self) -> Vec<i32> {
        self.inner.orders().to_vec()
    }
    #[getter]
    fn components(&self) -> Vec<&'static str> {
        self.inner
            .components()
            .iter()
            .map(|c| match c {
                fastsecdec::status::CoefficientComponent::Real => "real",
                fastsecdec::status::CoefficientComponent::Imag => "imag",
            })
            .collect()
    }
    #[getter]
    fn sector_count(&self) -> usize {
        self.inner.sectors().len()
    }
    #[getter]
    fn exact_coefficients(&self) -> Vec<f64> {
        self.inner.exact_coefficients().to_vec()
    }
    #[getter]
    fn backend(&self) -> &'static str {
        #[cfg(feature = "native")]
        {
            "native_o2"
        }
        #[cfg(not(feature = "native"))]
        {
            "portable_interpreted"
        }
    }
    fn snapshot(&self) -> PyGenerationSnapshot {
        PyGenerationSnapshot {
            inner: self.status.clone(),
        }
    }

    #[pyo3(signature = (settings=None))]
    fn session(&self, py: Python<'_>, settings: Option<&PyQmcSettings>) -> PyResult<PyQmcSession> {
        PyQmcSession::new(
            py,
            self.inner.clone(),
            settings.map(|s| s.inner.clone()).unwrap_or_default(),
        )
    }

    /// Restore complete accepted packages and replay state against these exact kernels.
    fn restore(&self, py: Python<'_>, checkpoint: &Bound<'_, PyBytes>) -> PyResult<PyQmcSession> {
        PyQmcSession::restore_native(py, self.inner.clone(), checkpoint.as_bytes())
    }
}
