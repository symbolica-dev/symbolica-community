use std::{ops::ControlFlow, rc::Rc, time::Instant};

use fastsecdec::{
    generation::{GeneratedIntegral, GenerationOptions, generate},
    parametric::ParametricIntegrand,
    status::{GenerationSnapshot, GenerationStage, GenerationTimings},
};
use pyo3::prelude::*;
use symbolica::{api::python::PythonExpression, symbol};

use super::{error, input::PyIntegral, kernels::PyKernels, status::PyGenerationSnapshot};

#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(
    name = "GeneratedIntegral",
    module = "symbolica.community.hepkit.fastsecdec",
    frozen
)]
pub(crate) struct PyGeneratedIntegral {
    inner: GeneratedIntegral,
    status: GenerationSnapshot,
}

/// None/True continues; False cancels. User exceptions preserve their Python type.
pub(crate) fn observe(
    py: Python<'_>,
    observer: Option<&Py<PyAny>>,
    status: &GenerationSnapshot,
) -> PyResult<bool> {
    py.check_signals()?;
    let Some(observer) = observer else {
        return Ok(true);
    };
    let result = observer.call1(
        py,
        (PyGenerationSnapshot {
            inner: status.clone(),
        },),
    )?;
    if result.is_none(py) {
        Ok(true)
    } else {
        result.extract::<bool>(py)
    }
}

#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyIntegral {
    /// Synchronous native generation. False from the observer cancels at a native event boundary.
    #[pyo3(signature = (max_order=0, *, observer=None))]
    fn generate(
        &self,
        py: Python<'_>,
        max_order: i32,
        observer: Option<Py<PyAny>>,
    ) -> PyResult<PyGeneratedIntegral> {
        let started = Instant::now();
        let mut status = GenerationSnapshot {
            stage: GenerationStage::Parametrization,
            completed: 0,
            total: None,
            sectors: 0,
            kernels: 0,
            elapsed_seconds: 0.0,
            timings: GenerationTimings::default(),
            coefficient_expansion: None,
            detail: "Parametrizing the native diagram".into(),
        };
        if !observe(py, observer.as_ref(), &status)? {
            return Err(error::cancelled(py, "generation"));
        }
        let parameters = (0..self.graph.powers().len())
            .map(|i| symbol!(format!("fastsecdec::hepkit::x{i}")))
            .collect();
        let input = ParametricIntegrand::from_graph(
            &self.graph,
            parameters,
            self.regulator,
            self.dimension.clone(),
        )
        .map_err(|e| error::native(py, "parametrization", e))?;
        status.timings.parametrization_seconds = started.elapsed().as_secs_f64();
        let options = GenerationOptions {
            max_order,
            ..GenerationOptions::default()
        };
        let mut callback_error = None;
        let mut cancelled = false;
        let result = generate(&input, &options, |event| {
            status.observe_generation(max_order, event);
            status.elapsed_seconds = started.elapsed().as_secs_f64();
            match observe(py, observer.as_ref(), &status) {
                Ok(true) => ControlFlow::Continue(()),
                Ok(false) => {
                    cancelled = true;
                    ControlFlow::Break(())
                }
                Err(e) => {
                    callback_error = Some(e);
                    ControlFlow::Break(())
                }
            }
        });
        if let Some(error) = callback_error {
            return Err(error);
        }
        if cancelled {
            return Err(error::cancelled(py, "generation"));
        }
        let inner = result.map_err(|e| error::native(py, "generation", e))?;
        Ok(PyGeneratedIntegral { inner, status })
    }
}

#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyGeneratedIntegral {
    #[getter]
    fn orders(&self) -> Vec<i32> {
        self.inner.orders().to_vec()
    }
    #[getter]
    fn sector_count(&self) -> usize {
        self.inner.sectors().len()
    }
    #[getter]
    fn exact_coefficients(&self) -> Vec<PythonExpression> {
        self.inner
            .exact_coefficients()
            .iter()
            .cloned()
            .map(|expr| PythonExpression { expr })
            .collect()
    }
    fn snapshot(&self) -> PyGenerationSnapshot {
        PyGenerationSnapshot {
            inner: self.status.clone(),
        }
    }

    /// Build the configured native O2 or portable interpreted evaluator.
    #[pyo3(signature = (*, observer=None))]
    fn compile(&self, py: Python<'_>, observer: Option<Py<PyAny>>) -> PyResult<PyKernels> {
        let started = Instant::now();
        let mut status = self.status.clone();
        let offset = status.elapsed_seconds;
        let mut callback_error = None;
        let mut cancelled = false;
        let result = self.inner.compile_with_progress(|event| {
            status.observe_compilation(event);
            status.elapsed_seconds = offset + started.elapsed().as_secs_f64();
            match observe(py, observer.as_ref(), &status) {
                Ok(true) => ControlFlow::Continue(()),
                Ok(false) => {
                    cancelled = true;
                    ControlFlow::Break(())
                }
                Err(e) => {
                    callback_error = Some(e);
                    ControlFlow::Break(())
                }
            }
        });
        if let Some(error) = callback_error {
            return Err(error);
        }
        if cancelled {
            return Err(error::cancelled(py, "compilation"));
        }
        let inner = result.map_err(|e| error::native(py, "compilation", e))?;
        status.stage = GenerationStage::Complete;
        status.timings.total_seconds = offset + started.elapsed().as_secs_f64();
        status.detail = "Kernels ready".into();
        status.elapsed_seconds = offset + started.elapsed().as_secs_f64();
        if !observe(py, observer.as_ref(), &status)? {
            return Err(error::cancelled(py, "compilation"));
        }
        Ok(PyKernels {
            inner: Rc::new(inner),
            status,
        })
    }
}
