//! Immutable Python views of the native numerical and generation snapshots.

use ::fastsecdec::{
    generation::{CoefficientExpansionMethod, CoefficientExpansionStage, CoefficientRequestCounts},
    integration::VectorEstimate,
    status::{
        CoefficientComponent, CoefficientExpansionSnapshot, EvaluationDiagnostics,
        GenerationSnapshot, GenerationStage, GenerationTimings, IntegrationMethod,
        IntegrationSnapshot, IntegrationStage, SectorSnapshot, StoppingReason, UncertaintyStatus,
    },
};
use pyo3::{prelude::*, types::PyModule};

/// Generation progress copied from the native caller-owned observer.
#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(
    frozen,
    from_py_object,
    module = "symbolica.community.hepkit.fastsecdec",
    name = "GenerationSnapshot"
)]
#[derive(Clone)]
pub(crate) struct PyGenerationSnapshot {
    pub(crate) inner: GenerationSnapshot,
}

#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyGenerationSnapshot {
    /// Stable snake_case name of the native generation stage.
    #[getter]
    fn stage(&self) -> &'static str {
        match self.inner.stage {
            GenerationStage::Input => "input",
            GenerationStage::Parametrization => "parametrization",
            GenerationStage::Geometry => "geometry",
            GenerationStage::Mapping => "mapping",
            GenerationStage::Symmetry => "symmetry",
            GenerationStage::Subtraction => "subtraction",
            GenerationStage::Expansion => "expansion",
            GenerationStage::CoefficientExpansion => "coefficient_expansion",
            GenerationStage::Compilation => "compilation",
            GenerationStage::Complete => "complete",
        }
    }

    #[getter]
    fn completed(&self) -> usize {
        self.inner.completed
    }

    #[getter]
    fn total(&self) -> Option<usize> {
        self.inner.total
    }

    #[getter]
    fn sectors(&self) -> usize {
        self.inner.sectors
    }

    #[getter]
    fn kernels(&self) -> usize {
        self.inner.kernels
    }

    #[getter]
    fn elapsed_seconds(&self) -> f64 {
        self.inner.elapsed_seconds
    }

    #[getter]
    fn timings(&self) -> PyGenerationTimings {
        PyGenerationTimings {
            inner: self.inner.timings.clone(),
        }
    }

    /// Current or last named-coefficient attempt; None for physical-only generation.
    #[getter]
    fn coefficient_expansion(&self) -> Option<PyCoefficientExpansionSnapshot> {
        self.inner
            .coefficient_expansion
            .clone()
            .map(|inner| PyCoefficientExpansionSnapshot { inner })
    }

    #[getter]
    fn detail(&self) -> String {
        self.inner.detail.clone()
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// Native observed generation wall times in seconds.
///
/// The total ends after preparing kernels and metadata, before artifact writing.
#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(
    frozen,
    from_py_object,
    module = "symbolica.community.hepkit.fastsecdec",
    name = "GenerationTimings"
)]
#[derive(Clone)]
pub(crate) struct PyGenerationTimings {
    inner: GenerationTimings,
}

#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyGenerationTimings {
    #[getter]
    fn input_seconds(&self) -> f64 {
        self.inner.input_seconds
    }

    #[getter]
    fn parametrization_seconds(&self) -> f64 {
        self.inner.parametrization_seconds
    }

    #[getter]
    fn domain_seconds(&self) -> f64 {
        self.inner.domain_seconds
    }

    #[getter]
    fn geometry_seconds(&self) -> f64 {
        self.inner.geometry_seconds
    }

    #[getter]
    fn mapping_seconds(&self) -> f64 {
        self.inner.mapping_seconds
    }

    #[getter]
    fn symmetry_seconds(&self) -> f64 {
        self.inner.symmetry_seconds
    }

    #[getter]
    fn subtraction_seconds(&self) -> f64 {
        self.inner.subtraction_seconds
    }

    #[getter]
    fn laurent_seconds(&self) -> f64 {
        self.inner.laurent_seconds
    }

    /// Entire named phase, including any exact physical fallback.
    #[getter]
    fn coefficient_expansion_seconds(&self) -> f64 {
        self.inner.coefficient_expansion_seconds
    }

    #[getter]
    fn compilation_seconds(&self) -> f64 {
        self.inner.compilation_seconds
    }

    #[getter]
    fn total_seconds(&self) -> f64 {
        self.inner.total_seconds
    }
}

/// Native per-representative coefficient work for the current or last attempt.
#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(
    frozen,
    from_py_object,
    module = "symbolica.community.hepkit.fastsecdec",
    name = "CoefficientExpansionSnapshot"
)]
#[derive(Clone)]
pub(crate) struct PyCoefficientExpansionSnapshot {
    inner: CoefficientExpansionSnapshot,
}

#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyCoefficientExpansionSnapshot {
    /// Zero-based representative being processed or most recently completed.
    #[getter]
    fn sector(&self) -> usize {
        self.inner.sector
    }

    #[getter]
    fn requested_method(&self) -> &'static str {
        coefficient_method(self.inner.requested_method)
    }

    /// Physical also identifies an exact unregulated-endpoint fallback.
    #[getter]
    fn effective_method(&self) -> &'static str {
        coefficient_method(self.inner.effective_method)
    }

    #[getter]
    fn stage(&self) -> &'static str {
        match self.inner.stage {
            CoefficientExpansionStage::Admission => "admission",
            CoefficientExpansionStage::RegularSeries => "regular_series",
            CoefficientExpansionStage::Naming => "naming",
            CoefficientExpansionStage::Endpoint => "endpoint",
            CoefficientExpansionStage::Composition => "composition",
            CoefficientExpansionStage::Coverage => "coverage",
            CoefficientExpansionStage::Lowering => "lowering",
            CoefficientExpansionStage::PhysicalFallback => "physical_fallback",
            CoefficientExpansionStage::Complete => "complete",
        }
    }

    /// One-based native attempt; zero means admission or exact physical fallback.
    #[getter]
    fn attempt(&self) -> usize {
        self.inner.attempt
    }

    /// Native relative width; zero has the same pre-attempt/fallback meaning.
    /// This is not an absolute Laurent cutoff.
    #[getter]
    fn relative_width(&self) -> i64 {
        self.inner.relative_width
    }

    /// Named-composition pieces in this attempt; zero for physical fallback.
    #[getter]
    fn formal_pieces(&self) -> usize {
        self.inner.formal_pieces
    }

    #[getter]
    fn requests(&self) -> PyCoefficientRequestCounts {
        PyCoefficientRequestCounts {
            inner: self.inner.requests,
        }
    }
}

fn coefficient_method(method: CoefficientExpansionMethod) -> &'static str {
    match method {
        CoefficientExpansionMethod::Physical => "physical",
        CoefficientExpansionMethod::NativeNamed => "native_named",
    }
}

/// Native representation counts, reset at each coefficient-series attempt.
#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(
    frozen,
    from_py_object,
    module = "symbolica.community.hepkit.fastsecdec",
    name = "CoefficientRequestCounts"
)]
#[derive(Clone)]
pub(crate) struct PyCoefficientRequestCounts {
    inner: CoefficientRequestCounts,
}

#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyCoefficientRequestCounts {
    #[getter]
    fn source_bodies(&self) -> usize {
        self.inner.source_bodies
    }

    #[getter]
    fn unique_requests(&self) -> usize {
        self.inner.unique_requests
    }

    #[getter]
    fn cached_partials(&self) -> usize {
        self.inner.cached_partials
    }

    #[getter]
    fn aliases(&self) -> usize {
        self.inner.aliases
    }

    #[getter]
    fn interleaved_requests(&self) -> usize {
        self.inner.interleaved_requests
    }

    #[getter]
    fn fallback_requests(&self) -> usize {
        self.inner.fallback_requests
    }
}

/// Coverage and estimates from the native caller-driven integration session.
#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(
    frozen,
    from_py_object,
    module = "symbolica.community.hepkit.fastsecdec",
    name = "IntegrationSnapshot"
)]
#[derive(Clone)]
pub(crate) struct PyIntegrationSnapshot {
    pub(crate) inner: IntegrationSnapshot,
}

#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyIntegrationSnapshot {
    #[getter]
    fn method(&self) -> &'static str {
        match self.inner.method {
            IntegrationMethod::DemocraticQmc => "democratic_qmc",
            IntegrationMethod::AdaptiveQmc => "adaptive_qmc",
            IntegrationMethod::HavanaMc => "havana_mc",
        }
    }

    #[getter]
    fn stage(&self) -> &'static str {
        match self.inner.stage {
            IntegrationStage::Pilot => "pilot",
            IntegrationStage::Production => "production",
        }
    }

    #[getter]
    fn completed_points(&self) -> u64 {
        self.inner.completed_points
    }

    #[getter]
    fn planned_points(&self) -> u64 {
        self.inner.planned_points
    }

    #[getter]
    fn complete_sectors(&self) -> usize {
        self.inner.complete_sectors
    }

    #[getter]
    fn sectors(&self) -> Vec<PySectorSnapshot> {
        self.inner
            .sectors
            .iter()
            .cloned()
            .map(|inner| PySectorSnapshot { inner })
            .collect()
    }

    /// Native validity state; statistical_failure retains its reason in uncertainty_detail.
    #[getter]
    fn uncertainty(&self) -> &'static str {
        match &self.inner.uncertainty {
            UncertaintyStatus::Available => "available",
            UncertaintyStatus::Exact => "exact",
            UncertaintyStatus::WaitingForCoverage => "waiting_for_coverage",
            UncertaintyStatus::PilotOnly => "pilot_only",
            UncertaintyStatus::StatisticalFailure { .. } => "statistical_failure",
        }
    }

    /// Native failure explanation, when the uncertainty state is statistical_failure.
    #[getter]
    fn uncertainty_detail(&self) -> Option<String> {
        match &self.inner.uncertainty {
            UncertaintyStatus::StatisticalFailure { reason } => Some(reason.clone()),
            UncertaintyStatus::Available
            | UncertaintyStatus::Exact
            | UncertaintyStatus::WaitingForCoverage
            | UncertaintyStatus::PilotOnly => None,
        }
    }

    /// Native full Laurent vector, or None when no valid estimate is available.
    #[getter]
    fn estimate(&self) -> Option<PyVectorEstimate> {
        self.inner
            .estimate
            .clone()
            .map(|inner| PyVectorEstimate { inner })
    }

    /// Sum of worker execution times; caller wall time is separate.
    #[getter]
    fn worker_seconds(&self) -> f64 {
        self.inner.worker_seconds
    }

    /// Native caller-supplied stopping reason, or None while none is recorded.
    #[getter]
    fn stop_reason(&self) -> Option<&'static str> {
        self.inner.stop_reason.as_ref().map(|reason| match reason {
            StoppingReason::TargetReached => "target_reached",
            StoppingReason::PlannedWorkComplete => "planned_work_complete",
            StoppingReason::WorkLimit => "work_limit",
            StoppingReason::TimeLimit => "time_limit",
            StoppingReason::Cancelled => "cancelled",
            StoppingReason::NumericalFailure(_) => "numerical_failure",
        })
    }

    /// Native failure explanation, when stop_reason is numerical_failure.
    #[getter]
    fn stop_detail(&self) -> Option<String> {
        match &self.inner.stop_reason {
            Some(StoppingReason::NumericalFailure(reason)) => Some(reason.clone()),
            Some(
                StoppingReason::TargetReached
                | StoppingReason::PlannedWorkComplete
                | StoppingReason::WorkLimit
                | StoppingReason::TimeLimit
                | StoppingReason::Cancelled,
            )
            | None => None,
        }
    }

    #[getter]
    fn evaluation_diagnostics(&self) -> Option<PyEvaluationDiagnostics> {
        self.inner
            .evaluation_diagnostics
            .clone()
            .map(|inner| PyEvaluationDiagnostics { inner })
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// Native per-sector work and complete-replica coverage.
#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(
    frozen,
    from_py_object,
    module = "symbolica.community.hepkit.fastsecdec",
    name = "SectorSnapshot"
)]
#[derive(Clone)]
pub(crate) struct PySectorSnapshot {
    inner: SectorSnapshot,
}

#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PySectorSnapshot {
    #[getter]
    fn id(&self) -> u64 {
        self.inner.id
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension
    }

    #[getter]
    fn completed_points(&self) -> u64 {
        self.inner.completed_points
    }

    #[getter]
    fn planned_points(&self) -> u64 {
        self.inner.planned_points
    }

    #[getter]
    fn complete_replicas(&self) -> usize {
        self.inner.complete_replicas
    }

    #[getter]
    fn planned_replicas(&self) -> usize {
        self.inner.planned_replicas
    }

    #[getter]
    fn worker_seconds(&self) -> f64 {
        self.inner.worker_seconds
    }
}

/// Native joint Laurent estimate with signed orders and full covariance.
///
/// Returned arrays are independent copies. The bridge does not reconstruct
/// uncertainty or combine components, sectors, or orders.
#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(
    frozen,
    from_py_object,
    module = "symbolica.community.hepkit.fastsecdec",
    name = "VectorEstimate"
)]
#[derive(Clone)]
pub(crate) struct PyVectorEstimate {
    pub(crate) inner: VectorEstimate,
}

#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyVectorEstimate {
    /// Delegate convergence to the native full-production and vector tolerance check.
    #[pyo3(signature = (*, absolute=0.0, relative=0.001))]
    fn meets(&self, py: Python<'_>, absolute: f64, relative: f64) -> PyResult<bool> {
        let tolerance = ::fastsecdec::integration::Tolerance::new(absolute, relative)
            .map_err(|e| super::error::native(py, "configuration", e))?;
        self.inner
            .meets(tolerance)
            .map_err(|e| super::error::native(py, "integration", e))
    }

    /// Signed epsilon powers, in exactly the native output order.
    #[getter]
    fn orders(&self) -> Vec<i32> {
        self.inner.orders.clone()
    }

    /// A real or imag label for each native output entry.
    #[getter]
    fn components(&self) -> Vec<&'static str> {
        self.inner
            .components
            .iter()
            .map(|component| match component {
                CoefficientComponent::Real => "real",
                CoefficientComponent::Imag => "imag",
            })
            .collect()
    }

    #[getter]
    fn mean(&self) -> Vec<f64> {
        self.inner.mean.clone()
    }

    #[getter]
    fn standard_error(&self) -> Vec<f64> {
        self.inner.standard_error.clone()
    }

    /// Full row-major covariance of the mean, including inter-order and
    /// real/imaginary correlations. Entry (i, j) is at i * len(orders) + j.
    #[getter]
    fn covariance_of_mean(&self) -> Vec<f64> {
        self.inner.covariance_of_mean.clone()
    }

    /// Only a complete production allocation can certify an accuracy stop.
    #[getter]
    fn production_complete(&self) -> bool {
        self.inner.production_complete
    }
}

/// Native caller-aggregated evaluation counters, including failed attempts.
#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(
    frozen,
    from_py_object,
    module = "symbolica.community.hepkit.fastsecdec",
    name = "EvaluationDiagnostics"
)]
#[derive(Clone)]
pub(crate) struct PyEvaluationDiagnostics {
    inner: EvaluationDiagnostics,
}

#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyEvaluationDiagnostics {
    #[getter]
    fn evaluations(&self) -> u64 {
        self.inner.evaluations
    }

    #[getter]
    fn conditioning_checks(&self) -> u64 {
        self.inner.conditioning_checks
    }

    #[getter]
    fn rescues(&self) -> u64 {
        self.inner.rescues
    }

    #[getter]
    fn max_precision_bits(&self) -> u32 {
        self.inner.max_precision_bits
    }

    #[getter]
    fn failures(&self) -> u64 {
        self.inner.failures
    }

    #[getter]
    fn weighted_checks(&self) -> u64 {
        self.inner.weighted_checks
    }

    #[getter]
    fn additional_replays(&self) -> u64 {
        self.inner.additional_replays
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyGenerationSnapshot>()?;
    module.add_class::<PyGenerationTimings>()?;
    module.add_class::<PyCoefficientExpansionSnapshot>()?;
    module.add_class::<PyCoefficientRequestCounts>()?;
    module.add_class::<PyIntegrationSnapshot>()?;
    module.add_class::<PySectorSnapshot>()?;
    module.add_class::<PyVectorEstimate>()?;
    module.add_class::<PyEvaluationDiagnostics>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_complex_vector_and_covariance_are_preserved() {
        let native = VectorEstimate {
            orders: vec![-2, -2, 1],
            components: vec![
                CoefficientComponent::Real,
                CoefficientComponent::Imag,
                CoefficientComponent::Real,
            ],
            mean: vec![2.0, -3.0, 7.0],
            standard_error: vec![1.0, 2.0, 3.0],
            covariance_of_mean: vec![1.0, -0.5, 0.25, -0.5, 4.0, -1.0, 0.25, -1.0, 9.0],
            production_complete: false,
        };
        let wrapped = PyVectorEstimate {
            inner: native.clone(),
        };
        assert_eq!(wrapped.orders(), native.orders);
        assert_eq!(wrapped.components(), vec!["real", "imag", "real"]);
        assert_eq!(wrapped.mean(), native.mean);
        assert_eq!(wrapped.standard_error(), native.standard_error);
        assert_eq!(wrapped.covariance_of_mean(), native.covariance_of_mean);
        assert!(!wrapped.production_complete());

        let mut detached = wrapped.covariance_of_mean();
        detached[1] = 0.0;
        assert_eq!(wrapped.covariance_of_mean()[1], -0.5);
    }

    #[test]
    fn missing_estimate_and_failure_payloads_remain_distinct() {
        let mut wrapped = PyIntegrationSnapshot {
            inner: IntegrationSnapshot {
                method: IntegrationMethod::DemocraticQmc,
                stage: IntegrationStage::Production,
                completed_points: 8,
                planned_points: 32,
                complete_sectors: 0,
                sectors: vec![SectorSnapshot {
                    id: 17,
                    dimension: 2,
                    completed_points: 8,
                    planned_points: 32,
                    complete_replicas: 0,
                    planned_replicas: 4,
                    worker_seconds: 0.25,
                }],
                uncertainty: UncertaintyStatus::WaitingForCoverage,
                estimate: None,
                worker_seconds: 0.25,
                stop_reason: None,
                evaluation_diagnostics: None,
            },
        };
        assert!(wrapped.estimate().is_none());
        assert_eq!(wrapped.uncertainty(), "waiting_for_coverage");
        assert!(wrapped.uncertainty_detail().is_none());
        assert!(wrapped.stop_reason().is_none());
        assert!(wrapped.stop_detail().is_none());
        assert!(wrapped.evaluation_diagnostics().is_none());
        let detached_sector = wrapped.sectors().remove(0);
        wrapped.inner.sectors[0].complete_replicas = 1;
        assert_eq!(detached_sector.complete_replicas(), 0);

        wrapped.inner.uncertainty = UncertaintyStatus::StatisticalFailure {
            reason: "native covariance range failure".into(),
        };
        wrapped.inner.stop_reason = Some(StoppingReason::NumericalFailure(
            "native evaluation failed".into(),
        ));
        assert!(wrapped.estimate().is_none());
        assert_eq!(wrapped.uncertainty(), "statistical_failure");
        assert_eq!(
            wrapped.uncertainty_detail().as_deref(),
            Some("native covariance range failure")
        );
        assert_eq!(wrapped.stop_reason(), Some("numerical_failure"));
        assert_eq!(
            wrapped.stop_detail().as_deref(),
            Some("native evaluation failed")
        );
    }
}
