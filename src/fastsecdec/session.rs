use std::rc::Rc;

use fastsecdec::{
    integration::{
        IntegrationProblem, Periodization, PublishedLattice, QmcSession, QmcSettings, QmcWorker,
        RuleSource,
    },
    kernel::{KernelSet, ReplayPolicy, ReplayState, WeightedEvaluationContext},
    results::{KernelResultManifest, ResultScope},
    status::{EvaluationDiagnostics, IntegrationMethod, StoppingReason},
};
use pyo3::{prelude::*, types::PyBytes};
use serde::{Deserialize, Serialize};

use super::{error, status::PyIntegrationSnapshot};

#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(
    name = "QmcSettings",
    module = "symbolica.community.hepkit.fastsecdec",
    from_py_object,
    frozen
)]
#[derive(Clone)]
pub(crate) struct PyQmcSettings {
    pub(crate) inner: QmcSettings,
}

#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyQmcSettings {
    /// Democratic QMC uses the native Kuo rule and retains full shift covariance.
    #[new]
    #[pyo3(signature = (*, points=4096, shifts=64, seed=0, package_points=1024, periodization="korobov3", rule="kuo_33002"))]
    #[allow(clippy::too_many_arguments)]
    fn py_new(
        py: Python<'_>,
        points: u64,
        shifts: u32,
        seed: u64,
        package_points: u64,
        periodization: &str,
        rule: &str,
    ) -> PyResult<Self> {
        let periodization = match periodization {
            "none" => Periodization::None,
            "korobov3" => Periodization::Korobov3,
            _ => {
                return Err(error::native(
                    py,
                    "configuration",
                    "periodization must be 'none' or 'korobov3'",
                ));
            }
        };
        let rule = match rule {
            "kuo_33002" => RuleSource::Kuo,
            "kuo_38005" => RuleSource::Published(PublishedLattice::Kuo38005),
            "kuo_39101" => RuleSource::Published(PublishedLattice::Kuo39101),
            "hkkn_alpha3" => RuleSource::Published(PublishedLattice::HkknAlpha3),
            _ => {
                return Err(error::native(
                    py,
                    "configuration",
                    "unknown published QMC rule",
                ));
            }
        };
        let inner = QmcSettings {
            points,
            shifts,
            seed,
            package_points,
            periodization,
            rule,
        };
        inner
            .validate()
            .map_err(|e| error::native(py, "configuration", e))?;
        Ok(Self { inner })
    }
    #[getter]
    fn points(&self) -> u64 {
        self.inner.points
    }
    #[getter]
    fn shifts(&self) -> u32 {
        self.inner.shifts
    }
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }
    #[getter]
    fn package_points(&self) -> u64 {
        self.inner.package_points
    }
    #[getter]
    fn periodization(&self) -> &'static str {
        match self.inner.periodization {
            Periodization::None => "none",
            Periodization::Korobov3 => "korobov3",
        }
    }
    #[getter]
    fn rule(&self) -> &'static str {
        match self.inner.rule {
            RuleSource::Kuo | RuleSource::Published(PublishedLattice::Kuo33002) => "kuo_33002",
            RuleSource::Published(PublishedLattice::Kuo38005) => "kuo_38005",
            RuleSource::Published(PublishedLattice::Kuo39101) => "kuo_39101",
            RuleSource::Published(PublishedLattice::HkknAlpha3) => "hkkn_alpha3",
            RuleSource::Supplied(_) => "supplied",
        }
    }
}

struct ActiveContext {
    sector: usize,
    worker: QmcWorker,
    context: WeightedEvaluationContext,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    version: u32,
    session: Vec<u8>,
    replay: Vec<ReplayState>,
    policy: ReplayPolicy,
    diagnostics: EvaluationDiagnostics,
    stop_reason: Option<StoppingReason>,
}

/// Caller-owned bounded execution, with no background loop or worker pool.
#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(
    name = "QmcSession",
    module = "symbolica.community.hepkit.fastsecdec",
    unsendable
)]
pub(crate) struct PyQmcSession {
    kernels: Rc<KernelSet>,
    session: QmcSession,
    replay: Vec<ReplayState>,
    policy: ReplayPolicy,
    active: Option<ActiveContext>,
    diagnostics: EvaluationDiagnostics,
    stop_reason: Option<StoppingReason>,
}

fn problem(py: Python<'_>, kernels: &KernelSet) -> PyResult<IntegrationProblem> {
    KernelResultManifest::from_kernels(kernels)
        .integration_problem(&ResultScope::FullIntegral, kernels.content_id())
        .map_err(|e| error::native(py, "configuration", e))
}

impl PyQmcSession {
    pub(crate) fn new(
        py: Python<'_>,
        kernels: Rc<KernelSet>,
        settings: QmcSettings,
    ) -> PyResult<Self> {
        let session = QmcSession::democratic(problem(py, &kernels)?, settings)
            .map_err(|e| error::native(py, "configuration", e))?;
        let policy = ReplayPolicy::default();
        let replay = (0..kernels.sectors().len())
            .map(|sector| kernels.replay_state(sector, policy.clone()))
            .collect::<Result<_, _>>()
            .map_err(|e| error::native(py, "configuration", e))?;
        Ok(Self {
            kernels,
            session,
            replay,
            policy,
            active: None,
            diagnostics: EvaluationDiagnostics::default(),
            stop_reason: None,
        })
    }

    pub(crate) fn restore_native(
        py: Python<'_>,
        kernels: Rc<KernelSet>,
        bytes: &[u8],
    ) -> PyResult<Self> {
        let state: Checkpoint =
            serde_json::from_slice(bytes).map_err(|e| error::native(py, "checkpoint", e))?;
        if state.version != 1 || state.replay.len() != kernels.sectors().len() {
            return Err(error::native(
                py,
                "checkpoint",
                "checkpoint version or replay sector count differs",
            ));
        }
        let session = QmcSession::restore(&state.session, &problem(py, &kernels)?)
            .map_err(|e| error::native(py, "checkpoint", e))?;
        if session.method() != IntegrationMethod::DemocraticQmc {
            return Err(error::native(
                py,
                "checkpoint",
                "this bridge supports democratic QMC checkpoints",
            ));
        }
        for (sector, replay) in state.replay.iter().enumerate() {
            kernels
                .validate_replay_state(sector, &state.policy, replay)
                .map_err(|e| error::native(py, "checkpoint", e))?;
        }
        Ok(Self {
            kernels,
            session,
            replay: state.replay,
            policy: state.policy,
            diagnostics: state.diagnostics,
            active: None,
            stop_reason: state.stop_reason,
        })
    }

    fn prepare(&mut self, py: Python<'_>, sector: usize) -> PyResult<()> {
        if self
            .active
            .as_ref()
            .is_none_or(|active| active.sector != sector)
        {
            // Release the prior large evaluator before making its replacement.
            self.active = None;
            let worker = self
                .session
                .worker_context(sector as u64)
                .map_err(|e| error::native(py, "integration", e))?;
            let context = self
                .kernels
                .restore_evaluation_context(sector, self.policy.clone(), &self.replay[sector])
                .map_err(|e| error::native(py, "integration", e))?;
            self.active = Some(ActiveContext {
                sector,
                worker,
                context,
            });
        }
        Ok(())
    }
}

#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyQmcSession {
    #[getter]
    fn complete(&self) -> bool {
        self.session.is_complete()
    }
    #[getter]
    fn settings(&self) -> PyQmcSettings {
        PyQmcSettings {
            inner: self.session.design().settings,
        }
    }

    fn snapshot(&self, py: Python<'_>) -> PyResult<PyIntegrationSnapshot> {
        let mut inner = self
            .session
            .diagnostic_observation()
            .map_err(|e| error::native(py, "integration", e))?
            .snapshot;
        inner.evaluation_diagnostics = Some(self.diagnostics.clone());
        inner.stop_reason = self.stop_reason.clone();
        Ok(PyIntegrationSnapshot { inner })
    }

    /// Execute at most max_packages, returning to the Python caller between steps.
    /// An observer receives immutable snapshots after accepted packages; False stops this call.
    #[pyo3(signature = (max_packages=1, *, observer=None))]
    fn step(
        &mut self,
        py: Python<'_>,
        max_packages: usize,
        observer: Option<Py<PyAny>>,
    ) -> PyResult<PyIntegrationSnapshot> {
        if max_packages == 0 {
            return Err(error::native(
                py,
                "configuration",
                "max_packages must be positive",
            ));
        }
        self.stop_reason = None;
        for _ in 0..max_packages {
            if let Err(e) = py.check_signals() {
                if e.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py) {
                    self.stop_reason = Some(StoppingReason::Cancelled);
                }
                return Err(e);
            }
            let Some(task) = self
                .session
                .next_work()
                .map_err(|e| error::native(py, "integration", e))?
            else {
                break;
            };
            let sector = task.sector_id() as usize;
            let mut signal_interrupted = false;
            let result = (|| {
                self.prepare(py, sector)?;
                let active = self.active.as_mut().expect("prepared context");
                let mut interrupted = None;
                let mut calls = 0usize;
                let value =
                    active
                        .worker
                        .evaluate_weighted(task.clone(), |point, weight, output| {
                            if calls.is_multiple_of(256)
                                && let Err(e) = py.check_signals()
                            {
                                interrupted = Some(e);
                                return Err("Python signal interrupted package".to_owned());
                            }
                            calls = calls.wrapping_add(1);
                            match active.context.evaluate_weighted(point, weight, output) {
                                Ok(report) => self
                                    .diagnostics
                                    .record_replay(report)
                                    .map_err(|e| e.to_string()),
                                Err(e) => {
                                    self.diagnostics
                                        .record_failure()
                                        .map_err(|e| e.to_string())?;
                                    Err(e.to_string())
                                }
                            }
                        });
                if let Some(e) = interrupted {
                    signal_interrupted = true;
                    return Err(e);
                }
                let value = value.map_err(|e| error::native(py, "integration", e))?;
                // Validate replay metadata before either accepted mutation.
                let mut accepted = self.replay[sector].clone();
                accepted
                    .merge(active.context.state())
                    .map_err(|e| error::native(py, "integration", e))?;
                self.session
                    .submit(value)
                    .map_err(|e| error::native(py, "integration", e))?;
                self.replay[sector] = accepted;
                Ok(())
            })();
            if let Err(e) = result {
                self.active = None;
                self.session
                    .retry(&task)
                    .map_err(|retry| error::native(py, "integration", retry))?;
                self.stop_reason = if e.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py)
                {
                    Some(StoppingReason::Cancelled)
                } else if signal_interrupted {
                    None
                } else {
                    Some(StoppingReason::NumericalFailure(e.to_string()))
                };
                return Err(e);
            }
            if let Some(observer) = &observer {
                let result = match observer.call1(py, (self.snapshot(py)?,)) {
                    Ok(value) => value,
                    Err(e) => {
                        if e.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py) {
                            self.stop_reason = Some(StoppingReason::Cancelled);
                        }
                        return Err(e);
                    }
                };
                if !result.is_none(py) && !result.extract::<bool>(py)? {
                    self.stop_reason = Some(StoppingReason::Cancelled);
                    break;
                }
            }
        }
        if self.session.is_complete() && self.stop_reason.is_none() {
            self.stop_reason = Some(StoppingReason::PlannedWorkComplete);
        }
        self.snapshot(py)
    }

    fn checkpoint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let state = Checkpoint {
            version: 1,
            session: self
                .session
                .checkpoint()
                .map_err(|e| error::native(py, "checkpoint", e))?,
            replay: self.replay.clone(),
            policy: self.policy.clone(),
            diagnostics: self.diagnostics.clone(),
            stop_reason: self.stop_reason.clone(),
        };
        let bytes = serde_json::to_vec(&state).map_err(|e| error::native(py, "checkpoint", e))?;
        Ok(PyBytes::new(py, &bytes))
    }
}
