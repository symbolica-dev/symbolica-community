use std::{collections::BTreeMap, sync::Arc};

use fastsecdec::{Atom, EdgeId, Symbol, input::GraphIntegral};
use feynkit_py::{PyFeynmanDiagram, PyKinematics};
use pyo3::{prelude::*, types::PyDict};
use symbolica::{api::python::PythonExpression, atom::AtomView};

use super::error;

/// Native diagram, assumptions and explicit normalized loop measure.
#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(
    name = "Integral",
    module = "symbolica.community.hepkit.fastsecdec",
    frozen
)]
pub(crate) struct PyIntegral {
    pub(crate) graph: GraphIntegral,
    pub(crate) regulator: Symbol,
    pub(crate) dimension: Atom,
}

fn symbol(py: Python<'_>, value: &PythonExpression, name: &str) -> PyResult<Symbol> {
    match value.expr.as_view() {
        AtomView::Var(value) => Ok(value.get_symbol()),
        _ => Err(error::native(
            py,
            "input",
            format!("{name} must be a Symbolica symbol"),
        )),
    }
}

#[cfg_attr(feature = "python_stubgen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyIntegral {
    /// Kinematics retains its symbolic tensor dimension; dimension defaults to 4-2*regulator.
    #[new]
    #[pyo3(signature = (diagram, kinematics, *, regulator, dimension=None, powers=None, scalar_values=None, auxiliary_momenta=None, measure_multiplier=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        diagram: &PyFeynmanDiagram,
        kinematics: &PyKinematics,
        regulator: &PythonExpression,
        dimension: Option<&PythonExpression>,
        powers: Option<BTreeMap<usize, u32>>,
        scalar_values: Option<&Bound<'_, PyDict>>,
        auxiliary_momenta: Option<Vec<PythonExpression>>,
        measure_multiplier: Option<&PythonExpression>,
    ) -> PyResult<Self> {
        let regulator = symbol(py, regulator, "regulator")?;
        // Guard partial selected subgraphs before borrowing the native owner.
        let graph = GraphIntegral::new(
            Arc::new(diagram.as_diagram()?.clone()),
            kinematics.as_kinematics(),
        )
        .map_err(|e| error::native(py, "input", e))?;
        let mut bindings = BTreeMap::new();
        if let Some(values) = scalar_values {
            for (key, value) in values.iter() {
                let key = key.extract::<PyRef<'_, PythonExpression>>()?;
                let value = value.extract::<PyRef<'_, PythonExpression>>()?;
                bindings.insert(symbol(py, &key, "scalar binding key")?, value.expr.clone());
            }
        }
        let powers = powers
            .unwrap_or_default()
            .into_iter()
            .map(|(id, power)| (EdgeId(id), power))
            .collect();
        let momenta: Vec<_> = auxiliary_momenta
            .unwrap_or_default()
            .into_iter()
            .map(|v| v.expr)
            .collect();
        let graph = graph
            .with_auxiliary_external_momenta(&momenta)
            .and_then(|g| g.with_scalar_values(&bindings))
            .and_then(|g| g.with_powers(&powers))
            .map_err(|e| error::native(py, "input", e))?
            .with_measure_multiplier(measure_multiplier.map_or_else(Atom::one, |v| v.expr.clone()));
        Ok(Self {
            graph,
            regulator,
            dimension: dimension.map_or_else(
                || Atom::num(4) - Atom::num(2) * Atom::var(regulator),
                |v| v.expr.clone(),
            ),
        })
    }

    #[getter]
    fn regulator(&self) -> PythonExpression {
        PythonExpression {
            expr: Atom::var(self.regulator),
        }
    }
    #[getter]
    fn dimension(&self) -> PythonExpression {
        PythonExpression {
            expr: self.dimension.clone(),
        }
    }
    #[getter]
    fn powers(&self) -> Vec<(usize, u32)> {
        self.graph
            .propagator_edges()
            .iter()
            .zip(self.graph.powers())
            .map(|(e, p)| (e.0, *p))
            .collect()
    }
}
