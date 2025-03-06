use pyo3::{exceptions::PyRuntimeError, prelude::*};

use spenso::{
    network::TensorNetwork,
    parametric::MixedTensor,
    structure::{representation::Rep, AtomStructure},
    symbolica_utils::SerializableAtom,
};
use symbolica::{api::python::PythonExpression, atom::Atom};

use super::{structure::PossiblyIndexed, ModuleInit, Spensor};
use pyo3_stub_gen::derive::*;

#[gen_stub_pyclass(module = "symbolica_community.tensors")]
#[pyclass(name = "TensorNetwork", module = "symbolica_community.tensors")]
#[derive(Clone)]
/// A tensor network.
///
/// This class is a wrapper around the `TensorNetwork` class from the `spenso` crate.
/// Such a network is a graph representing the arithmetic operations between tensors.
/// In the most basic case, edges represent the contraction of indices.
pub struct SpensoNet {
    pub network: TensorNetwork<MixedTensor<f64, AtomStructure<Rep>>, Atom>,
}

impl ModuleInit for SpensoNet {
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<SpensoNet>()
    }

    fn append_to_symbolica(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.getattr("Expression")?
            .setattr("to_net", wrap_pyfunction!(python_to_tensor_network, m)?)
    }
}

#[pyfunction(name = "to_net")]
pub fn python_to_tensor_network(a: &Bound<'_, PythonExpression>) -> anyhow::Result<SpensoNet> {
    SpensoNet::from_expression(a)
}

pub type ParsingNet = TensorNetwork<MixedTensor<f64, AtomStructure<Rep>>, SerializableAtom>;

#[pymethods]
impl SpensoNet {
    #[new]
    pub fn from_expression(expr: &Bound<'_, PythonExpression>) -> anyhow::Result<SpensoNet> {
        Ok(SpensoNet {
            network: ParsingNet::try_from(expr.borrow().expr.as_view())?.map_scalar(|r| r.0),
        })
    }

    fn contract(&mut self) -> PyResult<()> {
        self.network
            .contract()
            .map_err(|s| PyRuntimeError::new_err(s.to_string()))
    }

    fn result(&self) -> PyResult<Spensor> {
        Ok(Spensor {
            tensor: self
                .network
                .result_tensor_smart()
                .map_err(|s| PyRuntimeError::new_err(s.to_string()))?
                .map_structure(PossiblyIndexed::from),
        })
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(self.network.rich_graph().dot())
    }
}
