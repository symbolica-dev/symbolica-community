use std::collections::HashMap;

use anyhow::anyhow;
use network::SpensoNet;
use pyo3::{
    conversion::FromPyObject,
    exceptions::{self, PyIndexError, PyOverflowError, PyRuntimeError, PyTypeError},
    prelude::*,
    types::{PyComplex, PyFloat, PySlice, PyType},
    IntoPyObjectExt, PyClass,
};

use spenso::{
    complex::{RealOrComplex, RealOrComplexTensor},
    data::{
        DataTensor, DenseTensor, GetTensorData, SetTensorData, SparseOrDense, SparseTensor,
        StorageTensor,
    },
    parametric::{
        atomcore::TensorAtomOps, CompiledEvalTensor, ConcreteOrParam, LinearizedEvalTensor,
        MixedTensor, ParamOrConcrete, ParamTensor,
    },
    shadowing::{ExplicitKey, EXPLICIT_TENSOR_MAP},
    structure::{representation::Rep, AtomStructure, HasStructure, TensorStructure},
};
use structure::{PossiblyIndexed, SpensoIndices};
use symbolica::{
    api::python::PythonExpression,
    atom::Atom,
    domains::float::Complex,
    evaluate::{CompileOptions, FunctionMap, InlineASM, OptimizationSettings},
    poly::Variable,
};

use pyo3_stub_gen::{define_stub_info_gatherer, derive::*};

pub mod network;
pub mod structure;

trait ModuleInit: PyClass {
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<Self>()
    }

    fn append_to_symbolica(_m: &Bound<'_, PyModule>) -> PyResult<()> {
        Ok(())
    }
}

pub(crate) fn initialize_spenso(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let child_module = PyModule::new(m.py(), "tensors")?;

    SpensoNet::init(&child_module)?;
    SpensoNet::append_to_symbolica(m)?;
    Spensor::init(&child_module)?;
    Spensor::append_to_symbolica(m)?;
    SpensoIndices::init(&child_module)?;
    SpensoIndices::append_to_symbolica(m)?;
    m.add_submodule(&child_module)?;

    m.py()
        .import("sys")?
        .getattr("modules")?
        .set_item("symbolica_community.tensors", child_module)
}

/// A tensor class that can be either dense or sparse.
/// The data is either float or complex or a symbolica expression
/// It can be instantiated with data using the `sparse_empty` or `dense` module functions.
#[gen_stub_pyclass(module = "symbolica_community.tensors")]
#[pyclass(name = "Tensor", module = "symbolica_community.tensors")]
#[derive(Clone)]
pub struct Spensor {
    tensor: MixedTensor<f64, PossiblyIndexed>,
}

impl ModuleInit for Spensor {
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<Self>()?;
        m.add_function(wrap_pyfunction!(sparse_empty, m)?)?;
        m.add_function(wrap_pyfunction!(dense, m)?)?;
        m.add_function(wrap_pyfunction!(register, m)?)
    }
}

/// Create a new sparse empty tensor with the given structure and type.
/// The structure can be a list of integers, a list of representations, or a list of slots.
/// In the first two cases, no "indices" are assumed, and thus the tensor is indexless (i.e.) it has a shape but no proper way to contract it.
/// The structure can also be a proper `TensorIndices` object or `TensorStructure` object.
///
/// The type is either a float or a symbolica expression.
///
#[gen_stub_pyfunction(module = "symbolica_community.tensors")]
#[pyfunction]
pub fn sparse_empty(
    structure: Bound<'_, PyAny>,
    type_info: Bound<'_, PyType>,
) -> PyResult<Spensor> {
    let structure = PossiblyIndexed::extract_bound(&structure)?;

    if type_info.is_subclass_of::<PyFloat>()? {
        Ok(Spensor {
            tensor: SparseTensor::<f64, _>::empty(structure).into(),
        })
    } else if type_info.is_subclass_of::<PythonExpression>()? {
        Ok(Spensor {
            tensor: ParamOrConcrete::Param(ParamTensor::from(SparseTensor::<Atom, _>::empty(
                structure,
            ))),
        })
    } else {
        Err(PyTypeError::new_err("Only float type supported"))
    }
}

/// Create a new dense tensor with the given structure and data.
/// The structure can be a list of integers, a list of representations, or a list of slots.
/// In the first two cases, no "indices" are assumed, and thus the tensor is indexless (i.e.) it has a shape but no proper way to contract it.
/// The structure can also be a proper `TensorIndices` object or `TensorStructure` object.
///
/// The data is either a list of floats or a list of symbolica expressions, of length equal to the number of elements in the structure, in row-major order.
#[gen_stub_pyfunction(module = "symbolica_community.tensors")]
#[pyfunction]
pub fn dense(structure: Bound<'_, PyAny>, data: Bound<'_, PyAny>) -> PyResult<Spensor> {
    let structure = PossiblyIndexed::extract_bound(&structure)?;

    if let Ok(d) = data.extract::<Vec<f64>>() {
        Ok(Spensor {
            tensor: DenseTensor::<f64, _>::from_data(d, structure)
                .map_err(|e| PyOverflowError::new_err(e.to_string()))?
                .into(),
        })
    } else if let Ok(d) = data.extract::<Vec<PythonExpression>>() {
        let data = d.into_iter().map(|e| e.expr).collect();
        Ok(Spensor {
            tensor: ParamOrConcrete::Param(ParamTensor::from(
                DenseTensor::<Atom, _>::from_data(data, structure)
                    .map_err(|e| PyOverflowError::new_err(e.to_string()))?,
            )),
        })
    } else {
        Err(PyTypeError::new_err("Only float type supported"))
    }
}

#[gen_stub_pyfunction(module = "symbolica_community.tensors")]
#[pyfunction]
pub fn register(tensor: Spensor) -> PyResult<()> {
    EXPLICIT_TENSOR_MAP.write().unwrap().insert_explicit(
        tensor
            .tensor
            .clone()
            .map_structure_fallible(ExplicitKey::try_from)
            .map_err(|s| PyTypeError::new_err(s.to_string()))?,
    );
    Ok(())
}

#[derive(FromPyObject)]
pub enum SliceOrIntOrExpanded<'a> {
    Slice(Bound<'a, PySlice>),
    Int(usize),
    Expanded(Vec<usize>),
}

pub enum TensorElements {
    Real(Py<PyFloat>),
    Complex(Py<PyComplex>),
    Symbolica(PythonExpression),
}

impl<'py> IntoPyObject<'py> for TensorElements {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            TensorElements::Real(f) => f.into_bound_py_any(py),
            TensorElements::Complex(c) => c.into_bound_py_any(py),
            TensorElements::Symbolica(s) => s.into_bound_py_any(py),
        }
    }
}

impl From<ConcreteOrParam<RealOrComplex<f64>>> for TensorElements {
    fn from(value: ConcreteOrParam<RealOrComplex<f64>>) -> Self {
        match value {
            ConcreteOrParam::Concrete(RealOrComplex::Real(f)) => {
                TensorElements::Real(Python::with_gil(|py| {
                    PyFloat::new(py, f).as_unbound().to_owned()
                }))
            }
            ConcreteOrParam::Concrete(RealOrComplex::Complex(c)) => {
                TensorElements::Complex(Python::with_gil(|py| {
                    PyComplex::from_doubles(py, c.re, c.im)
                        .as_unbound()
                        .to_owned()
                }))
            }
            ConcreteOrParam::Param(p) => TensorElements::Symbolica(PythonExpression::from(p)),
        }
    }
}

// #[gen_stub_pymethods]
#[pymethods]
impl Spensor {
    pub fn structure<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        match self.tensor.structure() {
            PossiblyIndexed::Indexed(a) => a.clone().into_py_any(py),
            PossiblyIndexed::Unindexed(a) => a.clone().into_py_any(py),
        }
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_dense(&mut self) {
        self.tensor = self.tensor.clone().to_dense();
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_sparse(&mut self) {
        self.tensor = self.tensor.clone().to_sparse();
    }

    fn __repr__(&self) -> String {
        format!("Spensor(\n{})", self.tensor)
    }

    fn __str__(&self) -> String {
        format!("{}", self.tensor)
    }

    fn __len__(&self) -> usize {
        self.tensor.size().unwrap()
    }

    fn __getitem__<'py>(&self, item: SliceOrIntOrExpanded, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let out = match item {
            SliceOrIntOrExpanded::Int(i) => self
                .tensor
                .get_owned_linear(i.into())
                .ok_or(PyIndexError::new_err("flat index out of bounds"))?,
            SliceOrIntOrExpanded::Expanded(idxs) => self
                .tensor
                .get_owned(&idxs)
                .map_err(|s| PyIndexError::new_err(s.to_string()))?,
            SliceOrIntOrExpanded::Slice(s) => {
                let r = s.indices(self.tensor.size().unwrap() as isize)?;

                let start = if r.start < 0 {
                    (r.slicelength as isize + r.start) as usize
                } else {
                    r.start as usize
                };

                let end = if r.stop < 0 {
                    (r.slicelength as isize + r.stop) as usize
                } else {
                    r.stop as usize
                };

                let (range, step) = if r.step < 0 {
                    (end..start, -r.step as usize)
                } else {
                    (start..end, r.step as usize)
                };

                let slice: Option<Vec<TensorElements>> = range
                    .step_by(step)
                    .map(|i| {
                        self.tensor
                            .get_owned_linear(i.into())
                            .map(TensorElements::from)
                    })
                    .collect();

                if let Some(slice) = slice {
                    return slice.into_py_any(py);
                } else {
                    return Err(PyIndexError::new_err("slice out of bounds"));
                }
            }
        };

        TensorElements::from(out).into_py_any(py)
    }

    fn __setitem__<'py>(
        &mut self,
        item: Bound<'py, PyAny>,
        value: Bound<'py, PyAny>,
    ) -> anyhow::Result<()> {
        let value = if let Ok(v) = value.extract::<PythonExpression>() {
            ConcreteOrParam::Param(v.expr)
        } else if let Ok(v) = value.extract::<f64>() {
            ConcreteOrParam::Concrete(RealOrComplex::Real(v))
        } else {
            return Err(anyhow!("Value must be a PythonExpression or a float"));
        };

        if let Ok(flat_index) = item.extract::<usize>() {
            self.tensor.set_flat(flat_index.into(), value)
        } else if let Ok(expanded_idxs) = item.extract::<Vec<usize>>() {
            self.tensor.set(&expanded_idxs, value)
        } else {
            Err(anyhow!("Index must be an integer"))
        }
    }

    #[pyo3(signature =
           (constants,
           funs,
           params,
           iterations = 100,
           n_cores = 4,
           verbose = false),
           )]
    pub fn evaluator(
        &self,
        constants: HashMap<PythonExpression, PythonExpression>,
        funs: HashMap<(Variable, String, Vec<Variable>), PythonExpression>,
        params: Vec<PythonExpression>,
        iterations: usize,
        n_cores: usize,
        verbose: bool,
    ) -> PyResult<SpensoExpressionEvaluator> {
        let mut fn_map = FunctionMap::new();

        for (k, v) in constants {
            if let Ok(r) = v.expr.clone().try_into() {
                fn_map.add_constant(k.expr, r);
            } else {
                Err(exceptions::PyValueError::new_err(
                               "Constants must be rationals. If this is not possible, pass the value as a parameter",
                           ))?
            }
        }

        for ((symbol, rename, args), body) in funs {
            let symbol = symbol
                .to_id()
                .ok_or(exceptions::PyValueError::new_err(format!(
                    "Bad function name {}",
                    symbol
                )))?;
            let args: Vec<_> = args
                .iter()
                .map(|x| {
                    x.to_id().ok_or(exceptions::PyValueError::new_err(format!(
                        "Bad function name {}",
                        symbol
                    )))
                })
                .collect::<Result<_, _>>()?;

            fn_map
                .add_function(symbol, rename.clone(), args, body.expr)
                .map_err(|e| {
                    exceptions::PyValueError::new_err(format!("Could not add function: {}", e))
                })?;
        }

        let settings = OptimizationSettings {
            horner_iterations: iterations,
            n_cores,
            verbose,
            ..OptimizationSettings::default()
        };

        let params: Vec<_> = params.iter().map(|x| x.expr.clone()).collect();

        let mut evaltensor = match &self.tensor {
            ParamOrConcrete::Param(s) => s.to_evaluation_tree(&fn_map, &params).map_err(|e| {
                exceptions::PyValueError::new_err(format!("Could not create evaluator: {}", e))
            })?,
            ParamOrConcrete::Concrete(_) => return Err(PyRuntimeError::new_err("not atom")),
        };

        evaltensor.optimize_horner_scheme(
            settings.horner_iterations,
            settings.n_cores,
            settings.hot_start.clone(),
            settings.verbose,
        );

        evaltensor.common_subexpression_elimination();
        let linear = evaltensor.linearize(None);
        Ok(SpensoExpressionEvaluator {
            eval: linear.map_coeff(&|x| x.to_f64()),
        })
    }

    fn scalar(&self) -> PyResult<PythonExpression> {
        self.clone()
            .tensor
            .scalar()
            .map(|r| PythonExpression { expr: r.into() })
            .ok_or_else(|| PyRuntimeError::new_err("No scalar found"))
    }
}

impl From<DataTensor<f64, AtomStructure<Rep>>> for Spensor {
    fn from(value: DataTensor<f64, AtomStructure<Rep>>) -> Self {
        Spensor {
            tensor: MixedTensor::Concrete(RealOrComplexTensor::Real(
                value.map_structure(PossiblyIndexed::from),
            )),
        }
    }
}

impl From<DataTensor<f64, PossiblyIndexed>> for Spensor {
    fn from(value: DataTensor<f64, PossiblyIndexed>) -> Self {
        Spensor {
            tensor: MixedTensor::Concrete(RealOrComplexTensor::Real(value)),
        }
    }
}

impl From<DataTensor<Complex<f64>, AtomStructure<Rep>>> for Spensor {
    fn from(value: DataTensor<Complex<f64>, AtomStructure<Rep>>) -> Self {
        Spensor {
            tensor: MixedTensor::Concrete(RealOrComplexTensor::Complex(
                value
                    .map_structure(PossiblyIndexed::from)
                    .map_data(|c| c.into()),
            )),
        }
    }
}

impl From<DataTensor<Complex<f64>, PossiblyIndexed>> for Spensor {
    fn from(value: DataTensor<Complex<f64>, PossiblyIndexed>) -> Self {
        Spensor {
            tensor: MixedTensor::Concrete(RealOrComplexTensor::Complex(
                value.map_data(|c| c.into()),
            )),
        }
    }
}
/// An optimized evaluator for tensors.
///
#[gen_stub_pyclass(module = "symbolica_community.tensors")]
#[pyclass(name = "TensorEvaluator", module = "symbolica_community.tensors")]
#[derive(Clone)]
pub struct SpensoExpressionEvaluator {
    pub eval: LinearizedEvalTensor<f64, PossiblyIndexed>,
}

#[pymethods]
impl SpensoExpressionEvaluator {
    /// Evaluate the expression for multiple inputs and return the results.
    fn evaluate(&mut self, inputs: Vec<Vec<f64>>) -> Vec<Spensor> {
        inputs
            .iter()
            .map(|s| self.eval.evaluate(s).into())
            .collect()
    }

    /// Evaluate the expression for multiple inputs and return the results.
    fn evaluate_complex(&mut self, inputs: Vec<Vec<Complex<f64>>>) -> Vec<Spensor> {
        let mut eval = self.eval.clone().map_coeff(&|x| Complex::new(*x, 0.));

        inputs.iter().map(|s| eval.evaluate(s).into()).collect()
    }

    /// Compile the evaluator to a shared library using C++ and optionally inline assembly and load it.
    #[pyo3(signature =
        (function_name,
        filename,
        library_name,
        inline_asm = true,
        optimization_level = 3,
        compiler_path = None,
    ))]
    fn compile(
        &self,
        function_name: &str,
        filename: &str,
        library_name: &str,
        inline_asm: bool,
        optimization_level: u8,
        compiler_path: Option<&str>,
    ) -> PyResult<SpensoCompiledExpressionEvaluator> {
        let mut options = CompileOptions {
            optimization_level: optimization_level as usize,
            ..Default::default()
        };

        if let Some(compiler_path) = compiler_path {
            options.compiler = compiler_path.to_string();
        }

        Ok(SpensoCompiledExpressionEvaluator {
            eval: self
                .eval
                .export_cpp(
                    filename,
                    function_name,
                    true,
                    if inline_asm {
                        InlineASM::X64
                    } else {
                        InlineASM::None
                    },
                )
                .map_err(|e| exceptions::PyValueError::new_err(format!("Export error: {}", e)))?
                .compile(library_name, options)
                .map_err(|e| {
                    exceptions::PyValueError::new_err(format!("Compilation error: {}", e))
                })?
                .load()
                .map_err(|e| {
                    exceptions::PyValueError::new_err(format!("Library loading error: {}", e))
                })?,
        })
    }
}

/// A compiled and optimized evaluator for tensors.
///
#[gen_stub_pyclass(module = "symbolica_community.tensors")]
#[pyclass(
    name = "CompiledTensorEvaluator",
    module = "symbolica_community.tensors"
)]
#[derive(Clone)]
pub struct SpensoCompiledExpressionEvaluator {
    pub eval: CompiledEvalTensor<PossiblyIndexed>,
}

#[pymethods]
impl SpensoCompiledExpressionEvaluator {
    /// Evaluate the expression for multiple inputs and return the results.
    fn evaluate(&mut self, inputs: Vec<Vec<f64>>) -> Vec<Spensor> {
        inputs
            .iter()
            .map(|s| self.eval.evaluate(s).into())
            .collect()
    }

    /// Evaluate the expression for multiple inputs and return the results.
    fn evaluate_complex(&mut self, inputs: Vec<Vec<Complex<f64>>>) -> Vec<Spensor> {
        inputs
            .iter()
            .map(|s| self.eval.evaluate(s).into())
            .collect()
    }
}

define_stub_info_gatherer!(stub_info);
