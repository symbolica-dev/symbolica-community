use delegate::delegate;
use itertools::Itertools;
use pyo3::{
    exceptions::{self, PyIndexError, PyRuntimeError, PyTypeError},
    prelude::*,
    pybacked::PyBackedStr,
    types::PyTuple,
    IntoPyObjectExt,
};
use spenso::{
    shadowing::ExplicitKey,
    structure::{
        abstract_index::AbstractIndex,
        dimension::Dimension,
        representation::{ExtendibleReps, Rep, RepName, Representation},
        slot::{IsAbstractSlot, Slot},
        AtomStructure, HasName, IndexLess, IndexlessNamedStructure, StructureContract,
        TensorStructure, ToSymbolic, VecStructure,
    },
    symbolica_utils::{SerializableAtom, SerializableSymbol},
};
use symbolica::{api::python::PythonExpression, atom::AtomView, symbol};
use thiserror::Error;

use super::{ModuleInit, SliceOrIntOrExpanded};
use auto_enums::auto_enum;
use pyo3_stub_gen::derive::*;

#[gen_stub_pyclass(module = "symbolica_community.tensors")]
#[pyclass(name = "TensorIndices", module = "symbolica_community.tensors")]
#[derive(Clone)]
/// A structure that can be used to represent the "shape" of a tensor, along with a list of abstract indices.
/// This has an optional name, and accompanying symbolica expressions that are considered as additional non-indexed arguments.
/// The structure is essentially a list of `Slots` that are used to define the structure of the tensor.
pub struct SpensoIndices {
    pub structure: AtomStructure<Rep>,
}

impl From<AtomStructure<Rep>> for SpensoIndices {
    fn from(value: AtomStructure<Rep>) -> Self {
        SpensoIndices { structure: value }
    }
}

impl ModuleInit for SpensoIndices {
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<SpensoIndices>()?;
        m.add_class::<SpensoSlot>()?;
        m.add_class::<SpensoStucture>()?;
        m.add_class::<SpensoRepresentation>()?;
        Ok(())
    }
}

#[pymethods]
impl SpensoIndices {
    #[new]
    #[pyo3(signature =
           (name,
           *additional_args))]
    pub fn from_list(
        name: PythonExpression,
        additional_args: &Bound<'_, PyTuple>,
    ) -> PyResult<Self> {
        let mut args = Vec::new();
        let mut slots = Vec::new();
        for a in additional_args {
            if let Ok(s) = a.extract::<SpensoSlot>() {
                slots.push(s.slot);
            } else if let Ok(arg) = a.extract::<PythonExpression>() {
                args.push(arg.expr.into());
            } else {
                return Err(exceptions::PyTypeError::new_err(
                    "Only slots and expressions can be used",
                ));
            }
        }

        let args = if args.is_empty() { None } else { Some(args) };

        let id = match name.expr.as_view() {
            AtomView::Var(v) => v.get_symbol(),
            _ => {
                return Err(exceptions::PyTypeError::new_err(
                    "Only symbols can be used as names",
                ))
            }
        };

        Ok(SpensoIndices {
            structure: AtomStructure::<Rep>::from_iter(slots, id.into(), args),
        })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.structure)
    }

    fn __str__(&self) -> String {
        if let Some(atom) = self.structure.to_symbolic() {
            format!("{}", atom)
        } else {
            let args = self
                .structure
                .external_structure_iter()
                .map(|r| r.to_atom())
                .join(",");

            format!("({})", args.trim_end())
        }
    }

    fn to_expression(&self) -> PyResult<PythonExpression> {
        Ok(self
            .structure
            .to_symbolic()
            .ok_or(PyRuntimeError::new_err("No name"))?
            .into())
    }

    fn __len__(&self) -> usize {
        self.structure.size().unwrap()
    }

    fn __getitem__<'py>(&self, item: SliceOrIntOrExpanded, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match item {
            SliceOrIntOrExpanded::Int(i) => {
                let out: Vec<_> = self
                    .structure
                    .expanded_index(i.into())
                    .map_err(|s| PyIndexError::new_err(s.to_string()))?
                    .into();

                out.into_py_any(py)
            }
            SliceOrIntOrExpanded::Expanded(idxs) => {
                let out: usize = self
                    .structure
                    .flat_index(&idxs)
                    .map_err(|s| PyIndexError::new_err(s.to_string()))?
                    .into();

                out.into_py_any(py)
            }
            SliceOrIntOrExpanded::Slice(s) => {
                let r = s.indices(self.structure.size().unwrap() as isize)?;

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

                let slice: Result<Vec<Vec<usize>>, _> = range
                    .step_by(step)
                    .map(|i| {
                        self.structure
                            .expanded_index(i.into())
                            .map(Vec::<usize>::from)
                    })
                    .collect();

                match slice {
                    Ok(slice) => slice.into_py_any(py),
                    Err(e) => Err(PyIndexError::new_err(e.to_string())),
                }
            }
        }
    }
}

#[gen_stub_pyclass(module = "symbolica_community.tensors")]
#[pyclass(name = "TensorStructure", module = "symbolica_community.tensors")]
#[derive(Clone)]
/// A structure that can be used to represent the "shape" of a tensor.
/// This has an optional name, and accompanying symbolica expressions that are considered as additional non-indexed arguments.
/// The structure is essentially a list of `Representation` that are used to define the structure of the tensor.
pub struct SpensoStucture {
    pub structure: IndexlessNamedStructure<SerializableSymbol, Vec<SerializableAtom>, Rep>,
}

impl From<IndexlessNamedStructure<SerializableSymbol, Vec<SerializableAtom>, Rep>>
    for SpensoStucture
{
    fn from(
        value: IndexlessNamedStructure<SerializableSymbol, Vec<SerializableAtom>, Rep>,
    ) -> Self {
        SpensoStucture { structure: value }
    }
}

#[derive(Clone)]
pub enum PossiblyIndexed {
    Unindexed(SpensoStucture),
    Indexed(SpensoIndices),
}

impl<'py> FromPyObject<'py> for PossiblyIndexed {
    fn extract_bound(structure: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(structure) = structure.extract::<SpensoIndices>() {
            Ok(PossiblyIndexed::from(structure))
        } else if let Ok(structure) = structure.extract::<SpensoStucture>() {
            Ok(PossiblyIndexed::from(structure))
        } else if let Ok(s) = structure.extract::<Vec<SpensoSlot>>() {
            Ok(PossiblyIndexed::Indexed(SpensoIndices {
                structure: VecStructure::from_iter(s.into_iter().map(|s| s.slot)).into(),
            }))
        } else if let Ok(s) = structure.extract::<Vec<SpensoRepresentation>>() {
            Ok(PossiblyIndexed::Unindexed(SpensoStucture {
                structure: IndexLess::from_iter(s.into_iter().map(|s| s.representation)).into(),
            }))
        } else if let Ok(s) = structure.extract::<Vec<usize>>() {
            Ok(PossiblyIndexed::Unindexed(SpensoStucture {
                structure: IndexLess::from_iter(
                    s.into_iter().map(|s| ExtendibleReps::EUCLIDEAN.new_rep(s)),
                )
                .into(),
            }))
        } else {
            Err(PyTypeError::new_err("Internal tensor structure can only be build from TensorIndices, TensorStructure, lists of Representations or of Slots"))
        }
    }
}

impl From<SpensoIndices> for PossiblyIndexed {
    fn from(s: SpensoIndices) -> Self {
        PossiblyIndexed::Indexed(s)
    }
}

impl From<SpensoStucture> for PossiblyIndexed {
    fn from(s: SpensoStucture) -> Self {
        PossiblyIndexed::Unindexed(s)
    }
}
impl From<IndexlessNamedStructure<SerializableSymbol, Vec<SerializableAtom>, Rep>>
    for PossiblyIndexed
{
    fn from(s: IndexlessNamedStructure<SerializableSymbol, Vec<SerializableAtom>, Rep>) -> Self {
        PossiblyIndexed::Unindexed(s.into())
    }
}

impl From<AtomStructure<Rep>> for PossiblyIndexed {
    fn from(s: AtomStructure<Rep>) -> Self {
        PossiblyIndexed::Indexed(s.into())
    }
}

impl TensorStructure for PossiblyIndexed {
    type Slot = Slot<Rep>;

    fn dual(self) -> Self {
        match self {
            Self::Indexed(i) => Self::Indexed(SpensoIndices {
                structure: i.structure.dual(),
            }),
            Self::Unindexed(i) => Self::Unindexed(SpensoStucture {
                structure: i.structure.dual(),
            }),
        }
    }
    delegate! {
        to match self {
            PossiblyIndexed::Indexed(u) => u.structure,
            PossiblyIndexed::Unindexed(u) => u.structure,
        }{
            #[auto_enum(Iterator)]
            fn external_structure_iter(&self) -> impl Iterator<Item =  Slot<Rep>>;
            #[auto_enum(Iterator)]
            fn external_dims_iter(&self) -> impl Iterator<Item = Dimension>;
            #[auto_enum(Iterator)]
            fn external_reps_iter(
                &self,
            ) -> impl Iterator<Item = Representation<Rep>>;
            #[auto_enum(Iterator)]
            fn external_indices_iter(&self) -> impl Iterator<Item = AbstractIndex>;
            fn get_aind(&self, i: usize) -> Option<AbstractIndex>;
            fn get_rep(&self, i: usize) -> Option<Representation<Rep>>;
            fn get_dim(&self, i: usize) -> Option<Dimension>;
            fn get_slot(&self, i: usize) -> Option<Slot<Rep>>;

            fn order(&self) -> usize;
        }
    }
}

impl HasName for PossiblyIndexed {
    type Name = SerializableSymbol;
    type Args = Vec<SerializableAtom>;

    delegate! {
        to match self {
            PossiblyIndexed::Indexed(i) => i.structure,
            PossiblyIndexed::Unindexed(u) => u.structure,
        }{
            fn name(&self) -> Option<Self::Name>;
            fn args(&self) -> Option<Self::Args>;
            fn set_name(&mut self, name: Self::Name);
        }
    }
}

impl StructureContract for PossiblyIndexed {
    #[must_use]
    fn merge_at(&self, other: &Self, positions: (usize, usize)) -> Self {
        if let PossiblyIndexed::Indexed(i) = self {
            if let PossiblyIndexed::Indexed(j) = other {
                return i.structure.merge_at(&j.structure, positions).into();
            }
        }

        panic!("Cannot merge indexed and unindexed structures")
    }

    fn merge(&mut self, other: &Self) -> Option<usize> {
        if let PossiblyIndexed::Indexed(i) = self {
            if let PossiblyIndexed::Indexed(j) = other {
                return i.structure.merge(&j.structure);
            }
        }

        panic!("Cannot merge indexed and unindexed structures")
    }

    fn trace(&mut self, i: usize, j: usize) {
        match self {
            PossiblyIndexed::Indexed(s) => s.structure.trace(i, j),
            PossiblyIndexed::Unindexed(_) => panic!("Cannot trace unindexed structure"),
        }
    }

    fn trace_out(&mut self) {
        match self {
            PossiblyIndexed::Indexed(s) => s.structure.trace_out(),
            PossiblyIndexed::Unindexed(_) => panic!("Cannot trace uninidexed structure"),
        }
    }

    fn concat(&mut self, other: &Self) {
        if let PossiblyIndexed::Indexed(i) = self {
            if let PossiblyIndexed::Indexed(j) = other {
                i.structure.concat(&j.structure);
                return;
            }
        }

        panic!("Cannot concatenate indexed and unindexed structures")
    }
}

#[derive(Error, Debug)]
pub enum SpensoError {
    #[error("Must have a name to register")]
    NoName,
}

impl TryFrom<PossiblyIndexed> for ExplicitKey {
    type Error = SpensoError;
    fn try_from(s: PossiblyIndexed) -> Result<Self, Self::Error> {
        match s {
            PossiblyIndexed::Indexed(i) => Ok(ExplicitKey::from_iter(
                i.structure.external_reps_iter(),
                i.structure.name().ok_or(SpensoError::NoName)?.into(),
                i.structure
                    .args()
                    .map(|a| a.into_iter().map(|a| a.into()).collect()),
            )),
            PossiblyIndexed::Unindexed(i) => Ok(ExplicitKey::from_iter(
                i.structure.external_reps_iter(),
                i.structure.name().ok_or(SpensoError::NoName)?.into(),
                i.structure
                    .args()
                    .map(|a| a.into_iter().map(|a| a.into()).collect()),
            )),
        }
    }
}

#[pymethods]
impl SpensoStucture {
    #[new]
    #[pyo3(signature =
           (
           *additional_args,name=None))]
    pub fn from_list(
        additional_args: &Bound<'_, PyTuple>,
        name: Option<PythonExpression>,
    ) -> PyResult<Self> {
        let mut args = Vec::new();
        let mut slots = Vec::new();
        for a in additional_args {
            if let Ok(s) = a.extract::<SpensoRepresentation>() {
                slots.push(s.representation);
            } else if let Ok(arg) = a.extract::<PythonExpression>() {
                args.push(arg.expr.into());
            } else {
                return Err(exceptions::PyTypeError::new_err(
                    "Only slots and expressions can be used",
                ));
            }
        }

        let args = if args.is_empty() { None } else { Some(args) };

        let mut a: IndexlessNamedStructure<SerializableSymbol, Vec<SerializableAtom>, Rep> =
            IndexLess::from_iter(slots).into();
        if let Some(name) = name {
            match name.expr.as_view() {
                AtomView::Var(v) => a.set_name(v.get_symbol().into()),
                _ => {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only symbols can used as names",
                    ))
                }
            }
        };
        a.additional_args = args;

        Ok(SpensoStucture { structure: a })
    }

    fn __repr__(&self) -> String {
        format!("{}", self.structure.to_symbolic().unwrap())
    }

    fn __str__(&self) -> String {
        let slot = self
            .structure
            .external_reps()
            .into_iter()
            .map(|r| r.to_symbolic([]))
            .join(",");

        match (self.structure.name(), self.structure.args()) {
            (Some(name), Some(args)) => {
                let args = args.iter().join(",");
                format!("{}({})[{}]", name, args, slot)
            }
            (Some(name), None) => {
                format!("{}[{}]", name, slot)
            }
            (None, Some(args)) => {
                let args = args.iter().join(",");
                format!("({})[{}]", args, slot)
            }
            (None, None) => {
                format!("[{}]", slot)
            }
        }
    }

    fn __len__(&self) -> usize {
        self.structure.size().unwrap()
    }

    fn __getitem__<'py>(&self, item: SliceOrIntOrExpanded, py: Python<'py>) -> PyResult<Py<PyAny>> {
        match item {
            SliceOrIntOrExpanded::Int(i) => {
                let out: Vec<_> = self
                    .structure
                    .expanded_index(i.into())
                    .map_err(|s| PyIndexError::new_err(s.to_string()))?
                    .into();

                out.into_py_any(py)
            }
            SliceOrIntOrExpanded::Expanded(idxs) => {
                let out: usize = self
                    .structure
                    .flat_index(&idxs)
                    .map_err(|s| PyIndexError::new_err(s.to_string()))?
                    .into();

                out.into_py_any(py)
            }
            SliceOrIntOrExpanded::Slice(s) => {
                let r = s.indices(self.structure.size().unwrap() as isize)?;

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

                let slice: Result<Vec<Vec<usize>>, _> = range
                    .step_by(step)
                    .map(|i| {
                        self.structure
                            .expanded_index(i.into())
                            .map(Vec::<usize>::from)
                    })
                    .collect();

                match slice {
                    Ok(slice) => slice.into_py_any(py),
                    Err(e) => Err(PyIndexError::new_err(e.to_string())),
                }
            }
        }
    }
}

#[gen_stub_pyclass(module = "symbolica_community.tensors")]
#[pyclass(name = "Representation", module = "symbolica_community.tensors")]
#[derive(Clone)]
/// A representation class in the sense of representation theory. This class is used to represent the representation of a tensor. It is essentially a pair of a name and a dimension.
/// New representations are registered when constructing.
/// Some representations are dualizable, meaning that they have a dual representation.
/// Indices will only ever match across dual representations.
/// There are some already registered representations, such as:
///  EUCLIDEAN: Rep = Rep::SelfDual(0);
///  BISPINOR: Rep = Rep::SelfDual(1);
///  COLORADJ: Rep = Rep::SelfDual(2);
///  MINKOWSKI: Rep = Rep::SelfDual(3);
///
///  LORENTZ_UP: Rep = Rep::Dualizable(1);
///  LORENTZ_DOWN: Rep = Rep::Dualizable(-1);
///  SPINFUND: Rep = Rep::Dualizable(2);
///  SPINANTIFUND: Rep = Rep::Dualizable(-2);
///  COLORFUND: Rep = Rep::Dualizable(3);
///  COLORANTIFUND: Rep = Rep::Dualizable(-3);
///  COLORSEXT: Rep = Rep::Dualizable(4);
///  COLORANTISEXT: Rep = Rep::Dualizable(-4);
///
pub struct SpensoRepresentation {
    pub representation: Representation<Rep>,
}

// #[gen_stub_pymethods]
#[pymethods]
impl SpensoRepresentation {
    #[new]
    #[pyo3(signature =
           (
           name,dimension,dual=false))]
    /// Register a new representation with the given name and dimension. If dual is true, the representation will be dualizable, else it will be self-dual.
    pub fn register_new(name: Bound<'_, PyAny>, dimension: usize, dual: bool) -> PyResult<Self> {
        let name = name.extract::<PyBackedStr>()?;

        let rep = if dual {
            Rep::new_dual(&name).unwrap().new_rep(dimension)
        } else {
            Rep::new_self_dual(&name).unwrap().new_rep(dimension)
        };
        Ok(SpensoRepresentation {
            representation: rep,
        })
    }

    /// Generate a new slot with the given index, from this representation
    fn __call__(&self, aind: Bound<'_, PyAny>) -> PyResult<SpensoSlot> {
        if let Ok(i) = aind.extract::<isize>() {
            Ok(SpensoSlot {
                slot: self.representation.new_slot(i),
            })
        } else if let Ok(expr) = aind.extract::<PythonExpression>() {
            let id = match expr.expr.as_view() {
                AtomView::Var(v) => v.get_symbol(),
                _ => {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only symbols can be abstract indices",
                    ))
                }
            };

            let aind = AbstractIndex::Symbol(id.into());
            Ok(SpensoSlot {
                slot: self.representation.new_slot(aind),
            })
        } else if let Ok(s) = aind.extract::<PyBackedStr>() {
            let id = symbol!(&s);

            Ok(SpensoSlot {
                slot: self
                    .representation
                    .new_slot(AbstractIndex::Symbol(id.into())),
            })
        } else {
            Err(PyTypeError::new_err("aind must be an integer or a symbol"))
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.representation)
    }

    fn __str__(&self) -> String {
        format!("{}", self.representation.to_symbolic([]))
    }

    fn to_expression(&self) -> PythonExpression {
        PythonExpression::from(self.representation.to_symbolic([]))
    }
}

/// An abstract index slot for a tensor.
/// This is essentially a tuple of a `Representation` and an abstract index id.
///
/// The abstract index id can be either an integer or a symbol.
/// This is the building block for creating tensor structures that can be contracted.
#[gen_stub_pyclass(module = "symbolica_community.tensors")]
#[pyclass(name = "Slot", module = "symbolica_community.tensors")]
#[derive(Clone)]
pub struct SpensoSlot {
    pub slot: Slot<Rep>,
}

// #[gen_stub_pymethods]
#[pymethods]
impl SpensoSlot {
    fn __repr__(&self) -> String {
        format!("{:?}", self.slot)
    }

    fn __str__(&self) -> String {
        format!("{}", self.slot.to_atom())
    }

    #[new]
    #[pyo3(signature =
           (
           name,dimension,aind,dual=false))]
    /// Create a new slot from a name of a representation, a dimension and an abstract index.
    ///  If dual is true, the representation will be dualizable, else it will be self-dual.
    pub fn register_new(
        name: Bound<'_, PyAny>,
        dimension: usize,
        aind: Bound<'_, PyAny>,
        dual: bool,
    ) -> PyResult<Self> {
        let name = name.extract::<PyBackedStr>()?;
        let rep = if dual {
            Rep::new_dual(&name).unwrap().new_rep(dimension)
        } else {
            Rep::new_self_dual(&name).unwrap().new_rep(dimension)
        };
        if let Ok(i) = aind.extract::<isize>() {
            Ok(SpensoSlot {
                slot: rep.new_slot(i),
            })
        } else if let Ok(expr) = aind.extract::<PythonExpression>() {
            let id = match expr.expr.as_view() {
                AtomView::Var(v) => v.get_symbol(),
                _ => {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only symbols can be abstract indices",
                    ))
                }
            };

            let aind = AbstractIndex::Symbol(id.into());
            Ok(SpensoSlot {
                slot: rep.new_slot(aind),
            })
        } else if let Ok(s) = aind.extract::<PyBackedStr>() {
            let id = symbol!(&s);

            Ok(SpensoSlot {
                slot: rep.new_slot(AbstractIndex::Symbol(id.into())),
            })
        } else {
            Err(PyTypeError::new_err("aind must be an integer or a symbol"))
        }
    }

    fn to_expression(&self) -> PythonExpression {
        PythonExpression::from(self.slot.to_atom())
    }
}
