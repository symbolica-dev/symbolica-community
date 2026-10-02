//! Reduce directly to OneLoopMaster's symbols in the host's shared kernel.

use pyo3::{
    Bound, PyResult,
    types::{PyAnyMethods, PyModule, PyModuleMethods},
};
use symbolica::api::python::{Citation, SymbolicaCommunityModule};

pub fn get_citations() -> Vec<Citation> {
    let mut citations = oneloopreduce_python::CommunityModule::get_citations();
    #[cfg(not(target_arch = "wasm32"))]
    citations.extend(oneloop_native::CommunityModule::get_citations());
    citations
}

pub fn register(hep: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(not(target_arch = "wasm32"))]
    let module = {
        oneloop_native::register_hep_module(hep)?;
        hep.getattr("oneloop")?.cast_into::<PyModule>()?
    };
    #[cfg(target_arch = "wasm32")]
    let module = {
        let module = PyModule::new(hep.py(), "symbolica.community.hep.oneloop")?;
        hep.add("oneloop", &module)?;
        hep.py()
            .import("sys")?
            .getattr("modules")?
            .set_item("symbolica.community.hep.oneloop", &module)?;
        module
    };

    oneloopreduce_python::CommunityModule::register_module(&module)?;
    #[cfg(not(target_arch = "wasm32"))]
    module.add_function(pyo3::wrap_pyfunction!(
        native::reduction_coefficients,
        &module
    )?)?;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::collections::BTreeMap;

    use oneloop::ScalarIntegral;
    use oneloopreduce_python::{OneLoopMasters, Reduction, validate_namespace};
    use pyo3::{PyResult, exceptions::PyValueError, pyfunction};
    use symbolica::{
        api::python::PythonExpression,
        atom::{Atom, AtomCore, Symbol},
        poly::series::SeriesDepth,
    };

    /// Expand a reduction about d=4-2*eps and combine its master coefficients.
    ///
    /// Returns [finite, simple_pole, double_pole] with native evaluation hooks.
    /// Raises ValueError for coefficient poles at d=4 requiring unavailable
    /// positive-order master coefficients, fractional Taylor powers, or
    /// dimension-dependent kinematics or scale.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import S, E
    /// >>> from symbolica.community import hep
    /// >>> from symbolica.community.hep import oneloop
    /// >>> d, k, p, s = S("d", "k", "p", "s")
    /// >>> kin = hep.Kinematics(d, momenta=[k, p]).with_scalar_product(p, p, s)
    /// >>> family = hep.IntegralFamily([k], [p], [kin.scalar_product(k, k),
    /// ...     kin.scalar_product(k-p, k-p)], kinematics=kin)
    /// >>> reduction = oneloop.reduce(family, [1, 1])
    /// >>> coefficients = oneloop.reduction_coefficients(reduction)
    /// >>> assert len(coefficients) == 3
    ///
    /// Parameters
    /// ----------
    /// reduction : Reduction
    ///     Symbolic one-loop reduction with dimension dependence retained.
    /// mu_squared : Expression or None, optional
    ///     Squared scale; None uses one.
    #[pyfunction]
    #[pyo3(signature = (reduction, mu_squared = None))]
    pub fn reduction_coefficients(
        reduction: &Reduction,
        mu_squared: Option<PythonExpression>,
    ) -> PyResult<Vec<PythonExpression>> {
        let scale = mu_squared.map(|m| m.expr).unwrap_or_else(|| Atom::num(1));
        validate_namespace(&scale)?;
        let dimension = reduction.dimension_symbol();

        // Combine repeated masters before expanding, so removable poles in
        // their summed coefficient do not cause a false rejection.
        let mut terms = BTreeMap::<Atom, Atom>::new();
        for (coefficient, master) in reduction.terms_ref() {
            let master = OneLoopMasters.symbol_with_scale(master, &scale);
            if master.contains_symbol(dimension) {
                return Err(PyValueError::new_err(format!(
                    "master kinematics and mu_squared must be independent of the reduction dimension {dimension}"
                )));
            }
            *terms.entry(master).or_insert(Atom::Zero) += coefficient;
        }

        let mut result = [Atom::Zero, Atom::Zero, Atom::Zero];
        for (master, coefficient) in terms {
            let coefficient = coefficient.cancel();
            if coefficient.is_zero() {
                continue;
            }
            let series = coefficient
                .series(dimension, Atom::num(4), SeriesDepth::absolute(2))
                .map_err(|e| {
                    PyValueError::new_err(format!("cannot expand reduction coefficient: {e}"))
                })?;
            if series.get_trailing_exponent() < 0 {
                return Err(PyValueError::new_err(format!(
                    "reduction coefficient for {master} has a pole at d=4; \
                     positive-order epsilon coefficients of the master are required"
                )));
            }
            if series
                .terms()
                .any(|(power, value)| !power.is_integer() && !value.is_zero())
            {
                return Err(PyValueError::new_err(
                    "reduction coefficients must have an integer-power Taylor expansion at d=4",
                ));
            }
            // The series is in (d-4), so multiply Taylor coefficients by (-2)^n.
            let c0 = series.coefficient(0.into()).unwrap();
            let c1 = series.coefficient(1.into()).unwrap() * Atom::num(-2);
            let c2 = series.coefficient(2.into()).unwrap() * Atom::num(4);
            let (family, arguments) =
                oneloop::master_arguments(&master).map_err(PyValueError::new_err)?;
            let head: Symbol = match family {
                ScalarIntegral::A0 => oneloop::A0(),
                ScalarIntegral::B0 => oneloop::B0(),
                ScalarIntegral::DB0 => oneloop::dB0(),
                ScalarIntegral::C0 => oneloop::C0(),
                ScalarIntegral::D0 => oneloop::D0(),
            };
            let [finite, pole, double_pole] = [0, -1, -2].map(|tag| {
                let args = std::iter::once(Atom::num(tag))
                    .chain(arguments.iter().cloned())
                    .collect::<Vec<_>>();
                head.call(args.as_slice())
            });
            result[0] += &c0 * &finite + &c1 * &pole + &c2 * &double_pole;
            result[1] += &c0 * &pole + &c1 * &double_pole;
            result[2] += &c0 * &double_pole;
        }
        Ok(result.into_iter().map(Into::into).collect())
    }
}
