mod error;
mod zero_order;
use error::{RootFindingError, TuutalError};
use pyo3::exceptions::{PyRuntimeError, PyUserWarning, PyValueError};
use pyo3::prelude::*;
use zero_order::brent_bounded as brent_bounded_rs;
use zero_order::brent_root as brent_root_rs;
use zero_order::brentq as brentq_rs;

macro_rules! wrap_scalar_func {
    ($py:expr, $py_func:expr) => {
        |x: f64| {
            $py_func
                .call1($py, (x,))
                .expect("python objective function failed.")
                .extract::<f64>($py)
                .expect("python function should return a float-pointing number")
        }
    };
}

/// Brent algorithm for scalar function root finding.
#[pyfunction]
fn brent_root(
    py: Python,
    f: PyObject,
    a: f64,
    b: f64,
    xtol: f64,
    rtol: f64,
    maxiter: usize,
) -> PyResult<(f64, f64)> {
    let func = wrap_scalar_func!(py, f);
    return match brent_root_rs(func, a, b, xtol, rtol, maxiter) {
        Ok(val) => Ok(val),
        Err(error) => match error {
            RootFindingError::Bracketing { a: x, b: y } => Err(PyValueError::new_err(format!(
                "Bracketing condition f(a) * f(b) < 0, not satisfied by inputs a={x} and b={y}",
            ))),
            RootFindingError::Interpolation { a: x, b: y } => Err(PyValueError::new_err(format!(
                "Interpolation cannot be performed since f(a) = f(b) for a={x} and b={y}",
            ))),
        },
    };
}

#[pyfunction]
fn brentq(
    py: Python,
    f: PyObject,
    a: f64,
    b: f64,
    xtol: f64,
    rtol: f64,
    maxiter: usize,
) -> PyResult<(f64, f64, usize)> {
    let func = wrap_scalar_func!(py, f);
    return match brentq_rs(func, a, b, xtol, rtol, maxiter) {
        Ok(val) => Ok(val),
        Err(error) => match error {
            RootFindingError::Bracketing { a: x, b: y } => Err(PyValueError::new_err(format!(
                "Bracketing condition f(a) * f(b) < 0, not satisfied by inputs a={x} and b={y}",
            ))),
            RootFindingError::Interpolation { a: x, b: y } => Err(PyValueError::new_err(format!(
                "Interpolation cannot be performed since f(a) = f(b) for a={x} and b={y}",
            ))),
        },
    };
}

/// Brent algorithm for scalar function root finding.
#[pyfunction]
fn brent_bounded(
    py: Python,
    f: PyObject,
    bounds: (f64, f64),
    xatol: f64,
    maxiter: usize,
) -> PyResult<(f64, f64, usize)> {
    let func = wrap_scalar_func!(py, f);
    return match brent_bounded_rs(func, bounds, xatol, maxiter) {
        Ok(val) => Ok(val),
        Err(error) => match error {
            TuutalError::Infinity { x: _ } => {
                Err(PyValueError::new_err("One of the bounds is infinite."))
            }
            TuutalError::BoundOrder { lower: _, upper: _ } => Err(PyValueError::new_err(
                "The upper bound should be greater than the lower bound",
            )),
            TuutalError::Nan {
                x: (xf, a, b, fx, fa, fb, fcalls),
            } => Err(PyValueError::new_err(format!(
                "Nan value encountered in the final iterates {:?}",
                (xf, a, b, fx, fa, fb, fcalls)
            ))),
            TuutalError::Convergence {
                iterate: (xf, a, b, fx, fa, fb, fcalls),
                maxiter,
            } => Err(PyUserWarning::new_err(format!(
                "Maximum number of iteration reached before convergence {:?}",
                (xf, a, b, fx, fa, fb, fcalls)
            ))),
            _ => Err(PyRuntimeError::new_err("Unknown error")), // Should never come this far.
        },
    };
}

#[pymodule]
fn tuutal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(brent_root, m)?)?;
    m.add_function(wrap_pyfunction!(brentq, m)?)?;
    m.add_function(wrap_pyfunction!(brent_bounded, m)?)?;
    Ok(())
}
