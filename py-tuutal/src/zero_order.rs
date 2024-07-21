use pyo3::exceptions::{PyRuntimeError, PyUserWarning, PyValueError};
use pyo3::prelude::*;
use tuutal::{
    brent_bounded as brent_bounded_rs, brent_root as brent_root_rs,
    brent_unbounded as brent_unbounded_rs, brentq as brentq_rs, RootFindingError, TuutalError,
};

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
pub fn brent_root(
    py: Python,
    f: PyObject,
    a: f64,
    b: f64,
    xtol: f64,
    rtol: f64,
    maxiter: usize,
) -> PyResult<(f64, f64, usize)> {
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

/// Brent algorithm for scalar function root finding.
#[pyfunction]
pub fn brentq(
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
            _ => Err(PyRuntimeError::new_err("Unknown error")), // Should never come this far.
        },
    };
}

/// Brent bounded minimization algorithm for scalar function.
#[pyfunction]
pub fn brent_bounded(
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
                "Maximum number of iterations reached before convergence {:?}",
                (xf, a, b, fx, fa, fb, fcalls)
            ))),
            _ => Err(PyRuntimeError::new_err("Unknown error")), // Should never come this far.
        },
    };
}

/// Brent unbounded minimization algorithm for scalar function.
#[pyfunction]
pub fn brent_unbounded(
    py: Python,
    f: PyObject,
    xa: f64,
    xb: f64,
    maxiter: usize,
    tol: f64,
) -> PyResult<(f64, f64, usize)> {
    let func = wrap_scalar_func!(py, f);
    return match brent_unbounded_rs(func, xa, xb, maxiter, tol) {
        Ok(val) => Ok(val),
        Err(error) => match error {
            // Does not make a difference between bracketing
            // convergence and the actual brent algorithm convergence
            TuutalError::Convergence {
                iterate: (xa, xb, xc, fa, fb, fc, fcalls),
                maxiter,
            } => Err(PyUserWarning::new_err(format!(
                "Maximum number of iterations reached before convergence {:?}",
                (xa, xb, xc, fa, fb, fc, fcalls)
            ))),
            TuutalError::Bracketing {
                iterate: (xa, xb, xc, fa, fb, fc, fcalls),
            } => Err(PyValueError::new_err(format!(
                "Bracketing condition not satisfied by the final iterate: {:?}",
                (xa, xb, xc, fa, fb, fc, fcalls)
            ))),
            _ => Err(PyRuntimeError::new_err("Unknown error")), // Should never come this far.
        },
    };
}
