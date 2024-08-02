use crate::{wrap_scalar_func_scalar, wrap_vec_func_scalar};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tuutal::{
    brent_bounded as brent_bounded_rs, brent_root as brent_root_rs,
    brent_unbounded as brent_unbounded_rs, brentq as brentq_rs, nelder_mead as nelder_mead_rs,
    Array1, RootFindingError, TuutalError,
};

#[pyfunction]
pub fn nelder_mead<'py>(
    py: Python<'py>,
    f: PyObject,
    x0: PyReadonlyArray1<f64>,
    maxiter: Option<usize>,
    maxfev: Option<usize>,
    initial_simplex: Option<PyReadonlyArray2<f64>>,
    xatol: Option<f64>,
    fatol: Option<f64>,
    adaptive: Option<bool>,
    bounds: Option<(f64, f64)>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    match nelder_mead_rs(
        wrap_vec_func_scalar!(py, f, kwargs),
        &x0.as_array().to_owned(),
        maxfev,
        maxiter,
        initial_simplex.map(|sim| sim.as_array().to_owned()),
        xatol.unwrap_or(1e-4),
        fatol.unwrap_or(1e-4),
        adaptive.unwrap_or(false),
        bounds,
    ) {
        Ok(value) => Ok(value.into_pyarray_bound(py)),
        Err(error) => match error {
            // Maybe better to throw also the current iterate and the number
            // actual function calls for this exception.
            TuutalError::MaxFunCall { num: _ } => Err(PyRuntimeError::new_err(
                "Maximum number of function calls exceeded or will be exceeded.",
            )),
            TuutalError::EmptyDimension { x: _ } => {
                Err(PyValueError::new_err("Empty initial input vector"))
            }
            TuutalError::BoundOrder { lower: _, upper: _ } => Err(PyValueError::new_err(
                "The upper bound(s) should be greater than the lower bound(s)",
            )),
            TuutalError::Simplex {
                size: (nrows, ncols),
                msg: _,
            } => Err(PyValueError::new_err(format!(
                "Initial simplex shape = {:?}, but should be (N+1, N) with N = len(x0)",
                (nrows, ncols)
            ))),
            TuutalError::Convergence {
                iterate: x,
                maxiter: _,
            } => {
                println!("Maximum number of iterations reached before convergence");
                Ok(x.into_pyarray_bound(py))
            }
            err => Err(PyRuntimeError::new_err(err.to_string())), // Should never come this far.
        },
    }
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
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<(f64, f64, usize)> {
    let func = wrap_scalar_func_scalar!(py, f, kwargs);
    match brent_root_rs(func, a, b, xtol, rtol, maxiter) {
        Ok(val) => Ok(val),
        Err(error) => match error {
            RootFindingError::Bracketing { a: x, b: y } => Err(PyValueError::new_err(format!(
                "Bracketing condition f(a) * f(b) < 0, not satisfied by inputs a={x} and b={y}",
            ))),
            RootFindingError::Interpolation { a: x, b: y } => Err(PyValueError::new_err(format!(
                "Interpolation cannot be performed since f(a) = f(b) for a={x} and b={y}",
            ))),
        },
    }
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
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<(f64, f64, usize)> {
    let func = wrap_scalar_func_scalar!(py, f, kwargs);
    match brentq_rs(func, a, b, xtol, rtol, maxiter) {
        Ok(val) => Ok(val),
        Err(error) => match error {
            RootFindingError::Bracketing { a: x, b: y } => Err(PyValueError::new_err(format!(
                "Bracketing condition f(a) * f(b) < 0, not satisfied by inputs a={x} and b={y}",
            ))),
            err => Err(PyRuntimeError::new_err(err.to_string())), // Should never come this far.
        },
    }
}

/// Brent bounded minimization algorithm for scalar function.
#[pyfunction]
pub fn brent_bounded(
    py: Python,
    f: PyObject,
    bounds: (f64, f64),
    xatol: f64,
    maxiter: usize,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<(f64, f64, usize)> {
    let func = wrap_scalar_func_scalar!(py, f, kwargs);
    match brent_bounded_rs(func, bounds, xatol, maxiter) {
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
                iterate: (xf, _a, _b, fx, _fa, _fb, fcalls),
                maxiter: _,
            } => {
                println!("Maximum number of iterations reached before convergence");
                Ok((xf, fx, fcalls))
            }
            // Err(PyUserWarning::new_err(format!(
            //     "Maximum number of iterations reached before convergence {:?}",
            //     (xf, a, b, fx, fa, fb, fcalls)
            // ))),
            err => Err(PyRuntimeError::new_err(err.to_string())), // Should never come this far.
        },
    }
}

/// Brent unbounded minimization algorithm for scalar function.
#[pyfunction]
pub fn brent_unbounded(
    py: Python,
    f: PyObject,
    maxiter: usize,
    xtol: f64,
    brack: Option<(f64, f64)>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<(f64, f64, usize)> {
    let func = wrap_scalar_func_scalar!(py, f, kwargs);
    let brack = brack.map_or(None, |v| Some([v.0, v.1]));
    match if let Some(val) = brack {
        brent_unbounded_rs(func, Some(&[val[0], val[1]]), maxiter, xtol)
    } else {
        brent_unbounded_rs(func, None, maxiter, xtol)
    } {
        Ok(val) => Ok(val),
        Err(error) => match error {
            // Does not make a difference between bracketing
            // convergence and the actual brent algorithm convergence
            TuutalError::Convergence {
                iterate: (_xa, xb, _xc, _fa, fb, _fc, fcalls),
                maxiter: _,
            } => {
                println!("Maximum number of iterations reached before convergence");
                Ok((xb, fb, fcalls))
            }
            TuutalError::Bracketing {
                iterate: (xa, xb, xc, fa, fb, fc, fcalls),
            } => Err(PyValueError::new_err(format!(
                "Bracketing condition not satisfied by the final iterate: {:?}",
                (xa, xb, xc, fa, fb, fc, fcalls)
            ))),
            err => Err(PyRuntimeError::new_err(err.to_string())), // Should never come this far.
        },
    }
}
