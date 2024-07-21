mod error;
mod zero_order;
use error::RootFindingError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zero_order::brent_root as brent_root_rs;

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
fn brent_root(py: Python, f: PyObject, a: f64, b: f64) -> PyResult<(f64, f64)> {
    let func = wrap_scalar_func!(py, f);
    return match brent_root_rs(func, a, b) {
        Ok(val) => Ok(val),
        Err(error) => match error {
            RootFindingError::Bracketing { a: a, b: b } => Err(PyValueError::new_err(format!(
                "Bracketing condition f(a) * f(b) < 0, not satisfied by inputs a={a} and b={b}",
            ))),
            RootFindingError::Interpolation { a: a, b: b } => Err(PyValueError::new_err(
                "Interpolation cannot be performed du to dividion by 0.",
            )),
        },
    };
}

#[pymodule]
fn tuutal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(brent_root, m)?)?;
    Ok(())
}
