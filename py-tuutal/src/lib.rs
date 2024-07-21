mod error;
mod traits;
mod zero_order;
use error::RootFindingError;
use pyo3::conversion::FromPyObject;
use pyo3::prelude::*;
use pyo3::types::*;
use zero_order::brent_root as brent_root_rs;

#[pyfunction]
fn brent_root(py: Python, f: PyObject, a: f64, b: f64) -> PyResult<(f64, f64)> {
    let func = |x: f64| f.call1(py, (x,)).unwrap().extract::<f64>(py).unwrap();
    Ok(brent_root_rs(func, a, b).unwrap())
}

#[pymodule]
fn tuutal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test, m)?)?;
    m.add_function(wrap_pyfunction!(brent_root, m)?)?;
    Ok(())
}
