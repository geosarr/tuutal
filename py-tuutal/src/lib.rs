mod zero_order;
use pyo3::prelude::*;
pub use zero_order::{brent_bounded, brent_root, brent_unbounded, brentq, nelder_mead};

#[pymodule]
fn tuutal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(brent_root, m)?)?;
    m.add_function(wrap_pyfunction!(brentq, m)?)?;
    m.add_function(wrap_pyfunction!(brent_bounded, m)?)?;
    m.add_function(wrap_pyfunction!(brent_unbounded, m)?)?;
    m.add_function(wrap_pyfunction!(nelder_mead, m)?)?;
    Ok(())
}
