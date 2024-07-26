mod first_order;
mod interface;
mod zero_order;
pub use first_order::{adadelta, adagrad, armijo, powell_wolfe};
pub(crate) use interface::{wrap_scalar_func_scalar, wrap_vec_func_scalar, wrap_vec_func_vec};
use pyo3::prelude::*;
pub use zero_order::{brent_bounded, brent_root, brent_unbounded, brentq, nelder_mead};

#[pymodule]
fn tuutal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(brent_root, m)?)?;
    m.add_function(wrap_pyfunction!(brentq, m)?)?;
    m.add_function(wrap_pyfunction!(brent_bounded, m)?)?;
    m.add_function(wrap_pyfunction!(brent_unbounded, m)?)?;
    m.add_function(wrap_pyfunction!(nelder_mead, m)?)?;
    m.add_function(wrap_pyfunction!(armijo, m)?)?;
    m.add_function(wrap_pyfunction!(adadelta, m)?)?;
    m.add_function(wrap_pyfunction!(adagrad, m)?)?;
    m.add_function(wrap_pyfunction!(powell_wolfe, m)?)?;
    Ok(())
}
