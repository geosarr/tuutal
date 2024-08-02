use crate::{wrap_vec_func_scalar, wrap_vec_func_vec};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};
use tuutal::{descent, Array1, DescentParameter, TuutalError};

macro_rules! first_order_method {
    ($method:ident, $name:ident) => {
        #[pyfunction]
        pub fn $method<'py>(
            py: Python<'py>,
            f: PyObject,
            g: PyObject,
            x0: PyReadonlyArray1<f64>,
            gamma: f64,
            beta: f64,
            gtol: f64,
            maxiter: Option<usize>,
            f_kwargs: Option<&Bound<'_, PyDict>>,
            g_kwargs: Option<&Bound<'_, PyDict>>,
        ) -> PyResult<Bound<'py, PyArray1<f64>>> {
            match descent(
                wrap_vec_func_scalar!(py, f, f_kwargs),
                wrap_vec_func_vec!(py, g, g_kwargs),
                &x0.as_array().to_owned(),
                &DescentParameter::$name(gamma, beta),
                gtol,
                maxiter,
            ) {
                Ok(value) => Ok(value.into_pyarray_bound(py)),
                Err(error) => match error {
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
    };
}

first_order_method!(armijo, new_armijo);
first_order_method!(powell_wolfe, new_powell_wolfe);
first_order_method!(adagrad, new_adagrad);
first_order_method!(adadelta, new_adadelta);
