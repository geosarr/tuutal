use crate::{wrap_vec_func_scalar, wrap_vec_func_vec};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{conversion::FromPyObjectBound, exceptions::PyRuntimeError, intern, prelude::*};
use tuutal::{steepest_descent, SteepestDescentParameter, TuutalError, VecType};

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
        ) -> PyResult<Bound<'py, PyArray1<f64>>> {
            match steepest_descent(
                wrap_vec_func_scalar!(py, f),
                wrap_vec_func_vec!(py, g),
                &x0.as_array().to_owned(),
                &SteepestDescentParameter::$name(gamma, beta),
                gtol,
                maxiter.unwrap_or(x0.len().unwrap() * 1000),
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

// #[pyfunction]
// pub fn armijo<'py>(
//     py: Python<'py>,
//     f: PyObject,
//     g: PyObject,
//     x0: PyReadonlyArray1<f64>,
//     gamma: f64,
//     beta: f64,
//     gtol: f64,
//     maxiter: Option<usize>,
// ) -> PyResult<Bound<'py, PyArray1<f64>>> {
//     match steepest_descent(
//         wrap_vec_func_scalar!(py, f),
//         wrap_vec_func_vec!(py, g),
//         &x0.as_array().to_owned(),
//         &SteepestDescentParameter::new_armijo(gamma, beta),
//         gtol,
//         maxiter.unwrap_or(x0.len().unwrap() * 1000),
//     ) {
//         Ok(value) => Ok(value.into_pyarray_bound(py)),
//         Err(error) => match error {
//             TuutalError::Convergence {
//                 iterate: x,
//                 maxiter: _,
//             } => {
//                 println!("Maximum number of iterations reached before convergence");
//                 Ok(x.into_pyarray_bound(py))
//             }
//             err => Err(PyRuntimeError::new_err(err.to_string())), // Should never come this far.
//         },
//     }
// }

// #[pyfunction]
// pub fn adagrad<'py>(
//     py: Python<'py>,
//     f: PyObject,
//     g: PyObject,
//     x0: PyReadonlyArray1<f64>,
//     gamma: f64,
//     beta: f64,
//     gtol: f64,
//     maxiter: Option<usize>,
// ) -> PyResult<Bound<'py, PyArray1<f64>>> {
//     match steepest_descent(
//         wrap_vec_func_scalar!(py, f),
//         wrap_vec_func_vec!(py, g),
//         &x0.as_array().to_owned(),
//         &SteepestDescentParameter::new_adagr(gamma, beta),
//         gtol,
//         maxiter.unwrap_or(x0.len().unwrap() * 1000),
//     ) {
//         Ok(value) => Ok(value.into_pyarray_bound(py)),
//         Err(error) => match error {
//             TuutalError::Convergence {
//                 iterate: x,
//                 maxiter: _,
//             } => {
//                 println!("Maximum number of iterations reached before convergence");
//                 Ok(x.into_pyarray_bound(py))
//             }
//             err => Err(PyRuntimeError::new_err(err.to_string())), // Should never come this far.
//         },
//     }
// }
