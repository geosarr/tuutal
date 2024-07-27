macro_rules! wrap_scalar_func_scalar {
    ($py:expr, $py_func:expr, $kwds:ident) => {
        |x: f64| {
            $py_func
                .call_bound($py, (x,), $kwds)
                .expect("python objective function failed.")
                .extract::<f64>($py)
                .expect("python function should return a float-pointing number")
        }
    };
}

macro_rules! wrap_vec_func_scalar {
    ($py:expr, $py_func:expr, $kwds:ident) => {
        |x: &Array1<f64>| {
            $py_func
                .call_bound($py, (x.clone().into_pyarray_bound($py),), $kwds)
                .expect("python objective function failed.")
                .extract::<f64>($py)
                .expect("python function should return a float-pointing number")
        }
    };
}

macro_rules! wrap_vec_func_vec {
    ($py:expr, $py_func:expr, $kwds:ident) => {
        |x: &Array1<f64>| {
            $py_func
                .call_bound($py, (x.clone().into_pyarray_bound($py),), $kwds)
                .expect("python objective function failed.")
                .extract::<PyReadonlyArray1<f64>>($py)
                .expect("python function should return a python numpy.ndarray")
                .as_array()
                .to_owned()
        }
    };
}

pub(crate) use wrap_scalar_func_scalar;
pub(crate) use wrap_vec_func_scalar;
pub(crate) use wrap_vec_func_vec;
