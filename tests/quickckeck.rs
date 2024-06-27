// #![cfg(feature = "quickcheck")]
#[macro_use]
extern crate quickcheck;
extern crate tuutal;

use std::cmp::min;
use tuutal::{s, Array, SteepestDescentIterates, SteepestDescentParameter, VecType};

quickcheck! {
    fn descent_armijo(xs: Vec<f32>) -> bool {
        // When the objective function f is regular enough (e.g. continously differentiable),
        // then sequence (f(x_k))_k of output values of the iterates (x_k)_k, provided by the Armijo
        // rule is completely, should be strictly decreasing.
        let arr = Array::from_vec(xs);
        if arr.is_empty() {
            return true;
        }
        let arr = arr.slice(s!(..min(arr.shape()[0], 3)));
        if arr.iter().any(|val| (val.abs() > 1e6) | val.is_nan()) {
            // To avoid arbitrarily large or missing numbers.
            return true;
        }
        let eye = Array::ones(arr.shape()[0]);
        let f = |x: &VecType<f32>| 0.5 * x.dot(x).powi(2) + eye.dot(x) + 1.;
        let gradf = |x: &VecType<f32>| 2. * x * x.dot(x) + eye.clone();
        let param = SteepestDescentParameter::new_armijo(0.01, 0.5);
        let mut iterates = SteepestDescentIterates::new(f, gradf, arr.to_owned(), param, 1e-3);
        let mut x_prev = arr.to_owned();
        let mut iter = 1;
        while let Some(x) = iterates.next() {
            assert!(f(&x) < f(&x_prev) + f32::EPSILON);
            x_prev = x;
            iter += 1;
            if iter > 10 {
                break;
            }
        }
        true
    }
}
