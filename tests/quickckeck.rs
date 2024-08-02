#[macro_use]
extern crate quickcheck;
extern crate tuutal;

use core::cmp::min;
use tuutal::{s, Armijo, Array1, Optimizer, PowellWolfe, VarName};

quickcheck! {
    fn descent_armijo(xs: Vec<f32>) -> bool {
        // When the objective function f is regular enough (e.g. continously differentiable),
        // then sequence (f(x_k))_k of output values of the iterates (x_k)_k, provided by the Armijo
        // rule, should be strictly decreasing.
        let arr = Array1::from(xs);
        if arr.is_empty() {
            return true;
        }
        let arr = arr.slice(s!(..min(arr.shape()[0], 3)));
        if arr.iter().any(|val| (val.abs() > 1e6) | val.is_nan()) {
            // To avoid arbitrarily large or missing numbers.
            return true;
        }
        let eye = Array1::ones(arr.shape()[0]);
        let f = |x: &Array1<f32>| 0.5 * x.dot(x).powi(2) + eye.dot(x) + 1.;
        let gradf = |x: &Array1<f32>| 2. * x * x.dot(x) + eye.clone();
        let mut iterates = Armijo::new(f, gradf, arr.to_owned(), 0.01, 0.5, 1e-3);
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

    fn descent_powell_wolfe(xs: Vec<f32>) -> bool {
        // When the objective function f is regular enough (e.g. continously differentiable) and
        // then sequence (x_k)_k along with the descent directions (d_k)_k (which are the negative
        // gradient at x_k) generated by the Powell Wolfe rule satisfy for all k:
        //          inf_{t>=0} f(x_k + t * d_k) > - infinity
        // then the generated steps decrease the objective function value by at least a value proportional
        // to the norm of the gradient at the iterates.
        let arr = Array1::from(xs);
        if arr.is_empty() {
            return true;
        }
        let arr = arr.slice(s!(..min(arr.shape()[0], 3)));
        if arr.iter().any(|val| (val.abs() > 1e6) | val.is_nan()) {
            // To avoid arbitrarily large or missing numbers.
            return true;
        }
        let f = |x: &Array1<f32>| 0.5 * x.dot(x).powi(2) + 1.;
        let gradf = |x: &Array1<f32>| 2. * x * x.dot(x) ;
        let gamma = 0.001;
        let beta = 0.9;
        let mut iterates = PowellWolfe::new(f, gradf, arr.to_owned(), gamma, beta, 1e-3);
        let mut x_prev = arr.to_owned();
        let mut iter = 1;
        while let Some(x_next) = iterates.next() {
            // let gradfx_next = gradf(&x_next);
            let neg_gradfx_prev = -gradf(&x_prev);
            let gradfx_d = neg_gradfx_prev.dot(&neg_gradfx_prev);
            let intermed = iterates.intermediate();
            let step_size = intermed[&VarName::StepSize][0];
            assert!(f(&x_next) <= f(&x_prev) - step_size * gamma * gradfx_d);
            x_prev = x_next;
            iter += 1;
            if iter > 10 {
                break;
            }
        }
        true
    }
}
