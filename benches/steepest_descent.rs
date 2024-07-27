#![feature(test)]
extern crate test;
extern crate tuutal;

use test::Bencher;
use tuutal::{s, steepest_descent, Array, SteepestDescentParameter, Array1};

fn rosenbrock_nd() -> (
    impl Fn(&Array1<f32>) -> f32,
    impl Fn(&Array1<f32>) -> Array1<f32>,
) {
    // Rosenbrock function and its gradient function for arrays of size >= 3.
    let f = |x: &Array1<f32>| {
        let xi = x.slice(s![0..x.len() - 1]);
        let xi_plus1 = x.slice(s![1..]);
        let term = 100. * (xi_plus1.to_owned() - xi.map(|x| x.powi(2))).map(|x| x.powi(2))
            + (1. - xi.to_owned()).map(|x| x.powi(2));
        term.sum()
    };
    let gradf = |x: &Array1<f32>| {
        let n = x.len();
        let first_deriv = |i: usize| -400. * x[i] * (x[i + 1] - x[i].powi(2)) - 2. * (1. - x[i]);
        let last_deriv = |i: usize| 200. * (x[i] - x[i - 1].powi(2));
        let grad = (0..n).map(|i| {
            if i > 0 && i < n - 1 {
                first_deriv(i) + last_deriv(i)
            } else if i == 0 {
                first_deriv(0)
            } else {
                last_deriv(n - 1)
            }
        });
        Array::from_vec(grad.collect())
    };
    return (f, gradf);
}

#[bench]
fn armijo_bench(bench: &mut Bencher) {
    let (f, gradf) = rosenbrock_nd();
    static LENGTH: usize = 500;
    let x0 = Array::from_vec(vec![0_f32; LENGTH]);
    let params = SteepestDescentParameter::new_armijo(0.01, 0.01);
    bench.iter(|| {
        let _solution = steepest_descent(&f, &gradf, &x0, &params, 1e-6, 1000);
    });
}

#[bench]
fn powell_wolfe_bench(bench: &mut Bencher) {
    let (f, gradf) = rosenbrock_nd();
    static LENGTH: usize = 500;
    let x0 = Array::from_vec(vec![0_f32; LENGTH]);
    let params = SteepestDescentParameter::new_powell_wolfe(0.01, 0.1);
    bench.iter(|| {
        let _solution = steepest_descent(&f, &gradf, &x0, &params, 1e-6, 1000);
    });
}

#[bench]
fn adagrad_bench(bench: &mut Bencher) {
    let (f, gradf) = rosenbrock_nd();
    static LENGTH: usize = 500;
    let x0 = Array::from_vec(vec![0_f32; LENGTH]);
    let params = SteepestDescentParameter::new_adagrad(0.1, 0.0001);
    bench.iter(|| {
        let _solution = steepest_descent(&f, &gradf, &x0, &params, 1e-6, 1000);
    });
}

#[bench]
fn adadelta_bench(bench: &mut Bencher) {
    let (f, gradf) = rosenbrock_nd();
    static LENGTH: usize = 500;
    let x0 = Array::from_vec(vec![0_f32; LENGTH]);
    let params = SteepestDescentParameter::new_adadelta(0.1, 0.0001);
    bench.iter(|| {
        let _solution = steepest_descent(&f, &gradf, &x0, &params, 1e-6, 1000);
    });
}
