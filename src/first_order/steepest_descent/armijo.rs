use core::ops::{Add, Mul};

use crate::{
    first_order::macros::{descent_rule, impl_iterator_descent},
    traits::{Scalar, VecDot, Vector},
    Counter,
};
use hashbrown::HashMap;
use num_traits::{Float, One, Zero};

/// Computes a step size using the Armijo method.
pub(crate) fn armijo<F, A, X>(
    f: &F,
    x: &X,
    neg_gradfx: &X,
    squared_norm_2_gradfx: A,
    gamma: A,
    beta: A,
    fcalls: &mut usize,
) -> (X, A)
where
    A: Scalar<X>,
    F: Fn(&X) -> A,
    X: VecDot<X, Output = A> + Add<X, Output = X>,
    for<'a> &'a X: Add<X, Output = X>,
{
    let mut sigma = A::one();
    let mut x_next = x + sigma * neg_gradfx;
    let fx = f(x);
    *fcalls += 1;
    while f(&x_next) - fx > -sigma * gamma * squared_norm_2_gradfx {
        sigma = beta * sigma;
        x_next = x + sigma * neg_gradfx;
        *fcalls += 1;
    }
    (x_next, sigma)
}

descent_rule!(
    Armijo,
    [<X as Vector>::Elem; 1],
    [X::Elem::zero()],
    [].into()
);
impl_iterator_descent!(Armijo, [<X as Vector>::Elem; 1]);

impl<'a, X, F, G> Armijo<'a, X, F, G, [X::Elem; 1]>
where
    X: Vector + VecDot<X, Output = X::Elem>,
    for<'b> &'b X: Add<X, Output = X>,
    F: Fn(&X) -> X::Elem,
    G: Fn(&X) -> X,
{
    pub(crate) fn step(&mut self) {
        let mut sigma = X::Elem::one();
        let mut x_next = &self.x + sigma * &self.neg_gradfx;
        let fx = self.func(&self.x);
        self.counter.fcalls += 1;
        let (gamma, beta) = (self.hyper_params["gamma"], self.hyper_params["beta"]);
        // NB: self.stop_metrics is the squared L2-norm of gradf(&x).
        while self.func(&x_next) - fx > -sigma * gamma * self.stop_metrics {
            self.counter.fcalls += 1;
            sigma = beta * sigma;
            x_next = &self.x + sigma * &self.neg_gradfx;
        }
        self.x = x_next;
        self.sigma = [sigma];
    }
}
