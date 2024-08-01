use core::ops::Add;

use crate::{
    first_order::macros::steepest_descent_rule,
    traits::{Scalar, VecDot, Vector},
};
use num_traits::One;

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
        *fcalls += 1
    }
    (x_next, sigma)
}

steepest_descent_rule!(Armijo);

impl<X, F, G> Armijo<X, F, G>
where
    X: Vector + VecDot<X, Output = X::Elem>,
    for<'a> &'a X: Add<X, Output = X>,
    F: Fn(&X) -> X::Elem,
    G: Fn(&X) -> X,
{
    pub(crate) fn step(&mut self, neg_gradfx: &X, squared_norm_2_gradfx: X::Elem) {
        let mut sigma = X::Elem::one();
        let mut x_next = &self.x + sigma * neg_gradfx;
        let fx = self.func(&self.x);
        self.fcalls += 1;
        while self.func(&x_next) - fx > -sigma * self.gamma * squared_norm_2_gradfx {
            sigma = self.beta * sigma;
            x_next = &self.x + sigma * neg_gradfx;
            self.fcalls += 1;
        }
        self.x = x_next;
        self.sigma = sigma;
    }
}
