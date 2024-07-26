use std::ops::Add;

use crate::{traits::VecDot, Scalar};

/// Computes a step size using the Armijo method.
pub(crate) fn armijo<F, A, X>(
    f: &F,
    x: &X,
    neg_gradfx: &X,
    squared_norm_2_gradfx: A,
    gamma: A,
    beta: A,
) -> A
where
    A: Scalar<X>,
    F: Fn(&X) -> A,
    X: VecDot<X, Output = A> + Add<X, Output = X>,
    for<'a> &'a X: Add<X, Output = X>,
{
    let mut sigma = A::one();
    let mut x_next = x + sigma * neg_gradfx;
    let fx = f(x);
    while f(&x_next) - fx > -sigma * gamma * squared_norm_2_gradfx {
        sigma = beta * sigma;
        x_next = x + sigma * neg_gradfx;
    }
    sigma
}
