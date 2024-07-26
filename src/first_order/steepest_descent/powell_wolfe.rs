use std::ops::Add;

use crate::{traits::VecDot, Scalar};

/// Computes a step size using the Powell Wolfe method.
pub(crate) fn powell_wolfe<F, G, A, X>(
    f: &F,
    gradf: &G,
    x: &X,
    neg_gradfx: &X,
    squared_norm_2_gradfx: A,
    gamma: A,
    beta: A,
) -> A
where
    A: Scalar<X>,
    F: Fn(&X) -> A,
    G: Fn(&X) -> X,
    X: VecDot<X, Output = A> + Add<X, Output = X>,
    for<'a> &'a X: Add<X, Output = X>,
{
    let mut sigma_minus = A::one();
    let mut x_next = x + sigma_minus * neg_gradfx;
    let one_half = A::cast_from_f32(0.5);
    let fx = f(x);
    // The first if and else conditions guarantee having a segment [sigma_minus, sigma_plus]
    // such that sigma_minus satisfies the armijo condition and sigma_plus does not
    let mut sigma_plus = if f(&x_next) - fx <= -sigma_minus * gamma * squared_norm_2_gradfx {
        if gradf(&x_next).dot(neg_gradfx) >= -beta * squared_norm_2_gradfx {
            return sigma_minus;
        }
        // Computation of sigma_plus
        let two = A::cast_from_f32(2.);
        let mut sigma_plus = two;
        x_next = x + sigma_plus * neg_gradfx;
        while f(&x_next) - fx <= -sigma_plus * gamma * squared_norm_2_gradfx {
            sigma_plus = two * sigma_plus;
            x_next = x + sigma_plus * neg_gradfx;
        }
        // At this stage sigma_plus is the smallest 2^k that does not satisfy the Armijo rule
        sigma_minus = sigma_plus * one_half; // it satisfies the Armijo rule
        sigma_plus
    } else {
        sigma_minus = one_half;
        x_next = x + sigma_minus * neg_gradfx;
        while f(&x_next) - fx > -sigma_minus * gamma * squared_norm_2_gradfx {
            sigma_minus = one_half * sigma_minus;
            x_next = x + sigma_minus * neg_gradfx;
        }
        sigma_minus * (A::cast_from_f32(2.)) // does not satisfy the Armijo rule
    };
    x_next = x + sigma_minus * neg_gradfx;
    while gradf(&x_next).dot(neg_gradfx) < -beta * squared_norm_2_gradfx {
        let sigma = (sigma_minus + sigma_plus) * one_half;
        x_next = x + sigma * neg_gradfx;
        if f(&x_next) - fx <= -sigma * gamma * squared_norm_2_gradfx {
            sigma_minus = sigma;
        } else {
            sigma_plus = sigma;
        }
    }
    sigma_minus
}
