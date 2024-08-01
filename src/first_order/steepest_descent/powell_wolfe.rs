use core::ops::Add;

use crate::{
    first_order::macros::steepest_descent_rule,
    traits::{Number, VecDot, Vector},
    Scalar,
};
use num_traits::One;
/// Computes a step size using the Powell Wolfe method.
pub(crate) fn powell_wolfe<F, G, A, X>(
    funcs: (&F, &G),
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
    G: Fn(&X) -> X,
    X: VecDot<X, Output = A> + Add<X, Output = X>,
    for<'a> &'a X: Add<X, Output = X>,
{
    let (f, gradf) = funcs;
    let mut sigma_minus = A::one();
    let mut x_next = x + sigma_minus * neg_gradfx;
    let one_half = A::cast_from_f32(0.5);
    let fx = f(x);
    *fcalls += 1;
    // The first if and else conditions guarantee having a segment [sigma_minus, sigma_plus]
    // such that sigma_minus satisfies the armijo condition and sigma_plus does not
    let mut sigma_plus = if f(&x_next) - fx <= -sigma_minus * gamma * squared_norm_2_gradfx {
        *fcalls += 1;
        if gradf(&x_next).dot(neg_gradfx) >= -beta * squared_norm_2_gradfx {
            return (x_next, sigma_minus);
        }
        // Computation of sigma_plus
        let two = A::cast_from_f32(2.);
        let mut sigma_plus = two;
        x_next = x + sigma_plus * neg_gradfx;
        while f(&x_next) - fx <= -sigma_plus * gamma * squared_norm_2_gradfx {
            sigma_plus = two * sigma_plus;
            x_next = x + sigma_plus * neg_gradfx;
            *fcalls += 1;
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
            *fcalls += 1;
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
        *fcalls += 1;
    }
    (x + sigma_minus * neg_gradfx, sigma_minus)
}

steepest_descent_rule!(PowellWolfe);

impl<X, F, G> PowellWolfe<X, F, G>
where
    X: Vector + VecDot<X, Output = X::Elem> + Add<X, Output = X>,
    for<'a> &'a X: Add<X, Output = X>,
    F: Fn(&X) -> X::Elem,
    G: Fn(&X) -> X,
{
    pub(crate) fn step(&mut self, neg_gradfx: &X, squared_norm_2_gradfx: X::Elem) {
        let mut sigma_minus = X::Elem::one();
        let mut x_next = &self.x + sigma_minus * neg_gradfx;
        let one_half = X::Elem::cast_from_f32(0.5);
        let fx = self.func(&self.x);
        self.fcalls += 1;
        // The first if and else conditions guarantee having a segment [sigma_minus, sigma_plus]
        // such that sigma_minus satisfies the armijo condition and sigma_plus does not
        let mut sigma_plus =
            if self.func(&x_next) - fx <= -sigma_minus * self.gamma * squared_norm_2_gradfx {
                self.fcalls += 1;
                if self.grad(&x_next).dot(neg_gradfx) >= -self.beta * squared_norm_2_gradfx {
                    self.gcalls += 1;
                    (self.x, self.sigma) = (x_next, sigma_minus);
                    return;
                }
                // Computation of sigma_plus
                let two = X::Elem::cast_from_f32(2.);
                let mut sigma_plus = two;
                x_next = &self.x + sigma_plus * neg_gradfx;
                while self.func(&x_next) - fx <= -sigma_plus * self.gamma * squared_norm_2_gradfx {
                    sigma_plus = two * sigma_plus;
                    x_next = &self.x + sigma_plus * neg_gradfx;
                    self.fcalls += 1;
                }
                // At this stage sigma_plus is the smallest 2^k that does not satisfy the Armijo rule
                sigma_minus = sigma_plus * one_half; // it satisfies the Armijo rule
                sigma_plus
            } else {
                sigma_minus = one_half;
                x_next = &self.x + sigma_minus * neg_gradfx;
                while self.func(&x_next) - fx > -sigma_minus * self.gamma * squared_norm_2_gradfx {
                    sigma_minus = one_half * sigma_minus;
                    x_next = &self.x + sigma_minus * neg_gradfx;
                    self.fcalls += 1;
                }
                sigma_minus * (X::Elem::cast_from_f32(2.)) // does not satisfy the Armijo rule
            };
        x_next = &self.x + sigma_minus * neg_gradfx;
        while self.grad(&x_next).dot(neg_gradfx) < -self.beta * squared_norm_2_gradfx {
            self.gcalls += 1;
            let sigma = (sigma_minus + sigma_plus) * one_half;
            x_next = &self.x + sigma * neg_gradfx;
            if self.func(&x_next) - fx <= -sigma * self.gamma * squared_norm_2_gradfx {
                sigma_minus = sigma;
            } else {
                sigma_plus = sigma;
            }
            self.fcalls += 1;
        }
        self.x = &self.x + sigma_minus * neg_gradfx;
        self.sigma = sigma_minus;
    }
}
