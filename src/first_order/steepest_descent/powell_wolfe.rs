use core::ops::{Add, Mul};

use crate::{
    first_order::macros::{descent_rule, impl_optimizer_descent},
    traits::{Number, VecDot, Vector},
    Counter, Optimizer, VarName,
};
use hashbrown::HashMap;
use num_traits::{Float, One, Zero};

descent_rule!(
    PowellWolfe,
    [<X as Vector>::Elem; 1],
    [X::Elem::zero()],
    [].into()
);
impl_optimizer_descent!(PowellWolfe, [X::Elem; 1]);

impl<X, F, G> PowellWolfe<X, F, G, [X::Elem; 1]>
where
    X: Vector + VecDot<X, Output = X::Elem> + Add<X, Output = X>,
    for<'b> &'b X: Add<X, Output = X>,
    F: Fn(&X) -> X::Elem,
    G: Fn(&X) -> X,
{
    pub(crate) fn step(&mut self) {
        let mut sigma_minus = X::Elem::one();
        let mut x_next = &self.x + sigma_minus * &self.neg_gradfx;
        let one_half = X::Elem::cast_from_f32(0.5);
        let fx = self.func(&self.x);
        self.counter.fcalls += 1;
        let (gamma, beta) = (
            self.hyper_params[&VarName::Gamma],
            self.hyper_params[&VarName::Beta],
        );
        // The first if and else conditions guarantee having a segment [sigma_minus, sigma_plus]
        // such that sigma_minus satisfies the armijo condition and sigma_plus does not
        // NB: self.stop_metrics is the squared L2-norm of gradf(&x).
        let mut sigma_plus = if self.func(&x_next) - fx <= -sigma_minus * gamma * self.stop_metrics
        {
            self.counter.fcalls += 1;
            if self.grad(&x_next).dot(&self.neg_gradfx) >= -beta * self.stop_metrics {
                self.counter.gcalls += 1;
                (self.x, self.sigma[0]) = (x_next, sigma_minus);
                return;
            }
            // Computation of sigma_plus
            let two = X::Elem::cast_from_f32(2.);
            let mut sigma_plus = two;
            x_next = &self.x + sigma_plus * &self.neg_gradfx;
            while self.func(&x_next) - fx <= -sigma_plus * gamma * self.stop_metrics {
                sigma_plus = two * sigma_plus;
                x_next = &self.x + sigma_plus * &self.neg_gradfx;
                self.counter.fcalls += 1;
            }
            // At this stage sigma_plus is the smallest 2^k that does not satisfy the Armijo rule
            sigma_minus = sigma_plus * one_half; // it satisfies the Armijo rule
            sigma_plus
        } else {
            sigma_minus = one_half;
            x_next = &self.x + sigma_minus * &self.neg_gradfx;
            while self.func(&x_next) - fx > -sigma_minus * gamma * self.stop_metrics {
                sigma_minus = one_half * sigma_minus;
                x_next = &self.x + sigma_minus * &self.neg_gradfx;
                self.counter.fcalls += 1;
            }
            sigma_minus * (X::Elem::cast_from_f32(2.)) // does not satisfy the Armijo rule
        };
        x_next = &self.x + sigma_minus * &self.neg_gradfx;
        while self.grad(&x_next).dot(&self.neg_gradfx) < -beta * self.stop_metrics {
            self.counter.gcalls += 1;
            let sigma = (sigma_minus + sigma_plus) * one_half;
            x_next = &self.x + sigma * &self.neg_gradfx;
            if self.func(&x_next) - fx <= -sigma * gamma * self.stop_metrics {
                sigma_minus = sigma;
            } else {
                sigma_plus = sigma;
            }
            self.counter.fcalls += 1;
        }
        self.x = &self.x + sigma_minus * &self.neg_gradfx;
        self.sigma[0] = sigma_minus;
    }
}
