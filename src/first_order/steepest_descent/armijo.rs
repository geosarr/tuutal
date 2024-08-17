use core::ops::{Add, Mul};

use crate::{
    first_order::macros::{descent_rule, impl_optimizer_descent},
    traits::{VecDot, Vector},
    Counter, Optimizer,
};
use num_traits::{Float, One, Zero};
/// Hyperparameters used to compute step sizes in Armijo rule.
#[derive(Debug)]
pub struct ArmijoHyperParameter<T> {
    pub gamma: T,
    pub beta: T,
    pub epsilon: T,
}

descent_rule!(
    Armijo,
    [<X as Vector>::Elem; 1],
    [X::Elem::zero()],
    ArmijoHyperParameter,
    (),
    ()
);
impl_optimizer_descent!(Armijo, [<X as Vector>::Elem; 1], ArmijoHyperParameter, ());

impl<X, F, G, Farg, Garg>
    Armijo<X, F, G, [X::Elem; 1], ArmijoHyperParameter<X::Elem>, (), Farg, Garg>
where
    X: Vector + VecDot<X, Output = X::Elem>,
    for<'b> &'b X: Add<X, Output = X>,
    F: Fn(&X, &Farg) -> X::Elem,
    G: Fn(&X, &Garg) -> X,
{
    pub(crate) fn step(&mut self) {
        let mut sigma = X::Elem::one();
        let mut x_next = &self.x + sigma * &self.neg_gradfx;
        let fx = self.func(&self.x);
        self.counter.fcalls += 1;
        let (gamma, beta) = (self.hyper_params.gamma, self.hyper_params.beta);
        // NB: self.stop_metrics is the squared L2-norm of gradf(&x).
        while self.func(&x_next) - fx > -sigma * gamma * self.stop_metrics {
            self.counter.fcalls += 1;
            sigma = beta * sigma;
            x_next = &self.x + sigma * &self.neg_gradfx;
        }
        self.x = x_next;
        self.sigma[0] = sigma;
    }
}
