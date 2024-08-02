use core::ops::{Add, Mul};

use crate::{
    first_order::macros::{descent_rule, impl_optimizer_descent},
    traits::{VecDot, Vector},
    Counter, Optimizer,
};
use hashbrown::HashMap;
use num_traits::{Float, One, Zero};

descent_rule!(
    Armijo,
    [<X as Vector>::Elem; 1],
    [X::Elem::zero()],
    [].into()
);
impl_optimizer_descent!(Armijo, [<X as Vector>::Elem; 1]);

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
        self.sigma[0] = sigma;
    }
}
