use core::ops::{Add, Mul};

use crate::{
    first_order::macros::{descent_rule, impl_optimizer_descent},
    traits::{VecDot, Vector},
    Counter, Optimizer,
};
use num_traits::{Float, One};

/// Hyperparameters used to compute step sizes in Adaptive Delta rule.
pub struct AdaDeltaHyperParameter<T> {
    pub gamma: T,
    pub beta: T,
    pub epsilon: T,
}
/// Accumulators computed in Adaptive Delta rule.
#[derive(Debug)]
pub struct AdaDeltaAccumulators<X> {
    pub grad: X,
    pub update: X,
}

descent_rule!(
    AdaDelta,
    X,
    [].into_iter().collect::<X>(),
    AdaDeltaHyperParameter,
    AdaDeltaAccumulators<X>,
    AdaDeltaAccumulators {
        grad: X::zero(1),
        update: X::zero(1)
    } // broadcasting is assumed for X.
);

impl_optimizer_descent!(AdaDelta, X, AdaDeltaHyperParameter, AdaDeltaAccumulators<X>);

impl<X, F, G> AdaDelta<X, F, G, X, AdaDeltaHyperParameter<X::Elem>, AdaDeltaAccumulators<X>>
where
    X: Vector,
    for<'b> &'b X: Add<X, Output = X> + Mul<&'b X, Output = X>,
    F: Fn(&X) -> X::Elem,
    G: Fn(&X) -> X,
{
    pub(crate) fn step(&mut self) {
        let (gamma, beta) = (self.hyper_params.gamma, self.hyper_params.beta);
        let squared_grad = &self.neg_gradfx * &self.neg_gradfx;
        self.accumulators.grad =
            gamma * &self.accumulators.grad + (X::Elem::one() - gamma) * &squared_grad;
        self.sigma = (beta + &self.accumulators.update)
            .into_iter()
            .map(|x| x.sqrt())
            .collect::<X>()
            / (beta + &self.accumulators.grad)
                .into_iter()
                .map(|g| g.sqrt())
                .collect::<X>();
        self.accumulators.update = gamma * &self.accumulators.update
            + (X::Elem::one() - gamma) * (&self.sigma * &self.sigma) * squared_grad;
        self.x = &self.x + &self.sigma * &self.neg_gradfx;
    }
}
