use core::ops::{Add, Mul};

use num_traits::Float;

use crate::{
    first_order::macros::{descent_rule, impl_optimizer_descent},
    traits::{VecDot, Vector},
    Counter, Optimizer,
};
/// Hyperparameters used to compute step sizes in Adaptive Gradient rule.
#[derive(Debug)]
pub struct AdaGradHyperParameter<T> {
    pub gamma: T,
    pub beta: T,
    pub epsilon: T,
}
/// Accumulators computed in Adaptive Gradient rule.
#[derive(Debug)]
pub struct AdaGradAccumulators<X> {
    pub grad: X,
}

descent_rule!(
    AdaGrad,
    X,
    [].into_iter().collect::<X>(),
    AdaGradHyperParameter,
    AdaGradAccumulators<X>,
    AdaGradAccumulators { grad: X::zero(1) } // broadcasting is assumed for X.
);
impl_optimizer_descent!(AdaGrad, X, AdaGradHyperParameter, AdaGradAccumulators<X>);

impl<X, F, G> AdaGrad<X, F, G, X, AdaGradHyperParameter<X::Elem>, AdaGradAccumulators<X>>
where
    X: Vector,
    for<'b> &'b X: Add<X, Output = X> + Mul<&'b X, Output = X>,
    F: Fn(&X) -> X::Elem,
    G: Fn(&X) -> X,
{
    pub(crate) fn step(&mut self) {
        let (gamma, beta) = (self.hyper_params.gamma, self.hyper_params.beta);
        let squared_grad = &self.neg_gradfx * &self.neg_gradfx;
        self.accumulators.grad = &self.accumulators.grad + squared_grad;
        self.sigma = gamma
            / (beta + &self.accumulators.grad)
                .into_iter()
                .map(|g| g.sqrt())
                .collect::<X>();
        self.x = &self.x + &self.sigma * &self.neg_gradfx;
    }
}
