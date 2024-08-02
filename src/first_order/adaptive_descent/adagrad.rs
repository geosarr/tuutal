use core::ops::{Add, Mul};

use num_traits::Float;

use crate::{
    first_order::{
        adaptive_descent::ACCUM_GRAD,
        macros::{descent_rule, impl_optimizer_descent},
    },
    traits::{VecDot, Vector},
    Counter, Optimizer,
};
use hashbrown::HashMap;

descent_rule!(
    AdaGrad,
    X,
    [].into_iter().collect::<X>(),
    [(ACCUM_GRAD, X::zero(1))].into()
);
impl_optimizer_descent!(AdaGrad, X);

impl<'a, X, F, G> AdaGrad<'a, X, F, G, X>
where
    X: Vector,
    for<'b> &'b X: Add<X, Output = X> + Mul<&'b X, Output = X>,
    F: Fn(&X) -> X::Elem,
    G: Fn(&X) -> X,
{
    pub(crate) fn step(&mut self) {
        let squared_grad = &self.neg_gradfx * &self.neg_gradfx;
        self.accumulators
            .insert(ACCUM_GRAD, &self.accumulators[ACCUM_GRAD] + squared_grad);
        let (gamma, beta) = (self.hyper_params["gamma"], self.hyper_params["beta"]);
        self.sigma = gamma
            / (beta + &self.accumulators[ACCUM_GRAD])
                .into_iter()
                .map(|g| g.sqrt())
                .collect::<X>();
        self.x = &self.x + &self.sigma * &self.neg_gradfx;
    }
}
