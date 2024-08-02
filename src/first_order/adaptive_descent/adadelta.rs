use core::ops::{Add, Mul};

use crate::{
    first_order::{
        adaptive_descent::{ACCUM_GRAD, ACCUM_UPDATE},
        macros::{descent_rule, impl_optimizer_descent},
    },
    traits::{VecDot, Vector},
    Counter, Optimizer,
};
use hashbrown::HashMap;
use num_traits::{Float, One};

descent_rule!(
    AdaDelta,
    X,
    [].into_iter().collect::<X>(),
    [(ACCUM_GRAD, X::zero(1)), (ACCUM_UPDATE, X::zero(1))].into()
);
impl_optimizer_descent!(AdaDelta, X);

impl<'a, X, F, G> AdaDelta<'a, X, F, G, X>
where
    X: Vector,
    for<'b> &'b X: Add<X, Output = X> + Mul<&'b X, Output = X>,
    F: Fn(&X) -> X::Elem,
    G: Fn(&X) -> X,
{
    pub(crate) fn step(&mut self) {
        let (gamma, beta) = (self.hyper_params["gamma"], self.hyper_params["beta"]);
        let squared_grad = &self.neg_gradfx * &self.neg_gradfx;
        self.accumulators.insert(
            ACCUM_GRAD,
            gamma * &self.accumulators[ACCUM_GRAD] + (X::Elem::one() - gamma) * &squared_grad,
        );
        let accum_grad = &self.accumulators[ACCUM_GRAD];
        let accum_update = &self.accumulators[ACCUM_UPDATE];
        self.sigma = (beta + accum_update)
            .into_iter()
            .map(|x| x.sqrt())
            .collect::<X>()
            / (beta + accum_grad)
                .into_iter()
                .map(|g| g.sqrt())
                .collect::<X>();
        self.accumulators.insert(
            ACCUM_UPDATE,
            gamma * accum_update
                + (X::Elem::one() - gamma) * (&self.sigma * &self.sigma) * squared_grad,
        );
        self.x = &self.x + &self.sigma * &self.neg_gradfx;
    }
}
