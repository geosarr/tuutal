use core::ops::{Add, Mul};

use crate::{
    first_order::{
        // adaptive_descent::{VarName::AccumGrad, VarName::AccumUpdate},
        macros::{descent_rule, impl_optimizer_descent},
    },
    traits::{VecDot, Vector},
    Counter, Optimizer, VarName,
};
use hashbrown::HashMap;
use num_traits::{Float, One};

descent_rule!(
    AdaDelta,
    X,
    [].into_iter().collect::<X>(),
    [
        (VarName::AccumGrad, X::zero(1)),
        (VarName::AccumUpdate, X::zero(1))
    ]
    .into()
);
impl_optimizer_descent!(AdaDelta, X);

impl<X, F, G> AdaDelta<X, F, G, X>
where
    X: Vector,
    for<'b> &'b X: Add<X, Output = X> + Mul<&'b X, Output = X>,
    F: Fn(&X) -> X::Elem,
    G: Fn(&X) -> X,
{
    pub(crate) fn step(&mut self) {
        let (gamma, beta) = (
            self.hyper_params[&VarName::Gamma],
            self.hyper_params[&VarName::Beta],
        );
        let squared_grad = &self.neg_gradfx * &self.neg_gradfx;
        self.accumulators.insert(
            VarName::AccumGrad,
            gamma * &self.accumulators[&VarName::AccumGrad]
                + (X::Elem::one() - gamma) * &squared_grad,
        );
        let accum_grad = &self.accumulators[&VarName::AccumGrad];
        let accum_update = &self.accumulators[&VarName::AccumUpdate];
        self.sigma = (beta + accum_update)
            .into_iter()
            .map(|x| x.sqrt())
            .collect::<X>()
            / (beta + accum_grad)
                .into_iter()
                .map(|g| g.sqrt())
                .collect::<X>();
        self.accumulators.insert(
            VarName::AccumUpdate,
            gamma * accum_update
                + (X::Elem::one() - gamma) * (&self.sigma * &self.sigma) * squared_grad,
        );
        self.x = &self.x + &self.sigma * &self.neg_gradfx;
    }
}
