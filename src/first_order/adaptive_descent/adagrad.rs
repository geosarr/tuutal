use core::ops::{Add, Mul};

use num_traits::Float;

use crate::{
    first_order::macros::{descent_rule, impl_optimizer_descent},
    traits::{VecDot, Vector},
    Counter, Optimizer, VarName,
};
use hashbrown::HashMap;

descent_rule!(
    AdaGrad,
    X,
    [].into_iter().collect::<X>(),
    [(VarName::AccumGrad, X::zero(1))].into()
);
impl_optimizer_descent!(AdaGrad, X);

impl<X, F, G> AdaGrad<X, F, G, X>
where
    X: Vector,
    for<'b> &'b X: Add<X, Output = X> + Mul<&'b X, Output = X>,
    F: Fn(&X) -> X::Elem,
    G: Fn(&X) -> X,
{
    pub(crate) fn step(&mut self) {
        let squared_grad = &self.neg_gradfx * &self.neg_gradfx;
        self.accumulators.insert(
            VarName::AccumGrad,
            &self.accumulators[&VarName::AccumGrad] + squared_grad,
        );
        let (gamma, beta) = (
            self.hyper_params[&VarName::Gamma],
            self.hyper_params[&VarName::Beta],
        );
        self.sigma = gamma
            / (beta + &self.accumulators[&VarName::AccumGrad])
                .into_iter()
                .map(|g| g.sqrt())
                .collect::<X>();
        self.x = &self.x + &self.sigma * &self.neg_gradfx;
    }
}
