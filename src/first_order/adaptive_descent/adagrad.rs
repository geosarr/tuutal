use core::ops::{Add, Div, Mul};

use num_traits::Float;

use crate::{
    first_order::{
        adaptive_descent::ACCUM_GRAD,
        macros::{descent_rule, impl_iterator_descent},
    },
    traits::{VecDot, Vector},
    Counter,
};
use hashbrown::HashMap;

pub(crate) fn adagrad<A, X>(accum_grad: &mut X, squared_grad: X, gamma: A, epsilon: A) -> X
where
    for<'a> A: Float + Add<&'a X, Output = X> + Div<X, Output = X>,
    for<'b> &'b X: Add<X, Output = X>,
    X: FromIterator<A> + IntoIterator<Item = A> + Clone,
{
    *accum_grad = &*accum_grad + squared_grad;
    gamma
        / (epsilon + &*accum_grad)
            .into_iter()
            .map(|g| g.sqrt())
            .collect::<X>()
}

descent_rule!(
    AdaGrad,
    X,
    [].into_iter().collect::<X>(),
    [(ACCUM_GRAD, X::zero(1))].into()
);
impl_iterator_descent!(AdaGrad, X);

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
