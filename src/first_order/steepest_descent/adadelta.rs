use std::ops::{Add, Div, Mul};

use num_traits::Float;

pub(crate) fn adadelta<A, X>(
    accum_grad: &mut X,
    accum_update: &X,
    squared_grad: &X,
    gamma: A,
    epsilon: A,
) -> X
where
    for<'a> A: Float + Add<&'a X, Output = X> + Mul<&'a X, Output = X> + Mul<X, Output = X>,
    X: Add<X, Output = X> + Div<X, Output = X> + FromIterator<A> + IntoIterator<Item = A>,
{
    *accum_grad = gamma * &*accum_grad + (A::one() - gamma) * squared_grad;
    (epsilon + accum_update)
        .into_iter()
        .map(|g| g.sqrt())
        .collect::<X>()
        / (epsilon + &*accum_grad)
            .into_iter()
            .map(|g| g.sqrt())
            .collect::<X>()
}
