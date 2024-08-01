// use core::ops::{Add, Div};

// use num_traits::Float;

// pub(crate) fn adagrad<A, X>(accum_grad: &mut X, squared_grad: X, gamma: A, epsilon: A) -> X
// where
//     for<'a> A: Float + Add<&'a X, Output = X> + Div<X, Output = X>,
//     for<'b> &'b X: Add<X, Output = X>,
//     X: FromIterator<A> + IntoIterator<Item = A> + Clone,
// {
//     *accum_grad = &*accum_grad + squared_grad;
//     gamma
//         / (epsilon + &*accum_grad)
//             .into_iter()
//             .map(|g| g.sqrt())
//             .collect::<X>()
// }
