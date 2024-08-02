//! `tuutal` is a library providing tools to optimize functions with scalar or multidimensional inputs.
//! It aims at the moment to reproduce and improve if possible the [optimization submodule of Python scipy 1.13.1][scipy].
//! It is backed by [ndarray][ndarr] crate for multidimensional optimization. For compatibility purpose, some of ndarray's objects
//! are imported into this crate.
//!
//! [scipy]: https://docs.scipy.org/doc/scipy-1.13.1/reference/optimize.html
//! [ndarr]: https://crates.io/crates/ndarray

// #![no_std]

/// A set of error handling objects.
pub mod error;

/// A set of tools to optimize functions when gradient computation is provided.
pub mod first_order;
/// ["Black-Box Optimization"][bbox] module, it provides tools to optimize
/// functions when only function evaluation is permitted.
///
/// [bbox]: https://en.wikipedia.org/wiki/Derivative-free_optimization
pub mod zero_order;

mod traits;
mod utils;

pub use error::{RootFindingError, TuutalError};
pub use first_order::{descent, Armijo, DescentParameter, PowellWolfe};
pub use ndarray::{array, s, Array1, Array2};

use num_traits::Num;
#[allow(unused)]
pub(crate) use utils::{is_between, l2_diff};

pub use traits::{Bound, Number, Optimizer, Scalar};
pub use zero_order::{
    bracket, brent_bounded, brent_root, brent_unbounded, brentq, nelder_mead, powell,
    NelderMeadIterates, PowellIterates,
};

pub(crate) use zero_order::Bounds;

#[derive(Debug)]
pub(crate) struct Counter<T = usize> {
    iter: T,
    fcalls: T,
    gcalls: T,
}

impl<T> Counter<T>
where
    T: Num,
{
    pub(crate) fn new() -> Self {
        Self {
            iter: T::zero(),
            fcalls: T::zero(),
            gcalls: T::zero(),
        }
    }
}
