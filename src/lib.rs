//! `tuutal` is a library providing tools to optimize functions with scalar or multidimensional inputs.
//! It aims at the moment to reproduce and improve if possible the [optimization submodule of Python scipy 1.13.1][scipy].
//! It is backed by [ndarray][ndarr] crate for multidimensional optimization. For compatibility purpose, some of ndarray's objects
//! are imported into this crate.
//!
//! [scipy]: https://docs.scipy.org/doc/scipy-1.13.1/reference/optimize.html
//! [ndarr]: https://crates.io/crates/ndarray

/// A set of error handling objects.
pub mod error;

/// ["Black-Box Optimization"][bbox] module, it provides tools to optimize
/// functions when only function evaluation is permitted.
///
/// [bbox]: https://en.wikipedia.org/wiki/Derivative-free_optimization
pub mod zero_order;

/// A set of tools to optimize functions when gradient computation is provided.
pub mod first_order;

mod traits;

pub use error::{RootFindingError, TuutalError};
pub use first_order::{steepest_descent, SteepestDescentIterates, SteepestDescentParameter};
pub use ndarray::{array, s, Array};
use ndarray::{
    prelude::{ArrayBase, Dim},
    OwnedRepr,
};

pub use traits::{DefaultValue, Iterable, Scalar};
pub use zero_order::{bracket, brent_opt, brent_root};

/// Two dimensional owned matrix
pub type MatrixType<T> = ArrayBase<OwnedRepr<T>, Dim<[usize; 2]>>;

/// One dimensional owned matrix or vector.
pub type VecType<T> = ArrayBase<OwnedRepr<T>, Dim<[usize; 1]>>;

/// Generic function to launch an optimization routine when intermediate iterates are not needed.
pub(crate) fn optimize<X: Clone, I: Iterable<X>>(
    mut iterable: I,
    maxiter: usize,
) -> Result<X, TuutalError<X>> {
    let mut iterate_star = iterable.iterate().clone();
    while let Some(x) = iterable.next() {
        iterate_star = x;
        if iterable.nb_iter() > maxiter {
            return Err(TuutalError::Convergence {
                iterate: iterate_star,
                maxiter: maxiter.to_string(),
            });
        }
    }
    Ok(iterate_star)
}
