//! `tuutal` is a library providing tools to optimize functions with scalar or multidimensional inputs.
//! It aims at the moment to reproduce and improve if possible the optimization submodule of Python scipy 1.13.1.
//! It is backed by **ndarray** crate for multidimensional optimization, for compatibility purpose some of its objects
//! are imported.

/// A set of error handling objects.
pub mod error;

/// "**Black-Box Optimization**" module, it provides tools to optimize
/// functions when only function evaluation is permitted.
pub mod zero_order;

/// A set of tools to optimize functions when gradient computation is provided.
pub mod first_order;

mod traits;

pub use error::{RootFindingError, TuutalError};
pub use first_order::{steepest_descent, SteepestDescentIterates, SteepestDescentParameter};
pub use ndarray::{array, s};
use ndarray::{
    prelude::{ArrayBase, Dim},
    OwnedRepr,
};

pub use traits::DefaultValue;
pub use zero_order::{bracket, brent_opt, brent_root};

pub type MatrixType<T> = ArrayBase<OwnedRepr<T>, Dim<[usize; 2]>>;
pub type VecType<T> = ArrayBase<OwnedRepr<T>, Dim<[usize; 1]>>;
