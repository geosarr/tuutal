pub mod error;
pub mod scalar;
mod steepest_descent;
mod traits;
pub use error::{RootFindingError, TuutalError};
pub use ndarray::{
    array,
    prelude::{ArrayBase, Dim},
    s, OwnedRepr,
};
pub use scalar::bracket;
pub use steepest_descent::{steepest_descent, SteepestDescentIterates, SteepestDescentParameter};
pub use traits::DefaultValue;

pub type MatrixType<T> = ArrayBase<OwnedRepr<T>, Dim<[usize; 2]>>;
pub type VecType<T> = ArrayBase<OwnedRepr<T>, Dim<[usize; 1]>>;
