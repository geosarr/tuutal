mod steepest_descent;
mod traits;
use ndarray::{
    prelude::{ArrayBase, Dim},
    OwnedRepr,
};
pub use steepest_descent::{steepest_descent, SteepestDescentIterates, SteepestDescentParameter};
pub use traits::DefaultValue;

pub type MatrixType<T> = ArrayBase<OwnedRepr<T>, Dim<[usize; 2]>>;
pub type VecType<T> = ArrayBase<OwnedRepr<T>, Dim<[usize; 1]>>;
