mod adaptive_descent;
mod macros;
mod steepest_descent;

pub use steepest_descent::{descent, Armijo, DescentParameter, PowellWolfe};
