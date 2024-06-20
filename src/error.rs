use thiserror::Error;

/// Handles types of errors occuring during optimization.
#[derive(Error, Debug)]
pub enum TuutalError {
    /// This error occurs in a bracket finding algorithm.
    #[error("The algorithm terminated without finding a valid bracket.")]
    Bracketing,
    /// This error occurs when an optimization algorithm did not converge.
    #[error(
        "No valid solution was found before the maximum number of iterations `{0}` was reached."
    )]
    Convergence(String),
}

/// Handles types of errors occuring during a root finding algorithm.
#[derive(Error, Debug)]
pub enum RootFindingError {
    /// This error occurs when a mandatory condition f(a) * f(b) < 0 is not satisfied.
    #[error("The inputs a = {a:?} and b = {b:?} do not satisfy f(a) * f(b) < 0.")]
    Bracketing { a: String, b: String },
    /// This error occurs when interpolation cannot be done using inputs a and b with the same output.
    #[error("Cannot interpolate, a = {a:?} and b = {b:?} have the same output f(a) = f(b).")]
    Interpolation { a: String, b: String },
}
