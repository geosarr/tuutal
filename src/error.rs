use thiserror::Error;

/// Handles types of errors occuring during optimization.
#[derive(Error, Debug, PartialEq)]
pub enum TuutalError<X> {
    /// This error occurs when a bracket finding algorithm fails.
    ///
    /// It holds also the current iterate of the algorithm when this error is thrown.
    #[error("The algorithm terminated without finding a valid bracket.")]
    Bracketing { iterate: X },
    /// This error occurs when an optimization algorithm did not converge.
    ///
    /// It holds also the current iterate of the algorithm when this error is thrown.
    #[error(
        "No valid solution was found before the maximum number of iterations `{maxiter:?}` was reached."
    )]
    Convergence { iterate: X, maxiter: String },
}

/// Handles types of errors occuring during a root finding algorithm.
#[derive(Error, Debug, PartialEq)]
pub enum RootFindingError {
    /// This error occurs when a mandatory condition f(a) * f(b) < 0 is not satisfied.
    #[error("The inputs a = {a:?} and b = {b:?} do not satisfy f(a) * f(b) < 0.")]
    Bracketing { a: String, b: String },
    /// This error occurs when interpolation cannot be done using inputs a and b with the same output.
    #[error("Cannot interpolate, a = {a:?} and b = {b:?} have the same output f(a) = f(b).")]
    Interpolation { a: String, b: String },
}
