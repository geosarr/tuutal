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
    Convergence { iterate: X, maxiter: usize },
    /// This error occurs when at least one lower bound is greater than an upper bound.
    ///
    /// It holds the bounds values.
    #[error("At least one lower bound is greater than an upper bound")]
    BoundOrder { lower: X, upper: X },
    /// This error occurs when a simplex is not consistent with the variables dimension
    ///
    /// Its holds the simplex size and a explaining message.
    #[error("Simplex with the wrong size.")]
    Simplex { size: (usize, usize), msg: String },
    /// This error occurs when a vector of size 0 is wrongly used during an algorithm.
    ///
    /// It hols the empy vector.
    #[error("Empty vector.")]
    EmptyDimension { x: X },
    /// This error occurs when the maximum number of function calls was or will be reached.
    ///
    /// It holds the maximum number of function evaluation.
    #[error("Maximum function evaluation number is `{num:?}`")]
    MaxFunCall { num: usize },
    /// This error occurs when an undesired infinite value is encountered.
    ///
    /// It hols the value.
    #[error("Infinity value encoutered.")]
    Infinity { x: X },
    /// This error occurs when an undesired nan value is encountered.
    ///
    /// It hols the value.
    #[error("Nan value encoutered.")]
    Nan { x: X },
}

/// Handles types of errors occuring during a root finding algorithm.
#[derive(Error, Debug, PartialEq)]
pub enum RootFindingError<X> {
    /// This error occurs when a mandatory condition f(a) * f(b) < 0 is not satisfied.
    #[error("The inputs a = {a:?} and b = {b:?} do not satisfy f(a) * f(b) < 0.")]
    Bracketing { a: X, b: X },
    /// This error occurs when interpolation cannot be done using inputs a and b with the same output.
    #[error("Cannot interpolate, a = {a:?} and b = {b:?} have the same output f(a) = f(b).")]
    Interpolation { a: X, b: X },
}
