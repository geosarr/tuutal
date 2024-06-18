use thiserror::Error;

#[derive(Error, Debug)]
pub enum TuutalError {
    #[error("The algorithm terminated without finding a valid bracket.")]
    Bracketing,
    #[error(
        "No valid bracket was found before the maximum number of iterations `{0}` was reached."
    )]
    Convergence(String),
}

#[derive(Error, Debug)]
pub enum RootFindingError {
    #[error("The inputs a = {a:?} and b = {b:?} do not satisfy f(a) * f(b) < 0.")]
    Bracketing { a: String, b: String },
    #[error("Cannot interpolate, a = {a:?} and b = {b:?} have the same output f(a) = f(b).")]
    Interpolation { a: String, b: String },
}
