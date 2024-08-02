//! As such, the objective function output when convergence is reached and the number of function calls
//! during an algorithm is a simple yet arguably a good measure of optimality.

mod nelder_mead;
mod powell;
mod scalar;

pub use nelder_mead::{nelder_mead, NelderMeadIterates};
pub use powell::{powell, PowellIterates};
pub use scalar::{
    bracket, brent_bounded, brent_unbounded,
    root::{brent_root, brentq},
};

use crate::Array1;

#[derive(Debug)]
pub(crate) struct Bounds<A> {
    lower: Array1<A>,
    upper: Array1<A>,
}

impl<A> Bounds<A> {
    pub fn new(lower: Array1<A>, upper: Array1<A>) -> Self {
        Self { lower, upper }
    }
    pub fn lower_bound(&self) -> &Array1<A> {
        &self.lower
    }
    pub fn upper_bound(&self) -> &Array1<A> {
        &self.upper
    }
}

pub(crate) fn default_nb_iter(
    dim: usize,
    maxiter: Option<usize>,
    maxfev: Option<usize>,
    def: usize,
) -> (usize, usize) {
    if maxiter.is_none() && maxfev.is_none() {
        (dim * def, dim * def)
    } else if maxiter.is_none() {
        // Convert remaining Nones, to np.inf, unless the other is np.inf, in
        // which case use the default to avoid unbounded iteration
        let maxfev = maxfev.unwrap();
        let maxiter = if maxfev == usize::MAX {
            dim * def
        } else {
            usize::MAX
        };
        (maxiter, maxfev)
    } else if let Some(maxfcalls) = maxfev {
        (maxiter.unwrap(), maxfcalls)
    } else {
        let maxiter = maxiter.unwrap();
        let maxfev = if maxiter == usize::MAX {
            dim * def
        } else {
            usize::MAX
        };
        (maxiter, maxfev)
    }
}
