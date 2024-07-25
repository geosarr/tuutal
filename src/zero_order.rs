mod nelder_mead;
mod powell;
mod scalar;

pub use nelder_mead::{nelder_mead, NelderMeadIterates};
pub use powell::{powell, PowellIterates};
pub use scalar::{
    bracket, brent_bounded, brent_unbounded,
    root::{brent_root, brentq},
};

use crate::VecType;

#[derive(Debug)]
pub(crate) struct Bounds<A> {
    lower: VecType<A>,
    upper: VecType<A>,
}

impl<A> Bounds<A> {
    pub fn new(lower: VecType<A>, upper: VecType<A>) -> Self {
        Self { lower, upper }
    }
    pub fn lower_bound(&self) -> &VecType<A> {
        &self.lower
    }
    pub fn upper_bound(&self) -> &VecType<A> {
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
