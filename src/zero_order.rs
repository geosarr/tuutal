mod nelder_mead;
mod powell;
mod scalar;

pub use nelder_mead::{nelder_mead, NelderMeadIterates};
pub use powell::{powell, PowellIterates};
pub use scalar::{bounded, bracket, brent_opt, root::brent_root};

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
