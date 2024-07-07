mod nelder_mead;
mod scalar;

pub use nelder_mead::NelderMeadIterates;
pub use scalar::{bracket, brent_opt, root::brent_root};
