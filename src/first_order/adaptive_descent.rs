mod adadelta;
mod adagrad;

pub use adadelta::AdaDelta;
pub use adagrad::AdaGrad;

pub const ACCUM_GRAD: &str = "accum_grad";
pub const ACCUM_UPDATE: &str = "accum_update";
