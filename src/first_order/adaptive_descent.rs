mod adadelta;
mod adagrad;

pub(crate) use adadelta::AdaDelta;
pub(crate) use adagrad::AdaGrad;

pub(crate) const ACCUM_GRAD: &str = "accum_grad";
pub(crate) const ACCUM_UPDATE: &str = "accum_update";
