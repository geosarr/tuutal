mod adadelta;
mod adagrad;

pub(crate) use adadelta::adadelta;
pub(crate) use adagrad::adagrad;

pub(crate) const ACCUM_GRAD: &str = "accum_grad";
pub(crate) const ACCUM_UPDATE: &str = "accum_update";
