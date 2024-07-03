use std::ops::{Mul, Sub};

use crate::{MatrixType, VecType};

/// Complements num_traits float-pointing number Float trait by adding
/// conversion from f32 and provides easy access to exponential numbers.
pub trait DefaultValue: num_traits::Float {
    /// Returns b<sup>n</sup>.
    fn exp_base(b: usize, n: i32) -> Self;
    /// Converts from f32.
    fn from_f32(f: f32) -> Self;
}
macro_rules! impl_default_value(
  ( $( $t:ident ),* )=> {
      $(
          impl DefaultValue for $t {
            fn exp_base(b: usize, n: i32) -> $t {
              (b as $t).powi(n)
            }
            fn from_f32(f: f32) -> $t{
              f as $t
            }
          }
      )*
  }
);
impl_default_value!(f32, f64);

/// Implements scalar properties and matrices vs scalar operations.
pub trait Scalar<X>
where
    for<'a> Self: DefaultValue
        + Sub<Self, Output = Self>
        + Mul<Self, Output = Self>
        + Mul<X, Output = X>
        + Mul<&'a X, Output = X>
        + PartialOrd
        + Copy,
{
}
macro_rules! impl_scalar(
  ( $( $t:ident ),* )=> {
      $(
        impl Scalar<VecType<$t>> for $t {}
        impl Scalar<MatrixType<$t>> for $t {}
      )*
  }
);
impl_scalar!(f32, f64);

/// Implements an iterator counting the number of iterations don so far.
pub trait Iterable<X>: std::iter::Iterator<Item = X> {
    fn nb_iter(&self) -> usize;
}
