/// Complements num_traits float-pointing number Float trait by adding
/// easy access to the ten first positive integers (1. ..=10.).
pub trait DefaultValue: num_traits::Float {
    /// Tolerance value like 10<sup>n</sup> where n < 0 in general.
    fn tol(n: i32) -> Self;
    /// Converts from f32.
    fn from_f32(f: f32) -> Self;
}

macro_rules! impl_default_value(
  ( $( $t:ident ),* )=> {
      $(
          impl DefaultValue for $t {
            fn tol(n: i32) -> $t {
              (10 as $t).powi(n)
            }
            fn from_f32(f: f32) -> $t{
              f as $t
            }
          }
      )*
  }
);

impl_default_value!(f32, f64);
