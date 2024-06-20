/// Complements num_traits float-pointing number Float trait by adding
/// easy access to the ten first positive integers (1. ..=10.).
pub trait DefaultValue: num_traits::Float {
    /// Tolerance value like 10<sup>n</sup> where n < 0 in general.
    fn tol(n: i32) -> Self;
    /// Returns 0.5
    fn one_half() -> Self;
    /// Returns 2.
    fn two() -> Self;
    /// Returns 3.
    fn three() -> Self;
    /// Returns 4.
    fn four() -> Self;
    /// Returns 5.
    fn five() -> Self;
    /// Returns 6.
    fn six() -> Self;
    /// Returns 7.
    fn seven() -> Self;
    /// Returns 8.
    fn eight() -> Self;
    /// Returns 9.
    fn nine() -> Self;
    /// Returns 10.
    fn ten() -> Self;
}

macro_rules! impl_default_value(
  ( $( $t:ident ),* )=> {
      $(
          impl DefaultValue for $t {
            fn tol(n: i32) -> $t {
              (10 as $t).powi(n)
            }
            fn one_half() -> $t{
              0.5 as $t
            }
            fn two() -> $t {
              2 as $t
            }
            fn three() -> $t {
              3 as $t
            }
            fn four() -> $t {
              4 as $t
            }
            fn five() -> $t {
              5 as $t
            }
            fn six() -> $t {
              6 as $t
            }
            fn seven() -> $t {
              7 as $t
            }
            fn eight() -> $t {
              8 as $t
            }
            fn nine() -> $t {
              9 as $t
            }
            fn ten() -> $t {
              10 as $t
            }
          }
      )*
  }
);

impl_default_value!(f32, f64);
