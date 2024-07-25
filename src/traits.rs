use crate::{s, Array, Bounds, MatrixType, VecType};
use num_traits::{Float, FromPrimitive};
use std::ops::Mul;

/// Complements num_traits float-pointing number Float trait by adding
/// conversion from f32 and provides easy access to exponential numbers.
pub trait Number: Float + FromPrimitive {
    /// Returns b<sup>n</sup>.
    fn exp_base(b: usize, n: i32) -> Self;
    /// Converts from f32.
    fn cast_from_f32(f: f32) -> Self;
}
macro_rules! impl_default_value(
  ( $( $t:ident ),* )=> {
      $(
          impl Number for $t {
            fn exp_base(b: usize, n: i32) -> $t {
              (b as $t).powi(n)
            }
            fn cast_from_f32(f: f32) -> $t{
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
    for<'a> Self: Number + Mul<X, Output = X> + Mul<&'a X, Output = X>,
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

/// Implements an iterator counting the number of iterations done so far.
pub trait Iterable<X>: std::iter::Iterator<Item = X> {
    /// Number of iterations done so far.
    fn nb_iter(&self) -> usize;
    /// Current iterate.
    fn iterate(&self) -> X;
}

/// Implements the notion of upper and lower bounds
pub trait Bound<T> {
    fn lower(&self, dim: usize) -> VecType<T>;
    fn upper(&self, dim: usize) -> VecType<T>;
}
impl<T> Bound<T> for (T, T)
where
    T: Copy,
{
    fn lower(&self, dim: usize) -> VecType<T> {
        Array::from(vec![self.0; dim])
    }
    fn upper(&self, dim: usize) -> VecType<T> {
        Array::from(vec![self.1; dim])
    }
}
impl<T> Bound<T> for Vec<(T, T)>
where
    T: Copy,
{
    fn lower(&self, dim: usize) -> VecType<T> {
        assert!(dim <= self.len());
        (0..dim).map(|i| self[i].0).collect()
    }
    fn upper(&self, dim: usize) -> VecType<T> {
        assert!(dim <= self.len());
        (0..dim).map(|i| self[i].1).collect()
    }
}
impl<T, V> Bound<T> for Option<V>
where
    T: Copy,
    V: Bound<T>,
{
    fn lower(&self, dim: usize) -> VecType<T> {
        if let Some(bounds) = self {
            bounds.lower(dim)
        } else {
            panic!("No lower bounds for None")
        }
    }
    fn upper(&self, dim: usize) -> VecType<T> {
        if let Some(bounds) = self {
            bounds.upper(dim)
        } else {
            panic!("No upper bounds for None")
        }
    }
}

impl<T> Bound<T> for Bounds<T>
where
    T: Copy,
{
    fn lower(&self, dim: usize) -> VecType<T> {
        let bound = self.lower_bound();
        assert!(dim <= bound.len());
        bound.slice(s![..dim]).to_owned()
    }
    fn upper(&self, dim: usize) -> VecType<T> {
        let bound = self.upper_bound();
        assert!(dim <= bound.len());
        bound.slice(s![..dim]).to_owned()
    }
}
