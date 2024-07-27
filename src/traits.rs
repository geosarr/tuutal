use crate::{s, Array, Bounds, MatrixType, VecType};
use ndarray::linalg::Dot;
use num_traits::{Float, FromPrimitive, Zero};
use std::ops::{Add, Div, Mul, Sub};

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

/// Implements scalar properties and matrices/vectors vs scalar operations.
pub trait Scalar<X>
where
    for<'a> Self: Number
        + Add<X, Output = X>
        + Sub<X, Output = X>
        + Mul<X, Output = X>
        + Div<X, Output = X>
        + Add<&'a X, Output = X>
        + Sub<&'a X, Output = X>
        + Mul<&'a X, Output = X>
        + Div<&'a X, Output = X>,
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

/// Dot operation between vectors.
pub trait VecDot<Rhs = Self> {
    type Output;
    fn dot(&self, rhs: &Rhs) -> Self::Output;
}

impl<T> VecDot<VecType<T>> for VecType<T>
where
    Self: Dot<Self, Output = T>,
{
    type Output = T;
    fn dot(&self, rhs: &Self) -> Self::Output {
        Dot::dot(self, rhs)
    }
}

pub trait VecZero {
    fn zero(size: usize) -> Self;
}
impl<T> VecZero for VecType<T>
where
    T: Zero + Clone,
{
    fn zero(size: usize) -> Self {
        Array::from(vec![T::zero(); size])
    }
}

pub trait VecInfo {
    fn len(&self) -> usize;
}
impl<T> VecInfo for VecType<T> {
    fn len(&self) -> usize {
        VecType::len(self)
    }
}

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
