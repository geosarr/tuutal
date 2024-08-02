use crate::{s, Array1, Array2, Bounds, TuutalError};
extern crate alloc;
use alloc::vec::Vec;
use core::ops::{Add, Div, Mul, Neg, Sub};
use ndarray::linalg::Dot;
use num_traits::{Float, FromPrimitive};

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
        impl Scalar<Array1<$t>> for $t {}
        impl Scalar<Array2<$t>> for $t {}
      )*
  }
);
impl_scalar!(f32, f64);

pub trait Vector:
    Neg<Output = Self>
    + Add<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Sub<Output = Self>
    + FromIterator<Self::Elem>
    + IntoIterator<Item = Self::Elem>
    + Sized
where
    Self::Elem: Scalar<Self>,
{
    type Elem;
    fn len(&self) -> usize;
    fn zero(size: usize) -> Self;
}
impl<T> Vector for Array1<T>
where
    T: Scalar<Array1<T>>,
{
    type Elem = T;
    fn len(&self) -> usize {
        Array1::len(self)
    }
    fn zero(size: usize) -> Self {
        Array1::from_elem(size, T::zero())
    }
}

/// Dot operation between vectors.
pub trait VecDot<Rhs = Self> {
    type Output;
    fn dot(&self, rhs: &Rhs) -> Self::Output;
}
impl<T> VecDot<Array1<T>> for Array1<T>
where
    Self: Dot<Self, Output = T>,
{
    type Output = T;
    fn dot(&self, rhs: &Self) -> Self::Output {
        Dot::dot(self, rhs)
    }
}

/// Implements an iterator counting the number of iterations done so far and a full optimization routine.
pub trait Optimizer: core::iter::Iterator<Item = Self::Iterate> {
    type Iterate;
    /// Number of iterations done so far.
    fn nb_iter(&self) -> usize;
    /// Current iterate.
    fn iterate(&self) -> Self::Iterate;
    /// Full optimization routine.
    fn optimize(
        &mut self,
        maxiter: Option<usize>,
    ) -> Result<Self::Iterate, TuutalError<Self::Iterate>> {
        let maxiter = maxiter.unwrap_or(1000);
        while let Some(x) = self.next() {
            if self.nb_iter() > maxiter {
                return Err(TuutalError::Convergence {
                    iterate: x,
                    maxiter,
                });
            }
        }
        Ok(self.iterate())
    }
}

/// Implements the notion of upper and lower bounds
pub trait Bound<T> {
    fn lower(&self, dim: usize) -> Array1<T>;
    fn upper(&self, dim: usize) -> Array1<T>;
}
impl<T> Bound<T> for (T, T)
where
    T: Copy,
{
    fn lower(&self, dim: usize) -> Array1<T> {
        Array1::from_elem(dim, self.0)
    }
    fn upper(&self, dim: usize) -> Array1<T> {
        Array1::from_elem(dim, self.1)
    }
}
impl<T> Bound<T> for Vec<(T, T)>
where
    T: Copy,
{
    fn lower(&self, dim: usize) -> Array1<T> {
        assert!(dim <= self.len());
        (0..dim).map(|i| self[i].0).collect()
    }
    fn upper(&self, dim: usize) -> Array1<T> {
        assert!(dim <= self.len());
        (0..dim).map(|i| self[i].1).collect()
    }
}
impl<T, V> Bound<T> for Option<V>
where
    T: Copy,
    V: Bound<T>,
{
    fn lower(&self, dim: usize) -> Array1<T> {
        if let Some(bounds) = self {
            bounds.lower(dim)
        } else {
            panic!("No lower bounds for None")
        }
    }
    fn upper(&self, dim: usize) -> Array1<T> {
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
    fn lower(&self, dim: usize) -> Array1<T> {
        let bound = self.lower_bound();
        assert!(dim <= bound.len());
        bound.slice(s![..dim]).to_owned()
    }
    fn upper(&self, dim: usize) -> Array1<T> {
        let bound = self.upper_bound();
        assert!(dim <= bound.len());
        bound.slice(s![..dim]).to_owned()
    }
}
