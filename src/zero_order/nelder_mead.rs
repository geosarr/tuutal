use std::collections::HashMap;
#[cfg(test)]
mod unit_test;
use ndarray::s;

use crate::{Array, DefaultValue, MatrixType, Scalar, TuutalError, VecType};

pub trait Bounds<T> {
    fn lower(&self, dim: usize) -> VecType<T>;
    fn upper(&self, dim: usize) -> VecType<T>;
    fn length(&self) -> usize;
}

impl<T> Bounds<T> for (T, T)
where
    T: Copy,
{
    fn lower(&self, dim: usize) -> VecType<T> {
        Array::from(vec![self.0; dim])
    }
    fn upper(&self, dim: usize) -> VecType<T> {
        Array::from(vec![self.1; dim])
    }
    fn length(&self) -> usize {
        2
    }
}

impl<T> Bounds<T> for Vec<(T, T)>
where
    T: Copy,
{
    fn lower(&self, dim: usize) -> VecType<T> {
        (0..dim).map(|i| self[i].0).collect()
    }
    fn upper(&self, dim: usize) -> VecType<T> {
        (0..dim).map(|i| self[i].1).collect()
    }
    fn length(&self) -> usize {
        self.len()
    }
}

impl<T, V> Bounds<T> for Option<V>
where
    T: Copy,
    V: Bounds<T>,
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
    fn length(&self) -> usize {
        if let Some(bounds) = self {
            bounds.length()
        } else {
            panic!("No length for None")
        }
    }
}

fn initial_simplex<A>(
    x0: VecType<A>,
    init_sim: Option<MatrixType<A>>,
) -> Result<MatrixType<A>, TuutalError<VecType<A>>>
where
    A: DefaultValue,
{
    let dim = x0.len();
    if let Some(simplex) = init_sim {
        let (nrows, ncols) = (simplex.nrows(), simplex.ncols());
        if nrows != ncols + 1 {
            return Err(TuutalError::Simplex {
                size: (nrows, ncols),
                msg: "initial_simplex should be an array of shape (N+1, N)".to_string(),
            });
        }
        if dim != ncols {
            return Err(TuutalError::Simplex {
                size: (nrows, ncols),
                msg: "Size of initial_simplex is not consistent with x0".to_string(),
            });
        }
        Ok(simplex)
    } else {
        let mut simplex = Array::zeros((dim + 1, dim));
        for k in 0..dim {
            let mut y = x0.clone();
            if y[k] != A::zero() {
                y[k] = (A::one() + A::from_f32(0.05)) * y[k];
            } else {
                y[k] = A::from_f32(0.00025);
            }
            y.assign_to(simplex.slice_mut(s![k + 1, ..]));
        }
        x0.assign_to(simplex.slice_mut(s![0, ..]));
        Ok(simplex)
    }
}

fn clamp<A, B>(x0: VecType<A>, bounds: Option<B>) -> Result<VecType<A>, TuutalError<VecType<A>>>
where
    A: DefaultValue,
    B: Bounds<A>,
{
    let dim = x0.len();
    if let Some(_bounds) = bounds {
        let (lower_bound, upper_bound) = (_bounds.lower(dim), _bounds.upper(dim));
        // check bounds
        if lower_bound.iter().zip(&upper_bound).any(|(l, u)| l > u) {
            return Err(TuutalError::BoundOrder {
                lower: lower_bound,
                upper: upper_bound,
            });
        }
        if lower_bound.iter().zip(&x0).any(|(l, x)| l > x)
            | x0.iter().zip(&upper_bound).any(|(x, u)| x > u)
        {
            println!("Initial guess is not within the specified bounds");
        }
        Ok(x0
            .iter()
            .zip(&lower_bound)
            .zip(&upper_bound)
            .map(|((x, l), u)| (*x).clamp(*l, *u))
            .collect())
    } else {
        Ok(x0)
    }
}

fn simplex_parameters<'a, A>(
    x0: VecType<A>,
    adaptive: bool,
) -> Result<(VecType<A>, HashMap<&'a str, A>), TuutalError<VecType<A>>>
where
    A: DefaultValue,
{
    let dim = A::from_f32(x0.len() as f32);
    let (rho, chi, psi, sigma) = if adaptive && (dim > A::zero()) {
        (
            A::one(),
            A::one() + A::from_f32(2.) / dim,
            A::from_f32(0.75) - A::one() / (A::from_f32(2.) * dim),
            A::one() - A::one() / dim,
        )
    } else if !adaptive {
        (
            A::one(),
            A::from_f32(2.),
            A::from_f32(0.5),
            A::from_f32(0.5),
        )
    } else {
        return Err(TuutalError::EmptyDimension { x: x0.clone() });
    };
    Ok((
        x0,
        HashMap::from([("rho", rho), ("chi", chi), ("psi", psi), ("sigma", sigma)]),
    ))
}
pub struct NelderMeadIterates<'a, F, A> {
    f: F,
    // callback: Option<bool>,
    // maxiter: usize,
    maxfev: usize,
    // disp: bool,
    // return_all: bool,
    simplex: MatrixType<A>,
    xatol: A,
    fatol: A,
    // adaptive: bool,
    // bounds: Option<B>,
    sim_params: HashMap<&'a str, A>,
}

impl<'a, F, A> NelderMeadIterates<'a, F, A> {
    pub fn new<B>(
        f: F,
        x0: VecType<A>,
        maxfev: Option<usize>,
        simplex: Option<MatrixType<A>>,
        xatol: A,
        fatol: A,
        adaptive: bool,
        bounds: Option<B>,
    ) -> Result<Self, TuutalError<VecType<A>>>
    where
        A: DefaultValue,
        B: Bounds<A>,
    {
        let (x0, sim_params) = match simplex_parameters(x0, adaptive) {
            Err(error) => return Err(error),
            Ok(params) => params,
        };
        let x0 = match clamp(x0, bounds) {
            Err(error) => return Err(error),
            Ok(x0) => x0,
        };
        let simplex = match initial_simplex(x0, simplex) {
            Ok(sim) => sim,
            Err(error) => return Err(error),
        };
        let maxfev = if let Some(max) = maxfev {
            max
        } else {
            simplex.ncols() * 200
        };
        Ok(Self {
            f,
            maxfev,
            simplex,
            xatol,
            fatol,
            // adaptive,
            // bounds,
            sim_params,
        })
    }
}

impl<'a, F, A> std::iter::Iterator for NelderMeadIterates<'a, F, A>
where
    F: Fn(&VecType<A>) -> A,
    A: Scalar<VecType<A>>,
{
    type Item = VecType<A>;
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}
