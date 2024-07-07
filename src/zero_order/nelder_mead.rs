use std::collections::HashMap;

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

pub struct NelderMeadIterates<'a, F, A> {
    f: F,
    x0: VecType<A>,
    // callback: Option<bool>,
    // maxiter: usize,
    maxfev: Option<usize>,
    // disp: bool,
    // return_all: bool,
    simplex: MatrixType<A>,
    xatol: A,
    fatol: A,
    // adaptive: bool,
    // bounds: Option<B>,
    _params: HashMap<&'a str, A>,
}

impl<'a, F, A> NelderMeadIterates<'a, F, A> {
    pub fn new<B>(
        f: F,
        mut x0: VecType<A>,
        maxfev: Option<usize>,
        initial_simplex: Option<MatrixType<A>>,
        xatol: A,
        fatol: A,
        adaptive: bool,
        bounds: Option<B>,
    ) -> Result<Self, TuutalError<VecType<A>>>
    where
        A: DefaultValue,
        B: Bounds<A>,
    {
        let dim_usize = x0.len();
        let dim = A::from_f32(dim_usize as f32);
        let (rho, chi, psi, sigma) = if adaptive {
            (
                A::one(),
                A::one() + A::from_f32(2.) / dim,
                A::from_f32(0.75) - A::one() / (A::from_f32(2.) * dim),
                A::one() - A::one() / dim,
            )
        } else {
            (
                A::one(),
                A::from_f32(2.),
                A::from_f32(0.5),
                A::from_f32(0.5),
            )
        };
        let _params = HashMap::from([
            ("rho", rho),
            ("chi", chi),
            ("psi", psi),
            ("sigma", sigma),
            // ("nonzdelt", A::from_f32(0.05)),
            // ("zdelt", A::from_f32(0.00025)),
        ]);
        if let Some(_bounds) = bounds {
            let (lower_bound, upper_bound) = (_bounds.lower(dim_usize), _bounds.upper(dim_usize));
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
            x0 = x0
                .iter()
                .zip(&lower_bound)
                .zip(&upper_bound)
                .map(|((x, l), u)| (*x).clamp(*l, *u))
                .collect::<VecType<A>>();
        }
        let simplex = if let Some(simplex) = initial_simplex {
            let (nrows, ncols) = (simplex.shape()[0], simplex.shape()[1]);
            if nrows != ncols + 1 {
                return Err(TuutalError::Simplex {
                    size: (nrows, ncols),
                    msg: "initial_simplex should be an array of shape (N+1, N)".to_string(),
                });
            }
            if dim_usize != ncols {
                return Err(TuutalError::Simplex {
                    size: (nrows, ncols),
                    msg: "Size of initial_simplex is not consistent with x0".to_string(),
                });
            }
            simplex
        } else {
            let mut sim = Array::zeros((dim_usize + 1, dim_usize));
            for k in 0..dim_usize {
                let mut y = x0.clone();
                if y[k] != A::zero() {
                    y[k] = (A::one() + A::from_f32(0.05)) * y[k];
                } else {
                    y[k] = A::from_f32(0.00025);
                }
                y.assign_to(sim.slice_mut(s![k, ..]));
            }
            sim
        };
        Ok(Self {
            f,
            x0,
            maxfev,
            simplex,
            xatol,
            fatol,
            // adaptive,
            // bounds,
            _params,
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
#[test]
fn test_minimize() {
    use crate::VecType;
    let f = |x: &VecType<f32>| x.dot(x);
    // neldermead(f)
}
