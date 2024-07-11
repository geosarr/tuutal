use std::collections::HashMap;
#[cfg(test)]
mod unit_test;
use ndarray::s;

use crate::{Array, MatrixType, Number, Scalar, TuutalError, VecType};

type SimplexParameterResult<'a, A> =
    Result<(VecType<A>, HashMap<&'a str, A>), TuutalError<VecType<A>>>;
pub trait Bound<T> {
    fn lower(&self, dim: usize) -> VecType<T>;
    fn upper(&self, dim: usize) -> VecType<T>;
    fn length(&self) -> usize;
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
    fn length(&self) -> usize {
        2
    }
}

impl<T> Bound<T> for Vec<(T, T)>
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
    fn length(&self) -> usize {
        if let Some(bounds) = self {
            bounds.length()
        } else {
            panic!("No length for None")
        }
    }
}

/// Gets the hyperparameters of the Nelder-Mead Algorithm.
fn simplex_parameters<'a, A>(x0: VecType<A>, adaptive: bool) -> SimplexParameterResult<'a, A>
where
    A: Number,
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

/// Builds the k-th vector defining a simplex from vector x0.
///
/// k should be < x0.len().
fn simplex_vector<A>(x0: &VecType<A>, k: usize) -> VecType<A>
where
    A: Number,
{
    let mut y = x0.clone();
    if y[k] != A::zero() {
        y[k] = (A::one() + A::from_f32(0.05)) * y[k];
    } else {
        y[k] = A::from_f32(0.00025);
    }
    y
}

/// Function that makes sure that: x[k] = min(upper_bound[k], max(x[k], lower_bound[k]))
///
/// User must make sure that lower_bound[k] <= upper_bound[k] for all k.
fn clamp_vec<A>(x: VecType<A>, lower_bound: &VecType<A>, upper_bound: &VecType<A>) -> VecType<A>
where
    A: Number,
{
    x.iter()
        .zip(lower_bound)
        .zip(upper_bound)
        .map(|((x, l), u)| (*x).clamp(*l, *u))
        .collect()
}

/// Reflects a vector x into the interior of bound (lower_bound, upper_bound) then clamps its values.
fn reflect_then_clamp_vec<A>(
    x: VecType<A>,
    lower_bound: &VecType<A>,
    upper_bound: &VecType<A>,
) -> VecType<A>
where
    A: Number,
{
    x.iter()
        .zip(lower_bound)
        .zip(upper_bound)
        .map(|((x, l), u)| {
            if x <= u {
                *x
            } else {
                // Reflection
                A::from_f32(2.) * *u - *x
            }
            // Clamp to avoid x < l
            .clamp(*l, *u)
        })
        .collect()
}

/// Builds a simplex from a vector x0 and clamps its vectors.
fn default_simplex_with_bounds<A>(
    x0: VecType<A>,
    lower_bound: &VecType<A>,
    upper_bound: &VecType<A>,
) -> MatrixType<A>
where
    A: Number,
{
    let dim = x0.len();
    let mut simplex = Array::zeros((dim + 1, dim));
    for k in 0..dim {
        reflect_then_clamp_vec(simplex_vector(&x0, k), lower_bound, upper_bound)
            .assign_to(simplex.slice_mut(s![k + 1, ..]));
    }
    // Normally, x0 is already already clamped.
    clamp_vec(x0, lower_bound, upper_bound).assign_to(simplex.slice_mut(s![0, ..]));
    simplex
}

/// Builds a simplex from a vector x0 without clamping its vectors.
fn default_simplex_with_no_bounds<A>(x0: VecType<A>) -> MatrixType<A>
where
    A: Number,
{
    let dim = x0.len();
    let mut simplex = Array::zeros((dim + 1, dim));
    for k in 0..dim {
        simplex_vector(&x0, k).assign_to(simplex.slice_mut(s![k + 1, ..]));
    }
    x0.assign_to(simplex.slice_mut(s![0, ..]));
    simplex
}

fn check_simplex<A>(
    simplex: MatrixType<A>,
    x0: VecType<A>,
) -> Result<MatrixType<A>, TuutalError<VecType<A>>> {
    let dim = x0.len();
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
}

/// Reflects the vectors of the simplex (made by its rows) into the interior of the bounds.
/// and clamps their values.
fn reflect_then_clamp_simplex<A>(
    mut simplex: MatrixType<A>,
    lower_bound: &VecType<A>,
    upper_bound: &VecType<A>,
) -> MatrixType<A>
where
    A: Number,
{
    (0..simplex.nrows())
        .map(|idx| {
            let row = simplex.row(idx);
            let reflected_row = reflect_then_clamp_vec(row.to_owned(), lower_bound, upper_bound);
            simplex.row_mut(idx).assign(&reflected_row);
        })
        .for_each(drop);
    simplex
}

/// Applies a function to matrix row-wise.
fn vector_map_scalar<A, F>(
    mut fsim: VecType<A>,
    simplex: &MatrixType<A>,
    func: F,
    mut fcalls: usize,
) -> (VecType<A>, usize)
where
    A: Number,
    F: Fn(VecType<A>) -> A,
{
    (0..fsim.len())
        .map(|idx| {
            fsim[idx] = func(simplex.row(idx).to_owned());
            fcalls += 1;
        })
        .for_each(drop);
    (fsim, fcalls)
}

fn initial_simplex_with_bounds<A>(
    x0: VecType<A>,
    init_sim: Option<MatrixType<A>>,
    lower_bound: VecType<A>,
    upper_bound: VecType<A>,
) -> Result<Frontier<A>, TuutalError<VecType<A>>>
where
    A: Number,
{
    if let Some(simplex) = init_sim {
        let simplex =
            reflect_then_clamp_simplex(check_simplex(simplex, x0)?, &lower_bound, &upper_bound);
        Ok(Frontier::new(
            simplex,
            Some(Bounds::new(lower_bound, upper_bound)),
        ))
    } else {
        Ok(Frontier::new(
            default_simplex_with_bounds(x0, &lower_bound, &upper_bound),
            Some(Bounds::new(lower_bound, upper_bound)),
        ))
    }
}

fn initial_simplex_with_no_bounds<A>(
    x0: VecType<A>,
    init_sim: Option<MatrixType<A>>,
) -> Result<Frontier<A>, TuutalError<VecType<A>>>
where
    A: Number,
{
    if let Some(simplex) = init_sim {
        Ok(Frontier::new(check_simplex(simplex, x0)?, None))
    } else {
        Ok(Frontier::new(default_simplex_with_no_bounds(x0), None))
    }
}

fn clamp<A, B>(
    x0: VecType<A>,
    bounds: Option<B>,
    init_sim: Option<MatrixType<A>>,
) -> Result<Frontier<A>, TuutalError<VecType<A>>>
where
    A: Number,
    B: Bound<A>,
{
    if let Some(_bounds) = bounds {
        let dim = x0.len();
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
            println!("\nInitial guess is not within the specified bounds");
        }
        initial_simplex_with_bounds(
            clamp_vec(x0, &lower_bound, &upper_bound),
            init_sim,
            lower_bound,
            upper_bound,
        )
    } else {
        initial_simplex_with_no_bounds(x0, init_sim)
    }
}

#[derive(Debug)]
struct Bounds<A> {
    lower: VecType<A>,
    upper: VecType<A>,
}

impl<A> Bounds<A> {
    pub fn new(lower: VecType<A>, upper: VecType<A>) -> Self {
        Self { lower, upper }
    }
    pub fn lower(&self) -> &VecType<A> {
        &self.lower
    }
    pub fn upper(&self) -> &VecType<A> {
        &self.upper
    }
}

#[derive(Debug)]
struct Frontier<A> {
    simplex: MatrixType<A>,
    bounds: Option<Bounds<A>>,
}

impl<A> Frontier<A> {
    pub fn new(simplex: MatrixType<A>, bounds: Option<Bounds<A>>) -> Self {
        Self { simplex, bounds }
    }
    pub fn into_tuple(self) -> (MatrixType<A>, Option<Bounds<A>>) {
        (self.simplex, self.bounds)
    }
}

pub struct NelderMeadIterates<'a, F, A> {
    f: F,
    maxfev: usize,
    simplex: MatrixType<A>,
    xatol: A,
    fatol: A,
    sim_params: HashMap<&'a str, A>,
    bounds: Option<Bounds<A>>,
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
        A: Number,
        B: Bound<A>,
    {
        let (x0, sim_params) = simplex_parameters(x0, adaptive)?;
        let (simplex, bounds) = clamp(x0, bounds, simplex)?.into_tuple();
        let maxfev = if let Some(max) = maxfev {
            max
        } else {
            simplex.ncols() * 200
        };
        if maxfev < simplex.nrows() {
            return Err(TuutalError::MaxFunCall { num: maxfev });
        }
        Ok(Self {
            f,
            maxfev,
            simplex,
            xatol,
            fatol,
            sim_params,
            bounds,
        })
    }
    pub fn obj(&self) -> &F {
        &self.f
    }
    pub fn maxfev(&self) -> usize {
        self.maxfev
    }
    pub fn simplex(&self) -> &MatrixType<A> {
        &self.simplex
    }
    pub fn xatol(&self) -> &A {
        &self.xatol
    }
    pub fn fatol(&self) -> &A {
        &self.fatol
    }
    pub fn sim_params(&self) -> &HashMap<&'a str, A> {
        &self.sim_params
    }
    pub(crate) fn lower_bound(&self) -> &VecType<A> {
        if let Some(ref bounds) = self.bounds {
            bounds.lower()
        } else {
            panic!("No lower bound")
        }
    }
    pub(crate) fn upper_bound(&self) -> &VecType<A> {
        if let Some(ref bounds) = self.bounds {
            bounds.upper()
        } else {
            panic!("No upper bound")
        }
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
