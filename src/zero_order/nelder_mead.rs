use std::{fmt::Debug, ops::Mul};
#[cfg(test)]
mod unit_test;
use ndarray::{s, Axis};
use num_traits::FromPrimitive;

use crate::{Array, MatrixType, Number, TuutalError, VecType};

type SimplexParameterResult<'a, A> = Result<(VecType<A>, A, A, A, A), TuutalError<VecType<A>>>;
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
    Ok((x0, rho, chi, psi, sigma))
}

/// Builds the k-th vector defining a simplex from vector x0.
///
/// k should be < x0.len().
fn simplex_vector<A>(x0: &VecType<A>, k: usize) -> VecType<A>
where
    A: Number,
    A: Debug,
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
    A: Number + Debug,
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
    A: Number + Debug,
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

fn matrix_row_map<A, F>(simplex: &mut MatrixType<A>, func: F, from_axis: usize)
//-> MatrixType<A>
where
    A: Number,
    F: Fn(VecType<A>) -> VecType<A>,
{
    (from_axis..simplex.nrows())
        .map(|idx| {
            let frow = func(simplex.row(idx).to_owned());
            simplex.row_mut(idx).assign(&frow);
        })
        .for_each(drop);
    // simplex
}

/// Applies a function to matrix row-wise.
fn vector_map_scalar<A, F>(
    fsim: &mut VecType<A>,
    simplex: &MatrixType<A>,
    func: F,
    mut fcalls: usize,
    maxfev: usize,
) -> Result<usize, TuutalError<VecType<A>>>
where
    A: Number,
    F: Fn(&VecType<A>) -> A,
{
    let n = fsim.len();
    if fcalls + n < maxfev {
        (0..n)
            .map(|idx| {
                fsim[idx] = func(&simplex.row(idx).to_owned());
            })
            .for_each(drop);
        fcalls += n;
        Ok(fcalls)
    } else {
        return Err(TuutalError::MaxFunCall { num: maxfev });
    }
}

fn initial_simplex_with_bounds<A>(
    x0: VecType<A>,
    init_sim: Option<MatrixType<A>>,
    lower_bound: VecType<A>,
    upper_bound: VecType<A>,
) -> Result<Frontier<A>, TuutalError<VecType<A>>>
where
    A: Number + Debug,
{
    if let Some(simplex) = init_sim {
        let func = |x| reflect_then_clamp_vec(x, &lower_bound, &upper_bound);
        let mut simplex = check_simplex(simplex, x0)?;
        matrix_row_map(&mut simplex, func, 0);
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
    A: Number + Debug,
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
    A: Number + Debug,
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

// Taken and adapted from https://github.com/rust-ndarray/ndarray/issues/1145
pub fn argsort_by<A, F>(arr: &VecType<A>, mut compare: F) -> Vec<usize>
where
    A: Number,
    F: FnMut(&A, &A) -> std::cmp::Ordering,
{
    let mut indices: Vec<usize> = (0..arr.len()).collect();
    indices.sort_by(move |&i, &j| compare(&arr[i], &arr[j]));
    indices
}

pub struct NelderMeadIterates<F, A> {
    f: F,
    maxfev: usize,
    simplex: MatrixType<A>,
    xatol: A,
    fatol: A,
    bounds: Option<Bounds<A>>,
    fcalls: usize,
    fsim: VecType<A>,
    rho: A,
    chi: A,
    psi: A,
    sigma: A,
    convergence: bool,
}

impl<F, A> NelderMeadIterates<F, A> {
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
        A: Number + Debug,
        B: Bound<A>,
        F: Fn(&VecType<A>) -> A,
    {
        let maxfev = if let Some(max) = maxfev {
            max
        } else {
            x0.len() * 200
        };
        if maxfev < x0.len() + 1 {
            return Err(TuutalError::MaxFunCall { num: maxfev });
        }
        let (x0, rho, chi, psi, sigma) = simplex_parameters(x0, adaptive)?;
        let (simplex, bounds) = clamp(x0, bounds, simplex)?.into_tuple();
        let fcalls = 0;
        let mut fsim = Array::from(vec![A::zero(); simplex.nrows()]);
        let fcalls = vector_map_scalar(&mut fsim, &simplex, &f, fcalls, maxfev)?;
        let sorted_indices = argsort_by(&fsim, |x: &A, y: &A| x.partial_cmp(y).unwrap());
        let fsim = fsim.select(Axis(0), &sorted_indices);
        let simplex = simplex.select(Axis(0), &sorted_indices);
        Ok(Self {
            f,
            maxfev,
            simplex,
            xatol,
            fatol,
            bounds,
            fcalls,
            fsim,
            rho,
            chi,
            psi,
            sigma,
            convergence: false,
        })
    }

    pub fn obj(&self, x: &VecType<A>) -> A
    where
        F: Fn(&VecType<A>) -> A,
    {
        let f = &self.f;
        f(x)
    }

    pub(crate) fn fstop(&self) -> bool
    where
        A: Number,
    {
        let mut max = A::zero();
        for val in &self.fsim {
            let diff = (self.fsim[0] - *val).abs();
            if diff > max {
                max = diff;
            }
        }
        max <= self.fatol
    }

    pub(crate) fn xstop(&self) -> bool
    where
        A: Number,
    {
        let mut max = A::zero();
        let x0 = self.simplex.row(0).to_owned();
        for val in self.simplex.rows() {
            (&x0 - &val)
                .iter()
                .map(|x| {
                    if x.abs() > max {
                        max = x.abs()
                    }
                })
                .for_each(drop);
        }
        max <= self.xatol
    }

    pub(crate) fn centroid(&self) -> Option<VecType<A>>
    where
        A: Number + FromPrimitive,
    {
        let n = self.simplex.nrows();
        self.simplex.slice(s!(..n - 1, ..)).mean_axis(Axis(0))
    }

    pub(crate) fn affine(&self, centroid: &VecType<A>, a: A, b: A) -> VecType<A>
    where
        for<'b> A: Number
            + Mul<&'b VecType<A>, Output = VecType<A>>
            + Mul<VecType<A>, Output = VecType<A>>,
    {
        let xr = (A::from_f32(1.) + a * b) * centroid
            - a * b * self.simplex.row(self.simplex.nrows() - 1).to_owned();
        if let Some(ref bounds) = self.bounds {
            return clamp_vec(xr, &bounds.lower, &bounds.upper);
        }
        xr
    }

    pub(crate) fn sort(&mut self)
    // -> Result<usize, TuutalError<VecType<A>>>
    where
        A: Number,
        // F: Fn(&VecType<A>) -> A,
    {
        let sorted_indices = argsort_by(&self.fsim, |x: &A, y: &A| x.partial_cmp(y).unwrap());
        self.fsim = self.fsim.select(Axis(0), &sorted_indices);
        self.simplex = self.simplex.select(Axis(0), &sorted_indices);
    }
}

impl<F, A> std::iter::Iterator for NelderMeadIterates<F, A>
where
    F: Fn(&VecType<A>) -> A,
    for<'b> A: Number
        + Debug
        + FromPrimitive
        + Mul<&'b VecType<A>, Output = VecType<A>>
        + Mul<VecType<A>, Output = VecType<A>>,
{
    type Item = VecType<A>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.convergence {
            return None; // TODO
        } else if (self.fcalls + 1 > self.maxfev) | self.fstop() | self.xstop() {
            self.convergence = true;
            // self.fcalls = match self.sort() {
            //     Err(_) => {
            //         return None;
            //     } // TODO
            //     Ok(val) => val,
            // };
            return Some(self.simplex.row(0).to_owned());
        }
        let one = <A as Number>::from_f32(1.);
        let xbar = self.centroid().unwrap(); // Is it allways safe to .unwrap()?
                                             // println!("xbar:\n{:?}", xbar);
        let xr = self.affine(&xbar, self.rho, one);
        // println!("xr:\n{:?}", xr);
        let fxr = self.obj(&xr);
        self.fcalls += 1;
        let mut doshrink = false;
        let last = self.simplex.nrows() - 1;
        if fxr < self.fsim[0] {
            if self.fcalls + 1 > self.maxfev {
                return None; // TODO
            }
            let xe = self.affine(&xbar, self.rho, self.chi);
            let fxe = self.obj(&xe);
            self.fcalls += 1;
            if fxe < fxr {
                self.simplex.row_mut(last).assign(&xe);
                self.fsim[last] = fxe;
                return Some(xe);
            } else {
                self.simplex.row_mut(last).assign(&xr);
                self.fsim[last] = fxr;
                return Some(xr);
            }
        } else {
            if fxr < self.fsim[last - 1] {
                self.simplex.row_mut(last).assign(&xr);
                self.fsim[last] = fxr;
            } else {
                if fxr < self.fsim[last] {
                    if self.fcalls + 1 > self.maxfev {
                        return None; // TODO
                    }
                    let xc = self.affine(&xbar, self.psi, self.rho);
                    let fxc = self.obj(&xc);
                    self.fcalls += 1;
                    if fxc <= fxr {
                        self.simplex.row_mut(last).assign(&xc);
                        self.fsim[last] = fxc;
                    } else {
                        doshrink = true;
                    }
                } else {
                    // Perform an inside contraction
                    if self.fcalls + 1 > self.maxfev {
                        return None; // TODO
                    }
                    let xcc = self.affine(&xbar, self.psi, -one);
                    let fxcc = self.obj(&xcc);
                    self.fcalls += 1;
                    if fxcc < self.fsim[last] {
                        self.simplex.row_mut(last).assign(&xcc);
                        self.fsim[last] = fxcc;
                    } else {
                        doshrink = true;
                    }
                }
                if doshrink {
                    let row0 = self.simplex.row(0).to_owned();
                    let shrink = |row| &row0 + self.sigma * (row - &row0);
                    matrix_row_map(&mut self.simplex, shrink, 1);
                    if let Some(ref bounds) = self.bounds {
                        matrix_row_map(
                            &mut self.simplex,
                            |x| clamp_vec(x, &bounds.lower, &bounds.upper),
                            1,
                        );
                    }
                    self.fcalls = match vector_map_scalar(
                        &mut self.fsim,
                        &self.simplex,
                        &self.f,
                        self.fcalls,
                        self.maxfev,
                    ) {
                        Err(_) => {
                            return None;
                        } // TODO
                        Ok(val) => val,
                    };
                }
            }
        }
        self.sort();
        // self.fcalls = match self.sort() {
        //     Err(_) => {
        //         return None;
        //     } // TODO
        //     Ok(val) => val,
        // };
        None
    }
}
