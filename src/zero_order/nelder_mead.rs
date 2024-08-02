use core::ops::Mul;
mod unit_test;
use ndarray::{s, Axis};
extern crate alloc;
use crate::{Array1, Array2, Bound, Bounds, Number, Optimizer, Scalar, TuutalError};
use alloc::{string::ToString, vec::Vec};

use super::default_nb_iter;

type SimplexParameterResult<A> = Result<(A, A, A, A), TuutalError<Array1<A>>>;

/// The Nelder-Mead minimization algorithm.
///
/// It requires an initial guess x<sub>0</sub>.
/// ```
/// use tuutal::{array, nelder_mead, Array1};
/// // Example from python scipy.optimize.minimize_scalar
/// let f = |x: &Array1<f32>| (x[0] - 2.) * x[0] * (x[0] + 2.).powi(2);
/// let x0 = &array![-1.];
/// let x_star =
///     nelder_mead::<_, (f32, f32), _>(f, &x0, None, Some(100), None, 1e-5, 1e-5, true, None)
///     .unwrap();
/// assert!((-2. - x_star[0]).abs() <= 2e-4);
///
/// let f =
///     |arr: &Array1<f32>| 100. * (arr[1] - arr[0].powi(2)).powi(2) + (1. - arr[0]).powi(2);
/// let x0 = array![1., -0.5];
/// let x_star =
///     nelder_mead::<_, (f32, f32), _>(f, &x0, None, Some(100), None, 1e-5, 1e-5, true, None)
///     .unwrap();
/// assert!((1. - x_star[0]).abs() <= 1e-3);
/// assert!((1. - x_star[1]).abs() <= 2e-3);
/// ```
pub fn nelder_mead<A, B, F>(
    f: F,
    x0: &Array1<A>,
    maxfev: Option<usize>,
    maxiter: Option<usize>,
    simplex: Option<Array2<A>>,
    xatol: A,
    fatol: A,
    adaptive: bool,
    bounds: Option<B>,
) -> Result<Array1<A>, TuutalError<Array1<A>>>
where
    A: Scalar<Array1<A>>,
    B: Bound<A>,
    F: Fn(&Array1<A>) -> A,
{
    let (maxiter, maxfev) = default_nb_iter(x0.len(), maxiter, maxfev, 200);
    let mut nelder_mead = NelderMeadIterates::new(
        f,
        x0.clone(),
        Some(maxfev),
        simplex,
        xatol,
        fatol,
        adaptive,
        bounds,
    )?;
    nelder_mead.optimize(Some(maxiter))
}

/// Gets the hyperparameters of the Nelder-Mead Algorithm.
fn simplex_parameters<A>(x0: &Array1<A>, adaptive: bool) -> SimplexParameterResult<A>
where
    A: Number,
{
    let dim = A::cast_from_f32(x0.len() as f32);
    let (rho, chi, psi, sigma) = if adaptive && (dim > A::zero()) {
        (
            A::one(),
            A::one() + A::cast_from_f32(2.) / dim,
            A::cast_from_f32(0.75) - A::one() / (A::cast_from_f32(2.) * dim),
            A::one() - A::one() / dim,
        )
    } else if !adaptive {
        (
            A::one(),
            A::cast_from_f32(2.),
            A::cast_from_f32(0.5),
            A::cast_from_f32(0.5),
        )
    } else {
        return Err(TuutalError::EmptyDimension {
            x: Array1::from(Vec::new()),
        });
    };
    Ok((rho, chi, psi, sigma))
}

/// Builds the k-th vector defining a simplex from vector x0.
///
/// k should be < x0.len().
fn simplex_vector<A>(x0: &Array1<A>, k: usize) -> Array1<A>
where
    A: Number,
{
    let mut y = x0.clone();
    if y[k] != A::zero() {
        y[k] = (A::one() + A::cast_from_f32(0.05)) * y[k];
    } else {
        y[k] = A::cast_from_f32(0.00025);
    }
    y
}

/// Function that makes sure that: x[k] = min(upper_bound[k], max(x[k], lower_bound[k]))
///
/// User must make sure that lower_bound[k] <= upper_bound[k] for all k.
fn clamp_vec<A>(x: Array1<A>, lower_bound: &Array1<A>, upper_bound: &Array1<A>) -> Array1<A>
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
    x: Array1<A>,
    lower_bound: &Array1<A>,
    upper_bound: &Array1<A>,
) -> Array1<A>
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
                A::cast_from_f32(2.) * *u - *x
            }
            // Clamp to avoid x < l
            .clamp(*l, *u)
        })
        .collect()
}

/// Builds a simplex from a vector x0 and clamps
/// its vectors if both lower and upper bound are provided.
fn default_simplex<A>(
    x0: Array1<A>,
    lower_bound: Option<&Array1<A>>,
    upper_bound: Option<&Array1<A>>,
) -> Array2<A>
where
    A: Number,
{
    let dim = x0.len();
    let mut simplex = Array2::zeros((dim + 1, dim));
    if let Some(lb) = lower_bound {
        if let Some(ub) = upper_bound {
            for k in 0..dim {
                reflect_then_clamp_vec(simplex_vector(&x0, k), lb, ub)
                    .assign_to(simplex.slice_mut(s![k + 1, ..]));
            }
            // Normally, x0 is already already clamped.
            clamp_vec(x0, lb, ub).assign_to(simplex.slice_mut(s![0, ..]));
        } else {
            panic!("upper bound is not provided while lower bound is");
        }
    } else {
        for k in 0..dim {
            simplex_vector(&x0, k).assign_to(simplex.slice_mut(s![k + 1, ..]));
        }
        x0.assign_to(simplex.slice_mut(s![0, ..]));
    }
    simplex
}

fn check_simplex<A>(
    simplex: Array2<A>,
    x0: Array1<A>,
) -> Result<Array2<A>, TuutalError<Array1<A>>> {
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

fn row_map_matrix_mut<A, F>(simplex: &mut Array2<A>, func: F, from_axis: usize)
where
    A: Number,
    F: Fn(Array1<A>) -> Array1<A>,
{
    (from_axis..simplex.nrows())
        .map(|idx| {
            let frow = func(simplex.row(idx).to_owned());
            simplex.row_mut(idx).assign(&frow);
        })
        .for_each(drop);
}

/// Applies a function to matrix row-wise.
fn map_scalar_vector_mut<A, F>(
    fsim: &mut Array1<A>,
    simplex: &Array2<A>,
    func: F,
    mut fcalls: usize,
    maxfev: usize,
) -> Result<usize, TuutalError<Array1<A>>>
where
    A: Number,
    F: Fn(&Array1<A>) -> A,
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
        Err(TuutalError::MaxFunCall { num: maxfev })
    }
}

fn initial_simplex_with_bounds<A>(
    x0: Array1<A>,
    init_sim: Option<Array2<A>>,
    lower_bound: Array1<A>,
    upper_bound: Array1<A>,
) -> Result<Frontier<A>, TuutalError<Array1<A>>>
where
    A: Number,
{
    if let Some(simplex) = init_sim {
        let func = |x| reflect_then_clamp_vec(x, &lower_bound, &upper_bound);
        let mut simplex = check_simplex(simplex, x0)?;
        row_map_matrix_mut(&mut simplex, func, 0);
        Ok(Frontier::new(
            simplex,
            Some(Bounds::new(lower_bound, upper_bound)),
        ))
    } else {
        Ok(Frontier::new(
            default_simplex(x0, Some(&lower_bound), Some(&upper_bound)),
            Some(Bounds::new(lower_bound, upper_bound)),
        ))
    }
}

fn initial_simplex_with_no_bounds<A>(
    x0: Array1<A>,
    init_sim: Option<Array2<A>>,
) -> Result<Frontier<A>, TuutalError<Array1<A>>>
where
    A: Number,
{
    if let Some(simplex) = init_sim {
        Ok(Frontier::new(check_simplex(simplex, x0)?, None))
    } else {
        Ok(Frontier::new(default_simplex(x0, None, None), None))
    }
}

fn clamp<A, B>(
    x0: Array1<A>,
    bounds: Option<B>,
    init_sim: Option<Array2<A>>,
) -> Result<Frontier<A>, TuutalError<Array1<A>>>
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
            // println!("\nInitial guess is not within the specified bounds");
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
struct Frontier<A> {
    simplex: Array2<A>,
    bounds: Option<Bounds<A>>,
}

impl<A> Frontier<A> {
    pub fn new(simplex: Array2<A>, bounds: Option<Bounds<A>>) -> Self {
        Self { simplex, bounds }
    }
    pub fn into_tuple(self) -> (Array2<A>, Option<Bounds<A>>) {
        (self.simplex, self.bounds)
    }
}

// Taken and adapted from https://github.com/rust-ndarray/ndarray/issues/1145
pub fn argsort_by<A, F>(arr: &Array1<A>, mut compare: F) -> Vec<usize>
where
    A: Number,
    F: FnMut(&A, &A) -> core::cmp::Ordering,
{
    let mut indices: Vec<usize> = (0..arr.len()).collect();
    indices.sort_by(move |&i, &j| compare(&arr[i], &arr[j]));
    indices
}

/// Represents the sequence of iterates computed by the Nelder-Mead algorithm.
pub struct NelderMeadIterates<F, A> {
    f: F,
    maxfev: usize,
    sim: Array2<A>,
    xatol: A,
    fatol: A,
    bounds: Option<Bounds<A>>,
    fcalls: usize,
    fsim: Array1<A>,
    rho: A,
    chi: A,
    psi: A,
    sigma: A,
    convergence: bool,
    iter: usize,
}

impl<F, A> NelderMeadIterates<F, A> {
    pub fn new<B>(
        f: F,
        x0: Array1<A>,
        maxfev: Option<usize>,
        initial_simplex: Option<Array2<A>>,
        xatol: A,
        fatol: A,
        adaptive: bool,
        bounds: Option<B>,
    ) -> Result<Self, TuutalError<Array1<A>>>
    where
        A: Number,
        B: Bound<A>,
        F: Fn(&Array1<A>) -> A,
    {
        let maxfev = maxfev.unwrap_or(x0.len() * 200);
        if maxfev < x0.len() + 1 {
            return Err(TuutalError::MaxFunCall { num: maxfev });
        }
        let (rho, chi, psi, sigma) = simplex_parameters(&x0, adaptive)?;
        let (sim, bounds) = clamp(x0, bounds, initial_simplex)?.into_tuple();
        let fcalls = 0;
        let mut fsim = Array1::from_elem(sim.nrows(), A::zero());
        let fcalls = map_scalar_vector_mut(&mut fsim, &sim, &f, fcalls, maxfev)?;
        let sorted_indices = argsort_by(&fsim, |x: &A, y: &A| x.partial_cmp(y).unwrap());
        let fsim = fsim.select(Axis(0), &sorted_indices);
        let sim = sim.select(Axis(0), &sorted_indices);
        Ok(Self {
            f,
            maxfev,
            sim,
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
            iter: 0,
        })
    }

    /// Computes the objective function value for a given input vector.
    pub fn obj(&self, x: &Array1<A>) -> A
    where
        F: Fn(&Array1<A>) -> A,
    {
        let f = &self.f;
        f(x)
    }

    pub fn stop(&self) -> bool
    where
        A: Number,
    {
        self.fstop() | self.xstop()
    }

    /// Test whether or not the objective function outputs are almost the same for all the vertices.
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

    /// Test whether or not the simplex is almost degenerate.
    pub(crate) fn xstop(&self) -> bool
    where
        A: Number,
    {
        let mut max = A::zero();
        let x0 = self.sim.row(0).to_owned();
        for val in self.sim.rows() {
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

    /// Computes the centroid of the simplex.
    pub(crate) fn centroid(&self) -> Option<Array1<A>>
    where
        A: Number,
    {
        let last = self.sim.nrows() - 1;
        self.sim.slice(s!(..last, ..)).mean_axis(Axis(0))
    }

    /// Affine combination of a simplex centroid and the 'last' vertex of the simplex.
    pub(crate) fn affine(&self, centroid: &Array1<A>, a: A, b: A) -> Array1<A>
    where
        A: Scalar<Array1<A>>,
    {
        let last = self.sim.nrows() - 1;
        let c = a * b;
        // xaff = (1 + c) * centroid - c * last_vertex;
        let xaff = (A::cast_from_f32(1.) + c) * centroid - c * self.sim.row(last).to_owned();
        if let Some(ref bounds) = self.bounds {
            return clamp_vec(xaff, &bounds.lower, &bounds.upper);
        }
        xaff
    }

    pub(crate) fn shrink(&mut self)
    where
        F: Fn(&Array1<A>) -> A,
        A: Number + Mul<Array1<A>, Output = Array1<A>>,
    {
        let row0 = self.sim.row(0).to_owned();
        let shrink = |row| &row0 + self.sigma * (row - &row0);
        row_map_matrix_mut(&mut self.sim, shrink, 1);
        if let Some(ref bounds) = self.bounds {
            row_map_matrix_mut(
                &mut self.sim,
                |x| clamp_vec(x, &bounds.lower, &bounds.upper),
                1,
            );
        }
    }

    pub(crate) fn map_fsim(&mut self) -> Result<usize, TuutalError<Array1<A>>>
    where
        A: Number,
        F: Fn(&Array1<A>) -> A,
    {
        map_scalar_vector_mut(&mut self.fsim, &self.sim, &self.f, self.fcalls, self.maxfev)
    }

    /// Sorts the simplex vertices with respect to their objective function values in increasing order.
    pub(crate) fn sort(&mut self)
    where
        A: Number,
    {
        let sorted_indices = argsort_by(&self.fsim, |x: &A, y: &A| x.partial_cmp(y).unwrap());
        self.fsim = self.fsim.select(Axis(0), &sorted_indices);
        self.sim = self.sim.select(Axis(0), &sorted_indices);
    }
}

impl<F, A> core::iter::Iterator for NelderMeadIterates<F, A>
where
    F: Fn(&Array1<A>) -> A,
    A: Scalar<Array1<A>>,
{
    type Item = Array1<A>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.convergence {
            self.iter += 1;
            return None; // TODO
        } else if self.stop() {
            self.convergence = true;
            self.iter += 1;
            return Some(self.sim.row(0).to_owned());
        }
        if self.fcalls + 1 > self.maxfev {
            self.iter += 1;
            return None; // TODO
        }
        let one = A::cast_from_f32(1.);
        let xbar = self.centroid().unwrap(); // Is it allways safe to .unwrap()?
        let xr = self.affine(&xbar, self.rho, one);
        let fxr = self.obj(&xr);
        self.fcalls += 1;
        let mut doshrink = false;
        let last = self.sim.nrows() - 1;
        if fxr < self.fsim[0] {
            if self.fcalls + 1 > self.maxfev {
                self.iter += 1;
                return None; // TODO
            }
            let xe = self.affine(&xbar, self.rho, self.chi);
            let fxe = self.obj(&xe);
            self.fcalls += 1;
            if fxe < fxr {
                self.sim.row_mut(last).assign(&xe);
                self.fsim[last] = fxe;
            } else {
                self.sim.row_mut(last).assign(&xr);
                self.fsim[last] = fxr;
            }
        } else if fxr < self.fsim[last - 1] {
            self.sim.row_mut(last).assign(&xr);
            self.fsim[last] = fxr;
        } else if fxr < self.fsim[last] {
            if self.fcalls + 1 > self.maxfev {
                self.iter += 1;
                return None; // TODO
            }
            let xc = self.affine(&xbar, self.psi, self.rho);
            let fxc = self.obj(&xc);
            self.fcalls += 1;
            if fxc <= fxr {
                self.sim.row_mut(last).assign(&xc);
                self.fsim[last] = fxc;
            } else {
                doshrink = true;
            }
        } else if self.fcalls + 1 > self.maxfev {
            self.iter += 1;
            return None; // TODO
        } else {
            // Perform an inside contraction
            let xcc = self.affine(&xbar, self.psi, -one);
            let fxcc = self.obj(&xcc);
            self.fcalls += 1;
            if fxcc < self.fsim[last] {
                self.sim.row_mut(last).assign(&xcc);
                self.fsim[last] = fxcc;
            } else {
                doshrink = true;
            }
        }
        if doshrink {
            self.shrink();
            self.fcalls = match self.map_fsim() {
                Err(_) => {
                    self.iter += 1;
                    return None;
                } //TODO
                Ok(fcalls) => fcalls,
            };
        }
        self.sort();
        self.iter += 1;
        Some(self.sim.row(0).to_owned())
    }
}

impl<A, F> Optimizer for NelderMeadIterates<F, A>
where
    F: Fn(&Array1<A>) -> A,
    A: Scalar<Array1<A>>,
{
    type Iterate = Array1<A>;
    type Intermediate = ();
    fn nb_iter(&self) -> usize {
        self.iter
    }
    fn iterate(&self) -> Array1<A> {
        self.sim.row(0).to_owned()
    }
    fn intermediate(&self) {}
}
