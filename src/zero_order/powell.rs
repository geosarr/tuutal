mod unit_test;
use core::ops::Mul;
use ndarray::{Array, Axis};
extern crate alloc;
use alloc::vec::Vec;

use crate::{
    brent_bounded, brent_unbounded, Array1, Array2, Bound, Number, Optimizer, Scalar, TuutalError,
};

use super::{default_nb_iter, scalar::BrentOptResult};

/// The Powell minimization algorithm.
///
/// It requires an initial guess x<sub>0</sub>.
/// ```
/// use tuutal::{array, powell, Array1};
/// // Example from python scipy.optimize.minimize_scalar
/// let f = |x: &Array1<f32>| (x[0] - 2.) * x[0] * (x[0] + 2.).powi(2);
/// let x0 = &array![-1.];
/// let x_star =
///     powell::<_, (f32, f32), _>(f, &x0, None, Some(100), None, 1e-5, 1e-5, None)
///     .unwrap();
/// assert!((x_star[0] - 1.280776).abs() <= 1e-4);
///
/// let f =
///     |arr: &Array1<f32>| 100. * (arr[1] - arr[0].powi(2)).powi(2) + (1. - arr[0]).powi(2);
/// let x0 = array![1., -0.5];
/// let x_star =
///     powell::<_, (f32, f32), _>(f, &x0, None, Some(100), None, 1e-5, 1e-5, None)
///     .unwrap();
/// assert!((x_star[0] - 1.).abs() <= 1e-5);
/// assert!((x_star[1] - 1.).abs() <= 1e-5);
/// ```
pub fn powell<A, B, F>(
    f: F,
    x0: &Array1<A>,
    maxfev: Option<usize>,
    maxiter: Option<usize>,
    direc: Option<Array2<A>>,
    xtol: A,
    ftol: A,
    bounds: Option<B>,
) -> Result<Array1<A>, TuutalError<Array1<A>>>
where
    A: Scalar<Array1<A>> + core::fmt::Debug,
    B: Bound<A>,
    F: Fn(&Array1<A>) -> A,
{
    let (maxiter, maxfev) = default_nb_iter(x0.len(), maxiter, maxfev, 1000);
    let mut powell = PowellIterates::new(f, x0.clone(), Some(maxfev), direc, xtol, ftol, bounds)?;
    powell.optimize(Some(maxiter))
}

fn line_search_powell<A, F>(
    f: F,
    p: &Array1<A>,
    xi: &Array1<A>,
    tol: A,
    lower_bound: Option<&Array1<A>>,
    upper_bound: Option<&Array1<A>>,
    fval: A, // equal to f(p) (to avoid recomputing f(p))
    fcalls: usize,
) -> BrentOptResult<A>
where
    for<'a> A: Number + Mul<&'a Array1<A>, Output = Array1<A>> + core::fmt::Debug,
    F: Fn(&Array1<A>) -> A,
{
    let obj = |alpha: A| {
        let x = p + alpha * xi;
        f(&x)
    };
    if xi.iter().all(|x| *x == A::zero()) {
        return Ok((A::zero(), fval, fcalls));
    }
    if lower_bound.is_some() | upper_bound.is_some() {
        // Line for search.
        let (lb, ub) = (lower_bound.unwrap(), upper_bound.unwrap());
        let bounds = line_for_search(p, xi, lb, ub).unwrap(); // safe to .unwrap() since xi != 0 at this stage.
        if (bounds.0 == A::neg_infinity()) && (bounds.1 == A::infinity()) {
            // Unbounded minimization
            line_search_powell(f, p, xi, tol, None, None, fval, fcalls)
        } else if (bounds.0 != A::neg_infinity()) && (bounds.1 != A::infinity()) {
            // Bounded scalar minimization
            let xatol = tol / A::cast_from_f32(100.);
            return match brent_bounded(obj, bounds, xatol, 500) {
                Err(error) => Err(error),
                Ok((x, fx, fc)) => Ok((x, fx, fcalls + fc)),
            };
        } else {
            // One-sided bound minimization.
            let xatol = tol / A::cast_from_f32(100.);
            let bounds = (A::atan(bounds.0), A::atan(bounds.1));
            return match brent_bounded(|x| obj(A::tan(x)), bounds, xatol, 500) {
                Err(error) => Err(error),
                Ok((x, fx, fc)) => Ok((A::tan(x), fx, fcalls + fc)),
            };
        }
    } else {
        // Non-bounded minimization
        let (alpha_min, fret, fcalls) =
            match brent_unbounded(obj, None, 1000, A::cast_from_f32(1e-6)) {
                Err(error) => return Err(error),
                Ok(val) => val,
            };
        Ok((alpha_min, fret, fcalls))
    }
}

fn line_for_search<A>(
    x: &Array1<A>,
    d: &Array1<A>,
    lower_bound: &Array1<A>,
    upper_bound: &Array1<A>,
) -> Result<(A, A), TuutalError<Array1<A>>>
where
    A: Number + core::fmt::Debug,
{
    let (non_zero, _) = split_in_two(|i| d[*i] != A::zero(), d.len());
    let d = d.select(Axis(0), &non_zero);
    if d.is_empty() {
        return Err(TuutalError::EmptyDimension { x: d });
    }
    let (pos, neg) = split_in_two(|i| d[*i] > A::epsilon(), d.len());
    let x = x.select(Axis(0), &non_zero);
    let lb = lower_bound.select(Axis(0), &non_zero);
    let ub = upper_bound.select(Axis(0), &non_zero);
    let lower = (lb - &x) / &d;
    let upper = (ub - &x) / d;

    let lmin_pos = extremum(&lower.select(Axis(0), &pos), |a, b| a > b);
    let lmin_neg = extremum(&upper.select(Axis(0), &neg), |a, b| a > b);
    let lmin = min_max(lmin_pos, lmin_neg, true);

    let lmax_pos = extremum(&upper.select(Axis(0), &pos), |a, b| a < b);
    let lmax_neg = extremum(&lower.select(Axis(0), &neg), |a, b| a < b);
    let lmax = min_max(lmax_pos, lmax_neg, false);

    if lmin > lmax {
        Ok((A::zero(), A::zero()))
    } else {
        Ok((lmin, lmax))
    }
}

fn split_in_two<F>(func: F, dim: usize) -> (Vec<usize>, Vec<usize>)
where
    F: Fn(&usize) -> bool,
{
    let mut no_indices = Vec::with_capacity(dim);
    let yes_indices = (0..dim)
        .filter(|i| {
            if !func(i) {
                no_indices.push(*i);
            }
            func(i)
        })
        .collect();
    (yes_indices, no_indices)
}

fn extremum<A, F>(x: &Array1<A>, mut compare: F) -> Result<A, TuutalError<Array1<A>>>
where
    A: Number,
    F: FnMut(&A, &A) -> bool,
{
    if x.is_empty() {
        return Err(TuutalError::EmptyDimension { x: x.clone() });
    }
    let mut m = x[0];
    for val in x {
        if compare(val, &m) {
            m = *val;
        }
    }
    Ok(m)
}

fn min_max<A>(
    m1: Result<A, TuutalError<Array1<A>>>,
    m2: Result<A, TuutalError<Array1<A>>>,
    max: bool,
) -> A
where
    A: Number + core::fmt::Debug,
{
    match m1 {
        Err(_) => m2.unwrap(), // Assumes that m1 and m2 are not both Err.
        Ok(val1) => match m2 {
            Err(_) => val1,
            Ok(val2) => {
                if max {
                    val1.max(val2)
                } else {
                    val1.min(val2)
                }
            }
        },
    }
}
/// Represents the sequence of iterates computed by the Powell algorithm.
pub struct PowellIterates<F, A> {
    f: F,
    x: Array1<A>,
    x1: Array1<A>,
    maxfev: usize,
    direc: Array2<A>,
    xtol: A,
    ftol: A,
    lower: Option<Array1<A>>,
    upper: Option<Array1<A>>,
    fval: A,
    fcalls: usize,
    iter: usize,
}

impl<F, A> PowellIterates<F, A> {
    pub fn new<B>(
        f: F,
        x0: Array1<A>,
        maxfev: Option<usize>,
        direc: Option<Array2<A>>,
        xtol: A,
        ftol: A,
        bounds: Option<B>,
    ) -> Result<Self, TuutalError<Array1<A>>>
    where
        A: Scalar<Array1<A>> + core::fmt::Debug,
        B: Bound<A>,
        F: Fn(&Array1<A>) -> A,
    {
        let dim = x0.len();
        let maxfev = maxfev.unwrap_or(dim * 1000);
        if maxfev < x0.len() + 1 {
            return Err(TuutalError::MaxFunCall { num: maxfev });
        }
        let direc = if let Some(dir) = direc {
            // TODO check rank of the matrix.
            // println!("direc should be full rank.");
            dir
        } else {
            Array::eye(dim)
        };
        let (lower, upper) = if let Some(_bounds) = bounds {
            let dim = x0.len();
            let (lower_bound, upper_bound) = (_bounds.lower(dim), _bounds.upper(dim));
            // check bounds
            if lower_bound.iter().zip(&upper_bound).any(|(l, u)| l > u) {
                return Err(TuutalError::BoundOrder {
                    lower: lower_bound,
                    upper: upper_bound,
                });
            }
            (Some(lower_bound), Some(upper_bound))
        } else {
            (None, None)
        };
        let fval = f(&x0);
        Ok(Self {
            f,
            x: x0.clone(),
            x1: x0,
            maxfev,
            direc,
            xtol,
            ftol,
            lower,
            upper,
            fval,
            fcalls: 1,
            iter: 0,
        })
    }

    pub(crate) fn obj(&self, x: &Array1<A>) -> A
    where
        F: Fn(&Array1<A>) -> A,
    {
        let f = &self.f;
        f(x)
    }

    pub fn nb_iter(&self) -> usize {
        self.iter
    }
}

impl<F, A> core::iter::Iterator for PowellIterates<F, A>
where
    A: Scalar<Array1<A>> + core::fmt::Debug,
    F: Fn(&Array1<A>) -> A,
{
    type Item = Array1<A>;
    fn next(&mut self) -> Option<Self::Item> {
        let zero = A::zero();
        let one = A::one();
        let two = A::cast_from_f32(2.);
        let fx = self.fval;
        let mut bigind = 0;
        let mut delta = A::zero();
        for i in 0..self.x.len() {
            let direc1 = self.direc.row(i).to_owned();
            let fx2 = self.fval;
            let (alpha, fval, fcalls) = match line_search_powell(
                &self.f,
                &self.x,
                &direc1,
                self.xtol * A::cast_from_f32(100.),
                self.lower.as_ref(),
                self.upper.as_ref(),
                self.fval,
                self.fcalls,
            ) {
                Err(_) => {
                    self.iter += 1;
                    panic!("Error line search powell")
                } // TODO change
                Ok(val) => val,
            };
            self.x = &self.x + alpha * direc1;
            self.fval = fval;
            self.fcalls = fcalls;
            if (fx2 - fval) > delta {
                delta = fx2 - fval;
                bigind = i;
            }
        }

        let bnd = self.ftol * (fx.abs() + self.fval.abs()) + A::epsilon();
        if (two * (fx - self.fval) <= bnd)
            | (self.fcalls > self.maxfev)
            | (fx.is_nan() && self.fval.is_nan())
        {
            self.iter += 1;
            return None; // TODO change
                         // break;
        }
        // Construct the extrapolated point
        let mut direc1 = &self.x - &self.x1;
        self.x1 = self.x.clone();
        let lmax = if self.lower.is_none() && self.upper.is_none() {
            one
        } else {
            let bounds = line_for_search(
                &self.x,
                &direc1,
                self.lower.as_ref().unwrap(),
                self.upper.as_ref().unwrap(),
            )
            .unwrap(); // Safe to .unwrap() if direc1 is full rank, have to make sure of it.
            bounds.1
        };
        let x2 = &self.x + lmax.min(one) * &direc1;
        if self.fcalls + 1 > self.maxfev {
            self.iter += 1;
            return None; // TO change
        }
        let fx2 = self.obj(&x2);
        self.fcalls += 1;
        if fx > fx2 {
            let mut t = two * (fx + fx2 - two * self.fval);
            let mut temp = fx - self.fval - delta;
            t = t * temp * temp;
            temp = fx - fx2;
            t = t - delta * temp * temp;
            if t < zero {
                let (alpha, fval, fcalls) = match line_search_powell(
                    &self.f,
                    &self.x,
                    &direc1,
                    self.xtol * A::cast_from_f32(100.),
                    self.lower.as_ref(),
                    self.upper.as_ref(),
                    self.fval,
                    self.fcalls,
                ) {
                    Err(_) => panic!("Error line search powell"), // TODO change
                    Ok(val) => val,
                };
                self.fval = fval;
                self.fcalls = fcalls;
                direc1 = alpha * direc1;
                if direc1.iter().any(|d| d != &zero) {
                    let last = self.direc.nrows() - 1;
                    let last_row = self.direc.row(last).to_owned();
                    self.direc.row_mut(bigind).assign(&last_row);
                    self.direc.row_mut(last).assign(&direc1);
                }
                self.x = &self.x + direc1;
            }
        }
        self.iter += 1;
        Some(self.x.clone())
    }
}

impl<A, F> Optimizer for PowellIterates<F, A>
where
    A: Scalar<Array1<A>> + core::fmt::Debug,
    F: Fn(&Array1<A>) -> A,
{
    type Iterate = Array1<A>;
    type Intermediate = ();
    fn nb_iter(&self) -> usize {
        self.iter
    }
    fn iterate(&self) -> Array1<A> {
        self.x.clone()
    }
    fn intermediate(&self) {}
}
