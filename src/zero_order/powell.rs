mod unit_test;
use ndarray::{Array, Axis};
use std::ops::Mul;

use crate::{
    brent_bounded, brent_unbounded, optimize, Bound, Iterable, MatrixType, Number, TuutalError,
    VecType,
};

use super::scalar::BrentOptResult;

/// The Powell minimization algorithm.
///
/// It requires an initial guess x<sub>0</sub>.
/// ```
/// use tuutal::{array, powell, VecType};
/// // Example from python scipy.optimize.minimize_scalar
/// let f = |x: &VecType<f32>| (x[0] - 2.) * x[0] * (x[0] + 2.).powi(2);
/// let x0 = &array![-1.];
/// let x_star =
///     powell::<_, (f32, f32), _>(f, &x0, None, 100, None, 1e-5, 1e-5, None)
///     .unwrap();
/// assert!((-2. - x_star[0]).abs() <= 2e-4);
///
/// let f =
///     |arr: &VecType<f32>| 100. * (arr[1] - arr[0].powi(2)).powi(2) + (1. - arr[0]).powi(2);
/// let x0 = array![1., -0.5];
/// ```
pub fn powell<A, B, F>(
    f: F,
    x0: &VecType<A>,
    maxfev: Option<usize>,
    maxiter: usize,
    direc: Option<MatrixType<A>>,
    xtol: A,
    ftol: A,
    bounds: Option<B>,
) -> Result<VecType<A>, TuutalError<VecType<A>>>
where
    for<'b> A: Number
        + std::fmt::Debug
        + Mul<&'b VecType<A>, Output = VecType<A>>
        + Mul<VecType<A>, Output = VecType<A>>,
    B: Bound<A>,
    F: Fn(&VecType<A>) -> A,
{
    let iterates = PowellIterates::new(f, x0.clone(), maxfev, direc, xtol, ftol, bounds)?;
    optimize(iterates, maxiter)
}

pub fn line_search_powell<A, F>(
    f: F,
    p: &VecType<A>,
    xi: &VecType<A>,
    tol: A,
    lower_bound: Option<&VecType<A>>,
    upper_bound: Option<&VecType<A>>,
    fval: A, // equal to f(p) (to avoid recomputing f(p))
    fcalls: usize,
) -> BrentOptResult<A>
where
    for<'a> A: Number + Mul<&'a VecType<A>, Output = VecType<A>> + std::fmt::Debug,
    F: Fn(&VecType<A>) -> A,
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
            let xatol = tol / A::from_f32(100.);
            return match brent_bounded(obj, bounds, xatol, 500) {
                Err(error) => Err(error),
                Ok((x, fx, fc)) => Ok((x, fx, fcalls + fc)),
            };
        } else {
            // One-sided bound minimization.
            let xatol = tol / A::from_f32(100.);
            let bounds = (A::atan(bounds.0), A::atan(bounds.1));
            return match brent_bounded(|x| obj(A::tan(x)), bounds, xatol, 500) {
                Err(error) => Err(error),
                Ok((x, fx, fc)) => Ok((A::tan(x), fx, fcalls + fc)),
            };
        }
    } else {
        // Non-bounded minimization
        let (alpha_min, fret, fcalls) =
            match brent_unbounded(obj, A::zero(), A::one(), 1000, A::from_f32(1e-6)) {
                Err(error) => return Err(error),
                Ok(val) => val,
            };
        Ok((fret, alpha_min, fcalls))
    }
}

fn line_for_search<A>(
    x: &VecType<A>,
    d: &VecType<A>,
    lower_bound: &VecType<A>,
    upper_bound: &VecType<A>,
) -> Result<(A, A), TuutalError<VecType<A>>>
where
    A: Number + std::fmt::Debug,
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

fn extremum<A, F>(x: &VecType<A>, mut compare: F) -> Result<A, TuutalError<VecType<A>>>
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
    m1: Result<A, TuutalError<VecType<A>>>,
    m2: Result<A, TuutalError<VecType<A>>>,
    max: bool,
) -> A
where
    A: Number + std::fmt::Debug,
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
    x: VecType<A>,
    x1: VecType<A>,
    maxfev: usize,
    direc: MatrixType<A>,
    xtol: A,
    ftol: A,
    lower: Option<VecType<A>>,
    upper: Option<VecType<A>>,
    fval: A,
    fcalls: usize,
    iter: usize,
}

impl<F, A> PowellIterates<F, A> {
    pub fn new<B>(
        f: F,
        x0: VecType<A>,
        maxfev: Option<usize>,
        direc: Option<MatrixType<A>>,
        xtol: A,
        ftol: A,
        bounds: Option<B>,
    ) -> Result<Self, TuutalError<VecType<A>>>
    where
        for<'a> A: Number
            + Mul<&'a VecType<A>, Output = VecType<A>>
            + std::fmt::Debug
            + Mul<VecType<A>, Output = VecType<A>>,
        B: Bound<A>,
        F: Fn(&VecType<A>) -> A,
    {
        let dim = x0.len();
        let maxfev = if let Some(max) = maxfev {
            max
        } else {
            dim * 1000
        };
        if maxfev < x0.len() + 1 {
            return Err(TuutalError::MaxFunCall { num: maxfev });
        }
        let direc = if let Some(dir) = direc {
            // TODO check rank of the matrix.
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

    pub(crate) fn obj(&self, x: &VecType<A>) -> A
    where
        F: Fn(&VecType<A>) -> A,
    {
        let f = &self.f;
        f(x)
    }

    pub fn nb_iter(&self) -> usize {
        self.iter
    }
}

impl<F, A> std::iter::Iterator for PowellIterates<F, A>
where
    for<'a> A: Number
        + Mul<&'a VecType<A>, Output = VecType<A>>
        + std::fmt::Debug
        + Mul<VecType<A>, Output = VecType<A>>,
    F: Fn(&VecType<A>) -> A,
{
    type Item = VecType<A>;
    fn next(&mut self) -> Option<Self::Item> {
        let zero = A::zero();
        let one = A::one();
        let two = A::from_f32(2.);
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
                self.xtol * A::from_f32(100.),
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
        let direc1 = &self.x - &self.x1;
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
                    self.xtol * A::from_f32(100.),
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
                if direc1.iter().any(|d| d != &zero) {
                    let last = self.direc.nrows() - 1;
                    let last_row = self.direc.row(last).to_owned();
                    self.direc.row_mut(bigind).assign(&last_row);
                    self.direc.row_mut(last).assign(&direc1);
                }
                self.x = &self.x + alpha * direc1;
            }
        }
        self.iter += 1;
        Some(self.x.clone())
    }
}

impl<A, F> Iterable<VecType<A>> for PowellIterates<F, A>
where
    for<'a> A: Number
        + Mul<&'a VecType<A>, Output = VecType<A>>
        + std::fmt::Debug
        + Mul<VecType<A>, Output = VecType<A>>,
    F: Fn(&VecType<A>) -> A,
{
    fn nb_iter(&self) -> usize {
        self.iter
    }
    fn iterate(&self) -> VecType<A> {
        self.x.clone()
    }
}
