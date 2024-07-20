mod unit_test;
use ndarray::Axis;
use std::ops::Mul;

use crate::{brent_opt, Number, TuutalError, VecType};

pub fn line_search_powell<A, F>(
    f: F,
    p: &VecType<A>,
    xi: &VecType<A>,
    tol: A,
    lower_bound: Option<&VecType<A>>,
    upper_bound: Option<&VecType<A>>,
    fval: A, // equal to f(p) (to avoid recomputing f(p))
) -> Result<(A, A), TuutalError<(A, A, A, A, A, A, usize)>>
where
    for<'a> A: Number + Mul<&'a VecType<A>, Output = VecType<A>> + std::fmt::Debug,
    F: Fn(&VecType<A>) -> A,
{
    let obj = |alpha: A| {
        let x = p + alpha * xi;
        f(&x)
    };
    if xi.iter().all(|x| *x == A::zero()) {
        return Ok((fval, A::zero()));
    }
    if lower_bound.is_some() | upper_bound.is_some() {
        // Line for search.
        let (lb, ub) = (lower_bound.unwrap(), upper_bound.unwrap());
        let bound = line_for_search(p, xi, lb, ub).unwrap(); // safe to .unwrap() since xi != 0 at this stage.
        if (bound.0 == A::neg_infinity()) && (bound.1 == A::infinity()) {
            // Unboundend minimization
            return line_search_powell(f, p, xi, tol, None, None, fval);
        } else if bound.0.is_infinite() | bound.1.is_infinite() {
            let bound = (A::atan(bound.0), A::atan(bound.1));
        }
        Ok((A::zero(), A::zero())) // TODO change.
    } else {
        // Non-bounded minimization
        let (alpha_min, fret, _) =
            match brent_opt(&obj, A::zero(), A::one(), 1000, A::from_f32(1e-6)) {
                Err(error) => return Err(error),
                Ok(val) => val,
            };
        return Ok((fret, alpha_min));
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
    return (yes_indices, no_indices);
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
    return Ok(m);
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
                if val2 > val1 {
                    if max {
                        val2
                    } else {
                        val1
                    }
                } else {
                    if max {
                        val1
                    } else {
                        val2
                    }
                }
            }
        },
    }
}
