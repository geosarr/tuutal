use std::{
    mem::swap,
    ops::{Div, Mul, Sub},
};

use crate::{Number, RootFindingError};

/// Inverse quadratic interpolation
fn inv_quad_interpol<T>(a: T, fa: T, fb: T, fc: T) -> T
where
    T: Mul<Output = T> + Sub<Output = T> + Div<Output = T> + Copy,
{
    a * fb * fc / ((fa - fb) * (fa - fc))
}

/// [Brent's root finding][br] algorithm for a scalar function f.
///
/// # Returns
/// - Ok((x, f(x)) when a solution is found., f(x) is the output of x by f.
/// - [Err(RootFindingError::Bracketing)](../error/enum.RootFindingError.html) when f(a) * f(b) >= 0.
/// - [Err(RootFindingError::Interpolation)](../error/enum.RootFindingError.html) when interpolation could not be applied.
///
/// [br]: https://en.wikipedia.org/wiki/Brent%27s_method
///
/// ```
/// use tuutal::{brent_root, RootFindingError};
/// let res = brent_root(|x: f32| x.powi(2) - 4., 0., 3., 1e-6, 1e-6, 100).unwrap_or((0., 1., 0));
/// assert!((res.0 - 2.).abs() <= 1e-4);
/// assert!(res.1.abs() <= 1e-6);
///
/// let res = brent_root(|x: f32| x.powi(2) - 2., 0., 2., 1e-6, 1e-6, 100).unwrap_or((0., 1., 0));
/// assert!((res.0 - 1.4141).abs() <= 1e-3);
/// assert!(res.1.abs() <= 1e-6);
///
/// let res = brent_root(|x: f32| x.powi(3) + 27., -4., 5., 1e-6, 1e-6, 100).unwrap_or((0., 1., 0));
/// assert!((res.0 + 3.).abs() <= 1e-4);
/// assert!(res.1.abs() <= 1e-6);
///
/// let (a, b) = (0.5, 1.);
/// let error = RootFindingError::Bracketing {a, b};
/// assert_eq!(brent_root(|x: f32| x, a, b, 1e-6, 1e-6, 100).unwrap_err(), error);
/// ```
pub fn brent_root<T>(
    f: impl Fn(T) -> T,
    mut a: T,
    mut b: T,
    xtol: T,
    rtol: T,
    maxiter: usize,
) -> Result<(T, T, usize), RootFindingError<T>>
where
    T: Number,
{
    let mut fa = f(a);
    let mut fb = f(b);
    let mut fcalls = 2;
    let eps = T::epsilon();
    let zero = T::zero();
    if fa * fb > zero {
        return Err(RootFindingError::Bracketing { a, b });
    }
    if fa.abs() < eps {
        return Ok((a, fa, fcalls));
    }
    if fb.abs() < eps {
        return Ok((b, fb, fcalls));
    }
    if fa.abs() < fb.abs() {
        swap(&mut a, &mut b);
    }
    let mut c = a;
    let mut mflag = true;
    let two = T::cast_from_f32(2.);
    let three = T::cast_from_f32(3.);
    let four = T::cast_from_f32(4.);
    let delta = T::cast_from_f32(10.).powi(-5);
    let mut iter = 0;
    while (fb.abs() > eps)
        && (fa.abs() > eps)
        && (a - b).abs() > xtol + rtol * b.abs()
        && iter < maxiter
    {
        let fc = f(c);
        fcalls += 1;
        let mut s = if ((fa - fc).abs() > eps) && ((fb - fc).abs() > eps) {
            if fa == fb {
                return Err(RootFindingError::Interpolation { a, b });
            }
            // inverse quadratic interpolation
            inv_quad_interpol(a, fa, fb, fc)
                + inv_quad_interpol(b, fb, fa, fc)
                + inv_quad_interpol(c, fc, fa, fb)
        } else {
            // secant method
            b - fb * (b - a) / (fb - fa)
        };
        let a_b_bound = (three * a + b) / four;
        let condition_1 = !(((a_b_bound <= s) && (s <= b)) | ((b <= s) && (s <= a_b_bound))); // s not between (3a+b)/4 and b
        let condition_2 = mflag && ((s - b).abs() >= (b - c) / two);
        let d = c;
        let condition_3 = !mflag && ((s - b).abs() >= (c - d) / two);
        let condition_4 = mflag && ((b - c).abs() < delta);
        let condition_5 = !mflag && ((c - d).abs() < delta);
        if condition_1 | condition_2 | condition_3 | condition_4 | condition_5 {
            s = (a + b) / two; // bisection method
            mflag = true;
        } else {
            mflag = false;
        }
        c = b;
        fa = f(a);
        let fs = f(s);
        fcalls += 2;
        if fa * fs < T::zero() {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }
        if fa.abs() < fb.abs() {
            swap(&mut a, &mut b);
            // swap(&mut fa, &mut fb);
        }
        fa = f(a);
        fb = f(b);
        fcalls += 2;
        iter += 1;
    }
    Ok((b, f(b), fcalls))
}

/// Brent's root finding algorithm using inverse quadratic interpolation
///
/// The number of iterations coincides with the number of function evaluations - 2.
///
/// ```
/// use tuutal::{brentq, RootFindingError};
/// let res = brentq(|x: f32| x.powi(2) - 4., 0., 3., 1e-6, 1e-6, 100).unwrap_or((0., 1., 0));
/// assert!((res.0 - 2.).abs() <= 1e-4);
/// assert!(res.1.abs() <= 1e-6);
///
/// let res = brentq(|x: f32| x.powi(2) - 2., 0., 2., 1e-6, 1e-6, 100).unwrap_or((0., 1., 0));
/// assert!((res.0 - 1.4141).abs() <= 1e-3);
/// assert!(res.1.abs() <= 1e-6);
///
/// let res = brentq(|x: f32| x.powi(3) + 27., -4., 5., 1e-6, 1e-6, 100).unwrap_or((0., 1., 0));
/// assert!((res.0 + 3.).abs() <= 1e-4);
/// assert!(res.1.abs() <= 1e-6);
///
/// let (a, b) = (0.5, 1.);
/// let error = RootFindingError::Bracketing {a, b};
/// assert_eq!(brentq(|x: f32| x, a, b, 1e-6, 1e-6, 100).unwrap_err(), error);
/// ```
/// Adapted from [Scipy Optimize][brq]
///
/// [brq]: https://github.com/scipy/scipy/blob/v1.13.1/scipy/optimize/Zeros/brentq.c
pub fn brentq<F, T>(
    f: F,
    a: T,
    b: T,
    xtol: T,
    rtol: T,
    maxiter: usize,
) -> Result<(T, T, usize), RootFindingError<T>>
where
    T: Number,
    F: Fn(T) -> T,
{
    let (mut xpre, mut xcur) = (a, b);
    let zero = T::zero();
    let eps = T::epsilon();
    let two = T::cast_from_f32(2.);
    let three = T::cast_from_f32(3.);
    let (mut xblk, mut fblk, mut spre, mut scur) = (zero, zero, zero, zero);
    let mut fpre = f(xpre);
    let mut fcur = f(xcur);
    let mut fcalls = 2;
    if fpre.abs() < eps {
        return Ok((xpre, fpre, fcalls));
    }
    if fcur.abs() < eps {
        return Ok((xcur, fcur, fcalls));
    }
    if fpre * fcur >= T::zero() {
        return Err(RootFindingError::Bracketing { a, b });
    }
    let mut iter = 0;
    while iter < maxiter {
        iter += 1;
        if (fpre.abs() > eps) && (fcur.abs() > eps) && (fpre * fcur < T::zero()) {
            xblk = xpre;
            fblk = fpre;
            spre = xcur - xpre;
            scur = spre;
        }
        if fblk.abs() < fcur.abs() {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;

            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }

        let delta = (xtol + rtol * xcur.abs()) / two;
        let sbis = (xblk - xcur) / two;
        if (fcur.abs() < eps) | (sbis.abs() < delta) {
            return Ok((xcur, fcur, fcalls));
        }

        if (spre.abs() > delta) && (fcur.abs() < fpre.abs()) {
            let stry = if xpre == xblk {
                /* interpolate */
                -fcur * (xcur - xpre) / (fcur - fpre)
            } else {
                /* extrapolate */
                let dpre = (fpre - fcur) / (xpre - xcur);
                let dblk = (fblk - fcur) / (xblk - xcur);
                -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            };
            if two * stry.abs() < spre.abs().min(three * sbis.abs() - delta) {
                /* good short step */
                spre = scur;
                scur = stry;
            } else {
                /* bisect */
                spre = sbis;
                scur = sbis;
            }
        } else {
            /* bisect */
            spre = sbis;
            scur = sbis;
        }

        xpre = xcur;
        fpre = fcur;
        if scur.abs() > delta {
            xcur = xcur + scur;
        } else {
            xcur = xcur + if sbis > zero { delta } else { -delta };
        }

        fcur = f(xcur);
        fcalls += 1;
    }
    Ok((xcur, fcur, fcalls))
}
