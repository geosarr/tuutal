use crate::{Number, RootFindingError};

/// Brent's root finding algorithm using inverse quadratic interpolation.
///
/// The number of iterations coincides with the number of function evaluations - 2.
///
/// # Parameters
/// - **f**: Function with scalar input and scalar output.
/// - **a, b**: Initital points.
/// - **xtol**: Absolute tolerance on the solution for convergence.
/// - **rtol**: Relative tolerance on the solution for convergence.
/// - **maxiter**: Maximum number of iterations.
///
/// # Returns
/// - Ok((x, f(x), fcalls)) when a solution x is found, f(x) is the output of x by f, and fcalls is the number of function f
///   evaluation during the algorithm.
/// - [Err(RootFindingError::Bracketing)](../error/enum.RootFindingError.html) when f(a) * f(b) >= 0.
///
/// Adapted from [Scipy Optimize][brq]
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
    if fpre * fcur > T::zero() {
        return Err(RootFindingError::Bracketing { a, b });
    }
    let mut iter = 0;
    while iter < maxiter {
        iter += 1;
        if (fpre.abs() > eps) && (fcur.abs() > eps) && (fpre * fcur < T::zero()) {
            (xblk, fblk) = (xpre, fpre);
            spre = xcur - xpre;
            scur = spre;
        }
        if fblk.abs() < fcur.abs() {
            (xpre, fpre) = (xcur, fcur);
            (xcur, fcur) = (xblk, fblk);
            (xblk, fblk) = (xpre, fpre);
        }

        let delta = (xtol + rtol * xcur.abs()) / two;
        let sbis = (xblk - xcur) / two;
        if (fcur.abs() < eps) | (sbis.abs() < delta) {
            return Ok((xcur, fcur, fcalls));
        }

        (spre, scur) = if (spre.abs() > delta) && (fcur.abs() < fpre.abs()) {
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
                (scur, stry)
            } else {
                /* bisect */
                (sbis, sbis)
            }
        } else {
            /* bisect */
            (sbis, sbis)
        };

        (xpre, fpre) = (xcur, fcur);
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
