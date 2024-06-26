use num_traits::{Float, One};
pub mod root;
use crate::DefaultValue;
use crate::TuutalError;
use std::mem::swap;

/// Finds intervals that bracket the minimum of a scalar function f.
///
/// The algorithm attempts to find three finite scalars x<sub>a</sub>, x<sub>b</sub>, and x<sub>c</sub> satisfying **(bracketing condition)**:
/// - x<sub>b</sub> is strictly between x<sub>a</sub> and x<sub>c</sub>: (x<sub>a</sub> < x<sub>b</sub> < x<sub>c</sub>) or (x<sub>c</sub> < x<sub>b</sub> < x<sub>a</sub>)
/// - f<sub>b</sub>=f(x<sub>b</sub>) is below f<sub>a</sub>=f(x<sub>a</sub>) and f<sub>c</sub>=f(x<sub>c</sub>): (f<sub>b</sub> < f<sub>c</sub> and f<sub>b</sub> <= f<sub>a</sub>) or (f<sub>b</sub> < f<sub>a</sub> and f<sub>b</sub> <= f<sub>c</sub>)
///
/// # Returns
/// - Ok((x<sub>a</sub>, x<sub>b</sub>, x<sub>c</sub>, f<sub>a</sub>, f<sub>b</sub>, f<sub>c</sub>)) when it finds a solution triplet
/// (x<sub>a</sub>, x<sub>b</sub>, x<sub>c</sub>) satisfying the above conditions.
/// - [Err(TuutalError::Convergence(maxiter))](../error/enum.TuutalError.html) when the maximum number of iterations is reached without finding any solution.
/// - [Err(TuutalError::Bracketing)](../error/enum.TuutalError.html) when the algorithm found a bracket satisfying f<sub>b</sub> <= f<sub>c</sub> and did not satisfy at least one of
/// the above bracketing condition.
///
/// Adapted from [Scipy Optimize][opt]
///
/// [opt]: https://github.com/scipy/scipy/blob/v1.13.1/scipy/optimize/_optimize.py
///
/// ```
/// use tuutal::{bracket, TuutalError};
/// let f = |x: f32| 10. * x.powi(2) + 3. * x + 5.;
/// let (xa_star, xb_star, xc_star) = (1.0, 0.1, -1.3562305);
/// let (xa, xb, xc, fa, fb, fc) = bracket(f, 0.1, 1., 100).unwrap_or((0., 0., 0., 0., 0., 0.));
/// // Test bracketing condition
/// assert!((xa_star - xa).abs() < 1e-5);
/// assert!((xb_star - xb).abs() < 1e-5);
/// assert!((xc_star - xc).abs() < 1e-5);
/// assert!((xc < xb) && (xb < xa));
/// assert!((fb <= fc) && (fb < fa));
///
/// let low_maxiter = 20;
/// let convergence_error = TuutalError::Convergence(low_maxiter.to_string());
/// let bracketing_error = TuutalError::Bracketing;
/// assert_eq!(
///     bracket(|x: f32| x.powi(3), 0.5, 2., low_maxiter).unwrap_err(),
///     convergence_error
/// );
/// assert_eq!(
///     bracket(|x: f32| x, 0.5, 2., 500).unwrap_err(),
///     bracketing_error
/// );
/// ```
pub fn bracket<T>(
    f: impl Fn(T) -> T,
    mut xa: T,
    mut xb: T,
    maxiter: usize,
) -> Result<(T, T, T, T, T, T), TuutalError>
where
    T: One + Float + DefaultValue,
{
    let two = T::from_f32(2.);
    let _gold = (T::one() + T::from_f32(5.).sqrt()) / two; // golden ratio: (1.0+sqrt(5.0))/2.0
    let grow_limit = T::from_f32(110.);
    let mut fa = f(xa);
    let mut fb = f(xb);
    if fa < fb {
        swap(&mut xa, &mut xb);
        swap(&mut fa, &mut fb);
    }
    let mut xc = xb + _gold * (xb - xa);
    let mut fc = f(xc);
    let mut iter = 0;
    let zero = T::zero();
    while fc < fb {
        let tmp1 = (xb - xa) * (fb - fc);
        let tmp2 = (xb - xc) * (fb - fa);
        let val = tmp2 - tmp1;
        let denom = if val.abs() < T::epsilon() {
            two * T::epsilon()
        } else {
            two * val
        };
        let mut w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom;
        let wlim = xb + grow_limit * (xc - xb);
        if iter > maxiter {
            return Err(TuutalError::Convergence(maxiter.to_string()));
        }
        iter += 1;
        let mut fw = f(w);
        if (w - xc) * (xb - w) > zero {
            // fw = f(w);
            if fw < fc {
                xa = xb;
                xb = w;
                fa = fb;
                fb = fw;
                break;
            } else if fw > fb {
                xc = w;
                fc = fw;
                break;
            }
            // w = xc + _gold * (xc - xb);
            // fw = f(w);
        } else if (w - wlim) * (wlim - xc) >= zero {
            // w = wlim;
            // fw = f(w);
        } else if (w - wlim) * (xc - w) > zero {
            // fw = f(w);
            if fw < fc {
                xb = xc;
                xc = w;
                // w = xc + _gold * (xc - xb);
                fb = fc;
                fc = fw;
                // fw = f(w);
            }
        } else {
            w = xc + _gold * (xc - xb);
            fw = f(w);
            xa = xb;
            xb = xc;
            xc = w;
            fa = fb;
            fb = fc;
            fc = fw;
        }
    }
    let cond1 = ((fb < fc) && (fb <= fa)) | ((fb < fa) && (fb <= fc));
    let cond2 = ((xa < xb) && (xb < xc)) | ((xc < xb) && (xb < xa));
    let cond3 =
        (xa.abs() < T::infinity()) && (xb.abs() < T::infinity()) && (xc.abs() < T::infinity());

    if !(cond1 && cond2 && cond3) {
        return Err(TuutalError::Bracketing);
    }
    Ok((xa, xb, xc, fa, fb, fc))
}

/// Minimizes a scalar function f using Brent's algorithm.
///
/// The algorithm uses the [`bracket`] function to find bracket intervals, before finding a solution.
///
/// # Returns
/// - Ok((x, f(x))) if it finds a solution x minimizing f at least locally.
/// - [Err(TuutalError::SomeVariant)](../error/enum.TuutalError.html) if the function [`bracket`] fails.
///
/// Adapted from [Scipy Optimize][opt]
///
/// [opt]: https://github.com/scipy/scipy/blob/v1.13.1/scipy/optimize/_optimize.py
///
/// ```
/// use tuutal::brent_opt;
/// let f = |x: f32| (x - 2.) * x * (x + 2.).powi(2);
/// let (x, fx) = brent_opt(f, 0., 1., 1000, 1.48e-8).unwrap_or((0.0, 0.0));
/// assert!((x - 1.280776).abs() < 1e-4);
/// assert!((fx + 9.914950).abs() < 1e-4);
/// ```
pub fn brent_opt<T>(
    f: impl Fn(T) -> T,
    xa: T,
    xb: T,
    maxiter: usize,
    tol: T,
) -> Result<(T, T), TuutalError>
where
    T: One + Float + std::fmt::Debug + DefaultValue,
{
    match bracket(&f, xa, xb, maxiter) {
        Err(error) => Err(error),
        Ok((xa, xb, xc, _, fb, _)) => {
            let mut x = xb;
            let mut w = xb;
            let mut v = xb;
            let mut fw = fb;
            let mut fv = fb;
            let mut fx = fb;
            let (mut a, mut b) = if xa < xc { (xa, xc) } else { (xc, xa) };
            let zero = T::zero();
            let mut deltax = zero;
            let ten = T::from_f32(10.);
            let _mintol = ten.powi(-11);
            let _cg = T::from_f32(0.381_966);
            let one_half = T::from_f32(0.5);
            // fix of scipy rat variable initilization question.
            let mut rat = if x >= one_half * (a + b) {
                a - x
            } else {
                b - x
            } * _cg;
            let one = T::one();
            let two = T::from_f32(2.);
            let mut iter = 0;
            while iter < maxiter {
                let tol1 = tol * x.abs() + _mintol;
                let tol2 = two * tol1;
                let xmid = one_half * (a + b);
                // check for convergence
                if (x - xmid).abs() < (tol2 - one_half * (b - a)) {
                    break;
                }
                if deltax.abs() <= tol1 {
                    // do a golden section step
                    deltax = if x >= xmid { a - x } else { b - x };
                    rat = _cg * deltax;
                } else {
                    // do a parabolic step
                    let tmp1 = (x - w) * (fx - fv);
                    let tmp2 = (x - v) * (fx - fw);
                    let mut p = (x - v) * tmp2 - (x - w) * tmp1;
                    let mut tmp2 = two * (tmp2 - tmp1);
                    if tmp2 > zero {
                        p = -p;
                    }
                    tmp2 = tmp2.abs();
                    let dx_temp = deltax;
                    deltax = rat;
                    // check parabolic fit
                    if (p > tmp2 * (a - x))
                        && (p < tmp2 * (b - x))
                        && (p.abs() < (one_half * tmp2 * dx_temp).abs())
                    {
                        rat = p * one / tmp2; // if parabolic step is useful.
                        let u = x + rat;
                        if ((u - a) < tol2) | ((b - u) < tol2) {
                            if xmid >= x {
                                rat = tol1;
                            } else {
                                rat = -tol1;
                            }
                        }
                    } else {
                        if x >= xmid {
                            deltax = a - x; // if it's not do a golden section step
                        } else {
                            deltax = b - x;
                        }
                        rat = _cg * deltax;
                    }
                }
                let u = if rat.abs() < tol1 {
                    // update by at least tol1
                    if rat >= zero {
                        x + tol1
                    } else {
                        x - tol1
                    }
                } else {
                    x + rat
                };

                let fu = f(u);
                if fu > fx {
                    // if it's bigger than current
                    if u < x {
                        a = u;
                    } else {
                        b = u;
                    }
                    if (fu <= fw) | (w == x) {
                        v = w;
                        w = u;
                        fv = fw;
                        fw = fu;
                    } else if (fu <= fv) | (v == x) | (v == w) {
                        v = u;
                        fv = fu;
                    }
                } else {
                    if u >= x {
                        a = x;
                    } else {
                        b = x;
                    }
                    v = w;
                    w = x;
                    x = u;
                    fv = fw;
                    fw = fx;
                    fx = fu;
                }
                iter += 1;
            }
            Ok((x, fx))
        }
    }
}
