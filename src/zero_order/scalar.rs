use num_traits::{Float, One};
pub mod root;
use crate::DefaultValue;
use crate::TuutalError;
use std::mem::swap;

type BrentOptResult<T> = Result<(T, T, usize), TuutalError<(T, T, T, T, T, T, usize)>>;
type BracketResult<T> = Result<(T, T, T, T, T, T, usize), TuutalError<(T, T, T, T, T, T, usize)>>;

/// Finds intervals that bracket a minimum of a scalar function f, by searching in the downhill direction from initial points.
///
/// # Parameters
/// - **f**: Function with scalar input and scalar output.
/// - **x<sub>a</sub>**, **x<sub>b</sub>**: Initital points.
/// - **grow_limit**: Grow limit.
/// - **maxiter**: Maximum number of iterations.
///
/// # Returns
/// - Ok((x<sub>a</sub>, x<sub>b</sub>, x<sub>c</sub>, f<sub>a</sub>, f<sub>b</sub>, f<sub>c</sub>, fcalls)) when it finds a solution triplet
///   (x<sub>a</sub>, x<sub>b</sub>, x<sub>c</sub>) satisfying the bracketing condition below, fcalls is the number of function f evaluation during the algorithm.
/// - [Err(TuutalError::Convergence)](../error/enum.TuutalError.html) when the maximum number of iterations is reached without finding any solution.
/// - [Err(TuutalError::Bracketing)](../error/enum.TuutalError.html) when the algorithm found a bracket satisfying f<sub>b</sub> <= f<sub>c</sub> and did not satisfy at least one of
///   the bracketing condition.
///
/// # Notes
/// The algorithm attempts to find three finite scalars x<sub>a</sub>, x<sub>b</sub>, and x<sub>c</sub> satisfying **(bracketing condition)**:
/// - x<sub>b</sub> is strictly between x<sub>a</sub> and x<sub>c</sub>: (x<sub>a</sub> < x<sub>b</sub> < x<sub>c</sub>) or (x<sub>c</sub> < x<sub>b</sub> < x<sub>a</sub>)
/// - f<sub>b</sub>=f(x<sub>b</sub>) is below f<sub>a</sub>=f(x<sub>a</sub>) and f<sub>c</sub>=f(x<sub>c</sub>): (f<sub>b</sub> < f<sub>c</sub> and f<sub>b</sub> <= f<sub>a</sub>)
///   or (f<sub>b</sub> < f<sub>a</sub> and f<sub>b</sub> <= f<sub>c</sub>)
///
/// Adapted from [Scipy Optimize][opt]
///
/// [opt]: https://github.com/scipy/scipy/blob/v1.13.1/scipy/optimize/_optimize.py
///
/// ```
/// use tuutal::{bracket, TuutalError};
/// let f = |x: f32| 10. * x.powi(2) + 3. * x + 5.;
/// let (xa_star, xb_star, xc_star) = (1.0, 0.1, -1.3562305);
/// let (xa, xb, xc, fa, fb, fc, _fcalls) =
///     bracket(f, 0.1, 1., 110., 100).unwrap_or((0., 0., 0., 0., 0., 0., 0));
///
/// // Test bracketing condition
/// assert!((xa_star - xa).abs() < 1e-5);
/// assert!((xb_star - xb).abs() < 1e-5);
/// assert!((xc_star - xc).abs() < 1e-5);
/// assert!((xc < xb) && (xb < xa));
/// assert!((fb <= fc) && (fb < fa));
///
/// let low_maxiter = 20;
/// assert_eq!(
///     match bracket(|x: f32| x, 0.5, 2., 110., low_maxiter).unwrap_err() {
///         TuutalError::Convergence {
///             iterate: _,
///             maxiter,
///         } => maxiter,
///         _ => "-1".to_string(),
///     },
///     low_maxiter.to_string()
/// );
/// assert_eq!(
///     match bracket(|x: f32| x, 0.5, 2., 110., 500).unwrap_err() {
///         TuutalError::Bracketing { iterate } => iterate.1,
///         _ => 0.,
///     },
///     f32::NEG_INFINITY
/// );
/// ```
pub fn bracket<T, F>(f: F, mut xa: T, mut xb: T, grow_limit: T, maxiter: usize) -> BracketResult<T>
where
    T: One + Float + DefaultValue,
    F: Fn(T) -> T,
{
    let two = T::from_f32(2.);
    let _gold = (T::one() + T::from_f32(5.).sqrt()) / two; // golden ratio: (1.0+sqrt(5.0))/2.0
    let mut fa = f(xa);
    let mut fb = f(xb);
    let mut fcalls: usize = 2;
    if fa < fb {
        swap(&mut xa, &mut xb);
        swap(&mut fa, &mut fb);
    }
    let mut xc = xb + _gold * (xb - xa);
    let mut fc = f(xc);
    fcalls += 1;
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
            return Err(TuutalError::Convergence {
                iterate: (xa, xb, xc, fa, fb, fc, fcalls),
                maxiter: maxiter.to_string(),
            });
        }
        iter += 1;
        let fw = if (w - xc) * (xb - w) > zero {
            let mut fw = f(w);
            fcalls += 1;
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
            w = xc + _gold * (xc - xb);
            fw = f(w);
            fcalls += 1;
            fw
        } else if (w - wlim) * (wlim - xc) >= zero {
            w = wlim;
            let fw = f(w);
            fcalls += 1;
            fw
        } else if (w - wlim) * (xc - w) > zero {
            let mut fw = f(w);
            fcalls += 1;
            if fw < fc {
                xb = xc;
                xc = w;
                w = xc + _gold * (xc - xb);
                fb = fc;
                fc = fw;
                fw = f(w);
                fcalls += 1;
            }
            fw
        } else {
            w = xc + _gold * (xc - xb);
            let fw = f(w);
            fcalls += 1;
            fw
        };
        xa = xb;
        xb = xc;
        xc = w;
        fa = fb;
        fb = fc;
        fc = fw;
    }
    let cond1 = ((fb < fc) && (fb <= fa)) | ((fb < fa) && (fb <= fc));
    let cond2 = ((xa < xb) && (xb < xc)) | ((xc < xb) && (xb < xa));
    let cond3 =
        (xa.abs() < T::infinity()) && (xb.abs() < T::infinity()) && (xc.abs() < T::infinity());

    if !(cond1 && cond2 && cond3) {
        return Err(TuutalError::Bracketing {
            iterate: (xa, xb, xc, fa, fb, fc, fcalls),
        });
    }
    Ok((xa, xb, xc, fa, fb, fc, fcalls))
}

/// Minimizes a scalar function f using Brent's algorithm.
///
/// The algorithm uses the [`bracket`] function to find bracket intervals, before finding a solution.
///
/// # Returns
/// - Ok((x, f(x), fcalls)) if it finds a solution x minimizing f at least locally, f(x) is the output of f at x
///   and fcalls is the number of function f evaluations during the algorithm.
/// - [Err(TuutalError::SomeVariant)](../error/enum.TuutalError.html):
///     - if the function [`bracket`] fails.
///     - if convergence is not reached after success of [`bracket`].
///
/// Adapted from [Scipy Optimize][opt]
///
/// [opt]: https://github.com/scipy/scipy/blob/v1.13.1/scipy/optimize/_optimize.py
///
/// ```
/// use tuutal::brent_opt;
/// let f = |x: f32| (x - 2.) * x * (x + 2.).powi(2);
/// let (x, fx, fcalls) = brent_opt(f, 0., 1., 1000, 1.48e-8).unwrap_or((0.0, 0.0, 0));
/// assert!((x - 1.280776).abs() < 1e-4);
/// assert!((fx + 9.914950).abs() < 1e-10);
/// assert_eq!(fcalls, 25);
/// ```
pub fn brent_opt<T, F>(f: F, xa: T, xb: T, maxiter: usize, tol: T) -> BrentOptResult<T>
where
    T: One + Float + std::fmt::Debug + DefaultValue,
    F: Fn(T) -> T,
{
    match bracket(&f, xa, xb, T::from_f32(110.), maxiter) {
        Err(error) => Err(error),
        Ok((xa, xb, xc, _, fb, _, mut fcalls)) => {
            let mut x = xb;
            let mut w = xb;
            let mut v = xb;
            let mut fw = fb;
            let mut fv = fb;
            let mut fx = fb;
            let (mut a, mut b) = if xa < xc { (xa, xc) } else { (xc, xa) };
            let zero = T::zero();
            let mut deltax = zero;
            // small number that protects against trying to achieve fractional accuracy for a minimum that happens to be exactly zero
            // see for more details Press, W., S.A. Teukolsky, W.T. Vetterling, and B.P. Flannery. Numerical Recipes in C. Cambridge University Press
            let _mintol = T::epsilon();
            let _cg = T::from_f32(0.381_966);
            let one_half = T::from_f32(0.5);
            // fix of scipy rat variable initialization question.
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
                fcalls += 1;
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
            if iter < maxiter {
                Ok((x, fx, fcalls))
            } else {
                Err(TuutalError::Convergence {
                    iterate: (x, a, b, fx, f(a), f(b), fcalls + 2),
                    maxiter: maxiter.to_string(),
                })
            }
        }
    }
}
