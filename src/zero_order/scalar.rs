use num_traits::{Float, One};
pub mod root;
use crate::Number;
use crate::TuutalError;
use std::mem::swap;

pub(crate) type BrentOptError<T> = TuutalError<(T, T, T, T, T, T, usize)>;
pub(crate) type BrentOptResult<T> = Result<(T, T, usize), BrentOptError<T>>;
type BracketResult<T> = Result<(T, T, T, T, T, T, usize), BrentOptError<T>>;

/// Finds intervals that bracket a minimum of a scalar function f.
///
/// To do so, its searches in the downhill direction from initial points.
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
///         _ => 0,
///     },
///     low_maxiter
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
    T: One + Float + Number,
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
                maxiter,
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

/// Minimizes a scalar function f using Unbounded Brent's algorithm.
///
/// The algorithm uses the [`bracket`] function to find bracket intervals, before finding a solution.
///
/// # Parameters
/// - **f**: Objective function with scalar input and scalar output.
/// - **x<sub>a</sub>**, **x<sub>b</sub>**: Initital points for [`bracket`].
/// - **maxiter**: Maximum number of iterations for both the [`bracket`] function and this one.
/// - **tol**: Relative tolerance acceptable for convergence.
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
/// use tuutal::brent_unbounded;
/// let f = |x: f32| (x - 2.) * x * (x + 2.).powi(2);
/// let (x, fx, fcalls) = brent_unbounded(f, 0., 1., 1000, 1.48e-8).unwrap_or((0.0, 0.0, 0));
/// assert!((x - 1.280776).abs() < 1e-4);
/// assert!((fx + 9.914950).abs() < 1e-10);
/// assert_eq!(fcalls, 25);
/// ```
pub fn brent_unbounded<T, F>(f: F, xa: T, xb: T, maxiter: usize, tol: T) -> BrentOptResult<T>
where
    T: Number,
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
                    maxiter,
                })
            }
        }
    }
}

/// Minimizes a scalar function f using Bounded Brent algorithm.
///
/// In this algorithm, the number of iterations coincides with the number of function calls.
///
/// # Parameters
/// - **f**: Objective function with scalar input and scalar output.
/// - **bounds**: Lower (first element of **bounds**) and upper (second element of **bounds**)
///   bounds of the potentiel local mimimum.
/// - **xatol**: Absolute error acceptable for convergence.
/// - **maxiter**: Maximum number of iterations.
///
/// # Returns
/// - Ok((x, f(x), fcalls)) if it finds a solution x minimizing f at least locally, f(x) is the output of f at x
///   and fcalls is the number of function f evaluations during the algorithm.
/// - [Err(TuutalError::SomeVariant)](../error/enum.TuutalError.html):
///     - if one of the bounds is infinite.
///     - if the bounds are unordered i.e. bounds.0 > bounds.1
///     - if one of the final iterates is a NaN value
///     - if the number of iterations is reached before convergence.
///
/// Adapted from [Scipy Optimize][opt]
///
/// [opt]: https://github.com/scipy/scipy/blob/v1.13.1/scipy/optimize/_optimize.py
///
/// ```
/// use tuutal::brent_bounded;
/// let f = |x: f32| (x - 2.) * x * (x + 2.).powi(2);
/// let bounds = (0., 2.);
/// let (x, fx, fcalls) = brent_bounded(f, bounds, 1.48e-8, 1000).unwrap_or((0.0, 0.0, 0));
/// assert!((bounds.0 <= x) && (x <= bounds.1));
/// assert!((x - 1.280776).abs() < 1e-4);
/// assert!((fx + 9.914950).abs() < 1e-6);
/// assert_eq!(fcalls, 8);
/// ```
pub fn brent_bounded<T, F>(f: F, bounds: (T, T), xatol: T, maxiter: usize) -> BrentOptResult<T>
where
    T: Number,
    F: Fn(T) -> T,
{
    let (x1, x2) = bounds;

    if x1.is_infinite() {
        return Err(TuutalError::Infinity {
            x: (x1, x1, x1, x1, x1, x1, 0),
        });
    }
    if x2.is_infinite() {
        return Err(TuutalError::Infinity {
            x: (x2, x2, x2, x2, x2, x2, 0),
        });
    }
    if x1 > x2 {
        return Err(TuutalError::BoundOrder {
            lower: (x1, x1, x1, x1, x1, x1, 0),
            upper: (x2, x2, x2, x2, x2, x2, 0),
        });
    }

    let zero = T::zero();
    let two = T::from_f32(2.);
    let three = T::from_f32(3.);
    let one_half = T::from_f32(0.5);

    let sqrt_eps = T::epsilon().sqrt();
    let golden_mean = one_half * (three - T::from_f32(5.0).sqrt());
    let (mut a, mut b) = (x1, x2);
    let mut fulc = a + golden_mean * (b - a);
    let (mut nfc, mut xf) = (fulc, fulc);
    let mut rat = zero;
    let mut e = zero;
    let mut x = xf;
    let mut fx = f(x);
    let mut fcalls = 1;
    // let mut fmin_data = (1, xf, fx);
    let mut fu = T::infinity();

    let mut ffulc = fx;
    let mut fnfc = fx;
    let mut xm = one_half * (a + b);
    let mut tol1 = sqrt_eps * xf.abs() + xatol / three;
    let mut tol2 = two * tol1;

    while (xf - xm).abs() > tol2 - one_half * (b - a) {
        let mut golden = true;
        // Check for parabolic fit
        if e.abs() > tol1 {
            golden = false;
            let mut r = (xf - nfc) * (fx - ffulc);
            let mut q = (xf - fulc) * (fx - fnfc);
            let mut p = (xf - fulc) * q - (xf - nfc) * r;
            q = two * (q - r);
            if q > T::zero() {
                p = -p;
            }
            q = q.abs();
            r = e;
            e = rat;

            // Check for acceptability of parabola
            if (p.abs() < (one_half * q * r).abs()) && (p > q * (a - xf)) && (p < q * (b - xf)) {
                rat = (p + zero) / q;
                x = xf + rat;
                if ((x - a) < tol2) | ((b - x) < tol2) {
                    rat = tol1 * (xm - xf).signum();
                }
            } else {
                golden = true;
            }
        }
        if golden {
            // do a golden-section step
            e = if xf >= xm { a - xf } else { b - xf };
            rat = golden_mean * e;
        }
        let abs_rat = rat.abs();
        x = xf + rat.signum() * if abs_rat > tol1 { abs_rat } else { tol1 };
        fu = f(x);
        fcalls += 1;
        // fmin_data = (fcalls, x, fu);

        if fu <= fx {
            if x >= xf {
                a = xf
            } else {
                b = xf
            }
            (fulc, ffulc) = (nfc, fnfc);
            (nfc, fnfc) = (xf, fx);
            (xf, fx) = (x, fu)
        } else {
            if x < xf {
                a = x
            } else {
                b = x
            }
            if (fu <= fnfc) | (nfc == xf) {
                (fulc, ffulc) = (nfc, fnfc);
                (nfc, fnfc) = (x, fu);
            } else if (fu <= ffulc) | (fulc == xf) | (fulc == nfc) {
                (fulc, ffulc) = (x, fu)
            }
        }
        xm = one_half * (a + b);
        tol1 = sqrt_eps * xf.abs() + xatol / three;
        tol2 = two * tol1;

        if fcalls >= maxiter {
            break;
        }
    }
    if xf.is_nan() {
        return Err(TuutalError::Nan {
            x: (xf, a, b, fx, f(a), f(b), fcalls + 2),
        });
    }
    if fx.is_nan() {
        return Err(TuutalError::Nan {
            x: (xf, a, b, fx, f(a), f(b), fcalls + 2),
        });
    }
    if fu.is_nan() {
        return Err(TuutalError::Nan {
            x: (xf, a, b, fx, f(a), f(b), fcalls + 2),
        });
    }
    if fcalls >= maxiter {
        return Err(TuutalError::Convergence {
            iterate: (xf, a, b, fx, f(a), f(b), fcalls + 2),
            maxiter,
        });
    }
    Ok((xf, fx, fcalls))
}
