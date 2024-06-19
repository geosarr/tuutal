use num_traits::{Float, One};
pub mod root;
use crate::TuutalError;
use std::mem::swap;

/// Finds intervals that bracket the minimum of a scalar function
/// Adapted from https://github.com/scipy/scipy/blob/c22b657faf9e8cf19167a82b3bfe65a90a2c5afb/scipy/optimize/_optimize.py
/// ```
/// use tuutal::scalar::bracket;
/// let f = |x: f32| 10. * x.powi(2) + 3. * x + 5.;
/// let expected_bracket = (1.0, 0.1, -1.3562305);
/// let output_bracket = bracket(f, 0.1, 1., 100).unwrap_or((0., 0., 0., 0., 0., 0.));
/// assert!((expected_bracket.0 - output_bracket.0).abs() < 1e-5);
/// assert!((expected_bracket.1 - output_bracket.1).abs() < 1e-5);
/// assert!((expected_bracket.2 - output_bracket.2).abs() < 1e-5);
/// ```
pub fn bracket<T>(
    f: impl Fn(T) -> T,
    mut xa: T,
    mut xb: T,
    maxiter: usize,
) -> Result<(T, T, T, T, T, T), TuutalError>
where
    T: One + Float,
{
    let two = T::one() + T::one();
    let five = two + two + T::one();
    let _gold = (T::one() + five.sqrt()) / two; // golden ratio: (1.0+sqrt(5.0))/2.0
    let ten = two * five;
    let grow_limit = ten.powi(2) + ten; // 110.
    let mut fa = f(xa);
    let mut fb = f(xb);
    if fa < fb {
        swap(&mut xa, &mut xb);
        swap(&mut fa, &mut fb);
    }
    let mut xc = xb + _gold * (xb - xa);
    let mut fc = f(xc);
    let mut iter = 0;
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
        if (w - xc) * (xb - w) > T::zero() {
            fw = f(w);
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
        } else if (w - wlim) * (wlim - xc) >= T::zero() {
            w = wlim;
            fw = f(w);
        } else if (w - wlim) * (xc - w) > T::zero() {
            fw = f(w);
            if fw < fc {
                xb = xc;
                xc = w;
                w = xc + _gold * (xc - xb);
                fb = fc;
                fc = fw;
                fw = f(w);
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

/// Minimizes a scalar function using Brent's algorithm.
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
    T: One + Float + std::fmt::Debug,
{
    let bracket_info = bracket(&f, xa, xb, maxiter);
    if let Err(error) = bracket_info {
        Err(error)
    } else {
        let (xa, xb, xc, _, fb, _) =
            bracket_info.unwrap_or_else(|_| panic!("Bracket unwrap error."));
        let mut x = xb;
        let mut w = xb;
        let mut v = xb;
        let mut fw = fb;
        let mut fv = fb;
        let mut fx = fb;
        let (mut a, mut b) = if xa < xc { (xa, xc) } else { (xc, xa) };
        let mut deltax = T::zero();
        let mut iter = 0;
        let two = T::one() + T::one();
        let one_half = T::one() / two;
        let three = two + T::one();
        let five = two + three;
        let ten = two * five;
        let _mintol = ten.powi(-11);
        let _cg = three * ten.powi(-1) + (three + five) * ten.powi(-2) + ten.powi(-3); // 0.381

        // fix of scipy rat variable initilization question.
        let mut rat = if x >= one_half * (a + b) {
            a - x
        } else {
            b - x
        } * _cg;
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
                if tmp2 > T::zero() {
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
                    rat = p * T::one() / tmp2; // if parabolic step is useful.
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
                if rat >= T::zero() {
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
