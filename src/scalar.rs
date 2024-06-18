use num_traits::{Float, One};
pub mod root;
use crate::TuutalError;
use std::mem::swap;

/// Finds intervals that bracket the minimum of a function
/// Adapted from https://github.com/scipy/scipy/blob/c22b657faf9e8cf19167a82b3bfe65a90a2c5afb/scipy/optimize/_optimize.py
/// ```
/// use tuutal::scalar::bracket;
/// let f = |x: f32| 10. * x.powi(2) + 3. * x + 5.;
/// let expected_bracket = (1.0, 0.1, -1.3562305);
/// assert_eq!(
///     expected_bracket,
///     bracket(f, 0.1, 1., 100).unwrap_or((0., 0., 0.))
/// );
/// ```
pub fn bracket<T>(
    f: impl Fn(T) -> T,
    mut xa: T,
    mut xb: T,
    maxiter: usize,
) -> Result<(T, T, T), TuutalError>
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
    Ok((xa, xb, xc))
}
