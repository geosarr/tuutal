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
/// let res = brent_root(|x: f32| x.powi(2) - 4., 0., 3.).unwrap_or((0., 1.));
/// assert!((res.0 - 2.).abs() <= 1e-4);
/// assert!(res.1.abs() <= 1e-6);
///
/// let res = brent_root(|x: f32| x.powi(2) - 2., 0., 2.).unwrap_or((0., 1.));
/// assert!((res.0 - 1.4141).abs() <= 1e-3);
/// assert!(res.1.abs() <= 1e-6);
///
/// let res = brent_root(|x: f32| x.powi(3) + 27., -4., 5.).unwrap_or((0., 1.));
/// assert!((res.0 + 3.).abs() <= 1e-4);
/// assert!(res.1.abs() <= 1e-6);
///
/// let (a, b) = (0., 1.);
/// let error = RootFindingError::Bracketing {a, b};
/// assert_eq!(brent_root(|x: f32| x, a, b).unwrap_err(), error);
/// ```
pub fn brent_root<T>(f: impl Fn(T) -> T, mut a: T, mut b: T) -> Result<(T, T), RootFindingError<T>>
where
    T: Number,
{
    let fa = f(a);
    let fb = f(b);
    if fa * fb >= T::zero() {
        return Err(RootFindingError::Bracketing { a, b });
    }
    if fa.abs() < fb.abs() {
        swap(&mut a, &mut b);
    }
    let mut c = a;
    let mut mflag = true;
    let two = T::from_f32(2.);
    let three = T::from_f32(3.);
    let four = T::from_f32(4.);
    let delta = T::from_f32(10.).powi(-5);
    while (f(b).abs() > T::epsilon()) && (f(a).abs() > T::epsilon()) && (a - b).abs() > T::epsilon()
    {
        let fa = f(a);
        let fb = f(b);
        let fc = f(c);
        let mut s = if ((fa - fc).abs() > T::epsilon()) && ((fb - fc).abs() > T::epsilon()) {
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
        if f(a) * f(s) < T::zero() {
            b = s;
        } else {
            a = s;
        }
        if f(a).abs() < f(b).abs() {
            swap(&mut a, &mut b);
        }
    }
    Ok((b, f(b)))
}
