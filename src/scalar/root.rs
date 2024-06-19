use std::{
    mem::swap,
    ops::{Div, Mul, Sub},
};

use num_traits::Float;

use crate::RootFindingError;

/// Inverse quadratic interpolation
fn inv_quad_interpol<T>(a: T, fa: T, fb: T, fc: T) -> T
where
    T: Mul<Output = T> + Sub<Output = T> + Div<Output = T> + Copy,
{
    a * fb * fc / ((fa - fb) * (fa - fc))
}

/// [Brent's root finding][br] algorithm for a scalar function.
/// [br]: https://en.wikipedia.org/wiki/Brent%27s_method
/// ```
/// use tuutal::brent_root;
/// assert!((brent_root(|x: f32| x.powi(2) - 4., 0., 3.).unwrap_or(0.) - 2.).abs() <= 1e-4);
/// assert!((brent_root(|x: f32| x.powi(2) - 2., 0., 2.).unwrap_or(0.) - 1.4141).abs() <= 1e-3);
/// assert!((brent_root(|x: f32| x.powi(3) + 27., -4., 5.).unwrap_or(0.) + 3.).abs() <= 1e-4);
/// ```
pub fn brent_root<T>(f: impl Fn(T) -> T, mut a: T, mut b: T) -> Result<T, RootFindingError>
where
    T: Float + ToString,
{
    let fa = f(a);
    let fb = f(b);
    if fa * fb >= T::zero() {
        return Err(RootFindingError::Bracketing {
            a: a.to_string(),
            b: b.to_string(),
        });
    }
    if fa.abs() < fb.abs() {
        swap(&mut a, &mut b);
    }
    let mut c = a;
    let mut mflag = true;
    let two = T::one() + T::one();
    let three = T::one() + two;
    let four = T::one() + three;
    let ten = two * (four + T::one());
    let delta = T::powi(ten, -5);
    while (f(b).abs() > T::epsilon()) && (f(a).abs() > T::epsilon()) && (a - b).abs() > T::epsilon()
    {
        let fa = f(a);
        let fb = f(b);
        let fc = f(c);
        let mut s = if ((fa - fc).abs() > T::epsilon()) && ((fb - fc).abs() > T::epsilon()) {
            if fa == fb {
                return Err(RootFindingError::Interpolation {
                    a: a.to_string(),
                    b: b.to_string(),
                });
            }
            if fa == fc {
                return Err(RootFindingError::Interpolation {
                    a: a.to_string(),
                    b: c.to_string(),
                });
            }
            if fb == fc {
                return Err(RootFindingError::Interpolation {
                    a: b.to_string(),
                    b: c.to_string(),
                });
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
    Ok(b)
}