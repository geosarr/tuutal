#[cfg(test)]
mod unit_test;
use crate::DefaultValue;
use ndarray::linalg::Dot;
use std::{
    fmt::Debug,
    ops::{Add, Mul, Neg, Sub},
};

/// Alias of the output type of step size functions.
type StepSizeFunction<X, A> =
    fn(fn(&X) -> A, fn(&X) -> X, &X, &X, &SteepestDescentParameter<A>) -> A;

/// Parameters used in the steepest descent method.
#[derive(Debug, Clone, Copy)]
pub enum SteepestDescentParameter<T> {
    Armijo(T, T),
    PowellWolfe(T, T),
}

impl<T> Default for SteepestDescentParameter<T>
where
    T: DefaultValue,
{
    fn default() -> Self {
        Self::Armijo(T::tol(-3), T::from_f32(0.5))
    }
}

impl<T> SteepestDescentParameter<T> {
    pub fn gamma(&self) -> &T {
        match self {
            Self::Armijo(g, _) => g,
            Self::PowellWolfe(g, _) => g,
        }
    }
    pub fn beta(&self) -> &T {
        match self {
            Self::Armijo(_, b) => b,
            Self::PowellWolfe(_, b) => b,
        }
    }
}

/// Computes a step size using the Armijo method.
fn armijo_rule<F, G, A, X>(f: F, gradf: G, x: &X, d: &X, params: &SteepestDescentParameter<A>) -> A
where
    for<'a> A: DefaultValue
        + Sub<A, Output = A>
        + Mul<A, Output = A>
        + Mul<&'a X, Output = X>
        + PartialOrd
        + Copy,
    F: Fn(&X) -> A,
    G: Fn(&X) -> X,
    X: Dot<X, Output = A> + Add<X, Output = X> + Clone,
{
    let mut sigma = A::one();
    let mut x_next = x.clone() + sigma * d;
    let gradx_d = gradf(x).dot(d);
    while f(&x_next) - f(x) > sigma * *params.gamma() * gradx_d {
        sigma = *params.beta() * sigma;
        x_next = x.clone() + sigma * d;
    }
    sigma
}

/// Computes a step size using the Powell Wolfe method.
fn powell_wolfe_rule<F, G, A, X>(
    f: F,
    gradf: G,
    x: &X,
    d: &X,
    params: &SteepestDescentParameter<A>,
) -> A
where
    for<'a> A: DefaultValue
        + Sub<A, Output = A>
        + Add<A, Output = A>
        + Mul<A, Output = A>
        + Mul<&'a X, Output = X>
        + PartialOrd
        + Copy,
    F: Fn(&X) -> A,
    G: Fn(&X) -> X,
    X: Dot<X, Output = A> + Add<X, Output = X> + Clone,
{
    let mut sigma_minus = A::one();
    let mut sigma_plus = A::one();
    let mut sigma = A::one();
    let mut x_next = x.clone() + sigma_minus * d;
    let gradx_d = gradf(x).dot(d);
    let one_half = A::from_f32(0.5);
    // The first if and else conditions guarantee having a segment [sigma_minus, sigma_plus]
    // such that sigma_minus satisfies the armijo condition and sigma_plus does not
    if f(&x_next) - f(x) <= sigma_minus * *params.gamma() * gradx_d {
        if gradf(&x_next).dot(d) >= *params.beta() * gradx_d {
            return sigma_minus;
        }
        // Computation of sigma_plus
        let two = A::from_f32(2.);
        sigma_plus = two;
        x_next = x.clone() + sigma_plus * d;
        while f(&x_next) - f(x) <= sigma_plus * *params.gamma() * gradx_d {
            sigma_plus = two * sigma_plus;
            x_next = x.clone() + sigma_plus * d;
        }
        // At this stage sigma_plus is the smallest 2^k that does not satisfy the Armijo rule
        sigma_minus = sigma_plus * one_half; // it satisfies the Armijo rule
    } else {
        sigma_minus = one_half;
        x_next = x.clone() + sigma_minus * d;
        while f(&x_next) - f(x) > sigma_minus * *params.gamma() * gradx_d {
            sigma_minus = one_half * sigma_minus;
            x_next = x.clone() + sigma_minus * d;
        }
        sigma_plus = sigma_minus * (A::from_f32(2.)); // does not satisfy the Armijo rule
    }
    x_next = x.clone() + sigma_minus * d;
    while gradf(&x_next).dot(d) < *params.beta() * gradx_d {
        sigma = (sigma_minus + sigma_plus) * one_half;
        x_next = x.clone() + sigma * d;
        if f(&x_next) - f(x) <= sigma * *params.gamma() * gradx_d {
            sigma_minus = sigma;
        } else {
            sigma_plus = sigma;
        }
    }
    sigma_minus
}

/// The steepest descent algorithm using Armijo or Powell Wolfe step size methods.
/// It requires an initial guess **x0**.
/// ```
/// use tuutal::{VecType, array, steepest_descent, SteepestDescentParameter};
/// // Example from python scipy.optimize.minimize_scalar
/// let f = |x: &VecType<f32>| (x[0] - 2.) * x[0] * (x[0] + 2.).powi(2);
/// let gradf = |x: &VecType<f32>| array![2. * (x[0] + 2.) * (2. * x[0].powi(2) - x[0] - 1.)];
/// let x0 = &array![-1.];
///
/// let x_star = steepest_descent(f, gradf, &x0, &SteepestDescentParameter::Armijo(1e-2, 0.25), 1e-3, 10);
/// assert!((-2. - x_star.unwrap()[0]).abs() < 1e-10);
///
/// let x_star = steepest_descent(f, gradf, &x0, &SteepestDescentParameter::PowellWolfe(1e-2, 0.9), 1e-3, 10);
/// assert!((-2. - x_star.unwrap()[0]).abs() < 1e-10);
///
/// let x0 = &array![-0.5];
/// let x_star = steepest_descent(f, gradf, &x0, &Default::default(), 1e-3, 10);
/// assert!((-0.5 - x_star.unwrap()[0]).abs() < 1e-10);
///
/// let x0 = &array![0.];
/// let x_star = steepest_descent(f, gradf, &x0, &Default::default(), 1e-3, 10);
/// assert!((1. - x_star.unwrap()[0]).abs() < 1e-10);
///
/// // It also takes multivariate objective functions
/// let f = |arr: &VecType<f32>| 100. * (arr[1] - arr[0].powi(2)).powi(2) + (1. - arr[0]).powi(2);
/// let gradf = |arr: &VecType<f32>| {
///     array![
///         -400. * arr[0] * (arr[1] - arr[0].powi(2)) - 2. * (1. - arr[0]),
///          200. * (arr[1] - arr[0].powi(2))
///     ]
/// };
/// let x = array![1f32, -0.5f32];
/// let opt = steepest_descent(f, gradf, &x, &Default::default(), 1e-3, 10000).unwrap();
/// assert!((opt[0] - 1.).abs() <= 1e-2);
/// assert!((opt[1] - 1.).abs() <= 1e-2);
/// ```
pub fn steepest_descent<X, A>(
    f: fn(&X) -> A,
    gradf: fn(&X) -> X,
    x0: &X,
    params: &SteepestDescentParameter<A>,
    eps: A,
    nb_iter: usize,
) -> Option<X>
where
    for<'a> A: DefaultValue
        + Sub<A, Output = A>
        + Add<A, Output = A>
        + Mul<A, Output = A>
        + Mul<&'a X, Output = X>
        + Mul<X, Output = X>
        + PartialOrd
        + Copy,
    X: Dot<X, Output = A> + Neg<Output = X> + Add<X, Output = X> + Clone,
{
    let mut iter = 0;
    let mut iterates = SteepestDescentIterates::new(f, gradf, x0.clone(), *params, eps);
    let mut iterate_star = iterates.next();
    while iter < nb_iter {
        iterate_star = iterates.next();
        iter += 1;
    }
    iterate_star
}

/// Represents the sequence of iterates computed by a steepest algorithm.
pub struct SteepestDescentIterates<X, A> {
    f: fn(&X) -> A,
    gradf: fn(&X) -> X,
    params: SteepestDescentParameter<A>,
    x: X,
    eps: A,
    rule: StepSizeFunction<X, A>,
}

impl<X, A> SteepestDescentIterates<X, A> {
    pub fn new(
        f: fn(&X) -> A,
        gradf: fn(&X) -> X,
        x: X,
        params: SteepestDescentParameter<A>,
        eps: A,
    ) -> Self
    where
        for<'a> A: DefaultValue
            + Sub<A, Output = A>
            + Add<A, Output = A>
            + Mul<A, Output = A>
            + Mul<&'a X, Output = X>
            + PartialOrd
            + Copy,
        X: Dot<X, Output = A> + Add<X, Output = X> + Clone,
    {
        let rule = match params {
            SteepestDescentParameter::Armijo(_, _) => armijo_rule,
            SteepestDescentParameter::PowellWolfe(_, _) => powell_wolfe_rule,
        };
        Self {
            f,
            gradf,
            params,
            x,
            eps,
            rule,
        }
    }
    pub fn obj(&self) -> fn(&X) -> A {
        self.f
    }
    pub fn grad_obj(&self) -> fn(&X) -> X {
        self.gradf
    }
    pub fn step_size_rule(&self) -> StepSizeFunction<X, A> {
        self.rule
    }
}

impl<X, A> std::iter::Iterator for SteepestDescentIterates<X, A>
where
    for<'a> A: DefaultValue
        + Sub<A, Output = A>
        + Add<A, Output = A>
        + Mul<A, Output = A>
        + Mul<&'a X, Output = X>
        + Mul<X, Output = X>
        + PartialOrd
        + Copy,
    X: Dot<X, Output = A> + Neg<Output = X> + Add<X, Output = X> + Clone,
{
    type Item = X;
    fn next(&mut self) -> Option<Self::Item> {
        let neg_grad = -self.grad_obj()(&self.x);
        if neg_grad.dot(&neg_grad) <= (self.eps * self.eps) {
            Some(self.x.clone())
        } else {
            let sigma = self.step_size_rule()(
                self.obj(),
                self.grad_obj(),
                &self.x,
                &neg_grad,
                &self.params,
            );
            self.x = self.x.clone() + sigma * neg_grad;
            Some(self.x.clone())
        }
    }
}
