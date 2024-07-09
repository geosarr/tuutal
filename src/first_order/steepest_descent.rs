#[cfg(test)]
mod unit_test;
use crate::{optimize, Iterable, Number, Scalar, TuutalError};
use ndarray::linalg::Dot;
use std::{
    fmt::Debug,
    ops::{Add, Neg},
};

/// Parameters used in the steepest descent method.
///
/// The **gamma** parameter represents a magnitude of decrease in the objective function
/// in the negative gradient direction. The **beta** parameter controls:
/// - step size magnitude of decrease in the Armijo / AdaGrad rule.
/// - descent steepness for the Powell Wolfe strategy.
///
/// Use methods [`new_armijo`], [`new_powell_wolfe`] and [`new_adagrad`] to construct these parameters.
///
/// [`new_armijo`]: SteepestDescentParameter::new_armijo
///
/// [`new_powell_wolfe`]: SteepestDescentParameter::new_powell_wolfe
///
/// [`new_adagrad`]: SteepestDescentParameter::new_adagrad
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SteepestDescentParameter<T> {
    /// Armijo rule step size rule
    Armijo { gamma: T, beta: T },
    /// Powell Wolfe step size rule
    PowellWolfe { gamma: T, beta: T },
    /// Adaptive Gradient step size rule
    ///
    /// This variant uses Armijo rule (and the same parameters) to update the step size before normalizing it
    /// with the square root of the sum of the previous gradients L<sub>2</sub> norm, i.e at step each step t:
    /// - adagrad_step_size<sub>t</sub> = armijo_step_size<sub>t</sub> / ( sum<sub>k <= t</sub> ||g<sub>k</sub>||<sup>2</sup> ).sqrt()
    AdaGrad { gamma: T, beta: T },
}

impl<T> Default for SteepestDescentParameter<T>
where
    T: Number,
{
    fn default() -> Self {
        Self::Armijo {
            gamma: T::exp_base(10, -3),
            beta: T::from_f32(0.5),
        }
    }
}

impl<T> SteepestDescentParameter<T>
where
    T: Number,
{
    /// Constructs an Armijo rule parameter.
    ///
    /// # Panics
    /// When one of these conditions is not satisfied:
    /// - 0. < gamma < 1.
    /// - 0. < beta < 1.
    /// ```
    /// use tuutal::SteepestDescentParameter;
    /// let param = SteepestDescentParameter::new_armijo(0.1f32, 0.016f32);
    /// assert_eq!(param.gamma(), &0.1);
    /// assert_eq!(param.beta(), &0.016);
    /// ```
    pub fn new_armijo(gamma: T, beta: T) -> Self {
        assert!(
            (T::zero() < gamma) && (gamma < T::one()),
            "gamma should satisfy: 0. < gamma < 1."
        );
        assert!(
            (T::zero() < beta) && (beta < T::one()),
            "beta should satisfy: 0. < beta < 1."
        );
        Self::Armijo { gamma, beta }
    }
    /// Constructs a Powell-Wolfe rule parameter.
    ///
    /// # Panics
    /// When one of these conditions is not satisfied:
    /// - 0. < gamma < 1/2
    /// - gamma < beta < 1.
    /// ```
    /// use tuutal::SteepestDescentParameter;
    /// let param = SteepestDescentParameter::new_powell_wolfe(0.01f32, 0.75f32);
    /// assert_eq!(param.gamma(), &0.01);
    /// assert_eq!(param.beta(), &0.75);
    /// ```
    pub fn new_powell_wolfe(gamma: T, beta: T) -> Self {
        assert!(
            (T::zero() < gamma) && (gamma < T::from_f32(0.5)),
            "gamma should satisfy: 0 < gamma < 1/2"
        );
        assert!(
            (gamma < beta) && (beta < T::one()),
            "beta should satisfy: gamma < beta < 1."
        );
        Self::PowellWolfe { gamma, beta }
    }
    /// Constructs an Armijo rule parameter.
    ///
    /// # Panics
    /// When one of these conditions is not satisfied:
    /// - 0. < gamma < 1.
    /// - 0. < beta < 1.
    /// ```
    /// use tuutal::SteepestDescentParameter;
    /// let param = SteepestDescentParameter::new_adagrad(0.2f32, 0.8f32);
    /// assert_eq!(param.gamma(), &0.2);
    /// assert_eq!(param.beta(), &0.8);
    /// ```
    pub fn new_adagrad(gamma: T, beta: T) -> Self {
        assert!(
            (T::zero() < gamma) && (gamma < T::one()),
            "gamma should satisfy: 0. < gamma < 1."
        );
        assert!(
            (T::zero() < beta) && (beta < T::one()),
            "beta should satisfy: 0. < beta < 1."
        );
        Self::AdaGrad { gamma, beta }
    }
    /// Gets the gamma parameter.
    /// ```
    /// use tuutal::SteepestDescentParameter;
    /// let param = SteepestDescentParameter::new_powell_wolfe(0.49f32, 0.65f32);
    /// assert_eq!(param.gamma(), &0.49);
    /// ```
    pub fn gamma(&self) -> &T {
        match self {
            Self::Armijo { gamma: g, beta: _ } => g,
            Self::PowellWolfe { gamma: g, beta: _ } => g,
            Self::AdaGrad { gamma: g, beta: _ } => g,
        }
    }
    /// Gets the beta parameter.
    /// ```
    /// use tuutal::SteepestDescentParameter;
    /// let param = SteepestDescentParameter::new_powell_wolfe(0.23f32, 0.34f32);
    /// assert_eq!(param.beta(), &0.34);
    /// ```
    pub fn beta(&self) -> &T {
        match self {
            Self::Armijo { gamma: _, beta: b } => b,
            Self::PowellWolfe { gamma: _, beta: b } => b,
            Self::AdaGrad { gamma: _, beta: b } => b,
        }
    }
}

/// Computes a step size using the Armijo method.
fn armijo<F, A, X>(
    f: &F,
    x: &X,
    neg_gradfx: &X,
    squared_norm_2_gradfx: A,
    params: &SteepestDescentParameter<A>,
) -> A
where
    A: Scalar<X>,
    F: Fn(&X) -> A,
    X: Dot<X, Output = A> + Add<X, Output = X>,
    for<'a> &'a X: Add<X, Output = X>,
{
    let mut sigma = A::one();
    let mut x_next = x + sigma * neg_gradfx;
    let fx = f(x);
    while f(&x_next) - fx > -sigma * *params.gamma() * squared_norm_2_gradfx {
        sigma = *params.beta() * sigma;
        x_next = x + sigma * neg_gradfx;
    }
    sigma
}

/// Computes a step size using the Powell Wolfe method.
fn powell_wolfe<F, G, A, X>(
    f: &F,
    gradf: &G,
    x: &X,
    neg_gradfx: &X,
    squared_norm_2_gradfx: A,
    params: &SteepestDescentParameter<A>,
) -> A
where
    A: Scalar<X>,
    F: Fn(&X) -> A,
    G: Fn(&X) -> X,
    X: Dot<X, Output = A> + Add<X, Output = X>,
    for<'a> &'a X: Add<X, Output = X>,
{
    let mut sigma_minus = A::one();
    let mut x_next = x + sigma_minus * neg_gradfx;
    let one_half = A::from_f32(0.5);
    let fx = f(x);
    // The first if and else conditions guarantee having a segment [sigma_minus, sigma_plus]
    // such that sigma_minus satisfies the armijo condition and sigma_plus does not
    let mut sigma_plus =
        if f(&x_next) - fx <= -sigma_minus * *params.gamma() * squared_norm_2_gradfx {
            if gradf(&x_next).dot(neg_gradfx) >= -*params.beta() * squared_norm_2_gradfx {
                return sigma_minus;
            }
            // Computation of sigma_plus
            let two = A::from_f32(2.);
            let mut sigma_plus = two;
            x_next = x + sigma_plus * neg_gradfx;
            while f(&x_next) - fx <= -sigma_plus * *params.gamma() * squared_norm_2_gradfx {
                sigma_plus = two * sigma_plus;
                x_next = x + sigma_plus * neg_gradfx;
            }
            // At this stage sigma_plus is the smallest 2^k that does not satisfy the Armijo rule
            sigma_minus = sigma_plus * one_half; // it satisfies the Armijo rule
            sigma_plus
        } else {
            sigma_minus = one_half;
            x_next = x + sigma_minus * neg_gradfx;
            while f(&x_next) - fx > -sigma_minus * *params.gamma() * squared_norm_2_gradfx {
                sigma_minus = one_half * sigma_minus;
                x_next = x + sigma_minus * neg_gradfx;
            }
            sigma_minus * (A::from_f32(2.)) // does not satisfy the Armijo rule
        };
    x_next = x + sigma_minus * neg_gradfx;
    while gradf(&x_next).dot(neg_gradfx) < -*params.beta() * squared_norm_2_gradfx {
        let sigma = (sigma_minus + sigma_plus) * one_half;
        x_next = x + sigma * neg_gradfx;
        if f(&x_next) - fx <= -sigma * *params.gamma() * squared_norm_2_gradfx {
            sigma_minus = sigma;
        } else {
            sigma_plus = sigma;
        }
    }
    sigma_minus
}

/// Computes a step size using the Adaptive Gradient method.
fn adagrad<F, A, X>(
    f: &F,
    x: &X,
    neg_gradfx: &X,
    mut sum_squared_prev_grad: A,
    squared_norm_2_gradfx: A,
    params: &SteepestDescentParameter<A>,
) -> A
where
    A: Scalar<X> + std::fmt::Display,
    F: Fn(&X) -> A,
    X: Dot<X, Output = A> + Add<X, Output = X>,
    for<'a> &'a X: Add<X, Output = X>,
{
    sum_squared_prev_grad = sum_squared_prev_grad + squared_norm_2_gradfx;
    armijo(f, x, neg_gradfx, squared_norm_2_gradfx, params) / sum_squared_prev_grad.sqrt()
}

/// The steepest descent algorithm using Armijo or Powell Wolfe step size methods.
/// It requires an initial guess x<sub>0</sub>.
/// ```
/// use tuutal::{array, steepest_descent, SteepestDescentParameter, VecType};
/// // Example from python scipy.optimize.minimize_scalar
/// let f = |x: &VecType<f32>| (x[0] - 2.) * x[0] * (x[0] + 2.).powi(2);
/// let gradf = |x: &VecType<f32>| array![2. * (x[0] + 2.) * (2. * x[0].powi(2) - x[0] - 1.)];
/// let x0 = &array![-1.];
///
/// let x_star = steepest_descent(
///     f,
///     gradf,
///     &x0,
///     &SteepestDescentParameter::new_armijo(1e-2, 0.25),
///     1e-3,
///     10,
/// ).unwrap();
/// assert!((-2. - x_star[0]).abs() < 1e-10);
///
/// let x_star = steepest_descent(
///     &f,
///     &gradf,
///     &x0,
///     &SteepestDescentParameter::new_powell_wolfe(1e-2, 0.9),
///     1e-3,
///     10,
/// ).unwrap();
/// assert!((-2. - x_star[0]).abs() < 1e-10);
///
/// let x0 = &array![-0.5];
/// let x_star = steepest_descent(f, gradf, &x0, &Default::default(), 1e-3, 10).unwrap();
/// assert!((-0.5 - x_star[0]).abs() < 1e-10);
///
/// let x0 = &array![0.];
/// let x_star = steepest_descent(f, gradf, &x0, &Default::default(), 1e-3, 10).unwrap();
/// assert!((1. - x_star[0]).abs() < 1e-10);
///
/// // It also takes multivariate objective functions
/// let f =
///     |arr: &VecType<f32>| 100. * (arr[1] - arr[0].powi(2)).powi(2) + (1. - arr[0]).powi(2);
/// let gradf = |arr: &VecType<f32>| {
///     array![
///         -400. * arr[0] * (arr[1] - arr[0].powi(2)) - 2. * (1. - arr[0]),
///         200. * (arr[1] - arr[0].powi(2))
///     ]
/// };
/// let x = array![1f32, -0.5f32];
/// let opt = steepest_descent(f, gradf, &x, &Default::default(), 1e-3, 10000).unwrap();
/// assert!((opt[0] - 1.).abs() <= 1e-2);
/// assert!((opt[1] - 1.).abs() <= 1e-2);
/// ```
pub fn steepest_descent<X, F, G, A>(
    f: F,
    gradf: G,
    x0: &X,
    params: &SteepestDescentParameter<A>,
    eps: A,
    maxiter: usize,
) -> Result<X, TuutalError<X>>
where
    A: Scalar<X> + std::fmt::Display,
    X: Dot<X, Output = A> + Neg<Output = X> + Add<X, Output = X> + Clone,
    for<'a> &'a X: Add<X, Output = X>,
    F: Fn(&X) -> A,
    G: Fn(&X) -> X,
{
    let iterates = SteepestDescentIterates::new(f, gradf, x0.clone(), *params, eps);
    optimize(iterates, maxiter)
}

/// Represents the sequence of iterates computed by a steepest descent algorithm.
pub struct SteepestDescentIterates<X, F, G, A>
where
    F: Fn(&X) -> A,
    G: Fn(&X) -> X,
{
    f: F,
    gradf: G,
    params: SteepestDescentParameter<A>,
    x: X,
    eps: A,
    iter: usize,
    sigma: A,
    sum_squared_grad: A,
}

impl<X, F, G, A> SteepestDescentIterates<X, F, G, A>
where
    F: Fn(&X) -> A,
    G: Fn(&X) -> X,
{
    pub fn new(f: F, gradf: G, x: X, params: SteepestDescentParameter<A>, eps: A) -> Self
    where
        A: Scalar<X>,
        X: Dot<X, Output = A> + Add<X, Output = X>,
        for<'a> &'a X: Add<X, Output = X>,
    {
        Self {
            f,
            gradf,
            params,
            x,
            iter: 0,
            eps,
            sigma: A::zero(),
            sum_squared_grad: A::epsilon(), // to avoid division by zero for initial guess near stationnary points.
        }
    }
    /// Reference to the objective function
    pub(crate) fn obj(&self) -> &F {
        &self.f
    }
    /// Reference to the gradient of the objective function
    pub(crate) fn grad_obj(&self) -> &G {
        &self.gradf
    }
    /// Number of iterations done so far.
    pub fn nb_iter(&self) -> usize {
        self.iter
    }
    /// Current iterate.
    pub fn x(&self) -> &X {
        &self.x
    }
    /// Current step size.
    pub fn sigma(&self) -> &A {
        &self.sigma
    }
}

impl<X, F, G, A> std::iter::Iterator for SteepestDescentIterates<X, F, G, A>
where
    A: Scalar<X> + std::fmt::Display,
    X: Dot<X, Output = A> + Neg<Output = X> + Add<X, Output = X> + Clone,
    for<'a> &'a X: Add<X, Output = X>,
    F: Fn(&X) -> A,
    G: Fn(&X) -> X,
{
    type Item = X;
    fn next(&mut self) -> Option<Self::Item> {
        let neg_gradfx = -self.grad_obj()(&self.x);
        let squared_norm_2_gradfx = neg_gradfx.dot(&neg_gradfx);
        if squared_norm_2_gradfx <= (self.eps * self.eps) {
            self.iter += 1;
            None
        } else {
            self.sigma = match self.params {
                SteepestDescentParameter::Armijo { gamma: _, beta: _ } => armijo(
                    self.obj(),
                    &self.x,
                    &neg_gradfx,
                    squared_norm_2_gradfx,
                    &self.params,
                ),
                SteepestDescentParameter::PowellWolfe { gamma: _, beta: _ } => powell_wolfe(
                    self.obj(),
                    self.grad_obj(),
                    &self.x,
                    &neg_gradfx,
                    squared_norm_2_gradfx,
                    &self.params,
                ),
                SteepestDescentParameter::AdaGrad { gamma: _, beta: _ } => adagrad(
                    self.obj(),
                    &self.x,
                    &neg_gradfx,
                    self.sum_squared_grad,
                    squared_norm_2_gradfx,
                    &self.params,
                ),
            };
            self.x = &self.x + self.sigma * neg_gradfx;
            self.iter += 1;
            Some(self.x.clone())
        }
    }
}

impl<X, F, G, A> Iterable<X> for SteepestDescentIterates<X, F, G, A>
where
    A: Scalar<X> + std::fmt::Display,
    X: Dot<X, Output = A> + Neg<Output = X> + Add<X, Output = X> + Clone,
    for<'a> &'a X: Add<X, Output = X>,
    F: Fn(&X) -> A,
    G: Fn(&X) -> X,
{
    fn nb_iter(&self) -> usize {
        self.nb_iter()
    }
    fn iterate(&self) -> &X {
        self.x()
    }
}
