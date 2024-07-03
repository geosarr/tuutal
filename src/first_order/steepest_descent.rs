#[cfg(test)]
mod unit_test;
use crate::{optimize, traits::Scalar, DefaultValue, Iterable, TuutalError};
use ndarray::linalg::Dot;
use std::{
    fmt::Debug,
    ops::{Add, Neg},
};

/// Alias of the output type of step size functions.
type StepSizeFunction<X, F, G, A> = fn(&F, &G, &X, &X, &SteepestDescentParameter<A>) -> A;

/// Parameters used in the steepest descent method.
///
/// The **gamma** parameter represents a magnitude of decrease in the objective function
/// in the negative gradient direction. The **beta** parameter controls:
/// - step size magnitude of decrease in the Armijo rule.
/// - descent steepness for the Powell Wolfe strategy.
///
/// Use methods [`new_armijo`] and [`new_powell_wolfe`] to construct these parameters.
///
/// [`new_armijo`]: SteepestDescentParameter::new_armijo
///
/// [`new_powell_wolfe`]: SteepestDescentParameter::new_powell_wolfe
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SteepestDescentParameter<T> {
    /// Armijo rule step size rule
    Armijo { gamma: T, beta: T },
    /// Powell Wolfe step size rule
    PowellWolfe { gamma: T, beta: T },
}

impl<T> Default for SteepestDescentParameter<T>
where
    T: DefaultValue,
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
    T: DefaultValue,
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
        }
    }
}

/// Computes a step size using the Armijo method.
fn armijo_rule<F, G, A, X>(
    f: &F,
    gradf: &G,
    x: &X,
    d: &X,
    params: &SteepestDescentParameter<A>,
) -> A
where
    A: Scalar<X>,
    F: Fn(&X) -> A,
    G: Fn(&X) -> X,
    X: Dot<X, Output = A> + Add<X, Output = X> + Clone,
{
    let mut sigma = A::one();
    let mut x_next = x.clone() + sigma * d;
    let gradx_d = gradf(x).dot(d);
    let fx = f(x);
    while f(&x_next) - fx > sigma * *params.gamma() * gradx_d {
        sigma = *params.beta() * sigma;
        x_next = x.clone() + sigma * d;
    }
    sigma
}

/// Computes a step size using the Powell Wolfe method.
fn powell_wolfe_rule<F, G, A, X>(
    f: &F,
    gradf: &G,
    x: &X,
    d: &X,
    params: &SteepestDescentParameter<A>,
) -> A
where
    A: Scalar<X>,
    F: Fn(&X) -> A,
    G: Fn(&X) -> X,
    X: Dot<X, Output = A> + Add<X, Output = X> + Clone,
{
    let mut sigma_minus = A::one();
    let mut x_next = x.clone() + sigma_minus * d;
    let gradx_d = gradf(x).dot(d);
    let one_half = A::from_f32(0.5);
    let fx = f(x);
    // The first if and else conditions guarantee having a segment [sigma_minus, sigma_plus]
    // such that sigma_minus satisfies the armijo condition and sigma_plus does not
    let mut sigma_plus = if f(&x_next) - fx <= sigma_minus * *params.gamma() * gradx_d {
        if gradf(&x_next).dot(d) >= *params.beta() * gradx_d {
            return sigma_minus;
        }
        // Computation of sigma_plus
        let two = A::from_f32(2.);
        let mut sigma_plus = two;
        x_next = x.clone() + sigma_plus * d;
        while f(&x_next) - fx <= sigma_plus * *params.gamma() * gradx_d {
            sigma_plus = two * sigma_plus;
            x_next = x.clone() + sigma_plus * d;
        }
        // At this stage sigma_plus is the smallest 2^k that does not satisfy the Armijo rule
        sigma_minus = sigma_plus * one_half; // it satisfies the Armijo rule
        sigma_plus
    } else {
        sigma_minus = one_half;
        x_next = x.clone() + sigma_minus * d;
        while f(&x_next) - fx > sigma_minus * *params.gamma() * gradx_d {
            sigma_minus = one_half * sigma_minus;
            x_next = x.clone() + sigma_minus * d;
        }
        sigma_minus * (A::from_f32(2.)) // does not satisfy the Armijo rule
    };
    x_next = x.clone() + sigma_minus * d;
    while gradf(&x_next).dot(d) < *params.beta() * gradx_d {
        let sigma = (sigma_minus + sigma_plus) * one_half;
        x_next = x.clone() + sigma * d;
        if f(&x_next) - fx <= sigma * *params.gamma() * gradx_d {
            sigma_minus = sigma;
        } else {
            sigma_plus = sigma;
        }
    }
    sigma_minus
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
    A: Scalar<X>,
    X: Dot<X, Output = A> + Neg<Output = X> + Add<X, Output = X> + Clone,
    F: Fn(&X) -> A,
    G: Fn(&X) -> X,
{
    let iterates = SteepestDescentIterates::new(f, gradf, x0.clone(), *params, eps);
    return optimize(iterates, x0.clone(), maxiter);
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
    rule: StepSizeFunction<X, F, G, A>,
    sigma: A,
}

impl<X, F, G, A> SteepestDescentIterates<X, F, G, A>
where
    F: Fn(&X) -> A,
    G: Fn(&X) -> X,
{
    pub fn new(f: F, gradf: G, x: X, params: SteepestDescentParameter<A>, eps: A) -> Self
    where
        A: Scalar<X>,
        X: Dot<X, Output = A> + Add<X, Output = X> + Clone,
    {
        let rule = match params {
            SteepestDescentParameter::Armijo { gamma: _, beta: _ } => armijo_rule,
            SteepestDescentParameter::PowellWolfe { gamma: _, beta: _ } => powell_wolfe_rule,
        };
        Self {
            f,
            gradf,
            params,
            x,
            iter: 0,
            eps,
            rule,
            sigma: A::zero(),
        }
    }
    /// Reference to the objective function
    pub fn obj(&self) -> &F {
        &self.f
    }
    /// Reference to the gradient of the objective function
    pub fn grad_obj(&self) -> &G {
        &self.gradf
    }
    /// Step size finding algorithm
    pub fn step_size_rule(&self) -> StepSizeFunction<X, F, G, A> {
        self.rule
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
    A: Scalar<X>,
    X: Dot<X, Output = A> + Neg<Output = X> + Add<X, Output = X> + Clone,
    F: Fn(&X) -> A,
    G: Fn(&X) -> X,
{
    type Item = X;
    fn next(&mut self) -> Option<Self::Item> {
        let neg_grad = -self.grad_obj()(&self.x);
        if neg_grad.dot(&neg_grad) <= (self.eps * self.eps) {
            self.iter += 1;
            None
        } else {
            self.sigma = self.step_size_rule()(
                self.obj(),
                self.grad_obj(),
                &self.x,
                &neg_grad,
                &self.params,
            );
            self.x = self.x.clone() + self.sigma * neg_grad;
            self.iter += 1;
            Some(self.x.clone())
        }
    }
}

impl<X, F, G, A> Iterable<X> for SteepestDescentIterates<X, F, G, A>
where
    A: Scalar<X>,
    X: Dot<X, Output = A> + Neg<Output = X> + Add<X, Output = X> + Clone,
    F: Fn(&X) -> A,
    G: Fn(&X) -> X,
{
    fn nb_iter(&self) -> usize {
        self.nb_iter()
    }
}
