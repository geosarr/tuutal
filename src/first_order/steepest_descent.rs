mod armijo;
mod powell_wolfe;
mod unit_test;
use crate::first_order::adaptive_descent::{AdaDelta, AdaGrad};
use crate::{
    traits::{VecDot, Vector},
    Number, Optimizer, TuutalError,
};
pub use armijo::Armijo;
use core::{
    fmt::Debug,
    ops::{Add, Mul},
};
pub use powell_wolfe::PowellWolfe;

/// Parameters used in the a descent method.
///
/// The **gamma** parameter represents the:
/// - magnitude of decrease in the objective function in the negative gradient direction for Armijo and Powell rules.
/// - general step size for AdaGrad rule.
/// - magnitude of decay for previous gradients in the AdaDelta algorithms.
///
/// The **beta** parameter controls the:
/// - step size magnitude of decrease in the Armijo rule algorithm.
/// - descent steepness for the Powell Wolfe strategy.
/// - tolerance factor of the update denominator in the AdaGrad and AdaDelta algorithms.
///
/// Use methods [`new_armijo`], [`new_powell_wolfe`], [`new_adagrad`] and [`new_adadelta`], to construct these parameters.
///
/// [`new_armijo`]: DescentParameter::new_armijo
///
/// [`new_powell_wolfe`]: DescentParameter::new_powell_wolfe
///
/// [`new_adagrad`]: DescentParameter::new_adagrad
///
/// [`new_adadelta`]: DescentParameter::new_adadelta
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DescentParameter<T> {
    /// Armijo rule step size rule
    ///
    /// At each step t, the step size is a scalar.
    Armijo { gamma: T, beta: T },
    /// Powell Wolfe step size rule
    ///
    /// At each step t, the step size is a scalar.
    PowellWolfe { gamma: T, beta: T },
    /// Adaptive Gradient step size rule
    ///
    /// At each step t, the vector step size is given by:
    /// - adagrad_step_size<sub>t</sub> = gamma / ( sum<sub>k <= t</sub> g<sub>k</sub><sup>2</sup> + beta ).sqrt()
    ///   where g<sub>k</sub> is the gradient at step k.
    AdaGrad { gamma: T, beta: T },
    /// Adaptive Learning Rate DELTA step size rule
    ///
    /// At each step t, the vector step size is given by:
    /// - adadelta_step_size<sub>t</sub> = RMS(x<sub>t-1</sub>) / RMS(g<sub>t</sub>)
    ///   where :
    ///     - x<sub>k</sub> is the update at step k
    ///     - g<sub>k</sub> is the gradient of x<sub>k</sub>
    ///     - RMS(v<sub>t</sub>) = ( sum<sub>k <= t</sub> E[v<sup>2</sup>]<sub>k</sub> + beta ).sqrt()
    ///     - E[v<sup>2</sup>]<sub>k</sub> = gamma * E[v<sup>2</sup>]<sub>k-1</sub> + (1 - gamma) * v<sub>k</sub><sup>2</sup>
    ///       with E[v<sup>2</sup>]<sub>0</sub> = 0
    AdaDelta { gamma: T, beta: T },
}

impl<T> Default for DescentParameter<T>
where
    T: Number,
{
    fn default() -> Self {
        Self::Armijo {
            gamma: T::exp_base(10, -3),
            beta: T::cast_from_f32(0.5),
        }
    }
}

impl<T> DescentParameter<T>
where
    T: Number,
{
    /// Constructs an Armijo rule parameter.
    ///
    /// # Panics
    /// When one of these conditions is not satisfied:
    /// - 0. < gamma < 1.
    /// - 0. < beta < 1.
    ///
    /// ```
    /// use tuutal::DescentParameter;
    /// let param = DescentParameter::new_armijo(0.1f32, 0.016f32);
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
    ///
    /// ```
    /// use tuutal::DescentParameter;
    /// let param = DescentParameter::new_powell_wolfe(0.01f32, 0.75f32);
    /// ```
    pub fn new_powell_wolfe(gamma: T, beta: T) -> Self {
        assert!(
            (T::zero() < gamma) && (gamma < T::cast_from_f32(0.5)),
            "gamma should satisfy: 0 < gamma < 1/2"
        );
        assert!(
            (gamma < beta) && (beta < T::one()),
            "beta should satisfy: gamma < beta < 1."
        );
        Self::PowellWolfe { gamma, beta }
    }
    /// Constructs an AdaGrad rule parameter.
    ///
    /// # Panics
    /// When one of these conditions is not satisfied:
    /// - 0. < gamma.
    /// - beta > 0.
    /// ```
    /// use tuutal::DescentParameter;
    /// let param = DescentParameter::new_adagrad(0.01f32, 0.0001);
    /// ```
    pub fn new_adagrad(gamma: T, beta: T) -> Self {
        assert!(gamma > T::zero());
        assert!(beta > T::zero());
        Self::AdaGrad { gamma, beta }
    }
    /// Constructs an AdaDelta rule parameter.
    ///
    /// # Panics
    /// When one of these conditions is not satisfied:
    /// - 0. < gamma < 1.
    /// - beta > 0.
    /// ```
    /// use tuutal::DescentParameter;
    /// let param = DescentParameter::new_adadelta(0.2f32, 0.04);
    /// ```
    pub fn new_adadelta(gamma: T, beta: T) -> Self {
        assert!(gamma > T::zero() && gamma < T::one());
        assert!(beta > T::zero());
        Self::AdaDelta { gamma, beta }
    }
}

/// A descent algorithm using some step size method.
///
/// It requires an initial guess x<sub>0</sub>.
/// ```
/// use tuutal::{array, descent, DescentParameter, Array1};
/// // Example from python scipy.optimize.minimize_scalar
/// let f = |x: &Array1<f32>| (x[0] - 2.) * x[0] * (x[0] + 2.).powi(2);
/// let gradf = |x: &Array1<f32>| array![2. * (x[0] + 2.) * (2. * x[0].powi(2) - x[0] - 1.)];
/// let x0 = &array![-1.];
///
/// let x_star = descent(
///     f,
///     gradf,
///     &x0,
///     &DescentParameter::new_armijo(1e-2, 0.25),
///     1e-3,
///     10,
/// ).unwrap();
/// assert!((-2. - x_star[0]).abs() < 1e-10);
///
/// let x_star = descent(
///     &f,
///     &gradf,
///     &x0,
///     &DescentParameter::new_powell_wolfe(1e-2, 0.9),
///     1e-3,
///     10,
/// ).unwrap();
/// assert!((-2. - x_star[0]).abs() < 1e-10);
///
/// let x0 = &array![-0.5];
/// let x_star = descent(f, gradf, &x0, &Default::default(), 1e-3, 10).unwrap();
/// assert!((-0.5 - x_star[0]).abs() < 1e-10);
///
/// let x0 = &array![0.];
/// let x_star = descent(f, gradf, &x0, &Default::default(), 1e-3, 10).unwrap();
/// assert!((1. - x_star[0]).abs() < 1e-10);
///
/// // It also takes multivariate objective functions
/// let f =
///     |arr: &Array1<f32>| 100. * (arr[1] - arr[0].powi(2)).powi(2) + (1. - arr[0]).powi(2);
/// let gradf = |arr: &Array1<f32>| {
///     array![
///         -400. * arr[0] * (arr[1] - arr[0].powi(2)) - 2. * (1. - arr[0]),
///         200. * (arr[1] - arr[0].powi(2))
///     ]
/// };
/// let x = array![1f32, -0.5f32];
/// let opt = descent(f, gradf, &x, &Default::default(), 1e-3, 10000).unwrap();
/// assert!((opt[0] - 1.).abs() <= 1e-2);
/// assert!((opt[1] - 1.).abs() <= 1e-2);
/// ```
pub fn descent<X, F, G>(
    f: F,
    gradf: G,
    x0: &X,
    params: &DescentParameter<X::Elem>,
    gtol: X::Elem,
    maxiter: usize,
) -> Result<X, TuutalError<X>>
where
    X: Vector + Clone + VecDot<Output = X::Elem>,
    for<'a> &'a X: Add<X, Output = X> + Mul<&'a X, Output = X> + Mul<X, Output = X>,
    F: Fn(&X) -> X::Elem,
    G: Fn(&X) -> X,
{
    match params {
        DescentParameter::Armijo { gamma, beta } => {
            Armijo::new(f, gradf, x0.clone(), *gamma, *beta, gtol).optimize(Some(maxiter))
        }
        DescentParameter::PowellWolfe { gamma, beta } => {
            PowellWolfe::new(f, gradf, x0.clone(), *gamma, *beta, gtol).optimize(Some(maxiter))
        }
        DescentParameter::AdaDelta { gamma, beta } => {
            AdaDelta::new(f, gradf, x0.clone(), *gamma, *beta, gtol).optimize(Some(maxiter))
        }
        DescentParameter::AdaGrad { gamma, beta } => {
            AdaGrad::new(f, gradf, x0.clone(), *gamma, *beta, gtol).optimize(Some(maxiter))
        }
    }
}
