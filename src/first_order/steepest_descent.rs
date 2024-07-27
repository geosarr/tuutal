mod adadelta;
mod adagrad;
mod armijo;
mod powell_wolfe;
mod unit_test;
use crate::{
    optimize,
    traits::{VecDot, VecInfo, VecZero},
    Iterable, Number, Scalar, TuutalError,
};
use adadelta::adadelta;
use adagrad::adagrad;
use armijo::armijo;
use powell_wolfe::powell_wolfe;
use core::{
    fmt::Debug,
    ops::{Add, Div, Mul, Neg},
};

/// Parameters used in the steepest descent method.
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
/// [`new_armijo`]: SteepestDescentParameter::new_armijo
///
/// [`new_powell_wolfe`]: SteepestDescentParameter::new_powell_wolfe
///
/// [`new_adagrad`]: SteepestDescentParameter::new_adagrad
///
/// [`new_adadelta`]: SteepestDescentParameter::new_adadelta
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SteepestDescentParameter<T> {
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

impl<T> Default for SteepestDescentParameter<T>
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
    ///
    /// ```
    /// use tuutal::SteepestDescentParameter;
    /// let param = SteepestDescentParameter::new_armijo(0.1f32, 0.016f32);
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
    /// use tuutal::SteepestDescentParameter;
    /// let param = SteepestDescentParameter::new_powell_wolfe(0.01f32, 0.75f32);
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
    /// use tuutal::SteepestDescentParameter;
    /// let param = SteepestDescentParameter::new_adagrad(0.01f32, 0.0001);
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
    /// use tuutal::SteepestDescentParameter;
    /// let param = SteepestDescentParameter::new_adadelta(0.2f32, 0.04);
    /// ```
    pub fn new_adadelta(gamma: T, beta: T) -> Self {
        assert!(gamma > T::zero() && gamma < T::one());
        assert!(beta > T::zero());
        Self::AdaDelta { gamma, beta }
    }
    fn step_size_is_scalar(&self) -> bool {
        match self {
            Self::Armijo { gamma: _, beta: _ } => true,
            Self::PowellWolfe { gamma: _, beta: _ } => true,
            Self::AdaGrad { gamma: _, beta: _ } => false,
            Self::AdaDelta { gamma: _, beta: _ } => false,
        }
    }
}

/// The steepest descent algorithm using some step size method.
/// It requires an initial guess x<sub>0</sub>.
/// ```
/// use tuutal::{array, steepest_descent, SteepestDescentParameter, Array1};
/// // Example from python scipy.optimize.minimize_scalar
/// let f = |x: &Array1<f32>| (x[0] - 2.) * x[0] * (x[0] + 2.).powi(2);
/// let gradf = |x: &Array1<f32>| array![2. * (x[0] + 2.) * (2. * x[0].powi(2) - x[0] - 1.)];
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
///     |arr: &Array1<f32>| 100. * (arr[1] - arr[0].powi(2)).powi(2) + (1. - arr[0]).powi(2);
/// let gradf = |arr: &Array1<f32>| {
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
    A: Scalar<X> + core::fmt::Display,
    X: VecDot<X, Output = A> + Neg<Output = X> + Add<X, Output = X> + Clone + VecInfo + VecZero,
    for<'a> &'a X: Add<X, Output = X> + Mul<&'a X, Output = X> + Mul<X, Output = X>,
    F: Fn(&X) -> A,
    G: Fn(&X) -> X,
    X: FromIterator<A> + IntoIterator<Item = A> + Clone + Div<X, Output = X> + Mul<X, Output = X>,
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
    sigma: X,
    accum_grad: X,
    accum_update: X,
}

impl<X, F, G, A> SteepestDescentIterates<X, F, G, A>
where
    F: Fn(&X) -> A,
    G: Fn(&X) -> X,
{
    pub fn new(f: F, gradf: G, x: X, params: SteepestDescentParameter<A>, eps: A) -> Self
    where
        A: Scalar<X>,
        X: VecDot<X, Output = A> + Add<X, Output = X> + VecZero + VecInfo,
        for<'a> &'a X: Add<X, Output = X>,
    {
        let dim = x.len();
        if params.step_size_is_scalar() {
            Self {
                f,
                gradf,
                params,
                x,
                iter: 0,
                eps,
                sigma: X::zero(1),
                accum_grad: X::zero(1),
                accum_update: X::zero(1),
            }
        } else {
            Self {
                f,
                gradf,
                params,
                x,
                iter: 0,
                eps,
                sigma: X::zero(dim),
                accum_grad: X::zero(dim),
                accum_update: X::zero(dim),
            }
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
    pub fn sigma(&self) -> &X {
        &self.sigma
    }
}

impl<X, F, G, A> core::iter::Iterator for SteepestDescentIterates<X, F, G, A>
where
    A: Scalar<X> + core::fmt::Display,
    X: VecDot<X, Output = A> + Neg<Output = X> + Add<X, Output = X> + Clone,
    for<'a> &'a X: Add<X, Output = X> + Mul<&'a X, Output = X> + Mul<X, Output = X>,
    F: Fn(&X) -> A,
    G: Fn(&X) -> X,
    X: FromIterator<A> + IntoIterator<Item = A> + Clone + Div<X, Output = X> + Mul<X, Output = X>,
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
                SteepestDescentParameter::Armijo { gamma, beta } => [armijo(
                    self.obj(),
                    &self.x,
                    &neg_gradfx,
                    squared_norm_2_gradfx,
                    gamma,
                    beta,
                )]
                .into_iter()
                .collect::<X>(),
                SteepestDescentParameter::PowellWolfe { gamma, beta } => [powell_wolfe(
                    self.obj(),
                    self.grad_obj(),
                    &self.x,
                    &neg_gradfx,
                    squared_norm_2_gradfx,
                    gamma,
                    beta,
                )]
                .into_iter()
                .collect::<X>(),
                SteepestDescentParameter::AdaGrad { gamma, beta } => {
                    let squared_grad = &neg_gradfx * &neg_gradfx;
                    adagrad(&mut self.accum_grad, squared_grad, gamma, beta)
                }
                SteepestDescentParameter::AdaDelta { gamma, beta } => {
                    let squared_grad = &neg_gradfx * &neg_gradfx;
                    let step_size = adadelta(
                        &mut self.accum_grad,
                        &self.accum_update,
                        &squared_grad,
                        gamma,
                        beta,
                    );
                    self.accum_update = gamma * &self.accum_update
                        + (A::one() - gamma) * (&step_size * &step_size) * squared_grad;
                    step_size
                }
            };
            self.x = &self.x + &self.sigma * neg_gradfx;
            self.iter += 1;
            Some(self.x.clone())
        }
    }
}

impl<X, F, G, A> Iterable<X> for SteepestDescentIterates<X, F, G, A>
where
    A: Scalar<X> + core::fmt::Display,
    X: VecDot<X, Output = A> + Neg<Output = X> + Add<X, Output = X> + Clone,
    for<'a> &'a X: Add<X, Output = X> + Mul<&'a X, Output = X> + Mul<X, Output = X>,
    F: Fn(&X) -> A,
    G: Fn(&X) -> X,
    X: FromIterator<A> + IntoIterator<Item = A> + Clone + Div<X, Output = X> + Mul<X, Output = X>,
{
    fn nb_iter(&self) -> usize {
        self.nb_iter()
    }
    fn iterate(&self) -> X {
        self.x.clone()
    }
}
