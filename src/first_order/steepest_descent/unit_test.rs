#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::VecType;
    use ndarray::array;

    fn rosenbrock_2d() -> (fn(&VecType<f32>) -> f32, fn(&VecType<f32>) -> VecType<f32>) {
        let f =
            |arr: &VecType<f32>| 100. * (arr[1] - arr[0].powi(2)).powi(2) + (1. - arr[0]).powi(2);
        let gradf = |arr: &VecType<f32>| {
            array![
                -400. * arr[0] * (arr[1] - arr[0].powi(2)) - 2. * (1. - arr[0]),
                200. * (arr[1] - arr[0].powi(2))
            ]
        };
        return (f, gradf);
    }

    fn l2_diff<T>(a: &VecType<T>, b: &VecType<T>) -> T
    where
        for<'a> &'a T: std::ops::Sub<Output = T>,
        T: num_traits::Float + std::ops::Mul<Output = T> + std::iter::Sum,
    {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<T>()
            .sqrt()
    }

    #[test]
    fn test_armijo() {
        let armijo = SteepestDescentParameter::new_armijo(0.01, 0.5);
        let (f, gradf) = rosenbrock_2d();
        let x = array![1f32, -0.5f32];
        let opt = steepest_descent(f, gradf, &x, &armijo, 1e-4, 10000).unwrap();
        let expected = array![1., 1.];
        assert!(l2_diff(&opt, &expected) < 1e-3);
    }

    #[test]
    fn test_powell_wolfe() {
        let powolf = SteepestDescentParameter::new_powell_wolfe(0.0001, 0.9);
        let (f, gradf) = rosenbrock_2d();
        let x = array![1f32, -0.5f32];
        let opt = steepest_descent(f, gradf, &x, &powolf, 1e-4, 10000).unwrap();
        let expected = array![1., 1.];
        assert!(l2_diff(&opt, &expected) < 1e-3);
    }

    #[test]
    fn test_adagrad() {
        let adagrad = SteepestDescentParameter::AdaGrad {
            gamma: 0.01,
            beta: 0.5,
        };
        let (f, gradf) = rosenbrock_2d();
        let x = array![1f32, -0.5f32];
        let opt = steepest_descent(f, gradf, &x, &adagrad, 1e-3, 10000).unwrap_err();
        let expected = array![1., 1.];
        // Slow convergence due to gradient normalization.
        match opt {
            TuutalError::Convergence {
                iterate,
                maxiter: _,
            } => assert!(l2_diff(&iterate, &expected) < 2e-3),
            _ => panic!("Wrong error variant."),
        }
    }

    #[test]
    fn test_rosenbrock_3d() {
        let powolf = SteepestDescentParameter::new_powell_wolfe(0.0001, 0.9);
        let f = |x: &VecType<f32>| {
            100. * ((x[1] - x[0].powi(2)).powi(2) + (x[2] - x[1].powi(2)).powi(2))
                + (1. - x[0]).powi(2)
                + (1. - x[1]).powi(2)
        };
        let gradf = |x: &VecType<f32>| {
            2. * array![
                200. * x[0] * (x[0].powi(2) - x[1]) + (x[0] - 1.),
                100. * (x[1] - x[0].powi(2) + 2. * x[1] * (x[1].powi(2) - x[2])) + (x[1] - 1.),
                100. * (x[2] - x[1].powi(2))
            ]
        };
        let x = array![10f32, -15., -100.];
        let opt = steepest_descent(f, gradf, &x, &powolf, 1e-4, 10000).unwrap();
        let expected = array![1., 1., 1.];
        assert!(l2_diff(&opt, &expected) < 1e-3);
    }
}
