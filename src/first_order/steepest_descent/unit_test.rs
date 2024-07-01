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

    #[test]
    fn test_armijo() {
        let armijo = SteepestDescentParameter::new_armijo(0.01, 0.5);
        let (f, gradf) = rosenbrock_2d();
        let x = array![1f32, -0.5f32];
        let opt = steepest_descent(f, gradf, &x, &armijo, 1e-5, 10000);
        let expected = array![1., 1.];
        assert!(
            opt.iter()
                .zip(&expected)
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
                < 1e-4
        );
    }

    #[test]
    fn test_powell_wolfe() {
        let powolf = SteepestDescentParameter::new_powell_wolfe(0.0001, 0.9);
        let (f, gradf) = rosenbrock_2d();
        let x = array![1f32, -0.5f32];
        let opt = steepest_descent(f, gradf, &x, &powolf, 1e-4, 10000);
        let expected = array![1., 1.];
        assert!(
            opt.iter()
                .zip(&expected)
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
                < 1e-3
        );
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
        let opt = steepest_descent(f, gradf, &x, &powolf, 1e-5, 10000);
        let expected = array![1., 1., 1.];
        assert!(
            opt.iter()
                .zip(&expected)
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
                < 1e-4
        );
    }
}
