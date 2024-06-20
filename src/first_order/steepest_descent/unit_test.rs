#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::VecType;
    use ndarray::array;

    fn example() -> (fn(&VecType<f32>) -> f32, fn(&VecType<f32>) -> VecType<f32>) {
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
        let armijo = SteepestDescentParameter::Armijo(0.01, 0.5);
        let (f, gradf) = example();
        let x = array![1f32, -0.5f32];
        let opt = steepest_descent(f, gradf, &x, &armijo, 1e-3, 10000).unwrap();
        assert!((opt[0] - 1.).abs() <= 1e-2);
        assert!((opt[1] - 1.).abs() <= 1e-2);
    }

    #[test]
    fn test_powell_wolfe() {
        let powolf = SteepestDescentParameter::PowellWolfe(0.0001, 0.9);
        let (f, gradf) = example();
        let x = array![1f32, -0.5f32];
        let opt = steepest_descent(f, gradf, &x, &powolf, 1e-3, 10000).unwrap();
        assert!((opt[0] - 1.).abs() <= 1e-2);
        assert!((opt[1] - 1.).abs() <= 1e-2);
    }
}
