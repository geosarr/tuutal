#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::{array, is_between, l2_diff, Array};

    #[test]
    fn test_split_in_two() {
        let vec: Vec<f64> = vec![4., 2., 5., 8., 3., 1., -10.];
        let func = |i: &usize| (&vec[*i] % 2.) == 0.;
        let (res, no_res) = split_in_two(func, vec.len());
        assert_eq!(vec![0, 1, 3, 6], res);
        assert_eq!(vec![2, 4, 5], no_res);
    }

    #[test]
    fn test_min_max() {
        let d = Array::from(vec![1., -1., 2., 3., -2.]);
        let x = Array::from(vec![1., -1., 0., 0., -1.]);
        let lb = Array::from(vec![-2.; 5]);
        let ub = Array::from(vec![2.; 5]);
        let (lmin, lmax) = line_for_search(&x, &d, &lb, &ub).unwrap();
        let xmin: VecType<f64> = &x + lmin * &d;
        let xhalf: VecType<f64> = &x + ((lmin + lmax) / 2.) * &d;
        let xmax: VecType<f64> = &x + lmax * &d;

        // Testing whether or not the vectors x + l * d are between lb and ub component-wise.
        assert!(is_between(&xmin, &lb, &ub));
        assert!(is_between(&xhalf, &lb, &ub));
        assert!(is_between(&xmax, &lb, &ub));
    }

    #[test]
    fn test_powell() {
        let f = |arr: &VecType<f32>| arr.dot(arr);
        let x0 = array![-5., -5.];
        let x_star = powell::<_, (f32, f32), _>(f, &x0, None, 100, None, 1e-5, 1e-5, None).unwrap();
        // assert!(l2_diff(&x_star, &array![0., 0.]) < 2e-3);
        // assert!(f(&x_star) < 1e-5);
    }
}
