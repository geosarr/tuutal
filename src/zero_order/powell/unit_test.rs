#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::{is_between, Array};

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
    fn test_code() {
        use crate::bounded;
        let f = |x: f32| (x - 2.) * x * (x + 2.).powi(2);
        let bounds = (0., 2.);
        let (x, fx, fcalls) = bounded(f, bounds, 1.48e-8, 1000).unwrap_or((0.0, 0.0, 0));
        assert!((bounds.0 <= x) && (x <= bounds.1));
        println!("{:?}", (x, fx, fcalls));
        println!("{} {}", 1.2808123 - 1.280776, 9.914950 - 9.914949);
    }
}