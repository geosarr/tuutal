#[cfg(test)]
mod test {

    use super::super::*;
    use crate::{l2_diff, Array1};
    #[test]
    fn test_simplex() {
        let x0: Array1<f32> = Array1::from_iter([]);
        assert_eq!(
            TuutalError::EmptyDimension { x: x0.clone() },
            simplex_parameters(&x0, true).unwrap_err()
        );

        let x0: Array1<f32> = Array1::from_iter([-10., 36.]);
        assert_eq!((1., 2., 0.5, 0.5), simplex_parameters(&x0, true).unwrap());

        let x0: Array1<f32> = Array1::from_iter([-1., 1.]);
        assert_eq!((1., 2., 0.5, 0.5), simplex_parameters(&x0, false).unwrap());

        let x0 = Array1::from_iter([-1., 0., 1.]);
        let wrong_size_simplex = Array2::from_elem((2, 3), 1.);
        assert_eq!(
            initial_simplex_with_no_bounds(x0.clone(), Some(wrong_size_simplex)).unwrap_err(),
            TuutalError::Simplex {
                size: (2, 3),
                msg: "initial_simplex should be an array of shape (N+1, N)".to_string()
            }
        );
        let wrong_size_simplex = Array2::from_elem((3, 2), 1.);
        assert_eq!(
            initial_simplex_with_no_bounds(x0.clone(), Some(wrong_size_simplex)).unwrap_err(),
            TuutalError::Simplex {
                size: (3, 2),
                msg: "Size of initial_simplex is not consistent with x0".to_string()
            }
        );
        let right_size_simplex = Array2::from_elem((4, 3), 1.);
        assert_eq!(
            initial_simplex_with_no_bounds(x0.clone(), Some(right_size_simplex.clone()))
                .unwrap()
                .simplex,
            right_size_simplex
        );
        // Simplex default builder
        let built_frontier = initial_simplex_with_no_bounds(x0.clone(), None).unwrap();
        let mut expected_simplex = Vec::with_capacity(12);
        expected_simplex
            .extend_from_slice(&[-1., 0., 1., -1.05, 0., 1., -1., 0.00025, 1., -1., 0., 1.05]);
        // [-1., 0., 1.],
        // [-1.05, 0., 1.],
        // [-1., 0.00025, 1.],
        // [-1., 0., 1.05]
        assert!(
            built_frontier
                .simplex
                .into_iter()
                .zip(&expected_simplex)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                < 1e-12
        );
    }
    #[test]
    fn test_clamp() {
        let tuple_bound: (f32, f32) = (-1., 1.);
        let x0 = Array1::from_iter([-3.5, 1., 0., 2.]);
        assert_eq!(
            clamp(x0.clone(), Some(tuple_bound), None)
                .unwrap()
                .simplex
                .slice(s![0, ..]),
            Array1::from_iter([-1., 1., 0., 1.])
        );

        assert_eq!(
            clamp::<f32, (f32, f32)>(x0.clone(), None, None)
                .unwrap()
                .simplex
                .slice(s![0, ..]),
            x0.clone()
        );

        let wrong_order_bound: (f32, f32) = (1., 0.);
        let lower = Array1::from_iter([1., 1., 1., 1.]);
        let upper = Array1::from_iter([0., 0., 0., 0.]);
        assert_eq!(
            clamp(x0.clone(), Some(wrong_order_bound), None).unwrap_err(),
            TuutalError::BoundOrder {
                lower: lower,
                upper: upper
            }
        );
    }

    #[test]
    #[should_panic]
    fn test_wrong_size_bound() {
        // At the moment Bounds of type Vec<(T, T)> must have same length than the dimension
        // of x0, should maybe throw an error instead of panicking.
        let x0 = Array1::from_iter([1., 2.]);
        let mut vec_bounds = Vec::with_capacity(1);
        vec_bounds.push((-1., 1.));
        let _ = clamp(x0.clone(), Some(vec_bounds), None);
    }

    #[test]
    fn test_simplex_reflection_interior() {
        let mut vec = Vec::with_capacity(6);
        vec.extend_from_slice(&[0., 1., -1., 2., 2., -5.]);
        let mut simplex = Array2::from_shape_vec((3, 2), vec).unwrap();
        let lower = Array1::from_iter([-1., -1.]);
        let upper = Array1::from_iter([1., 1.]);

        let mut vec = Vec::with_capacity(6);
        vec.extend_from_slice(&[0., 1., -1., 0., 0., -1.]);
        let expected = Array2::from_shape_vec((3, 2), vec).unwrap();
        let func = |x| reflect_then_clamp_vec(x, &lower, &upper);
        row_map_matrix_mut(&mut simplex, func, 0);
        for k in 0..3 {
            assert!(l2_diff(&simplex.row(k).to_owned(), &expected.row(k).to_owned()) < 1e-6);
        }
    }

    #[test]
    fn test_nelder_mead() {
        let f = |arr: &Array1<f32>| arr.dot(arr);
        let x0 = Array1::from_iter([-5., -5.]);
        let x_star =
            nelder_mead::<_, (f32, f32), _>(f, &x0, None, Some(100), None, 1e-5, 1e-5, true, None)
                .unwrap();
        assert!(l2_diff(&x_star, &Array1::from_iter([0., 0.])) < 2e-3);
        assert!(f(&x_star) < 1e-5);
    }
}
