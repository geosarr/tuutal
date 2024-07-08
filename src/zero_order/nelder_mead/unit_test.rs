#[cfg(test)]
mod test {
    use super::super::*;
    use crate::{array, VecType};
    #[test]
    fn test_simplex() {
        let x0: VecType<f32> = array![];
        assert_eq!(
            TuutalError::EmptyDimension { x: x0.clone() },
            simplex_parameters(x0, true).unwrap_err()
        );

        let x0: VecType<f32> = array![-10., 36.];
        assert_eq!(
            (
                x0.clone(),
                HashMap::from([("rho", 1.), ("chi", 2.), ("psi", 0.5), ("sigma", 0.5)])
            ),
            simplex_parameters(x0, true).unwrap()
        );

        let x0: VecType<f32> = array![-1., 1.];
        assert_eq!(
            (
                x0.clone(),
                HashMap::from([("rho", 1.), ("chi", 2.), ("psi", 0.5), ("sigma", 0.5)])
            ),
            simplex_parameters(x0, false).unwrap()
        );

        let x0 = array![-1., 0., 1.];
        let wrong_size_simplex = Array::from_shape_vec((2, 3), vec![1.; 6]).unwrap();
        assert_eq!(
            initial_simplex(x0.clone(), Some(wrong_size_simplex)).unwrap_err(),
            TuutalError::Simplex {
                size: (2, 3),
                msg: "initial_simplex should be an array of shape (N+1, N)".to_string()
            }
        );
        let wrong_size_simplex = Array::from_shape_vec((3, 2), vec![1.; 6]).unwrap();
        assert_eq!(
            initial_simplex(x0.clone(), Some(wrong_size_simplex)).unwrap_err(),
            TuutalError::Simplex {
                size: (3, 2),
                msg: "Size of initial_simplex is not consistent with x0".to_string()
            }
        );
        let right_size_simplex = Array::from_shape_vec((4, 3), vec![1.; 12]).unwrap();
        assert_eq!(
            initial_simplex(x0.clone(), Some(right_size_simplex.clone())).unwrap(),
            right_size_simplex
        );
        // Simplex default builder
        let built_simplex = initial_simplex(x0.clone(), None).unwrap();
        let expected_simplex = array![
            [-1., 0., 1.],
            [-1.05, 0., 1.],
            [-1., 0.00025, 1.],
            [-1., 0., 1.05],
        ];
        assert!(
            built_simplex
                .into_iter()
                .zip(&expected_simplex)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                < 1e-12
        );
    }
}
