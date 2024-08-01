macro_rules! steepest_descent_rule {
    ($rule:ident) => {
        pub struct $rule<X, F, G>
        where
            X: Vector,
            F: Fn(&X) -> X::Elem,
            G: Fn(&X) -> X,
        {
            f: F,
            gradf: G,
            x: X,
            eps: X::Elem,
            iter: usize,
            gamma: X::Elem,
            beta: X::Elem,
            sigma: X::Elem,
            fcalls: usize,
            gcalls: usize,
        }
        impl<X, F, G> $rule<X, F, G>
        where
            X: Vector,
            F: Fn(&X) -> X::Elem,
            G: Fn(&X) -> X,
        {
            pub fn new(f: F, gradf: G, x: X, gamma: X::Elem, beta: X::Elem, eps: X::Elem) -> Self
            where
                X: Vector,
                for<'a> &'a X: Add<X, Output = X>,
            {
                Self {
                    f,
                    gradf,
                    x,
                    eps,
                    iter: 0,
                    gamma,
                    beta,
                    sigma: X::Elem::one(),
                    fcalls: 0,
                    gcalls: 0,
                }
            }

            pub(crate) fn func(&self, x: &X) -> X::Elem {
                let f = &self.f;
                f(x)
            }
            pub(crate) fn grad(&self, x: &X) -> X {
                let g = &self.gradf;
                g(x)
            }
        }

        impl<X, F, G> core::iter::Iterator for $rule<X, F, G>
        where
            X: Vector + VecDot<Output = X::Elem> + Clone,
            for<'a> &'a X: Add<X, Output = X>,
            F: Fn(&X) -> X::Elem,
            G: Fn(&X) -> X,
        {
            type Item = X::Elem;
            fn next(&mut self) -> Option<Self::Item> {
                let neg_gradfx = -self.grad(&self.x);
                self.gcalls += 1;
                let squared_norm_2_gradfx = neg_gradfx.dot(&neg_gradfx);
                if squared_norm_2_gradfx <= (self.eps * self.eps) {
                    self.iter += 1;
                    None
                } else {
                    self.step(&neg_gradfx, squared_norm_2_gradfx);
                    self.iter += 1;
                    Some(squared_norm_2_gradfx)
                }
            }
        }

        // impl<X, F, G> Optimizer<X> for $rule<X, F, G>
        // where
        //     X: Vector + VecDot<Output = X::Elem> + Clone,
        //     for<'a> &'a X: Add<X, Output = X>,
        //     F: Fn(&X) -> X::Elem,
        //     G: Fn(&X) -> X,
        // {
        //     type Iterate = X;
        //     fn nb_iter(&self) -> usize {
        //         self.iter
        //     }
        //     fn iterate(&self) -> X {
        //         self.x.clone()
        //     }
        // }
    };
}

pub(crate) use steepest_descent_rule;
