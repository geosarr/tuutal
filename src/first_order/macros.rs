macro_rules! descent_rule {
    ($rule:ident, $step:ty, $sigma:expr, $accum:expr) => {
        #[derive(Debug)]
        #[allow(dead_code)]
        pub struct $rule<'a, X, F, G, S>
        where
            X: Vector,
        {
            f: F,                                    // objective function
            gradf: G,                                // gradient of the objective function
            x: X,                                    // candidate solution
            neg_gradfx: X,                           // negative gradient of f at x,
            sigma: S,                                // step size
            hyper_params: HashMap<&'a str, X::Elem>, // hyper-parameters of the algorithm like tolerance for convergence.
            counter: Counter<usize>, // [nb of iterations, number of f calls, nb of gradf calls]
            stop_metrics: X::Elem,   // metrics used to stop the algorithm,
            accumulators: HashMap<&'a str, X>, // accumulators during corresponding algorithms.
        }

        impl<'a, X, F, G> $rule<'a, X, F, G, $step>
        where
            X: Vector,
        {
            #[allow(dead_code)]
            pub(crate) fn func(&self, x: &X) -> X::Elem
            where
                F: Fn(&X) -> X::Elem,
            {
                let f = &self.f;
                f(x)
            }
            pub(crate) fn grad(&self, x: &X) -> X
            where
                G: Fn(&X) -> X,
            {
                let g = &self.gradf;
                g(x)
            }
            pub(crate) fn stop(&self) -> bool {
                self.stop_metrics <= self.hyper_params["eps"].powi(2)
            }

            pub fn new(f: F, gradf: G, x: X, gamma: X::Elem, beta: X::Elem, eps: X::Elem) -> Self
            where
                X: Vector,
                G: Fn(&X) -> X,
            {
                let neg_gradfx = -gradf(&x);
                let mut optimizer = Self {
                    f,
                    gradf,
                    x,
                    neg_gradfx: neg_gradfx,
                    sigma: $sigma,
                    hyper_params: [("gamma", gamma), ("beta", beta), ("eps", eps)].into(),
                    counter: Counter::new(),
                    stop_metrics: X::Elem::infinity(),
                    accumulators: $accum,
                };
                optimizer.counter.gcalls += 1;
                // Not needed when broadcasting is allowed ?
                // For descent method with adaptive step size
                // let dim = x.len();
                // if $sigma.len() > 1 {
                //     for (_, val) in optimizer.accumulators.iter_mut() {
                //         *val = X::zero(dim);
                //     }
                // }
                optimizer
            }
        }
    };
}

macro_rules! impl_iterator_descent {
    ($rule:ident, $step:ty) => {
        impl<'a, X, F, G> core::iter::Iterator for $rule<'a, X, F, G, $step>
        where
            X: Vector + VecDot<Output = X::Elem>,
            for<'b> &'b X: Add<X, Output = X> + Mul<&'b X, Output = X>,
            F: Fn(&X) -> X::Elem,
            G: Fn(&X) -> X,
        {
            type Item = X::Elem;
            fn next(&mut self) -> Option<Self::Item> {
                if self.stop() {
                    None
                } else {
                    self.stop_metrics = self.neg_gradfx.dot(&self.neg_gradfx);
                    self.step();
                    self.counter.iter += 1;
                    self.neg_gradfx = -self.grad(&self.x);
                    self.counter.gcalls += 1;
                    Some(self.stop_metrics)
                }
            }
        }
    };
}

pub(crate) use descent_rule;
pub(crate) use impl_iterator_descent;
