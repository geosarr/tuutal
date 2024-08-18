macro_rules! descent_rule {
    ($rule:ident, $step:ty, $sigma:expr, $hp:ident, $accum:ty, $accumnew:expr) => {
        #[derive(Debug)]
        #[allow(dead_code)]
        pub struct $rule<X, F, G, S, H, A>
        where
            X: Vector,
            F: Fn(&X) -> X::Elem,
            G: Fn(&X) -> X,
        {
            f: F,                    // objective function
            gradf: G,                // gradient of the objective function
            x: X,                    // candidate solution
            neg_gradfx: X,           // negative gradient of f at x,
            sigma: S,                // step size
            hyper_params: H, // hyper-parameters of the algorithm like tolerance for convergence.
            counter: Counter<usize>, // [nb of iterations, number of f calls, nb of gradf calls]
            stop_metrics: X::Elem, // metrics used to stop the algorithm,
            accumulators: A, // accumulators during corresponding algorithms.
        }

        impl<X, F, G> $rule<X, F, G, $step, $hp<X::Elem>, $accum>
        where
            X: Vector,
            F: Fn(&X) -> X::Elem,
            G: Fn(&X) -> X,
        {
            /// New algorithm
            pub fn new(f: F, gradf: G, x: X, hyper_params: $hp<X::Elem>) -> Self {
                let neg_gradfx = -gradf(&x);
                let mut optimizer = Self {
                    f,
                    gradf,
                    x,
                    neg_gradfx,
                    sigma: $sigma,
                    hyper_params,
                    counter: Counter::new(),
                    stop_metrics: X::Elem::infinity(),
                    accumulators: $accumnew,
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
            #[allow(dead_code)]
            pub(crate) fn func(&self, x: &X) -> X::Elem {
                let f = &self.f;
                f(x)
            }
            pub(crate) fn grad(&self, x: &X) -> X {
                let g = &self.gradf;
                g(x)
            }
            pub(crate) fn stop(&self) -> bool {
                self.stop_metrics <= self.hyper_params.epsilon.powi(2)
            }
        }
    };
}

macro_rules! impl_optimizer_descent {
    ($rule:ident, $step:ty, $hp:ident, $accum:ty) => {
        impl<X, F, G> core::iter::Iterator for $rule<X, F, G, $step, $hp<X::Elem>, $accum>
        where
            X: Vector + VecDot<Output = X::Elem> + Clone,
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
        impl<X, F, G> Optimizer for $rule<X, F, G, $step, $hp<X::Elem>, $accum>
        where
            X: Vector + VecDot<Output = X::Elem> + Clone,
            for<'b> &'b X: Add<X, Output = X> + Mul<&'b X, Output = X>,
            F: Fn(&X) -> X::Elem,
            G: Fn(&X) -> X,
        {
            type Iterate = X;
            type Intermediate = $step;
            type ObjectiveOutput = X::Elem;
            fn nb_iter(&self) -> usize {
                self.counter.iter
            }
            fn iterate(&self) -> X {
                self.x.clone()
            }
            fn intermediate(&self) -> Self::Intermediate {
                self.sigma.clone()
            }
            fn objective_output(&mut self) -> Self::ObjectiveOutput {
                let fx = self.func(&self.x);
                self.counter.fcalls += 1;
                fx
            }
        }
    };
}

pub(crate) use descent_rule;
pub(crate) use impl_optimizer_descent;
