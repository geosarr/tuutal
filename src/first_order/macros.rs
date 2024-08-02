macro_rules! descent_rule {
    ($rule:ident, $step:ty, $sigma:expr, $accum:expr) => {
        #[derive(Debug)]
        #[allow(dead_code)]
        pub struct $rule<X, F, G, S>
        where
            X: Vector,
        {
            f: F,                                    // objective function
            gradf: G,                                // gradient of the objective function
            x: X,                                    // candidate solution
            neg_gradfx: X,                           // negative gradient of f at x,
            sigma: S,                                // step size
            hyper_params: HashMap<VarName, X::Elem>, // hyper-parameters of the algorithm like tolerance for convergence.
            counter: Counter<usize>, // [nb of iterations, number of f calls, nb of gradf calls]
            stop_metrics: X::Elem,   // metrics used to stop the algorithm,
            accumulators: HashMap<VarName, X>, // accumulators during corresponding algorithms.
        }

        impl<X, F, G> $rule<X, F, G, $step>
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
                self.stop_metrics <= self.hyper_params[&VarName::Epsilon].powi(2)
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
                    hyper_params: [
                        (VarName::Gamma, gamma),
                        (VarName::Beta, beta),
                        (VarName::Epsilon, eps),
                    ]
                    .into(),
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

macro_rules! impl_optimizer_descent {
    ($rule:ident, $step:ty) => {
        impl<X, F, G> core::iter::Iterator for $rule<X, F, G, $step>
        where
            X: Vector + VecDot<Output = X::Elem> + Clone,
            for<'b> &'b X: Add<X, Output = X> + Mul<&'b X, Output = X>,
            F: Fn(&X) -> X::Elem,
            G: Fn(&X) -> X,
        {
            type Item = X;
            fn next(&mut self) -> Option<Self::Item> {
                if self.stop() {
                    None
                } else {
                    self.stop_metrics = self.neg_gradfx.dot(&self.neg_gradfx);
                    self.step();
                    self.counter.iter += 1;
                    self.neg_gradfx = -self.grad(&self.x);
                    self.counter.gcalls += 1;
                    Some(self.x.clone())
                }
            }
        }
        impl<X, F, G> Optimizer for $rule<X, F, G, $step>
        where
            X: Vector + VecDot<Output = X::Elem> + Clone,
            for<'b> &'b X: Add<X, Output = X> + Mul<&'b X, Output = X>,
            F: Fn(&X) -> X::Elem,
            G: Fn(&X) -> X,
        {
            type Iterate = X;
            type Intermediate = HashMap<VarName, $step>;
            fn nb_iter(&self) -> usize {
                self.counter.iter
            }
            fn iterate(&self) -> X {
                self.x.clone()
            }
            fn intermediate(&self) -> Self::Intermediate {
                HashMap::from([(VarName::StepSize, self.sigma.clone())])
            }
        }
    };
}

pub(crate) use descent_rule;
pub(crate) use impl_optimizer_descent;
