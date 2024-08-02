![](assets/logo_tuutal.png)

# tuutal

Optimization library for scalar or multidimensional inputs functions.

It aims at the moment to reproduce and improve if possible the optimization submodule of Python scipy 1.13.1.
It is backed by ndarray crate for multidimensional optimization. For compatibility purpose, some of ndarray's objects
are imported into this crate.

# Implementations

For now, the supported algorithms are:
( - ) `Nelder-Mead`, `Powell` for multidimensional derivative-free optimization,
( - ) `Unbounded and Bounded Brent`'s algorithms for unidimensional derivative-free optimization,
( - ) `Brent`'s root finding algorithm for unidimensional functions
( - ) `AdaGrad`, `AdaDelta` for adaptive step size,
( - ) `Armijo` and `PowellWolfe` for steepest descent algorithms.
