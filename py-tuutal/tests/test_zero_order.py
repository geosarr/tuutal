from tuutal import brent_bounded, brent_unbounded, brentq as brentq_rs, brent_root

from scipy.optimize._optimize import _minimize_scalar_bounded, _minimize_scalar_brent
from scipy.optimize._zeros_py import brentq
import numpy as np


def test_brent_bounded():
    f = lambda x: 2 * x**3 + np.exp(x**2) + np.cos(x - 1)
    bounds = (-1, 1)
    rust = brent_bounded(f, bounds=bounds, xatol=1e-5, maxiter=500)
    py = _minimize_scalar_bounded(f, bounds=bounds, xatol=1e-5, maxiter=500)
    assert abs(rust[0] - py.x) < 1e-9
    assert abs(rust[1] - py.fun) < 1e-9
    assert rust[2] == py.nfev


def test_brent_unbounded():
    f = lambda x: (np.exp(x) - 1 / 2) / (np.exp(x) + 1)
    rust = brent_unbounded(f, brack=(0, 1), xtol=1e-5, maxiter=500)
    py = _minimize_scalar_brent(f, brack=(0, 1), xtol=1e-5, maxiter=500)
    assert abs(rust[0] - py.x) < 1e-6
    assert abs(rust[1] - py.fun) < 1e-9
    assert rust[2] == py.nfev


def test_brentq():
    f = lambda x: np.log(x + 1) / (np.cos(x) + 1.5)
    rust = brentq_rs(f, a=-0.9, b=0.9, xtol=1e-4, rtol=1e-4, maxiter=1000)
    py = brentq(f, a=-0.9, b=0.9, xtol=1e-4, rtol=1e-4, maxiter=1000, full_output=True)
    assert abs(rust[0] - py[1].root) < 1e-20
    assert abs(rust[1]) < 1e-7
    assert rust[2] == py[1].function_calls

    # Less satisfying results.
    rust = brent_root(f, a=-0.9, b=0.9, xtol=1e-4, rtol=1e-4, maxiter=1000)
    assert abs(rust[0]) < 1e-4
    assert abs(rust[0]) < 9e-4
