"""Who moved first?  Levy area between positioning and narrative (paper Section 7).

Theorem 7.3: for jointly stationary smooth paths, E[A_T] / T -> C'(0), the slope of
the cross-covariance at zero.  If Y is a lagged copy of X (Y_t = gamma X_{t-l} + noise),
the rate has the sign of gamma * l.  With X = positioning and Y = headline gloom, a
positive area rate means money moved first and the story later turned against it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["levy_area", "levy_area_rate", "rolling_levy_area", "standardise"]


def levy_area(X: np.ndarray, Y: np.ndarray) -> float:
    """Exact Levy area of the piecewise-linear path through (X_i, Y_i).

    A = 1/2 sum_i [(X_i - X_0)(Y_{i+1} - Y_i) - (Y_i - Y_0)(X_{i+1} - X_i)]
    (Proposition, appendix).  Antisymmetric, bilinear, reparametrisation invariant,
    and equal to the signed enclosed area for closed loops.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape or X.ndim != 1:
        raise ValueError("X and Y must be 1-D arrays of equal length")
    x, y = X[:-1] - X[0], Y[:-1] - Y[0]
    return float(0.5 * np.sum(x * np.diff(Y) - y * np.diff(X)))


def levy_area_rate(X: np.ndarray, Y: np.ndarray, dt: float = 1.0) -> float:
    """Levy area per unit time, the estimator of C'(0) in Theorem 7.3."""
    return levy_area(X, Y) / (dt * (len(X) - 1))


def rolling_levy_area(X: np.ndarray, Y: np.ndarray, window: int) -> np.ndarray:
    """Levy area over trailing windows (NaN until the first full window)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    out = np.full(X.size, np.nan)
    for t in range(window - 1, X.size):
        out[t] = levy_area(X[t - window + 1: t + 1], Y[t - window + 1: t + 1])
    return out


def standardise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.0)
