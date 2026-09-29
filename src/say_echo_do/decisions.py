"""From signals to decisions (paper Sections 8 and 9).

* causal Gaussian HMM regime filter (informational / overreaction / strategic),
* exponential detectability bound for strategic episodes (Theorem 8.2),
* adaptive conformal intervals with deterministic long-run coverage
  (Gibbs & Candes, 2021; Theorem 8.3),
* Kelly shrinkage under estimation risk (Theorem 9.1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import multivariate_normal

__all__ = [
    "REGIMES",
    "RegimeFilter",
    "detection_bound",
    "events_needed",
    "AdaptiveConformal",
    "kelly_shrinkage",
    "expected_log_growth",
]

REGIMES = ("informational", "overreaction", "strategic")


@dataclass
class RegimeFilter:
    """Forward (causal) filter for a Gaussian-emission hidden Markov model.

    Only filtered probabilities are produced; smoothed probabilities use future
    data and must never be used for forecasting.
    """

    means: np.ndarray
    covs: np.ndarray
    transition: np.ndarray
    prior: np.ndarray | None = None
    names: tuple = REGIMES
    _pi: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.means = np.atleast_2d(np.asarray(self.means, dtype=float))
        self.covs = np.asarray(self.covs, dtype=float)
        self.transition = np.asarray(self.transition, dtype=float)
        K = self.means.shape[0]
        if not np.allclose(self.transition.sum(1), 1):
            raise ValueError("transition rows must sum to one")
        self._pi = np.full(K, 1.0 / K) if self.prior is None else np.asarray(self.prior, dtype=float)

    def update(self, obs: np.ndarray) -> np.ndarray:
        pred = self.transition.T @ self._pi
        lik = np.array([multivariate_normal(self.means[k], self.covs[k]).pdf(obs) for k in range(len(pred))])
        post = pred * lik
        self._pi = post / post.sum()
        return self._pi.copy()

    def filter(self, observations: np.ndarray) -> np.ndarray:
        return np.array([self.update(o) for o in np.atleast_2d(observations)])

    def mixture_forecast(self, expert_forecasts: np.ndarray) -> float:
        """E[r | F_t] = sum_k pi_t(k) E[r | F_t, K_t = k]."""
        return float(self._pi @ np.asarray(expert_forecasts, dtype=float))


def detection_bound(delta: float, n: int | np.ndarray, pi0: float = 0.5) -> np.ndarray:
    """Upper bound on MAP misclassification of an n-event episode: sqrt(pi0 pi1) exp(-n delta^2 / 8)."""
    return math.sqrt(pi0 * (1 - pi0)) * np.exp(-np.asarray(n) * delta**2 / 8.0)


def events_needed(delta: float, eps: float) -> int:
    """Smallest n guaranteeing error <= eps (equal priors): n >= 8 / delta^2 * log(1 / (2 eps))."""
    return int(math.ceil(8.0 / delta**2 * math.log(1.0 / (2.0 * eps))))


@dataclass
class AdaptiveConformal:
    """Adaptive conformal inference with symmetric absolute-residual intervals.

    alpha_{t+1} = alpha_t + gamma (alpha - err_t).  For every outcome sequence,
    |mean(err) - alpha| <= (max(alpha_1, 1 - alpha_1) + gamma) / (gamma T).
    """

    alpha: float = 0.1
    gamma: float = 0.005
    alpha_t: float | None = None
    residuals: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.alpha_t is None:
            self.alpha_t = self.alpha

    def radius(self) -> float:
        """Interval half-width; inf means the whole line, a negative value means the empty set."""
        if self.alpha_t <= 0 or not self.residuals:
            return math.inf
        if self.alpha_t >= 1:
            return -1.0
        return float(np.quantile(np.abs(self.residuals), 1 - self.alpha_t))

    def interval(self, prediction: float) -> tuple[float, float] | None:
        r = self.radius()
        return None if r < 0 else (prediction - r, prediction + r)

    def update(self, prediction: float, outcome: float) -> bool:
        """Record an outcome, return whether it was a miss, and adapt alpha_t."""
        r = self.radius()
        err = bool(r < 0 or abs(outcome - prediction) > r)
        self.errors.append(err)
        self.residuals.append(outcome - prediction)
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha - float(err))
        return err

    def coverage_gap_bound(self, alpha1: float | None = None) -> float:
        a1 = self.alpha if alpha1 is None else alpha1
        return (max(a1, 1 - a1) + self.gamma) / (self.gamma * max(len(self.errors), 1))


def kelly_shrinkage(mu: float, s: float) -> float:
    """Growth-optimal fraction of full Kelly under estimation risk: c* = mu^2 / (mu^2 + s^2)."""
    return mu**2 / (mu**2 + s**2) if (mu or s) else 0.0


def expected_log_growth(c: float, mu: float, sigma: float, s: float) -> float:
    """E g = (c mu^2 - c^2 (mu^2 + s^2) / 2) / sigma^2 for f = c mu_hat / sigma^2."""
    return (c * mu**2 - 0.5 * c**2 * (mu**2 + s**2)) / sigma**2
