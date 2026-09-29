"""Three-voice linear--Gaussian market (paper Section 3).

Implements the closed forms of

* Theorem 3.4  (follow-or-fade loadings),
* Corollary 3.5 / 3.5 (fade the echo, follow the news),
* Theorem 3.7  (Say--Do gap optimality),
* Theorem 3.8  (absorption sign),

together with a simulator that generates data from the model, so every
closed form can be checked by regression.

Model
-----
    v ~ N(0, sigma_v^2)                      true value
    y = h v + nu,  nu ~ N(0, Sigma_nu)       public channels (Say, news, echo, ...)
    u ~ N(0, sigma_u^2)                      noise demand
    x = beta (v - p)                         informed demand
    p = phi' y + lam (x + u)                 price
    r = v - p                                forward return
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["ThreeVoiceMarket", "SayDoCoefficients", "kyle_intensities"]


@dataclass(frozen=True)
class SayDoCoefficients:
    """Output of :meth:`ThreeVoiceMarket.say_do`.

    Attributes
    ----------
    c_x:
        Loading of the forecast on the positioning proxy ``x_hat``.
    theta:
        Shrinkage factor applied to the narrative loadings, in (0, 1].
    y_loadings:
        Loadings on the public channels once positioning is observed.
    gamma:
        Exchange rate between words and deeds for the chosen Say channel,
        so that the forecast depends on (m, x_hat) through ``-c_x (gamma m - x_hat)``.
    variance:
        Conditional variance of the return given (y, x_hat).
    """

    c_x: float
    theta: float
    y_loadings: np.ndarray
    gamma: float
    variance: float


@dataclass
class ThreeVoiceMarket:
    """Linear--Gaussian economy with a narrative-sensitive crowd.

    Parameters
    ----------
    sigma_v:
        Standard deviation of fundamental value.
    h:
        Signal loadings, shape (K,).
    Sigma_nu:
        Channel noise covariance, shape (K, K).
    phi:
        Crowd narrative multipliers, shape (K,).
    beta:
        Informed trading intensity (> 0).
    lam:
        Price impact (> 0).
    sigma_u:
        Standard deviation of noise demand.
    """

    sigma_v: float
    h: np.ndarray
    Sigma_nu: np.ndarray
    phi: np.ndarray
    beta: float
    lam: float
    sigma_u: float
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.h = np.asarray(self.h, dtype=float)
        self.Sigma_nu = np.atleast_2d(np.asarray(self.Sigma_nu, dtype=float))
        self.phi = np.asarray(self.phi, dtype=float)
        K = self.h.shape[0]
        if self.Sigma_nu.shape != (K, K) or self.phi.shape != (K,):
            raise ValueError("h, Sigma_nu and phi must have compatible shapes (K,), (K,K), (K,)")
        if self.beta <= 0 or self.lam <= 0:
            raise ValueError("beta and lam must be positive")

    # ------------------------------------------------------------------ basics
    @property
    def kappa(self) -> float:
        """kappa = 1 / (1 + lam * beta)."""
        return 1.0 / (1.0 + self.lam * self.beta)

    @property
    def Sigma_y(self) -> np.ndarray:
        return self.sigma_v**2 * np.outer(self.h, self.h) + self.Sigma_nu

    @property
    def bayes_weights(self) -> np.ndarray:
        """w* such that E[v | y] = w*' y."""
        return self.sigma_v**2 * np.linalg.solve(self.Sigma_y, self.h)

    @property
    def residual_value_variance(self) -> float:
        """sigma^2_{v|y} = Var(v | y)."""
        return float(self.sigma_v**2 - self.sigma_v**4 * self.h @ np.linalg.solve(self.Sigma_y, self.h))

    # ------------------------------------------------------- Theorem 3.4
    def return_loadings(self) -> np.ndarray:
        """Predictive loadings of r on y: kappa (w* - phi).  Negative entries = fade."""
        return self.kappa * (self.bayes_weights - self.phi)

    def return_variance(self) -> float:
        """varsigma^2 = Var(r | y) = kappa^2 (sigma^2_{v|y} + lam^2 sigma_u^2)."""
        return self.kappa**2 * (self.residual_value_variance + self.lam**2 * self.sigma_u**2)

    def follow_or_fade(self) -> list[str]:
        """'fade' where the crowd overweights a channel, 'follow' where it underweights it."""
        w = self.bayes_weights
        return ["fade" if p > wk else "follow" if p < wk else "neutral" for p, wk in zip(self.phi, w)]

    # ------------------------------------------------------- Theorem 3.7
    def say_do(self, sigma_x: float, say_index: int = 0) -> SayDoCoefficients:
        """Optimal forecast once a positioning proxy x_hat = x + N(0, sigma_x^2) is observed."""
        vs2 = self.return_variance()
        denom = self.beta**2 * vs2 + sigma_x**2
        c_x = self.beta * vs2 / denom
        theta = sigma_x**2 / denom
        y_load = theta * self.return_loadings()
        w = self.bayes_weights
        gamma = theta * self.kappa * (self.phi[say_index] - w[say_index]) / c_x if c_x > 0 else np.inf
        return SayDoCoefficients(c_x=c_x, theta=theta, y_loadings=y_load, gamma=float(gamma), variance=theta * vs2)

    # ------------------------------------------------------- Theorem 3.8
    def absorption_coefficient(self) -> float:
        """c_A in E[r | y, a] = E[r | y] + c_A (a - E[a | y]).

        Positive -> absorption predicts continuation; negative -> reversal.
        """
        s2 = self.residual_value_variance
        num = self.beta * s2 - self.lam * self.sigma_u**2
        den = self.lam * (self.beta**2 * s2 + self.sigma_u**2)
        return float(num / den)

    def absorption_regime(self) -> str:
        c = self.absorption_coefficient()
        return "continuation" if c > 0 else "reversal" if c < 0 else "uninformative"

    # ------------------------------------------------------------ simulation
    def simulate(self, n: int, rng: np.random.Generator | None = None, sigma_x: float | None = None) -> dict:
        """Draw n i.i.d. economies.  Returns arrays v, y, u, p, r, x, a (and x_hat)."""
        rng = np.random.default_rng(rng)
        K = self.h.shape[0]
        v = rng.normal(0.0, self.sigma_v, n)
        nu = rng.multivariate_normal(np.zeros(K), self.Sigma_nu, n)
        y = v[:, None] * self.h[None, :] + nu
        u = rng.normal(0.0, self.sigma_u, n)
        k = self.kappa
        p = k * (y @ self.phi + self.lam * self.beta * v + self.lam * u)
        r = v - p
        x = self.beta * r
        a = p - y @ self.phi
        out = dict(v=v, y=y, u=u, p=p, r=r, x=x, a=a)
        if sigma_x is not None:
            out["x_hat"] = x + rng.normal(0.0, sigma_x, n)
        return out


def kyle_intensities(sigma_v_given_y: float, sigma_u: float) -> tuple[float, float]:
    """Single-period Kyle (1985) equilibrium: beta = sigma_u / sigma_v, lam = sigma_v / (2 sigma_u)."""
    return sigma_u / sigma_v_given_y, sigma_v_given_y / (2.0 * sigma_u)
