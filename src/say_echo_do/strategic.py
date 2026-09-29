"""Strategic speech and the economics of false alarms (paper Section 4).

An institution knows value v, publishes a message m, and trades x.  The crowd
prices the (noisy) message with credulity ``phi``; price impact is ``lam``;
misreporting costs ``k/2 (m - v)^2``.  With ``a = 2 lam k``:

* Theorem 4.3  optimal speech m* = psi v and trade x* = chi v,
* Corollary 4.4 four-regime taxonomy (truth, shading, false alarm, exaggeration),
* Theorem 4.5  Say--Do covariance identity  Cov(r, Say) = lam Cov(Do, Say) - phi s_eps^2,
* Proposition 4.6 manipulation windows in the media branching ratio,
* Theorem 4.7  credibility cycles of adaptive trust  phi_{t+1} = (a - phi^2) / (a - phi).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

__all__ = [
    "Regime",
    "StrategicInstitution",
    "effective_credulity",
    "manipulation_window",
    "trust_map",
    "trust_map_derivative",
    "iterate_trust",
    "two_cycle",
    "two_cycle_multiplier",
    "trust_lyapunov_exponent",
    "trust_regime",
]


class Regime(str, Enum):
    TRUTH = "truth"
    SHADING = "shading"
    FALSE_ALARM = "false_alarm"
    EXAGGERATION = "exaggeration"
    UNBOUNDED = "unbounded"


@dataclass(frozen=True)
class StrategicInstitution:
    """Linear--quadratic institution that speaks while it trades.

    Parameters
    ----------
    phi:
        Crowd credulity (price response per unit of message), >= 0.
    lam:
        Price impact of the institution's trade, > 0.
    k:
        Cost coefficient of misreporting, > 0.
    """

    phi: float
    lam: float
    k: float

    def __post_init__(self) -> None:
        if self.phi < 0 or self.lam <= 0 or self.k <= 0:
            raise ValueError("require phi >= 0, lam > 0, k > 0")

    @property
    def a(self) -> float:
        """Deterrence--impact product a = 2 lam k."""
        return 2.0 * self.lam * self.k

    @property
    def has_interior_optimum(self) -> bool:
        return self.a > self.phi**2

    # ------------------------------------------------------- Theorem 4.3
    @property
    def speech_slope(self) -> float:
        """psi = (a - phi) / (a - phi^2):  m* = psi v."""
        self._require_interior()
        return (self.a - self.phi) / (self.a - self.phi**2)

    @property
    def trade_slope(self) -> float:
        """chi = k (1 - phi) / (a - phi^2):  x* = chi v."""
        self._require_interior()
        return self.k * (1.0 - self.phi) / (self.a - self.phi**2)

    def objective(self, x: float, m: float, v: float) -> float:
        """J(x, m) = x (v - phi m - lam x) - k/2 (m - v)^2."""
        return x * (v - self.phi * m - self.lam * x) - 0.5 * self.k * (m - v) ** 2

    def optimal_policy(self, v: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v = np.asarray(v, dtype=float)
        return self.speech_slope * v, self.trade_slope * v

    # ------------------------------------------------------ Corollary 4.4
    def regime(self, tol: float = 1e-12) -> Regime:
        if not self.has_interior_optimum:
            return Regime.UNBOUNDED
        if abs(self.phi - 1.0) <= tol:
            return Regime.TRUTH
        if self.phi > 1.0:
            return Regime.EXAGGERATION
        return Regime.SHADING if self.a > self.phi else Regime.FALSE_ALARM

    # ------------------------------------------------------- Theorem 4.5
    def say_do_covariance(self, sigma_v2: float) -> float:
        """Equilibrium Cov(Do, Say) = psi chi sigma_v^2 (message noise is independent)."""
        return self.speech_slope * self.trade_slope * sigma_v2

    def return_say_covariance(self, sigma_v2: float, sigma_eps2: float) -> float:
        """Cov(r, Say) = lam Cov(Do, Say) - phi sigma_eps^2 (Theorem 4.5)."""
        return self.lam * self.say_do_covariance(sigma_v2) - self.phi * sigma_eps2

    def simulate(self, v: np.ndarray, eps: np.ndarray, message_rule=None) -> dict:
        """Simulate prices and returns.

        If ``message_rule`` (callable v -> m) is given, the institution uses it and
        trades optimally given the message; otherwise the equilibrium policy is used.
        The identity r = lam x - phi eps holds pathwise either way.
        """
        v = np.asarray(v, dtype=float)
        eps = np.asarray(eps, dtype=float)
        m = message_rule(v) if message_rule is not None else self.speech_slope * v
        x = (v - self.phi * m) / (2.0 * self.lam)
        m_obs = m + eps
        p = self.phi * m_obs + self.lam * x
        return dict(m=m, m_obs=m_obs, x=x, p=p, r=v - p)

    def _require_interior(self) -> None:
        if not self.has_interior_optimum:
            raise ValueError(f"no interior optimum: a = {self.a:.4g} <= phi^2 = {self.phi**2:.4g}")


# --------------------------------------------------------- Proposition 4.6
def effective_credulity(phi0: float, branching_ratio: float) -> float:
    """phi = phi0 / (1 - n): per-article credulity amplified by a Hawkes cascade."""
    if not 0.0 <= branching_ratio < 1.0:
        raise ValueError("branching ratio must lie in [0, 1)")
    return phi0 / (1.0 - branching_ratio)


def manipulation_window(phi0: float, a: float) -> dict:
    """Window of media branching ratios n in which speech is manipulative.

    Returns ``{"regime": Regime, "n_low": float, "n_high": float}``; the window may
    be empty (n_low >= n_high).  ``n_unbounded`` is the ratio above which no
    interior optimum exists.
    """
    if not 0.0 < phi0 < 1.0 or a <= 0:
        raise ValueError("require 0 < phi0 < 1 and a > 0")
    n_unb = 1.0 - phi0 / math.sqrt(a)
    if a < 1.0:
        lo, regime = max(0.0, 1.0 - phi0 / a), Regime.FALSE_ALARM
    else:
        lo, regime = 1.0 - phi0, Regime.EXAGGERATION
    return {"regime": regime, "n_low": lo, "n_high": n_unb, "n_unbounded": n_unb}


# ------------------------------------------------------------ Theorem 4.7
def trust_map(phi: float | np.ndarray, a: float) -> np.ndarray:
    """Adaptive-trust update F_a(phi) = (a - phi^2) / (a - phi) = 1 / psi(phi)."""
    phi = np.asarray(phi, dtype=float)
    return (a - phi**2) / (a - phi)


def trust_map_derivative(phi: float | np.ndarray, a: float) -> np.ndarray:
    phi = np.asarray(phi, dtype=float)
    return (a - 2 * a * phi + phi**2) / (a - phi) ** 2


def _admissible(phi: float, a: float) -> bool:
    return np.isfinite(phi) and phi**2 < a and phi != a and (a - phi) != 0 and (a - phi**2) != 0


def iterate_trust(a: float, phi0: float, steps: int) -> np.ndarray:
    """Iterate the trust map.  Entries become NaN once the orbit leaves the admissible set."""
    out = np.full(steps + 1, np.nan)
    phi = phi0
    for t in range(steps + 1):
        if not _admissible(phi, a):
            break
        out[t] = phi
        phi = float(trust_map(phi, a))
    return out


def two_cycle(a: float) -> tuple[float, float]:
    """Closed-form 2-cycle phi_pm = (a -/+ sqrt(a (2 - a))) / 2, valid for 1 < a < 2."""
    if not 1.0 < a < 2.0:
        raise ValueError("the 2-cycle exists for 1 < a < 2")
    r = math.sqrt(a * (2.0 - a))
    return (a - r) / 2.0, (a + r) / 2.0


def two_cycle_multiplier(a: float) -> float:
    """(F_a^2)'(phi_pm) = (5a - 9) / (a - 1).  Stable iff 5/3 < a < 2."""
    return (5.0 * a - 9.0) / (a - 1.0)


def trust_lyapunov_exponent(a: float, phi0: float = 0.97, burn: int = 1000, steps: int = 20000) -> float:
    """Numerical Lyapunov exponent of the trust map; NaN if the orbit escapes."""
    phi, acc, cnt = phi0, 0.0, 0
    for t in range(burn + steps):
        if not _admissible(phi, a):
            return float("nan")
        if t >= burn:
            d = abs(float(trust_map_derivative(phi, a)))
            acc += math.log(d) if d > 0 else -np.inf
            cnt += 1
        phi = float(trust_map(phi, a))
    return acc / cnt


def trust_regime(a: float) -> str:
    """'converges' (a > 2), 'two_cycle' (5/3 < a < 2), 'complex' (1 < a < 5/3), 'no_truth' (a <= 1)."""
    if a > 2.0:
        return "converges"
    if a > 5.0 / 3.0:
        return "two_cycle"
    if a > 1.0:
        return "complex"
    return "no_truth"
