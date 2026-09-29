"""Composite follow-or-fade signal (paper Section 10).

    D = w_news s_new - w_echo s_echo + w_do Do + switch * w_say gamma Say + w_abs A + w_levy A_levy

where ``switch = sign(Cov(Say, Do))`` (Theorem 4.5).  In manipulative regimes
(switch = -1) the Say and Do terms combine into ``-G`` with G = gamma Say - Do, the
Say--Do gap of Theorem 3.7.  Every sign is fixed by a theorem:

=====================  ===========  ======================================
component              sign         source
=====================  ===========  ======================================
news sentiment         +            Corollary 3.6 (follow the news)
echo sentiment         -            Corollary 3.5 (fade the echo)
positioning (Do)       +            Theorem 4.5 (deeds are always followed)
statements (Say)       sign of Cov  Theorems 3.7 and 4.5
absorption             +            Theorem 3.8 (informed regime)
Levy area (signed)     +            Theorem 7.3 (money moved first)
=====================  ===========  ======================================
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["SignalWeights", "EventSignals", "composite_score", "say_do_gap", "rolling_covariance", "say_do_switch"]


@dataclass(frozen=True)
class SignalWeights:
    """Non-negative magnitudes; signs are fixed by the theory."""

    news: float = 1.0
    echo: float = 1.0
    do: float = 1.0
    say: float = 1.0
    absorption: float = 1.0
    levy: float = 1.0

    def __post_init__(self) -> None:
        if min(asdict(self).values()) < 0:
            raise ValueError("weights are magnitudes; the signs are fixed by the theory")


@dataclass
class EventSignals:
    """Standardised signals for one event (positive = bullish)."""

    s_news: float
    s_echo: float
    say: float
    do: float
    absorption: float
    levy_area: float
    say_do_cov_sign: float = -1.0
    gamma: float = 1.0

    @property
    def gap(self) -> float:
        return say_do_gap(self.say, self.do, self.gamma)

    def contributions(self, w: SignalWeights = SignalWeights()) -> dict:
        switch = 1.0 if self.say_do_cov_sign > 0 else -1.0
        return {
            "news": float(w.news * self.s_news),
            "echo": float(-w.echo * self.s_echo),
            "do": float(w.do * self.do),
            "say": float(switch * w.say * self.gamma * self.say),
            "absorption": float(w.absorption * self.absorption),
            "levy": float(w.levy * self.levy_area),
        }

    def score(self, w: SignalWeights = SignalWeights()) -> float:
        return float(sum(self.contributions(w).values()))


def composite_score(signals: EventSignals, weights: SignalWeights = SignalWeights()) -> float:
    """Positive -> lean long; negative -> lean short."""
    return signals.score(weights)


def say_do_gap(say: float, do: float, gamma: float = 1.0) -> float:
    """G = gamma * Say - Do (Theorem 3.7).  Negative: talks bearish, acts bullish."""
    return gamma * say - do


def rolling_covariance(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.full(x.size, np.nan)
    for t in range(window - 1, x.size):
        xs, ys = x[t - window + 1: t + 1], y[t - window + 1: t + 1]
        out[t] = np.mean((xs - xs.mean()) * (ys - ys.mean()))
    return out


def say_do_switch(say: np.ndarray, do: np.ndarray, window: int) -> np.ndarray:
    """Rolling sign of Cov(Say, Do): +1 follow words, -1 fade words (Theorem 4.5)."""
    return np.sign(rolling_covariance(say, do, window))
