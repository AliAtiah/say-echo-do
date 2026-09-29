import numpy as np
import pytest

from say_echo_do.decisions import (
    AdaptiveConformal, RegimeFilter, detection_bound, events_needed, expected_log_growth, kelly_shrinkage,
)
from say_echo_do.leadlag import levy_area, levy_area_rate
from say_echo_do.signal import EventSignals, SignalWeights, say_do_gap, say_do_switch


# ------------------------------------------------------------------ Levy area
def test_levy_area_of_unit_square():
    X = np.array([0, 1, 1, 0, 0.0])
    Y = np.array([0, 0, 1, 1, 0.0])
    assert levy_area(X, Y) == pytest.approx(1.0)
    assert levy_area(Y, X) == pytest.approx(-1.0)          # antisymmetry
    assert levy_area(2 * X + 3, -Y + 1) == pytest.approx(-2.0)  # bilinearity, shift invariance


def test_levy_area_reparametrisation_invariance():
    t = np.linspace(0, 2 * np.pi, 4001)
    s = np.linspace(0, 1, 4001) ** 3 * 2 * np.pi
    assert levy_area(np.cos(t), np.sin(t)) == pytest.approx(np.pi, rel=1e-5)
    assert levy_area(np.cos(s), np.sin(s)) == pytest.approx(np.pi, rel=1e-4)


def test_levy_area_rate_identifies_lead(rng):  # Theorem 7.3
    dt, T = 0.01, 2000
    n = int(T / dt)
    w = rng.normal(size=n + 600)
    ker = np.exp(-(np.arange(-300, 301) * dt) ** 2 / 2)
    X = np.convolve(w, ker, "same")[300:300 + n]
    X /= X.std()
    lag = int(0.5 / dt)
    Y_lead = np.r_[np.zeros(lag), X[:-lag]]      # Y follows X: X leads
    Y_lag = np.r_[X[lag:], np.zeros(lag)]        # Y precedes X
    R = lambda h: np.mean(X[h:] * X[:-h])
    theory = -(R(lag + 1) - R(lag - 1)) / (2 * dt)   # C'(0) = -R'(lag)
    assert levy_area_rate(X, Y_lead, dt) == pytest.approx(theory, rel=0.05)
    assert levy_area_rate(X, Y_lead, dt) > 0 > levy_area_rate(X, Y_lag, dt)


# ------------------------------------------------------------------ decisions
def test_kelly_shrinkage_monte_carlo(rng):  # Theorem 9.1
    mu, sigma, s = 0.05, 0.2, 0.07
    mu_hat = rng.normal(mu, s, 2_000_000)
    for c in (kelly_shrinkage(mu, s), 1.0):
        f = c * mu_hat / sigma**2
        g = np.mean(f * mu - 0.5 * f**2 * sigma**2)
        assert g == pytest.approx(expected_log_growth(c, mu, sigma, s), rel=5e-3, abs=2e-5)
    assert expected_log_growth(1.0, mu, sigma, s) < 0 < expected_log_growth(kelly_shrinkage(mu, s), mu, sigma, s)


def test_adaptive_conformal_deterministic_coverage(rng):  # Theorem 8.3
    aci = AdaptiveConformal(alpha=0.1, gamma=0.02)
    # adversarial, regime-switching outcome sequence
    for t in range(5000):
        scale = 1.0 if (t // 500) % 2 == 0 else 8.0
        aci.update(0.0, rng.standard_cauchy() * scale)
    gap = abs(np.mean(aci.errors) - 0.1)
    assert gap <= aci.coverage_gap_bound() + 1e-12


def test_detection_bound_holds(rng):  # Theorem 8.2
    delta, n = 1.0, 12
    m0, m1 = np.zeros(2), np.array([delta, 0.0])
    errs = 0
    trials = 20000
    for _ in range(trials):
        k = rng.integers(2)
        obs = rng.normal(size=(n, 2)) + (m1 if k else m0)
        llr = (obs @ (m1 - m0) - 0.5 * (m1 @ m1 - m0 @ m0)).sum()
        errs += int((llr > 0) != bool(k))
    assert errs / trials <= detection_bound(delta, n)
    assert detection_bound(delta, events_needed(delta, 0.01)) <= 0.01


def test_regime_filter_is_causal_and_normalised(rng):
    means = np.array([[1.0, 1.0], [0.0, -1.0], [-1.0, 2.0]])
    covs = np.array([np.eye(2)] * 3)
    P = np.full((3, 3), 0.05) + np.eye(3) * 0.85
    obs = rng.normal(size=(50, 2)) + means[2]
    f = RegimeFilter(means, covs, P)
    probs = f.filter(obs)
    np.testing.assert_allclose(probs.sum(1), 1)
    assert probs[-1, 2] > 0.95
    # causality: probabilities at t do not change when later data are appended
    f2 = RegimeFilter(means, covs, P)
    np.testing.assert_allclose(f2.filter(obs[:20]), probs[:20])


# ------------------------------------------------------------------ signal
def test_composite_signal_false_alarm_is_bullish():
    ev = EventSignals(s_news=-0.5, s_echo=-2.3, say=-1.5, do=1.8, absorption=1.6, levy_area=0.8, say_do_cov_sign=-1)
    c = ev.contributions()
    assert c["echo"] > 0 and c["say"] > 0 and c["do"] > 0 and ev.score() > 0
    assert c["say"] + c["do"] == pytest.approx(-say_do_gap(-1.5, 1.8))   # = -G in manipulative regimes
    follow = EventSignals(s_news=-2.0, s_echo=-0.2, say=-1.5, do=-1.2, absorption=-0.3, levy_area=-0.2, say_do_cov_sign=+1)
    assert follow.score() < 0
    with pytest.raises(ValueError):
        SignalWeights(news=-1)


def test_say_do_switch(rng):
    v = rng.normal(size=300)
    assert np.nanmean(say_do_switch(-v, v, 50)) == pytest.approx(-1)
    assert np.nanmean(say_do_switch(v, v + 0.1 * rng.normal(size=300), 50)) == pytest.approx(1)
