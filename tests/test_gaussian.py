import numpy as np
import pytest

from say_echo_do import ThreeVoiceMarket, kyle_intensities
from conftest import ols

N = 600_000


def test_follow_or_fade_loadings_match_regression(market, rng):  # Theorem 3.4
    d = market.simulate(N, rng)
    b = ols(d["y"], d["r"])
    np.testing.assert_allclose(b, market.return_loadings(), atol=4e-3)
    resid = d["r"] - d["y"] @ b
    assert resid.var() == pytest.approx(market.return_variance(), rel=1e-2)


def test_echo_has_zero_bayes_weight(market):  # Corollary 3.5
    w = market.bayes_weights
    assert abs(w[2]) < 1e-12
    assert market.return_loadings()[2] == pytest.approx(-market.kappa * market.phi[2])
    assert market.follow_or_fade()[2] == "fade"


def test_news_weight_in_unit_interval(market):  # Corollary 3.6
    assert 0 < market.bayes_weights[1] < 1
    assert market.follow_or_fade()[1] == "follow"


def test_say_do_coefficients(market, rng):  # Theorem 3.7
    sx = 0.6
    d = market.simulate(N, rng, sigma_x=sx)
    b = ols(np.c_[d["y"], d["x_hat"]], d["r"])
    c = market.say_do(sx)
    np.testing.assert_allclose(b[:3], c.y_loadings, atol=4e-3)
    assert b[3] == pytest.approx(c.c_x, abs=4e-3)
    # forecast depends on (m, x_hat) through -c_x (gamma m - x_hat)
    assert -c.c_x * c.gamma == pytest.approx(c.y_loadings[0], rel=1e-10)
    assert c.variance < market.return_variance()


def test_absorption_coefficient(market, rng):  # Theorem 3.8
    d = market.simulate(N, rng)
    b = ols(np.c_[d["y"], d["a"]], d["r"])
    assert b[3] == pytest.approx(market.absorption_coefficient(), abs=5e-3)


def test_absorption_limits_and_kyle():
    base = dict(sigma_v=1.0, h=[1.0], Sigma_nu=[[1.0]], phi=[0.3])
    s2 = ThreeVoiceMarket(beta=1, lam=1, sigma_u=1, **base).residual_value_variance
    beta, lam = kyle_intensities(np.sqrt(s2), 0.7)
    kyle = ThreeVoiceMarket(beta=beta, lam=lam, sigma_u=0.7, **base)
    assert kyle.absorption_coefficient() == pytest.approx(0.5)
    tiny_noise = ThreeVoiceMarket(beta=2.0, lam=0.5, sigma_u=1e-9, **base)
    assert tiny_noise.absorption_coefficient() == pytest.approx(1 / (0.5 * 2.0), rel=1e-6)
    no_informed = ThreeVoiceMarket(beta=1e-9, lam=0.5, sigma_u=1.0, **base)
    assert no_informed.absorption_coefficient() == pytest.approx(-1.0, rel=1e-6)


def test_absorption_robust_to_misspecified_multiplier(market, rng):  # Lemma 3.9
    d = market.simulate(200_000, rng)
    wrong = np.array([0.1, 0.9, -0.4])
    a_hat = d["p"] - d["y"] @ wrong
    f1 = np.c_[d["y"], d["a"]] @ ols(np.c_[d["y"], d["a"]], d["r"])
    f2 = np.c_[d["y"], a_hat] @ ols(np.c_[d["y"], a_hat], d["r"])
    np.testing.assert_allclose(f1, f2, atol=1e-8)
