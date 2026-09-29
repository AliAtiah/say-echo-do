import numpy as np
import pytest

from say_echo_do.embeddings import (
    analog_forecast, certified_fraction, racl_loss, soft_targets, surprise, topic_orthogonal,
)


def test_loss_lower_bound_and_kl_decomposition(rng):  # Theorem 6.1 (inequality)
    rho = rng.normal(size=(30, 3))
    Z = rng.normal(size=(30, 8))
    res = racl_loss(Z, rho, tau=0.1, b=1.0)
    assert res.loss >= res.entropy_bound
    assert res.loss - res.entropy_bound == pytest.approx(res.excess_kl.sum())
    assert np.all(res.excess_kl >= -1e-12)


def test_isometric_embedding_attains_the_bound(rng):  # Theorem 6.1 (equality)
    # Outcomes on a sphere of radius r with tau r^2 / b^2 = 1 are exactly realisable by z = rho / r.
    tau, b, r = 0.2, 1.0, np.sqrt(1.0 / 0.2)
    rho = rng.normal(size=(25, 4))
    rho = r * rho / np.linalg.norm(rho, axis=1, keepdims=True)
    res = racl_loss(rho / r, rho, tau=tau, b=b)
    assert res.loss == pytest.approx(res.entropy_bound, rel=1e-10)
    assert np.abs(res.excess_kl).max() < 1e-10
    # squared embedding distances are affine in squared outcome distances
    Z = rho / r
    dz = ((Z[:, None] - Z[None]) ** 2).sum(-1)
    dr = ((rho[:, None] - rho[None]) ** 2).sum(-1)
    iu = np.triu_indices(25, 1)
    slope = np.polyfit(dr[iu], dz[iu], 1)[0]
    assert slope == pytest.approx(tau / b**2)


def test_certificate_never_wrong_and_tightens_with_training(rng):  # Theorem 6.3
    tau, b, r = 0.2, 1.0, np.sqrt(5.0)
    rho = rng.normal(size=(20, 3))
    rho = r * rho / np.linalg.norm(rho, axis=1, keepdims=True)
    exact = rho / r                                   # the exact minimiser
    fractions = []
    for noise in (0.0, 0.02, 0.1, 0.5):
        out = certified_fraction(exact + noise * rng.normal(size=exact.shape), rho, tau, b)
        assert out["violations"] == 0                 # certified pairs are always correctly ordered
        fractions.append(out["fraction"])
    assert fractions[0] == pytest.approx(1.0)         # zero excess loss certifies every strict pair
    assert fractions == sorted(fractions, reverse=True)  # less training -> fewer guarantees


def test_topic_orthogonal_is_invariant_to_topic_shifts(rng):
    U = rng.normal(size=(10, 3))
    w = rng.normal(size=10)
    wp = topic_orthogonal(w, U)
    z = rng.normal(size=10)
    t = rng.normal(size=3)
    assert wp @ (z + U @ t) == pytest.approx(wp @ z)
    assert np.abs(U.T @ wp).max() < 1e-10


def test_surprise_is_chi_square_calibrated(rng):
    D = 6
    A = rng.normal(size=(D, D))
    Sigma = A @ A.T + np.eye(D)
    L = np.linalg.cholesky(Sigma)
    w = rng.normal(size=D)
    S = [surprise(L @ rng.normal(size=D), np.zeros(D), Sigma, w).mahalanobis for _ in range(20_000)]
    assert np.mean(S) == pytest.approx(D, rel=0.03)
    assert np.var(S) == pytest.approx(2 * D, rel=0.08)


def test_analog_forecast_recovers_local_mean(rng):
    Zh = rng.normal(size=(2000, 5))
    Zh /= np.linalg.norm(Zh, axis=1, keepdims=True)
    mu = lambda Z: -6.0 * Z[:, 0]  # stories aligned with axis 0 are "alarming"
    ar = mu(Zh) + rng.normal(0, 1.0, 2000)
    z = np.eye(5)[0]
    f = analog_forecast(z, Zh, ar, h=0.05)
    assert f.mean == pytest.approx(-6.0, abs=0.6)
    assert f.n_eff > 20


def test_exact_certificate_is_sound_and_dominates_pinsker(rng):  # Theorem 6.3 (ii)
    from say_echo_do.embeddings import certify_anchor
    from scipy.special import rel_entr
    for _ in range(500):
        n = int(rng.integers(3, 25))
        q = rng.dirichlet(np.ones(n) * rng.uniform(0.2, 3))
        p = q * np.exp(rng.normal(0, rng.uniform(0.01, 1.0), n))
        p /= p.sum()
        eps = rel_entr(q, p).sum()
        exact, pins = certify_anchor(q, eps), certify_anchor(q, eps, method="pinsker")
        assert not (exact & ~(p[:, None] > p[None, :])).any()   # never wrong
        assert not (pins & ~exact).any()                        # certifies everything Pinsker does


def test_exact_certificate_is_tight(rng):
    from say_echo_do.embeddings import certify_anchor
    from scipy.special import rel_entr
    q = rng.dirichlet(np.ones(8))
    k, l = np.argsort(-q)[:2]
    p = q.copy()
    p[k] = p[l] = (q[k] + q[l]) / 2                             # a tie within the bound
    eps = rel_entr(q, p).sum()
    assert not certify_anchor(q, eps * (1 + 1e-9))[k, l]
    assert certify_anchor(q, eps * (1 - 1e-6))[k, l]
