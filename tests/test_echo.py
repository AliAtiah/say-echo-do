import itertools

import numpy as np
import pytest

from say_echo_do.echo import (
    HawkesParams, fit_hawkes, hawkes_intensity_at_events, log_vmf_normaliser, news_probability,
    sample_vmf, semantic_declustering, simulate_hawkes, split_sentiment,
)


def brute_force_posterior(times, Z, p, kappa):
    """Enumerate every labelling and normalise the joint density (Theorem 5.7 check)."""
    N, D = Z.shape
    logC = log_vmf_normaliser(kappa, D)
    from say_echo_do.echo import log_uniform_sphere
    lf0 = log_uniform_sphere(D)

    def w(k, j):
        if k == 0:
            return np.exp(np.log(p.mu) + lf0)
        l = k - 1
        return np.exp(np.log(p.alpha) - p.beta * (times[j] - times[l]) + logC + kappa * Z[j] @ Z[l])

    post, tot = {}, 0.0
    for lab in itertools.product(*[range(j + 1) for j in range(N)]):
        val = np.prod([w(lab[j], j) for j in range(N)])
        post[lab] = val
        tot += val
    return {k: v / tot for k, v in post.items()}


def test_semantic_declustering_is_exact_and_factorises(rng):
    p = HawkesParams(mu=0.8, alpha=1.2, beta=1.3)
    times = np.array([0.2, 0.5, 0.9, 1.2, 1.6])
    Z = rng.normal(size=(5, 3))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    kappa = 4.0
    brute = brute_force_posterior(times, Z, p, kappa)
    res = semantic_declustering(times, Z, p, kappa=kappa)
    for lab, prob in brute.items():
        prod = np.prod([res.parent_probs[j][lab[j]] for j in range(5)])
        assert prob == pytest.approx(prod, abs=1e-14)


def test_timing_only_reduces_to_stochastic_declustering(rng):
    p = HawkesParams(mu=0.5, alpha=0.8, beta=1.6)
    times, _, _ = simulate_hawkes(p, 200, rng)
    res = semantic_declustering(times, None, p, kappa=0.0)
    np.testing.assert_allclose(res.p_news, news_probability(times, p))


def test_intensity_recursion_matches_direct_sum(rng):
    p = HawkesParams(mu=0.3, alpha=0.7, beta=1.1)
    t = np.sort(rng.uniform(0, 50, 300))
    direct = np.array([p.mu + p.alpha * np.exp(-p.beta * (tj - t[:j])).sum() for j, tj in enumerate(t)])
    np.testing.assert_allclose(hawkes_intensity_at_events(t, p), direct, rtol=1e-12)


def test_echo_share_equals_branching_ratio(rng):  # Proposition 5.4
    p = HawkesParams(mu=1.0, alpha=0.6, beta=1.0)
    _, parents, _ = simulate_hawkes(p, 20_000, rng)
    assert (parents >= 0).mean() == pytest.approx(p.branching_ratio, abs=0.01)


def test_fit_recovers_parameters(rng):
    true = HawkesParams(mu=0.8, alpha=0.9, beta=1.5)
    times, _, _ = simulate_hawkes(true, 5000, rng)
    est = fit_hawkes(times, 5000)
    assert est.branching_ratio == pytest.approx(true.branching_ratio, abs=0.05)
    assert est.mu == pytest.approx(true.mu, rel=0.1)


def test_semantics_improves_parent_attribution(rng):
    """Adding meaning should identify true parents more often than timing alone."""
    p = HawkesParams(mu=0.6, alpha=1.0, beta=1.25)
    kappa, D = 25.0, 16
    news = lambda r: (lambda z: z / np.linalg.norm(z))(r.normal(size=D))
    echo = lambda parent, r: sample_vmf(parent, kappa, r)
    times, parents, Z = simulate_hawkes(p, 400, rng, mark_news=news, mark_echo=echo)
    timing = semantic_declustering(times, None, p, kappa=0.0)
    sem = semantic_declustering(times, Z, p, kappa=kappa)
    acc_t = (timing.map_parent == parents).mean()
    acc_s = (sem.map_parent == parents).mean()
    assert acc_s > acc_t + 0.1
    # expected posterior entropy never increases (on average)
    ent = lambda r: np.mean([-(q[q > 0] * np.log(q[q > 0])).sum() for q in r.parent_probs])
    assert ent(sem) <= ent(timing)


def test_vmf_sampler_mean_resultant(rng):
    mu = np.eye(5)[0]
    kappa = 10.0
    draws = np.array([sample_vmf(mu, kappa, rng) for _ in range(4000)])
    from scipy.special import ive
    A = ive(5 / 2, kappa) / ive(5 / 2 - 1, kappa)  # E<z, mu> for vMF on S^4
    assert (draws @ mu).mean() == pytest.approx(A, abs=0.01)


def test_split_sentiment():
    s_new, s_echo = split_sentiment([1.0, -2.0, 3.0], [1.0, 0.25, 0.5])
    assert s_new == pytest.approx(1.0 - 0.5 + 1.5)
    assert s_echo == pytest.approx(0.0 - 1.5 + 1.5)
