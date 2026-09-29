"""Echo: separating news from repetition with Hawkes processes (paper Section 5).

* Proposition 5.4 -- the long-run echo share equals the branching ratio n = alpha / beta.
* Theorem 5.7     -- semantic declustering: exact, factorised parent posteriors that
  combine timing (exponential Hawkes kernel) with meaning (von Mises--Fisher marks
  on the unit sphere).  Timing-only stochastic declustering (Zhuang et al., 2002)
  is the special case ``kappa = 0``.

Everything is causal: the posterior parentage of article j depends only on
articles published before it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, ive, logsumexp

__all__ = [
    "HawkesParams",
    "hawkes_intensity_at_events",
    "hawkes_loglik",
    "fit_hawkes",
    "simulate_hawkes",
    "sample_vmf",
    "log_vmf_normaliser",
    "log_uniform_sphere",
    "news_probability",
    "semantic_declustering",
    "DeclusteringResult",
    "split_sentiment",
]


@dataclass(frozen=True)
class HawkesParams:
    """Exponential Hawkes process lambda(t) = mu + sum alpha exp(-beta (t - tau_l))."""

    mu: float
    alpha: float
    beta: float

    @property
    def branching_ratio(self) -> float:
        """n = alpha / beta: expected echoes per article and long-run echo share."""
        return self.alpha / self.beta

    @property
    def mean_intensity(self) -> float:
        return self.mu / (1.0 - self.branching_ratio)


# ------------------------------------------------------------------ Hawkes
def hawkes_intensity_at_events(times: np.ndarray, p: HawkesParams) -> np.ndarray:
    """Left-limit intensity lambda(tau_j) at every event, in O(N)."""
    t = np.asarray(times, dtype=float)
    out = np.empty_like(t)
    acc = 0.0
    for j in range(t.size):
        if j > 0:
            acc = math.exp(-p.beta * (t[j] - t[j - 1])) * (acc + 1.0)
        out[j] = p.mu + p.alpha * acc
    return out


def hawkes_loglik(times: np.ndarray, T: float, p: HawkesParams) -> float:
    t = np.asarray(times, dtype=float)
    lam = hawkes_intensity_at_events(t, p)
    compensator = p.mu * T + (p.alpha / p.beta) * np.sum(1.0 - np.exp(-p.beta * (T - t)))
    return float(np.sum(np.log(lam)) - compensator)


def fit_hawkes(times: np.ndarray, T: float, init: HawkesParams | None = None) -> HawkesParams:
    """Maximum-likelihood fit with stationarity enforced (0 < n < 1) by reparametrisation."""
    t = np.sort(np.asarray(times, dtype=float))
    if init is None:
        rate = max(t.size / T, 1e-6)
        init = HawkesParams(mu=0.5 * rate, alpha=0.5, beta=1.0)
    n0 = min(max(init.branching_ratio, 1e-3), 0.999)
    x0 = np.array([math.log(init.mu), math.log(init.beta), math.log(n0 / (1 - n0))])

    def unpack(x):
        mu, beta = math.exp(x[0]), math.exp(x[1])
        n = 1.0 / (1.0 + math.exp(-x[2]))
        return HawkesParams(mu=mu, alpha=n * beta, beta=beta)

    res = minimize(lambda x: -hawkes_loglik(t, T, unpack(x)), x0, method="Nelder-Mead",
                   options=dict(maxiter=4000, xatol=1e-7, fatol=1e-9))
    return unpack(res.x)


def simulate_hawkes(p: HawkesParams, T: float, rng=None, mark_news=None, mark_echo=None):
    """Simulate by the cluster (branching) representation.

    Returns ``(times, parents, marks)`` sorted by time; ``parents[j] = -1`` for news,
    otherwise the index of the parent article.  Optional ``mark_news(rng)`` and
    ``mark_echo(parent_mark, rng)`` generate marks (e.g. embeddings).
    """
    rng = np.random.default_rng(rng)
    n = p.branching_ratio
    if n >= 1:
        raise ValueError("simulation requires a stationary process (alpha < beta)")
    k0 = rng.poisson(p.mu * T)
    times = list(rng.uniform(0, T, k0))
    parents = [-1] * k0
    marks = [mark_news(rng) if mark_news else None for _ in range(k0)]
    frontier = list(range(k0))
    while frontier:
        nxt = []
        for idx in frontier:
            for _ in range(rng.poisson(n)):
                tc = times[idx] + rng.exponential(1.0 / p.beta)
                if tc < T:
                    times.append(tc)
                    parents.append(idx)
                    marks.append(mark_echo(marks[idx], rng) if mark_echo else None)
                    nxt.append(len(times) - 1)
        frontier = nxt
    order = np.argsort(times)
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    t_sorted = np.asarray(times)[order]
    par = np.array([-1 if parents[i] < 0 else rank[parents[i]] for i in order], dtype=int)
    mk = [marks[i] for i in order]
    if mark_news is not None:
        mk = np.asarray(mk)
    return t_sorted, par, mk


# ------------------------------------------------------------ sphere densities
def log_vmf_normaliser(kappa: float, dim: int) -> float:
    """log C_D(kappa) for the von Mises--Fisher density C_D(kappa) exp(kappa <z, mu>)."""
    if kappa <= 0:
        return log_uniform_sphere(dim)
    nu = dim / 2.0 - 1.0
    return nu * math.log(kappa) - (dim / 2.0) * math.log(2 * math.pi) - (math.log(ive(nu, kappa)) + kappa)


def log_uniform_sphere(dim: int) -> float:
    """log density of the uniform law on S^{D-1}: log Gamma(D/2) - log 2 - (D/2) log pi."""
    return float(gammaln(dim / 2.0) - math.log(2.0) - (dim / 2.0) * math.log(math.pi))


def sample_vmf(mu: np.ndarray, kappa: float, rng=None) -> np.ndarray:
    """One draw from vMF(mu, kappa) on S^{D-1} (Wood, 1994)."""
    rng = np.random.default_rng(rng)
    mu = np.asarray(mu, dtype=float)
    mu = mu / np.linalg.norm(mu)
    d = mu.size
    if kappa <= 0:
        z = rng.normal(size=d)
        return z / np.linalg.norm(z)
    b = (-2 * kappa + math.sqrt(4 * kappa**2 + (d - 1) ** 2)) / (d - 1)
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d - 1) * math.log(1 - x0**2)
    while True:
        zb = rng.beta((d - 1) / 2, (d - 1) / 2)
        w = (1 - (1 + b) * zb) / (1 - (1 - b) * zb)
        if kappa * w + (d - 1) * math.log(1 - x0 * w) - c >= math.log(rng.uniform()):
            break
    v = rng.normal(size=d)
    v -= v.dot(mu) * mu
    v /= np.linalg.norm(v)
    return w * mu + math.sqrt(max(1 - w**2, 0.0)) * v


# --------------------------------------------------------------- declustering
def news_probability(times: np.ndarray, p: HawkesParams) -> np.ndarray:
    """Timing-only posterior probability that each article is news: mu / lambda(tau_j)."""
    return p.mu / hawkes_intensity_at_events(times, p)


@dataclass
class DeclusteringResult:
    p_news: np.ndarray
    """Posterior probability that each article is news (immigrant)."""
    map_parent: np.ndarray
    """Most likely parent (-1 = news)."""
    parent_probs: list
    """For each j, a vector over (news, parent_0, ..., parent_{j-1}) within the lag window."""
    candidate_parents: list
    """Indices of the candidate parents matching ``parent_probs[j][1:]``."""

    def echo_share(self) -> float:
        return float(1.0 - self.p_news.mean())


def semantic_declustering(
    times: np.ndarray,
    embeddings: np.ndarray | None,
    p: HawkesParams,
    kappa: float = 0.0,
    log_f0=None,
    max_lag: float | None = None,
) -> DeclusteringResult:
    """Exact parent posteriors of Theorem 5.7.

    Parameters
    ----------
    times:
        Sorted publication times, shape (N,).
    embeddings:
        Unit-norm embeddings, shape (N, D).  Ignored if ``kappa == 0``.
    p:
        Hawkes parameters (exponential kernel).
    kappa:
        Concentration of the vMF echo density f(z | z_parent) propto exp(kappa <z, z_parent>).
        ``kappa = 0`` reduces to timing-only stochastic declustering.
    log_f0:
        Callable returning the log density of news embeddings; default uniform on the sphere.
    max_lag:
        Only articles within this lag are considered as candidate parents (speed-up).
    """
    t = np.asarray(times, dtype=float)
    if np.any(np.diff(t) < 0):
        raise ValueError("times must be sorted")
    N = t.size
    use_marks = kappa > 0 and embeddings is not None
    if use_marks:
        Z = np.asarray(embeddings, dtype=float)
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        D = Z.shape[1]
        logC = log_vmf_normaliser(kappa, D)
        lf0 = (lambda z: np.full(z.shape[0], log_uniform_sphere(D))) if log_f0 is None else log_f0
        log_news_mark = np.asarray(lf0(Z), dtype=float)
    p_news = np.empty(N)
    map_parent = np.full(N, -1, dtype=int)
    probs, cands = [], []
    lo = 0
    for j in range(N):
        if max_lag is not None:
            while lo < j and t[j] - t[lo] > max_lag:
                lo += 1
        idx = np.arange(lo, j)
        lw_news = math.log(p.mu) + (log_news_mark[j] if use_marks else 0.0)
        lw = math.log(p.alpha) - p.beta * (t[j] - t[idx])
        if use_marks:
            lw = lw + logC + kappa * (Z[idx] @ Z[j])
        allw = np.concatenate([[lw_news], lw])
        post = np.exp(allw - logsumexp(allw))
        p_news[j] = post[0]
        k = int(np.argmax(post))
        map_parent[j] = -1 if k == 0 else int(idx[k - 1])
        probs.append(post)
        cands.append(idx)
    return DeclusteringResult(p_news=p_news, map_parent=map_parent, parent_probs=probs, candidate_parents=cands)


def split_sentiment(sentiment: np.ndarray, p_news: np.ndarray) -> tuple[float, float]:
    """MMSE split of total sentiment into news and echo components (Theorem 5.7)."""
    s = np.asarray(sentiment, dtype=float)
    pn = np.asarray(p_news, dtype=float)
    return float(np.sum(pn * s)), float(np.sum((1.0 - pn) * s))
