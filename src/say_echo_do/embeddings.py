"""Market-aligned embeddings (paper Section 6).

* Theorem 6.1 -- the return-aligned soft-contrastive loss is bounded below by the
  entropy of the outcome-similarity targets, with equality iff embedding
  similarities are an affine function of squared outcome distances.
* Theorem 6.3 -- robust neighbour preservation: the leftover per-anchor KL
  certifies which analog rankings are correct, by Pinsker (i) or by the exact
  three-cell bound (ii), which is the tightest certificate using only q and the KL.
* Topic-orthogonal sentiment, calibrated (chi-square) surprise, and the
  "deja vu" analog forecaster used to estimate the narrative-implied reaction.

The core functions are pure NumPy.  :func:`racl_loss_torch` provides a
differentiable version for fine-tuning an encoder with PyTorch (optional).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from scipy.stats import chi2

__all__ = [
    "soft_targets",
    "model_probabilities",
    "RACLResult",
    "racl_loss",
    "certify_anchor",
    "certified_fraction",
    "topic_orthogonal",
    "Surprise",
    "surprise",
    "AnalogForecast",
    "analog_forecast",
    "absorption",
    "racl_loss_torch",
]


def _normalise(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def soft_targets(rho: np.ndarray, b: float) -> np.ndarray:
    """q_{jk} propto exp(-||rho_j - rho_k||^2 / (2 b^2)), k != j, rows sum to one."""
    rho = np.asarray(rho, dtype=float)
    D = ((rho[:, None, :] - rho[None, :, :]) ** 2).sum(-1)
    logits = -D / (2 * b**2)
    np.fill_diagonal(logits, -np.inf)
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def model_probabilities(Z: np.ndarray, tau: float) -> np.ndarray:
    """p_{jk}(Z) propto exp(<z_j, z_k> / tau), k != j."""
    Z = _normalise(Z)
    logits = Z @ Z.T / tau
    np.fill_diagonal(logits, -np.inf)
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


@dataclass
class RACLResult:
    loss: float
    """Soft-InfoNCE loss  -sum_j sum_k q_jk log p_jk."""
    entropy_bound: float
    """Lower bound sum_j H(q_j.) attained only by outcome-isometric embeddings."""
    excess_kl: np.ndarray
    """Per-anchor KL(q_j. || p_j.): the certificate inputs of Theorem 6.3."""


def racl_loss(Z: np.ndarray, rho: np.ndarray, tau: float, b: float) -> RACLResult:
    q = soft_targets(rho, b)
    p = model_probabilities(Z, tau)
    mask = ~np.eye(q.shape[0], dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        ce = -np.where(mask & (q > 0), q * np.log(p), 0.0).sum(1)
        ent = -np.where(mask & (q > 0), q * np.log(q), 0.0).sum(1)
    return RACLResult(loss=float(ce.sum()), entropy_bound=float(ent.sum()), excess_kl=ce - ent)


def certify_anchor(q_row: np.ndarray, eps: float, method: str = "exact") -> np.ndarray:
    """Boolean matrix C[k, l]: embedding provably ranks k closer than l to the anchor.

    Theorem 6.3 (i), Pinsker:  q_jk - q_jl > sqrt(2 eps_j).
    Theorem 6.3 (ii), exact:   q_jk > q_jl and
        q_jk log(2 q_jk / (q_jk + q_jl)) + q_jl log(2 q_jl / (q_jk + q_jl)) > eps_j,
    i.e. no distribution within KL eps_j of q_j. ranks l at least as high as k.
    The exact test certifies every pair the Pinsker test does, and usually many more.
    Either way, a certified pair is ranked correctly by the embedding.
    """
    q_row = np.asarray(q_row, dtype=float)
    eps = max(float(eps), 0.0)
    if method == "pinsker":
        return (q_row[:, None] - q_row[None, :]) > np.sqrt(2.0 * eps)
    if method != "exact":
        raise ValueError("method must be 'exact' or 'pinsker'")
    qk, ql = q_row[:, None], q_row[None, :]
    tot = qk + ql
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(qk > 0, qk * np.log(2 * qk / tot), 0.0) + np.where(ql > 0, ql * np.log(2 * ql / tot), 0.0)
    return (qk > ql) & (v > eps)


def certified_fraction(Z: np.ndarray, rho: np.ndarray, tau: float, b: float, method: str = "exact") -> dict:
    """Fraction of ordered outcome-neighbour pairs certified by the leftover loss.

    Also verifies each certified pair against the actual embedding distances
    (``violations`` must be 0 by Theorem 6.3).
    """
    Zn = _normalise(Z)
    res = racl_loss(Zn, rho, tau, b)
    q = soft_targets(rho, b)
    N = q.shape[0]
    cert = viol = total = 0
    for j in range(N):
        others = np.array([i for i in range(N) if i != j])
        C = certify_anchor(q[j, others], res.excess_kl[j], method=method)
        d = np.linalg.norm(Zn[others] - Zn[j], axis=1)
        closer = d[:, None] < d[None, :]
        qo = q[j, others]
        total += int(((qo[:, None] - qo[None, :]) > 0).sum())
        cert += int(C.sum())
        viol += int((C & ~closer).sum())
    return {"certified": cert, "ordered_pairs": total, "fraction": cert / max(total, 1), "violations": viol,
            "mean_excess_kl": float(np.mean(res.excess_kl))}


def topic_orthogonal(w: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Project a concept direction off the span of topic directions U (columns)."""
    w = np.asarray(w, dtype=float)
    Q, _ = np.linalg.qr(np.atleast_2d(np.asarray(U, dtype=float)).reshape(w.size, -1))
    return w - Q @ (Q.T @ w)


@dataclass
class Surprise:
    mahalanobis: float
    """S = eps' Sigma^{-1} eps ~ chi^2_D under a calibrated predictive model."""
    sentiment_surprise: float
    """s = <w, eps>: how much more alarming (negative) or reassuring (positive) than expected."""
    standardised_sentiment: float
    p_value: float


def surprise(z: np.ndarray, z_hat: np.ndarray, Sigma: np.ndarray, w: np.ndarray) -> Surprise:
    eps = np.asarray(z, dtype=float) - np.asarray(z_hat, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    S = float(eps @ np.linalg.solve(Sigma, eps))
    s = float(np.asarray(w) @ eps)
    sd = float(np.sqrt(np.asarray(w) @ Sigma @ np.asarray(w)))
    return Surprise(mahalanobis=S, sentiment_surprise=s, standardised_sentiment=s / sd,
                    p_value=float(chi2.sf(S, eps.size)))


@dataclass
class AnalogForecast:
    mean: float
    std: float
    n_eff: float
    weights: np.ndarray


def analog_forecast(
    z: np.ndarray,
    Z_hist: np.ndarray,
    ar_hist: np.ndarray,
    h: float = 0.1,
    state: np.ndarray | None = None,
    state_hist: np.ndarray | None = None,
    state_bandwidth: float = 1.0,
    age: np.ndarray | None = None,
    half_life: float | None = None,
) -> AnalogForecast:
    """"Deja vu" estimate of the narrative-implied reaction (Proposition 6.9).

    Weights combine a vMF kernel on story similarity, a Gaussian kernel on market
    state, and exponential recency.  Only pass events whose outcome window has closed.
    """
    z = np.asarray(z, dtype=float) / np.linalg.norm(z)
    Zh = _normalise(Z_hist)
    logw = Zh @ z / h
    if state is not None and state_hist is not None:
        ds = ((np.asarray(state_hist) - np.asarray(state)) ** 2).sum(-1)
        logw = logw - ds / (2 * state_bandwidth**2)
    if age is not None and half_life is not None:
        logw = logw - np.log(2) * np.asarray(age) / half_life
    w = np.exp(logw - logsumexp(logw))
    ar = np.asarray(ar_hist, dtype=float)
    mean = float(w @ ar)
    std = float(np.sqrt(max(w @ (ar - mean) ** 2, 0.0)))
    return AnalogForecast(mean=mean, std=std, n_eff=float(1.0 / (w**2).sum()), weights=w)


def absorption(actual_reaction: float, expected_reaction: float, scale: float) -> float:
    """Standardised interpretation gap A = (actual - expected) / scale (Theorem 3.8)."""
    return (actual_reaction - expected_reaction) / scale


def racl_loss_torch(Z, rho, tau: float, b: float):
    """Differentiable return-aligned soft-InfoNCE loss (requires PyTorch).

    ``Z``: (N, D) embeddings from an encoder; ``rho``: (N, M) post-event response vectors.
    """
    import torch  # optional dependency
    import torch.nn.functional as F

    Z = F.normalize(Z, dim=-1)
    N = Z.shape[0]
    eye = torch.eye(N, dtype=torch.bool, device=Z.device)
    D = torch.cdist(rho, rho) ** 2
    q = torch.softmax((-D / (2 * b**2)).masked_fill(eye, float("-inf")), dim=1)
    logp = torch.log_softmax((Z @ Z.T / tau).masked_fill(eye, float("-inf")), dim=1)
    return -(q * logp.masked_fill(eye, 0.0)).sum()
