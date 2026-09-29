"""Text experiment: do return-aligned embeddings organise headlines by consequence?

A synthetic corpus lets us control what matters.  Every headline mixes topic
words, tone words and one short cue that decides whether the move is *overdone*
(it reverses within a week) or *fundamental* (it continues).  Topic words are
numerous and irrelevant to outcomes, so a generic bag-of-words similarity groups
headlines by topic; the question is whether training with the paper's
return-aligned loss (Theorem 6.1) moves the geometry towards consequence, and
whether the neighbour certificate (Theorem 6.3, Pinsker and exact forms) tightens as
training proceeds.

Usage:  python experiments/text_experiment.py [seed] [outcome-noise scale] [bandwidth scale]
        -> results/text_experiment.json (seed 0, noise 1) or results/text_experiment_seed{s}_noise{x}.json
"""

from __future__ import annotations

import json
from pathlib import Path

import autograd.numpy as anp
import numpy as np
from autograd import grad
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from say_echo_do.embeddings import analog_forecast, certified_fraction

OUT = Path(__file__).resolve().parents[1] / "results"

TOPICS = {
    "earnings": "quarterly profit revenue earnings margin eps sales outlook results forecast".split(),
    "merger": "merger acquisition deal takeover bid buyout stake offer target combination".split(),
    "regulation": "regulator probe antitrust fine lawsuit ruling approval license compliance inquiry".split(),
    "macro": "rates inflation tariff demand supply currency slowdown recession stimulus exports".split(),
}
TONE = {+1: "soars jumps beats surges rallies upgrade record strong gains tops".split(),
        -1: "plunges slumps misses tumbles warns downgrade weak losses cuts falls".split()}
CUES = {"overdone": "reportedly rumor sources speculation unconfirmed chatter".split(),
        "fundamental": "filing confirmed official disclosed audited announced".split()}
FILLER = "shares stock company group firm investors market analysts week today".split()


def make_corpus(n, rng, noise=1.0):
    """noise scales the outcome noise; noise=0 gives each headline its expected consequence."""
    docs, rho, meta = [], [], []
    for _ in range(n):
        topic = rng.choice(list(TOPICS))
        tone = rng.choice([-1, 1])
        kind = rng.choice(["overdone", "fundamental"])
        words = (list(rng.choice(TOPICS[topic], 4, replace=False)) + list(rng.choice(TONE[tone], 2, replace=False))
                 + [rng.choice(CUES[kind])] + list(rng.choice(FILLER, 3, replace=False)) + [f"co{rng.integers(200)}"])
        rng.shuffle(words)
        docs.append(" ".join(words))
        day1 = 1.0 * tone + noise * rng.normal(0, 0.3)
        week = (0.8 if kind == "fundamental" else -0.8) * tone + noise * rng.normal(0, 0.3)
        div = (0.5 if kind == "overdone" else -0.2) + noise * rng.normal(0, 0.2)
        rho.append([day1, week, div])
        meta.append((topic, tone, kind))
    return docs, np.array(rho), meta


def racl_loss(W, X, rho, tau, b):
    Z = anp.dot(X, W)
    Z = Z / anp.sqrt(anp.sum(Z**2, axis=1, keepdims=True) + 1e-9)
    n = X.shape[0]
    eye = anp.eye(n)
    D = anp.sum((rho[:, None, :] - rho[None, :, :]) ** 2, axis=-1)
    lq = -D / (2 * b**2) - 1e9 * eye
    q = anp.exp(lq - anp.max(lq, axis=1, keepdims=True))
    q = q / anp.sum(q, axis=1, keepdims=True)
    ls = anp.dot(Z, Z.T) / tau - 1e9 * eye
    logp = ls - anp.max(ls, axis=1, keepdims=True)
    logp = logp - anp.log(anp.sum(anp.exp(logp), axis=1, keepdims=True))
    return -anp.sum(q * logp * (1 - eye)) / n


def embed(X, W):
    Z = X @ W
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def evaluate(Z_tr, Z_te, rho_tr, rho_te, meta_tr, meta_te, h=0.05, k=10):
    preds = np.array([analog_forecast(z, Z_tr, rho_tr[:, 1], h=h).mean for z in Z_te])
    ic = float(np.corrcoef(preds, rho_te[:, 1])[0, 1])
    S = Z_te @ Z_tr.T
    top = np.argsort(-S, axis=1)[:, :k]
    kind_tr = np.array([m[2] for m in meta_tr]); kind_te = np.array([m[2] for m in meta_te])
    top_tr = np.array([m[0] for m in meta_tr]); top_te = np.array([m[0] for m in meta_te])
    tone_tr = np.array([m[1] for m in meta_tr]); tone_te = np.array([m[1] for m in meta_te])
    same_cons = float(np.mean((kind_tr[top] == kind_te[:, None]) & (tone_tr[top] == tone_te[:, None])))
    same_topic = float(np.mean(top_tr[top] == top_te[:, None]))
    return dict(analog_ic=ic, neighbours_same_consequence=same_cons, neighbours_same_topic=same_topic)


def main(seed=0, n_train=2000, n_test=1000, dim=16, tau=0.1, steps=600, batch=256, lr=0.05, bw_scale=1.0, noise=1.0, tag=""):
    rng = np.random.default_rng(seed)
    docs_tr, rho_tr, meta_tr = make_corpus(n_train, rng, noise)
    docs_te, rho_te, meta_te = make_corpus(n_test, rng, noise)
    vec = TfidfVectorizer().fit(docs_tr)
    X_tr = vec.transform(docs_tr).toarray()
    X_te = vec.transform(docs_te).toarray()
    V = X_tr.shape[1]
    D = np.sqrt(((rho_tr[:400, None] - rho_tr[None, :400]) ** 2).sum(-1))
    b = bw_scale * float(np.median(D[np.triu_indices(400, 1)]))       # median heuristic (optionally rescaled)

    results = {}
    # generic baselines
    results["TF-IDF (generic)"] = evaluate(X_tr / np.linalg.norm(X_tr, axis=1, keepdims=True),
                                           X_te / np.linalg.norm(X_te, axis=1, keepdims=True), rho_tr, rho_te, meta_tr, meta_te)
    W0 = rng.normal(0, 1 / np.sqrt(V), (V, dim))
    results["Random projection (generic, 16-d)"] = evaluate(embed(X_tr, W0), embed(X_te, W0), rho_tr, rho_te, meta_tr, meta_te)
    # supervised baseline: predict the week-ahead move directly
    ridge = Ridge(alpha=1.0).fit(X_tr, rho_tr[:, 1])
    results["Ridge regression (supervised, one target)"] = dict(
        analog_ic=float(np.corrcoef(ridge.predict(X_te), rho_te[:, 1])[0, 1]),
        neighbours_same_consequence=None, neighbours_same_topic=None)
    gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed).fit(X_tr, rho_tr[:, 1])
    results["Gradient boosting (supervised, one target)"] = dict(
        analog_ic=float(np.corrcoef(gbm.predict(X_te), rho_te[:, 1])[0, 1]),
        neighbours_same_consequence=None, neighbours_same_topic=None)

    # return-aligned contrastive training (Adam)
    g = grad(racl_loss)
    W = W0.copy()
    m, v = np.zeros_like(W), np.zeros_like(W)
    cert_idx = rng.choice(n_train, 150, replace=False)
    trace = []
    for step in range(1, steps + 1):
        idx = rng.choice(n_train, batch, replace=False)
        gr = g(W, X_tr[idx], rho_tr[idx], tau, b)
        m = 0.9 * m + 0.1 * gr
        v = 0.999 * v + 0.001 * gr**2
        W -= lr * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
        if step in (1, 25, 50, 100, 200, 400, 600):
            cf = certified_fraction(embed(X_tr[cert_idx], W), rho_tr[cert_idx], tau, b)
            cfp = certified_fraction(embed(X_tr[cert_idx], W), rho_tr[cert_idx], tau, b, method="pinsker")
            ev = evaluate(embed(X_tr, W), embed(X_te, W), rho_tr, rho_te, meta_tr, meta_te)
            loss = float(racl_loss(W, X_tr[cert_idx], rho_tr[cert_idx], tau, b))
            trace.append(dict(step=step, loss=loss, excess_kl=cf["mean_excess_kl"], certified=cf["fraction"],
                              violations=cf["violations"], certified_pinsker=cfp["fraction"],
                              violations_pinsker=cfp["violations"], **ev))
            print(trace[-1], flush=True)
    results["Return-aligned embedding (16-d)"] = evaluate(embed(X_tr, W), embed(X_te, W), rho_tr, rho_te, meta_tr, meta_te)
    cf0 = certified_fraction(embed(X_tr[cert_idx], W0), rho_tr[cert_idx], tau, b)
    out = dict(results=results, trace=trace, certified_at_init=cf0["fraction"], bandwidth=b, bw_scale=bw_scale, noise=noise, seed=seed, vocab=V)
    OUT.mkdir(exist_ok=True)
    (OUT / f"text_experiment{tag}.json").write_text(json.dumps(out, indent=1))
    for k, r in results.items():
        print(k, r)
    return out


if __name__ == "__main__":
    import sys
    # python text_experiment.py [seed] [outcome-noise scale] [bandwidth scale]
    sd = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nz = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    bs = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    tag = "" if (sd == 0 and nz == 1.0 and bs == 1.0) else f"_seed{sd}_noise{nz:g}" + ("" if bs == 1.0 else f"_bw{bs:g}")
    main(seed=sd, noise=nz, bw_scale=bs, tag=tag)
