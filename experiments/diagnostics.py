"""Diagnostics behind the interpretation of the controlled experiments (Section 12).

For each main-market seed this script reports:

* the oracle IC, from forecasting with the true mispricing v - p: the ceiling for any forecast;
* where absorption's reversal comes from, by comparing the estimated absorption with
  the true one, lambda * (x + u), and with a narrative model in which the statement's
  price impact scales with the size of its echo cascade;
* the Say-in-the-fade-group test with echo sentiment alone as a control, and
  Do's loading with Say removed.

Usage:  python experiments/diagnostics.py  ->  results/sim/diagnostics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import absorption, build_features, cluster_ols, fit_kappa, fit_pooled_hawkes, split  # noqa: E402

from say_echo_do.simulation import SimConfig, simulate_market  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "sim"


def one(seed):
    cfg = SimConfig(seed=seed)
    sim = simulate_market(cfg)
    F, E = sim["say"].shape
    tr, te = split(E)
    train_arts = [sim["articles"][f][e] for f in range(F) for e in range(E) if tr[e]]
    params = fit_pooled_hawkes([a["times"] for a in train_arts], cfg.window)
    rng = np.random.default_rng(seed)
    sub = [train_arts[i] for i in rng.choice(len(train_arts), 600, replace=False)]
    feat = build_features(sim, params, fit_kappa(sub, params, cfg.dim))
    A = absorption(sim, feat, tr)

    firms = sim["firms"]
    lam = np.array([f.lam for f in firms])[:, None]
    phi0 = np.array([f.phi0 for f in firms])
    narr = np.zeros((F, E))
    cascade = np.zeros((F, E))
    for f in range(F):
        for e in range(E):
            a = sim["articles"][f][e]
            narr[f, e] = (np.where(a["kind"] == "say", phi0[f], cfg.phi_news) * a["sent"]).sum()
            cascade[f, e] = (a["kind"] == "say").sum()          # statement plus its echoes
    p, r, v = sim["p"], sim["r"], sim["v"]
    A_true = p - narr                                            # = lambda * (x + u)

    firm = np.repeat(np.arange(F)[:, None], E, 1)[:, te].ravel()
    y = r[:, te].ravel()
    sd = lambda z: z[:, te].ravel() / z[:, tr].std()
    ctrl = np.c_[sd(sim["say"]), sd(sim["do"]), sd(feat["s_new"]), sd(feat["s_echo"])]
    trades = np.repeat((sim["regime"] != "honest")[:, None], E, 1)[:, te].ravel()
    out = {}

    out["oracle_ic"] = float(np.mean([np.corrcoef(v[:, e] - p[:, e], r[:, e])[0, 1] for e in np.where(te)[0]]))

    tv = lambda z: z[:, te].var()
    out["absorption_variance_share"] = dict(informed=float(tv(lam * sim["x"]) / tv(A)),
                                            noise=float(tv(A_true - lam * sim["x"]) / tv(A)),
                                            narrative_misfit=float(tv(A_true - A) / tv(A)))
    for name, Z in (("estimated", A), ("true", A_true)):
        b, t = cluster_ols(y, np.c_[sd(Z) * trades, sd(Z) * ~trades, ctrl], firm)
        out[f"absorption_{name}"] = dict(trades_coef=float(b[0]), trades_t=float(t[0]),
                                         no_trade_coef=float(b[1]), no_trade_t=float(t[1]))
    # narrative model with the statement's impact scaling with its cascade size
    X = np.stack([sim["say"], feat["tone"], feat["s_new"], feat["s_echo"], sim["say"] * cascade], -1)
    c = np.linalg.lstsq(np.c_[X[:, tr].reshape(-1, 5), np.ones(tr.sum() * F)], p[:, tr].ravel(), rcond=None)[0]
    A_cas = p - (np.c_[X.reshape(-1, 5), np.ones(F * E)] @ c).reshape(F, E)
    b, t = cluster_ols(y, np.c_[sd(A_cas), ctrl], firm)
    b0, t0 = cluster_ols(y, np.c_[sd(A), ctrl], firm)
    out["absorption_all_t"] = dict(linear_narrative=float(t0[0]), cascade_scaled_narrative=float(t[0]))

    # Say in the fade group: which control removes it?
    neg = feat["switch"][:, te].ravel() < 0
    say = sd(sim["say"])
    for name, extra in (("do_only", []), ("do_echo", [sd(feat["s_echo"])]), ("do_news", [sd(feat["s_new"])]),
                        ("do_news_echo", [sd(feat["s_new"]), sd(feat["s_echo"])])):
        b, t = cluster_ols(y, np.column_stack([say * neg, say * ~neg, sd(sim["do"])] + extra), firm)
        out[f"say_fade_t_{name}"] = float(t[0])
    b, t = cluster_ols(y, np.c_[sd(sim["do"]), sd(feat["s_new"]), sd(feat["s_echo"])], firm)
    out["do_t_without_say"] = float(t[0])
    # Say's loading by regime (no controls)
    reg = sim["regime"]
    out["say_t_by_regime"] = {}
    for g in ("honest", "shading", "false_alarm", "exaggeration"):
        m = np.repeat((reg == g)[:, None], E, 1)[:, te].ravel()
        b, t = cluster_ols(y[m], np.c_[say[m]], firm[m])
        out["say_t_by_regime"][g] = float(t[0])
    return out


def main(seeds=range(5)):
    res = {}
    for s in seeds:
        res[str(s)] = one(s)
        print(s, json.dumps(res[str(s)]), flush=True)
    keys = [k for k, v in res["0"].items() if isinstance(v, float)]
    res["mean"] = {k: float(np.mean([res[str(s)][k] for s in seeds])) for k in keys}
    (OUT / "diagnostics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["mean"], indent=1))


if __name__ == "__main__":
    main()
