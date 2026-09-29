"""Causal feature pipeline and evaluation for the controlled experiments.

Everything a strategy uses at event (f, e) is computed from information available
at that time: the firm's earlier events, the current event's articles and
reaction, and parameters estimated on the training half only (events e < E/2).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from say_echo_do.echo import HawkesParams, log_uniform_sphere, log_vmf_normaliser, semantic_declustering
from say_echo_do.leadlag import levy_area


# ------------------------------------------------------------------ Hawkes fit
def _pairs(events):
    """Flatten per-event articles into point arrays and within-event pair lists."""
    pt_event, pt_time, pi, pd = [], [], [], []
    base = 0
    for k, t in enumerate(events):
        n = len(t)
        pt_event += [k] * n
        pt_time += list(t)
        if n > 1:
            ii, jj = np.tril_indices(n, -1)          # ii > jj
            pi.append(base + ii)
            pd.append(t[ii] - t[jj])
        base += n
    return (np.array(pt_time), np.concatenate(pi) if pi else np.zeros(0, int),
            np.concatenate(pd) if pd else np.zeros(0))


def fit_pooled_hawkes(events, window):
    """MLE of an exponential Hawkes process pooled over independent event windows."""
    t, pi, pd = _pairs(events)
    n_ev = len(events)

    def nll(x):
        mu, beta = math.exp(x[0]), math.exp(x[1])
        n = 1 / (1 + math.exp(-x[2]))
        alpha = n * beta
        A = np.bincount(pi, weights=np.exp(-beta * pd), minlength=t.size)
        lam = mu + alpha * A
        comp = mu * window * n_ev + n * np.sum(1 - np.exp(-beta * (window - t)))
        return -(np.log(lam).sum() - comp)

    x0 = np.array([math.log(t.size / (n_ev * window) / 2), 0.0, 0.0])
    res = minimize(nll, x0, method="Nelder-Mead", options=dict(maxiter=3000, xatol=1e-6, fatol=1e-6))
    mu, beta = math.exp(res.x[0]), math.exp(res.x[1])
    n = 1 / (1 + math.exp(-res.x[2]))
    return HawkesParams(mu=mu, alpha=n * beta, beta=beta)


def fit_kappa(arts, params, dim, grid=(1, 60)):
    """Choose the vMF concentration maximising the marked likelihood sum_j log Lambda_j."""
    lf0 = log_uniform_sphere(dim)

    def neg(logk):
        k = math.exp(logk)
        logC = log_vmf_normaliser(k, dim)
        tot = 0.0
        for a in arts:
            t, Z = a["times"], a["Z"]
            for j in range(len(t)):
                lw = [math.log(params.mu) + lf0]
                if j:
                    lw = np.concatenate([lw, math.log(params.alpha) - params.beta * (t[j] - t[:j]) + logC + k * (Z[:j] @ Z[j])])
                tot += logsumexp(lw)
        return -tot

    r = minimize_scalar(neg, bounds=(math.log(grid[0]), math.log(grid[1])), method="bounded", options=dict(xatol=1e-2))
    return math.exp(r.x)


# ------------------------------------------------------------------ features
def build_features(sim, params, kappa, W=20, corr_threshold=0.2, use_semantics=True):
    F, E = sim["say"].shape
    feat = {k: np.zeros((F, E)) for k in
            ["s_new", "s_echo", "tone", "n_art", "echo_share", "switch", "corr_sd", "levy", "levy_signed", "pos_move"]}
    for f in range(F):
        for e in range(E):
            a = sim["articles"][f][e]
            res = semantic_declustering(a["times"], a["Z"] if use_semantics else None, params,
                                        kappa=kappa if use_semantics else 0.0)
            pn = res.p_news.copy()
            s = a["sent"]
            not_stmt = np.arange(len(s)) != 0          # article 0 is the statement itself
            feat["s_new"][f, e] = np.sum(pn[not_stmt] * s[not_stmt])
            feat["s_echo"][f, e] = np.sum((1 - pn) * s)
            feat["tone"][f, e] = s.sum()
            feat["n_art"][f, e] = len(s)
            feat["echo_share"][f, e] = 1 - pn.mean()
            lo = max(0, e - W)
            if e - lo >= 5:
                c = np.corrcoef(sim["say"][f, lo:e], sim["do"][f, lo:e])[0, 1]
            else:
                c = 0.0
            feat["corr_sd"][f, e] = c
            feat["switch"][f, e] = -1.0 if c < -corr_threshold else 1.0
            X, Y = sim["pos_path"][f, e], -sim["tone_path"][f, e]
            feat["pos_move"][f, e] = X[-1] - X[0]
            feat["levy"][f, e] = levy_area(X, Y)
    feat["levy_signed"] = np.sign(feat["pos_move"]) * feat["levy"]
    return feat


def split(E):
    tr = np.zeros(E, bool)
    tr[: E // 2] = True
    return tr, ~tr


def zscore(x, tr):
    m, s = x[:, tr].mean(), x[:, tr].std()
    return (x - m) / (s if s > 0 else 1.0)


def absorption_firm(sim, feat, tr):
    """Absorption with a firm-specific crowd multiplier (the model's phi is market-specific)."""
    F, E = sim["say"].shape
    A = np.zeros((F, E))
    for f in range(F):
        X = np.c_[sim["say"][f], feat["s_new"][f], feat["s_echo"][f], np.ones(E)]
        coef = np.linalg.lstsq(X[tr], sim["p"][f, tr], rcond=None)[0]
        A[f] = sim["p"][f] - X @ coef
    return A


def absorption(sim, feat, tr):
    """Price reaction net of the narrative-implied move, phi estimated on training events."""
    X = np.stack([sim["say"], feat["tone"], feat["s_new"], feat["s_echo"]], -1)
    Xtr = X[:, tr].reshape(-1, X.shape[-1])
    ptr = sim["p"][:, tr].ravel()
    coef = np.linalg.lstsq(np.c_[Xtr, np.ones(len(Xtr))], ptr, rcond=None)[0]
    implied = np.c_[X.reshape(-1, X.shape[-1]), np.ones(X[..., 0].size)] @ coef
    return sim["p"] - implied.reshape(sim["p"].shape)


# ------------------------------------------------------------------ strategies
THEORY_TERMS = ("news", "echo", "do", "say", "absorption", "levy")


def theory_score(Z, drop=None, no_switch=False):
    terms = {
        "news": Z["s_new"],
        "echo": -Z["s_echo"],
        "do": Z["do"],
        "say": (0.0 if no_switch else -1.0) * (Z["switch"] < 0) * Z["say"],
        "absorption": Z["A"],
        "levy": Z["levy_signed"],
    }
    return sum(v for k, v in terms.items() if k != drop)


def design(sim, feat, A, raw_only=False):
    cols = [sim["say"], sim["do"], feat["tone"], sim["p"], feat["n_art"]]
    if not raw_only:
        cols += [feat["s_new"], feat["s_echo"], A, feat["levy_signed"], feat["switch"] * sim["say"], feat["corr_sd"],
                 feat["echo_share"]]
    return np.stack(cols, -1)


def all_scores(sim, feat, seed=0):
    F, E = sim["say"].shape
    tr, te = split(E)
    A = absorption(sim, feat, tr)
    Af = absorption_firm(sim, feat, tr)
    Z = {k: zscore(v, tr) for k, v in
         dict(s_new=feat["s_new"], s_echo=feat["s_echo"], do=sim["do"], say=sim["say"], A=A,
              levy_signed=feat["levy_signed"], tone=feat["tone"], p=sim["p"]).items()}
    Z["switch"] = feat["switch"]
    Z["A_firm"] = zscore(Af, tr)
    sc = {
        "Naive sentiment (follow tone)": Z["tone"],
        "Naive contrarian (fade tone)": -Z["tone"],
        "Price reversal (fade reaction)": -Z["p"],
        "Follow institutions (Say)": Z["say"],
        "Follow positioning (Do)": Z["do"],
        "News/echo split only": Z["s_new"] - Z["s_echo"],
        "Theory composite (unit weights)": theory_score(Z),
        "Theory composite, firm-level absorption": theory_score({**Z, "A": Z["A_firm"]}),
    }
    y = sim["r"]
    for name, raw in (("Linear, raw voices", True), ("Linear, raw + theory features", False)):
        X = design(sim, feat, A, raw_only=raw)
        b = np.linalg.lstsq(np.c_[X[:, tr].reshape(-1, X.shape[-1]), np.ones(tr.sum() * F)], y[:, tr].ravel(), rcond=None)[0]
        sc[name] = (np.c_[X.reshape(-1, X.shape[-1]), np.ones(F * E)] @ b).reshape(F, E)
    for name, raw in (("Gradient boosting, raw voices", True), ("Gradient boosting, raw + theory", False)):
        X = design(sim, feat, A, raw_only=raw)
        gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed)
        gb.fit(X[:, tr].reshape(-1, X.shape[-1]), y[:, tr].ravel())
        sc[name] = gb.predict(X.reshape(-1, X.shape[-1])).reshape(F, E)
    return sc, Z, A


def evaluate(score, r, te):
    """Cross-sectional IC per test period, hit rate and long-short Sharpe per period."""
    ics, pnl = [], []
    for e in np.where(te)[0]:
        s, y = score[:, e], r[:, e]
        if s.std() == 0:
            ics.append(0.0)
            pnl.append(0.0)
            continue
        ics.append(np.corrcoef(s, y)[0, 1])
        z = (s - s.mean()) / s.std()
        w = z / np.abs(z).sum()
        pnl.append(w @ y)
    ics, pnl = np.array(ics), np.array(pnl)
    hit = np.mean(np.sign(score[:, te]) == np.sign(r[:, te]))
    return dict(ic=float(ics.mean()), ic_t=float(ics.mean() / ics.std() * np.sqrt(len(ics))),
                hit=float(hit), sharpe=float(pnl.mean() / pnl.std()))


def detection_auc(sim, feat, A, seed=0):
    """Out-of-sample AUC for flagging false-alarm events from observable features."""
    F, E = sim["say"].shape
    tr, te = split(E)
    lab = np.repeat((sim["regime"] == "false_alarm")[:, None], E, 1)
    X = np.stack([feat["corr_sd"], feat["levy"], A, feat["echo_share"], sim["say"] * sim["do"]], -1)
    mu, sd = X[:, tr].reshape(-1, X.shape[-1]).mean(0), X[:, tr].reshape(-1, X.shape[-1]).std(0)
    X = (X - mu) / sd
    clf = LogisticRegression(max_iter=1000).fit(X[:, tr].reshape(-1, X.shape[-1]), lab[:, tr].ravel())
    p = clf.predict_proba(X[:, te].reshape(-1, X.shape[-1]))[:, 1]
    out = {"all features": roc_auc_score(lab[:, te].ravel(), p)}
    names = ["Say-Do correlation (rolling)", "Levy area", "absorption", "echo share", "Say x Do"]
    for i, nm in enumerate(names):
        s = X[:, te, i].ravel()
        auc = roc_auc_score(lab[:, te].ravel(), s)
        out[nm] = max(auc, 1 - auc)
    return out


# ------------------------------------------------------------------ theory tests
def cluster_ols(y, X, groups):
    """OLS with firm-clustered standard errors."""
    X = np.c_[X, np.ones(len(X))]
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(groups):
        m = groups == g
        sg = X[m].T @ u[m]
        meat += np.outer(sg, sg)
    G = len(np.unique(groups))
    V = XtX_inv @ meat @ XtX_inv * G / (G - 1)
    return b[:-1], b[:-1] / np.sqrt(np.diag(V)[:-1])


def theory_tests(sim, feat, A):
    """The regressions a real-data study would run, on the test half only."""
    F, E = sim["say"].shape
    tr, te = split(E)
    firm = np.repeat(np.arange(F)[:, None], E, 1)[:, te].ravel()
    y = sim["r"][:, te].ravel()
    sd = lambda x: x[:, te].ravel() / x[:, tr].std()
    out = {}
    # T1: news vs echo (controls: Say, Do)
    b, t = cluster_ols(y, np.c_[sd(feat["s_new"]), sd(feat["s_echo"]), sd(sim["say"]), sd(sim["do"])], firm)
    out["T1 news"] = (b[0], t[0]); out["T1 echo"] = (b[1], t[1])
    # T2: words by Say-Do switch group
    neg = (feat["switch"][:, te].ravel() < 0)
    say = sd(sim["say"])
    b, t = cluster_ols(y, np.c_[say * neg, say * ~neg, sd(sim["do"]), sd(feat["s_new"]), sd(feat["s_echo"])], firm)
    out["T2 Say | Cov<0"] = (b[0], t[0]); out["T2 Say | Cov>=0"] = (b[1], t[1]); out["T2 Do"] = (b[2], t[2])
    # T3: absorption, overall and where the institution trades vs not
    Ax = sd(A)
    ctrl = np.c_[sd(sim["say"]), sd(sim["do"]), sd(feat["s_new"]), sd(feat["s_echo"])]
    b, t = cluster_ols(y, np.c_[Ax, ctrl], firm)
    out["T3 absorption"] = (b[0], t[0])
    trades = np.repeat((sim["regime"] != "honest")[:, None], E, 1)[:, te].ravel()
    b, t = cluster_ols(y, np.c_[Ax * trades, Ax * ~trades, ctrl], firm)
    out["T3 absorption | informed trading"] = (b[0], t[0]); out["T3 absorption | no informed trading"] = (b[1], t[1])
    Af = sd(absorption_firm(sim, feat, tr))
    b, t = cluster_ols(y, np.c_[Af, ctrl], firm)
    out["T3 absorption, firm-level multiplier"] = (b[0], t[0])
    b, t = cluster_ols(y, np.c_[Af * trades, Af * ~trades, ctrl], firm)
    out["T3 firm-level | informed trading"] = (b[0], t[0]); out["T3 firm-level | no informed trading"] = (b[1], t[1])
    # T4: who moved first
    b, t = cluster_ols(y, np.c_[sd(feat["levy_signed"]), ctrl], firm)
    out["T4 signed Levy area"] = (b[0], t[0])
    return {k: (float(v[0]), float(v[1])) for k, v in out.items()}


def run_one(cfg, W=20, use_semantics=True, kappa_fit_events=600):
    from say_echo_do.simulation import simulate_market
    sim = simulate_market(cfg)
    F, E = sim["say"].shape
    tr, te = split(E)
    train_arts = [sim["articles"][f][e] for f in range(F) for e in range(E) if tr[e]]
    params = fit_pooled_hawkes([a["times"] for a in train_arts], cfg.window)
    rng = np.random.default_rng(cfg.seed)
    sub = [train_arts[i] for i in rng.choice(len(train_arts), min(kappa_fit_events, len(train_arts)), replace=False)]
    kap = fit_kappa(sub, params, cfg.dim) if use_semantics else 0.0
    feat = build_features(sim, params, kap, W=W, use_semantics=use_semantics)
    sc, Z, A = all_scores(sim, feat, seed=cfg.seed)
    perf = {k: evaluate(v, sim["r"], te) for k, v in sc.items()}
    abl = {}
    for term in THEORY_TERMS:
        abl[term] = evaluate(theory_score(Z, drop=term), sim["r"], te)["ic"]
    abl["none"] = perf["Theory composite (unit weights)"]["ic"]
    true_echo = float(np.mean([(a["parent"] >= 0).mean() for row in sim["articles"] for a in row]))
    return dict(perf=perf, ablation=abl, detection=detection_auc(sim, feat, A, seed=cfg.seed),
                tests=theory_tests(sim, feat, A),
                hawkes=dict(mu=params.mu, alpha=params.alpha, beta=params.beta, kappa=kap),
                echo_share=dict(true=true_echo, estimated=float(feat["echo_share"].mean())),
                _sim=sim, _feat=feat)
