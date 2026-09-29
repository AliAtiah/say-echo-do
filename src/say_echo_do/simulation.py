"""Agent-based market with three voices, used for the controlled experiments.

Each firm hosts one institution with fixed parameters.  Institutions are either
*honest* (they report value and do not trade) or *strategic* (Section 4): they
choose a message and a trade given crowd credulity ``phi``, price impact ``lam``
and misreporting cost ``k``.  Strategic firms are drawn inside the shading,
false-alarm or exaggeration regions of Corollary 4.4.

For every event:

1. value ``v`` is drawn (Gaussian, or Student-t for robustness tests);
2. the institution speaks (``say`` = message + noise) and, if strategic, trades
   during the ten days *before* speaking;
3. the media publish a Hawkes cascade: the institution's statement and
   independent news items arrive as immigrants; echoes copy their parent's
   sentiment and embedding (von Mises--Fisher around the parent);
4. the crowd prices every article (per-article impact ``phi0`` for statements
   and their echoes, ``phi_news`` for news and theirs), informed and noise
   trades add impact, and the forward return is ``v - p`` plus noise.

The simulator records only what an econometrician could observe (article times,
embeddings and sentiment; noisy positioning; the price reaction; daily paths)
together with the latent truth needed for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .echo import sample_vmf

__all__ = ["SimConfig", "Firm", "simulate_market"]

REGIMES = ("honest", "shading", "false_alarm", "exaggeration")


@dataclass
class SimConfig:
    n_firms: int = 300
    n_events: int = 60
    dim: int = 16                 # embedding dimension
    kappa: float = 30.0           # echo concentration (vMF)
    beta: float = 1.5             # Hawkes decay (per day)
    window: float = 10.0          # article window after the statement (days)
    news_rate: float = 0.4        # independent news immigrants per day
    phi_news: float = 0.04        # per-article price response to news and its echoes
    sigma_say: float = 0.3        # noise in the published statement
    sigma_news: float = 1.5       # noise in each independent news article
    sigma_article: float = 0.15   # noise added when an article is echoed
    sigma_x: float = 0.5          # noise in observed positioning
    disclosed: float = 1.0        # share of the trade disclosed by decision time
    sigma_u: float = 0.4          # noise-trader demand
    sigma_eta: float = 10.0       # later information: calibrated so a raw linear model reaches IC ~0.17
    value_dist: str = "gauss"     # "gauss" or "t3"
    crowd: str = "linear"         # "linear" or "tanh" (nonlinear crowd)
    kernel: str = "exp"           # "exp" or "powerlaw" (misspecified echo timing)
    regime_mix: tuple = (0.25, 0.25, 0.25, 0.25)
    pre_days: int = 10            # days of trading before the statement
    seed: int = 0


@dataclass
class Firm:
    regime: str
    phi0: float
    n: float          # media branching ratio
    phi: float        # effective credulity phi0 / (1 - n)
    lam: float
    k: float
    psi: float        # speech slope
    chi: float        # trade slope


def _draw_firm(regime: str, rng: np.random.Generator) -> Firm:
    lam = rng.uniform(0.3, 1.0)
    phi0 = rng.uniform(0.2, 0.45)
    if regime == "honest":
        phi = rng.uniform(0.4, 0.95)
        n = 1 - phi0 / phi if phi > phi0 else 0.0
        return Firm(regime, phi0, max(n, 0.0), phi, lam, np.inf, 1.0, 0.0)
    if regime == "shading":
        phi = rng.uniform(0.35, 0.9)
        a = rng.uniform(phi + 0.05, 2.5)
    elif regime == "false_alarm":
        phi = rng.uniform(0.5, 0.95)
        a = rng.uniform(phi**2 + 0.2 * (phi - phi**2), phi - 0.2 * (phi - phi**2))
    else:  # exaggeration
        phi = rng.uniform(1.05, 1.5)
        a = rng.uniform(1.15 * phi**2, 2.0 * phi**2)
    phi0 = min(phi0, 0.95 * phi)
    n = 1 - phi0 / phi
    k = a / (2 * lam)
    psi = (a - phi) / (a - phi**2)
    chi = k * (1 - phi) / (a - phi**2)
    return Firm(regime, phi0, n, phi, lam, k, psi, chi)


def _delay(cfg: SimConfig, rng) -> float:
    if cfg.kernel == "exp":
        return rng.exponential(1.0 / cfg.beta)
    # Lomax (power-law tail) delays with the same mean 1/beta
    shape = 2.5
    return (rng.pareto(shape)) * (shape - 1) / cfg.beta


def _cascade(root_time, root_s, root_z, root_kind, n, cfg, rng):
    """Offspring cascade of one immigrant article."""
    out = [(root_time, root_s, root_z, root_kind, -1)]
    frontier = [0]
    while frontier:
        nxt = []
        for idx in frontier:
            t0, s0, z0, _, _ = out[idx]
            for _ in range(rng.poisson(n)):
                t = t0 + _delay(cfg, rng)
                if t < cfg.window:
                    z = sample_vmf(z0, cfg.kappa, rng)
                    out.append((t, s0 + rng.normal(0, cfg.sigma_article), z, root_kind, idx))
                    nxt.append(len(out) - 1)
        frontier = nxt
    return out


def _unit(rng, d):
    z = rng.normal(size=d)
    return z / np.linalg.norm(z)


def simulate_market(cfg: SimConfig = SimConfig()) -> dict:
    """Simulate a panel of firms x events.  Returns a dict of arrays and per-event article lists."""
    rng = np.random.default_rng(cfg.seed)
    regimes = rng.choice(REGIMES, size=cfg.n_firms, p=np.asarray(cfg.regime_mix) / sum(cfg.regime_mix))
    firms = [_draw_firm(r, rng) for r in regimes]
    F, E, T0 = cfg.n_firms, cfg.n_events, cfg.pre_days
    days = np.arange(-T0, int(cfg.window) + 1)

    keys = ["v", "say", "do", "p", "r", "x", "news_mean"]
    out = {k: np.zeros((F, E)) for k in keys}
    out["articles"] = [[None] * E for _ in range(F)]
    out["pos_path"] = np.zeros((F, E, days.size))
    out["tone_path"] = np.zeros((F, E, days.size))

    for f, firm in enumerate(firms):
        for e in range(E):
            v = rng.normal() if cfg.value_dist == "gauss" else rng.standard_t(3) / np.sqrt(3)
            m = firm.psi * v
            say = m + rng.normal(0, cfg.sigma_say)
            x = firm.chi * v
            # --- media cascade ---------------------------------------------------
            arts = _cascade(0.0, say, _unit(rng, cfg.dim), "say", firm.n, cfg, rng)
            n_news = rng.poisson(cfg.news_rate * cfg.window)
            news_sig = []
            for _ in range(n_news):
                sig = v + rng.normal(0, cfg.sigma_news)
                news_sig.append(sig)
                arts += _cascade(rng.uniform(0, cfg.window), sig, _unit(rng, cfg.dim), "news", firm.n, cfg, rng)
            # rebuild parent indices cascade by cascade (each cascade starts with parent -1)
            fixed = []
            i = 0
            while i < len(arts):
                # each cascade starts with parent == -1
                j = i + 1
                while j < len(arts) and arts[j][4] != -1:
                    j += 1
                for t, s, z, kind, par in arts[i:j]:
                    fixed.append((t, s, z, kind, -1 if par < 0 else par + i))
                i = j
            order = np.argsort([a[0] for a in fixed], kind="stable")
            rank = np.empty(len(order), dtype=int)
            rank[order] = np.arange(len(order))
            times = np.array([fixed[o][0] for o in order])
            sent = np.array([fixed[o][1] for o in order])
            Z = np.array([fixed[o][2] for o in order])
            kind = np.array([fixed[o][3] for o in order])
            parent = np.array([-1 if fixed[o][4] < 0 else rank[fixed[o][4]] for o in order])
            # --- prices -------------------------------------------------------
            w = np.where(kind == "say", firm.phi0, cfg.phi_news)
            contrib = w * sent
            if cfg.crowd == "tanh":
                contrib = w * 1.5 * np.tanh(sent / 1.5)
            u = rng.normal(0, cfg.sigma_u)
            p = contrib.sum() + firm.lam * (x + u)
            r = v - p + rng.normal(0, cfg.sigma_eta)
            # --- observed positioning and daily paths --------------------------
            x_obs = cfg.disclosed * x + rng.normal(0, cfg.sigma_x)
            daily_trade = np.zeros(days.size)
            if firm.regime != "honest":
                daily_trade[:T0] = x / T0          # trades before the statement
            daily_trade += rng.normal(0, cfg.sigma_x / np.sqrt(days.size), days.size)
            pos = np.cumsum(daily_trade)
            tone = np.array([sent[times <= d].sum() if d >= 0 else 0.0 for d in days])
            out["v"][f, e], out["say"][f, e], out["do"][f, e] = v, say, x_obs
            out["p"][f, e], out["r"][f, e], out["x"][f, e] = p, r, x
            out["news_mean"][f, e] = np.mean(news_sig) if news_sig else 0.0
            out["articles"][f][e] = dict(times=times, sent=sent, Z=Z, kind=kind, parent=parent)
            out["pos_path"][f, e] = pos
            out["tone_path"][f, e] = tone
    out["firms"] = firms
    out["regime"] = np.array([fm.regime for fm in firms])
    out["days"] = days
    out["config"] = cfg
    return out
