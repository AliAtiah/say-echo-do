"""Reproduce Table 1 of the paper: every closed form against an independent check.

Usage:  python scripts/reproduce_table1.py [--n 2000000] [--seed 0] [--out results]

Theory values are computed exactly from model parameters (no sample estimates);
simulation values come from regression on simulated data, brute-force
enumeration, or numerical optimisation.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from say_echo_do import ThreeVoiceMarket
from say_echo_do.decisions import expected_log_growth, kelly_shrinkage
from say_echo_do.echo import HawkesParams, log_uniform_sphere, log_vmf_normaliser, semantic_declustering
from say_echo_do.leadlag import levy_area_rate
from say_echo_do.strategic import StrategicInstitution, iterate_trust, two_cycle, two_cycle_multiplier


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def main(n: int, seed: int, out: Path) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    add = lambda res, q, th, sim: rows.append(dict(result=res, quantity=q, theory=th, simulation=sim))

    # ---------------------------------------------------------------- Section 3
    Sigma = np.array([[0.81, 0.0, 0.81], [0.0, 1.69, 0.0], [0.81, 0.0, 1.06]])
    mk = ThreeVoiceMarket(1.0, [1.0, 1.0, 1.0], Sigma, [0.6, 0.1, 0.3], beta=0.9, lam=0.7, sigma_u=0.8)
    d = mk.simulate(n, rng, sigma_x=0.6)
    add("Corollary 3.5", "Bayes weight on echo", mk.bayes_weights[2], ols(d["y"], d["v"])[2])
    add("Theorem 3.4", "return loading, echo channel", mk.return_loadings()[2], ols(d["y"], d["r"])[2])
    add("Theorem 3.7", "loading on positioning c_x", mk.say_do(0.6).c_x, ols(np.c_[d["y"], d["x_hat"]], d["r"])[3])
    add("Theorem 3.8", "absorption coefficient c_A", mk.absorption_coefficient(), ols(np.c_[d["y"], d["a"]], d["r"])[3])

    # ---------------------------------------------------------------- Section 4
    inst = StrategicInstitution(phi=0.6, lam=0.5, k=0.5)
    v0 = 1.0
    opt = minimize(lambda z: -inst.objective(z[0], z[1], v0), [0.1, 0.1], method="BFGS", options={"gtol": 1e-12}).x
    add("Theorem 4.3", "speech slope psi (phi=0.6, a=0.5)", inst.speech_slope, opt[1] / v0)
    add("Theorem 4.3", "trade slope chi (phi=0.6, a=0.5)", inst.trade_slope, opt[0] / v0)
    for phi, lam, k, label in [(0.7, 0.5, 0.6, "false alarm"), (1.2, 0.8, 1.0, "exaggeration")]:
        s = StrategicInstitution(phi, lam, k)
        v = rng.standard_t(3, n)
        eps = rng.laplace(0, 0.3 / np.sqrt(2), n)
        sim = s.simulate(v, eps)
        # theory side uses the identity with population moments of the t3 / Laplace laws
        sv2, se2 = 3.0, 0.3**2
        add("Theorem 4.5", f"Cov(r, Say), t3 values, {label}", s.return_say_covariance(sv2, se2),
            np.cov(sim["r"], sim["m_obs"])[0, 1])
    lo, hi = two_cycle(1.95)
    orbit = iterate_trust(1.95, 0.95, 4000)
    add("Theorem 4.7", "2-cycle point phi_+ at a=1.95", hi, float(max(orbit[-2:])))
    add("Theorem 4.7", "2-cycle point phi_- at a=1.95", lo, float(min(orbit[-2:])))
    add("Theorem 4.7", "2-cycle multiplier at a=1.8", two_cycle_multiplier(1.8), float(
        np.prod([(1.8 - 2 * 1.8 * p + p**2) / (1.8 - p) ** 2 for p in two_cycle(1.8)])))

    # ---------------------------------------------------------------- Sections 5 and 7
    p = HawkesParams(mu=0.8, alpha=1.2, beta=1.3)
    times = np.array([0.2, 0.5, 0.9, 1.2, 1.6])
    Z = rng.normal(size=(5, 3))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    kappa = 4.0
    res = semantic_declustering(times, Z, p, kappa=kappa)
    logC, lf0 = log_vmf_normaliser(kappa, 3), log_uniform_sphere(3)
    w = lambda kk, j: np.exp(np.log(p.mu) + lf0) if kk == 0 else np.exp(
        np.log(p.alpha) - p.beta * (times[j] - times[kk - 1]) + logC + kappa * Z[j] @ Z[kk - 1])
    labs = list(itertools.product(*[range(j + 1) for j in range(5)]))
    joint = np.array([np.prod([w(l[j], j) for j in range(5)]) for l in labs])
    joint /= joint.sum()
    prod = np.array([np.prod([res.parent_probs[j][l[j]] for j in range(5)]) for l in labs])
    add("Theorem 5.7", "max |posterior - product| (exact enumeration)", 0.0, float(np.abs(joint - prod).max()))

    dt, T = 0.01, 2000
    m = int(T / dt)
    wn = rng.normal(size=m + 600)
    X = np.convolve(wn, np.exp(-(np.arange(-300, 301) * dt) ** 2 / 2), "same")[300:300 + m]
    X /= X.std()
    lag = int(0.5 / dt)
    Y = np.r_[np.zeros(lag), X[:-lag]]
    R = lambda h: np.mean(X[h:] * X[:-h])
    add("Theorem 7.3", "Levy area rate, lag 0.5", -(R(lag + 1) - R(lag - 1)) / (2 * dt), levy_area_rate(X, Y, dt))

    # ---------------------------------------------------------------- Section 9
    mu, sig, s = 0.05, 0.2, 0.07
    mh = rng.normal(mu, s, n)
    for c, lab in [(kelly_shrinkage(mu, s), "growth at shrunk Kelly c*"), (1.0, "growth at full Kelly")]:
        f = c * mh / sig**2
        add("Theorem 9.1", lab, expected_log_growth(c, mu, sig, s), float(np.mean(f * mu - 0.5 * f**2 * sig**2)))

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "table1.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)
    lines = ["| Result | Quantity | Theory | Simulation |", "|---|---|---:|---:|"]
    lines += [f"| {r['result']} | {r['quantity']} | {r['theory']:.4g} | {r['simulation']:.4g} |" for r in rows]
    (out / "table1.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    main(a.n, a.seed, a.out)
