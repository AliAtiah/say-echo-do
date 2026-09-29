"""Controlled experiments for the paper (Section: Experiments).

Usage:
    python experiments/run_simulation_experiments.py main        # 5 seeds + power sweeps
    python experiments/run_simulation_experiments.py robustness  # reference + 7 variants x 3 seeds
    python experiments/run_simulation_experiments.py robustness base,fat_tails   # a subset
    python experiments/run_simulation_experiments.py samples     # events-per-firm sweep

Results are written to results/sim/*.json.  All design choices (simulator
calibration, features, strategies, tests) were fixed before any strategy result
was inspected; the post-window noise was set so that a linear model on the raw
voices, which uses no theory features, reaches an out-of-sample IC of about 0.17.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import build_features, cluster_ols, run_one, split  # noqa: E402

from say_echo_do.simulation import SimConfig  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "sim"
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    "base": dict(),                    # reference world on the robustness seeds
    "fat_tails": dict(value_dist="t3"),
    "nonlinear_crowd": dict(crowd="tanh"),
    "powerlaw_kernel": dict(kernel="powerlaw"),
    "partial_disclosure": dict(disclosed=0.3, sigma_x=1.0),
    "weak_semantics": dict(kappa=5.0),
    "no_semantics": dict(),            # base world, declustering uses timing only
    "honest_only": dict(regime_mix=(1.0, 0.0, 0.0, 0.0)),
}


def save(name, res):
    clean = {k: v for k, v in res.items() if not k.startswith("_")}
    (OUT / f"{name}.json").write_text(json.dumps(clean, indent=1))


def switch_power(sim, feat_base, windows=(5, 10, 20, 40), thr=0.2):
    """How well the rolling Say-Do correlation identifies firms whose words should be faded."""
    F, E = sim["say"].shape
    tr, te = split(E)
    should_fade = np.repeat(np.isin(sim["regime"], ["false_alarm", "exaggeration"])[:, None], E, 1)
    out = {}
    for W in windows:
        sw = np.ones((F, E))
        for f in range(F):
            for e in range(E):
                lo = max(0, e - W)
                if e - lo >= min(5, W):
                    c = np.corrcoef(sim["say"][f, lo:e], sim["do"][f, lo:e])[0, 1]
                    sw[f, e] = -1.0 if c < -thr else 1.0
        acc = float(np.mean((sw[:, te] < 0) == should_fade[:, te]))
        firm = np.repeat(np.arange(F)[:, None], E, 1)[:, te].ravel()
        neg = sw[:, te].ravel() < 0
        say = sim["say"][:, te].ravel() / sim["say"][:, tr].std()
        do = sim["do"][:, te].ravel() / sim["do"][:, tr].std()
        b, t = cluster_ols(sim["r"][:, te].ravel(), np.c_[say * neg, say * ~neg, do], firm)
        out[str(W)] = dict(accuracy=acc, t_fade=float(t[0]), t_follow=float(t[1]))
    return out


def main_block(seeds=range(5)):
    for s in seeds:
        t0 = time.time()
        res = run_one(SimConfig(seed=s))
        res["switch_power"] = switch_power(res["_sim"], res["_feat"])
        save(f"main_seed{s}", res)
        print(f"main seed {s} done in {time.time() - t0:.0f}s", flush=True)


def robustness_block(seeds=range(3), only=None):
    for name, kw in VARIANTS.items():
        if only and name not in only:
            continue
        for s in seeds:
            t0 = time.time()
            cfg = dataclasses.replace(SimConfig(seed=100 + s), **kw)
            res = run_one(cfg, use_semantics=(name != "no_semantics"))
            save(f"robust_{name}_seed{s}", res)
            print(f"{name} seed {s} done in {time.time() - t0:.0f}s", flush=True)


def samples_block(seeds=range(2), sizes=(20, 40, 80)):
    for E in sizes:
        for s in seeds:
            t0 = time.time()
            res = run_one(SimConfig(seed=200 + s, n_events=E))
            save(f"samples_E{E}_seed{s}", res)
            print(f"E={E} seed {s} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    block = sys.argv[1] if len(sys.argv) > 1 else "main"
    if block == "robustness" and len(sys.argv) > 2:      # e.g. robustness base,fat_tails
        robustness_block(only=sys.argv[2].split(","))
    else:
        {"main": main_block, "robustness": robustness_block, "samples": samples_block}[block]()
