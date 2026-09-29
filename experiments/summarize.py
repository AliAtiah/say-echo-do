"""Aggregate per-seed results into the tables reported in the paper.

Usage:  python experiments/summarize.py  ->  results/sim/summary.json and results/sim/summary.md
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1] / "results" / "sim"

# predicted sign of each test (None = theory allows either sign)
PREDICTED = {
    "T1 news": +1, "T1 echo": -1,
    "T2 Say | Cov<0": -1, "T2 Say | Cov>=0": None, "T2 Do": +1,
    "T3 absorption": None, "T3 absorption | informed trading": None, "T3 absorption | no informed trading": -1,
    "T3 absorption, firm-level multiplier": None, "T3 firm-level | informed trading": None,
    "T3 firm-level | no informed trading": -1,
    "T4 signed Levy area": +1,
}


def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(str(R / pattern)))]


def ms(x):
    x = np.asarray(x, float)
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else 0.0, n=len(x))


def summarize_runs(runs):
    out = {}
    out["perf"] = {k: {m: ms([r["perf"][k][m] for r in runs]) for m in ("ic", "ic_t", "hit", "sharpe")}
                   for k in runs[0]["perf"]}
    tests = {}
    for k in runs[0]["tests"]:
        b = [r["tests"][k][0] for r in runs]
        t = [r["tests"][k][1] for r in runs]
        pred = PREDICTED.get(k)
        sig = sum(1 for tt in t if abs(tt) > 2 and (pred is None or np.sign(tt) == pred))
        tests[k] = dict(coef=ms(b), t=ms(t), predicted=pred, significant_as_predicted=sig, n=len(t))
    out["tests"] = tests
    out["detection"] = {k: ms([r["detection"][k] for r in runs]) for k in runs[0]["detection"]}
    out["ablation"] = {k: ms([r["ablation"][k] for r in runs]) for k in runs[0]["ablation"]}
    out["echo_share"] = dict(true=ms([r["echo_share"]["true"] for r in runs]),
                             estimated=ms([r["echo_share"]["estimated"] for r in runs]))
    out["hawkes"] = {k: ms([r["hawkes"][k] for r in runs]) for k in runs[0]["hawkes"]}
    if "switch_power" in runs[0]:
        out["switch_power"] = {W: {m: ms([r["switch_power"][W][m] for r in runs]) for m in runs[0]["switch_power"][W]}
                               for W in runs[0]["switch_power"]}
    return out


def text_summary():
    """Text experiment: default runs (outcome noise 1, median bandwidth) over seeds, plus the certificate sweep."""
    T = R.parent
    runs = []
    for f in sorted(T.glob("text_experiment*.json")):
        r = json.load(open(f))
        runs.append((f.name, r.get("seed", 0), r.get("noise", 1.0), r.get("bw_scale", 1.0), r))
    if not runs:
        return []
    L = ["", "## Text experiment", ""]
    default = [r for _, _, nz, bs, r in runs if nz == 1.0 and bs == 1.0]
    L += [f"Default setting, {len(default)} seeds (mean ± sd):", "", "| Method | analog IC | top-10 neighbours: same consequence | same topic |", "|---|---:|---:|---:|"]
    for k in default[0]["results"]:
        vals = [r["results"][k] for r in default]
        ic = ms([v["analog_ic"] for v in vals])
        sc = [v["neighbours_same_consequence"] for v in vals]
        st = [v["neighbours_same_topic"] for v in vals]
        fmt = lambda x: "—" if x[0] is None else f"{np.mean(x):.3f}"
        L.append(f"| {k} | {ic['mean']:+.3f} ± {ic['sd']:.3f} | {fmt(sc)} | {fmt(st)} |")
    L += ["", "Neighbour certificate (Theorem 6.3) on 150 training anchors after training:", "",
          "| seed | outcome noise | bandwidth scale | final loss | certified fraction | violations | analog IC |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for name, sd, nz, bs, r in sorted(runs, key=lambda x: (x[2], x[3], x[1])):
        t = r["trace"][-1]
        L.append(f"| {sd} | {nz:g} | {bs:g} | {t['loss']:.4f} | {t['certified']:.4f} | {t['violations']} | {t['analog_ic']:+.3f} |")
    return L


def main():
    summary = {"main": summarize_runs(load("main_seed*.json"))}
    variants = sorted({Path(f).name.split("_seed")[0].replace("robust_", "") for f in glob.glob(str(R / "robust_*.json"))})
    summary["robustness"] = {v: summarize_runs(load(f"robust_{v}_seed*.json")) for v in variants}
    sizes = sorted({int(Path(f).name.split("_E")[1].split("_")[0]) for f in glob.glob(str(R / "samples_E*.json"))})
    summary["samples"] = {str(E): summarize_runs(load(f"samples_E{E}_seed*.json")) for E in sizes}
    (R / "summary.json").write_text(json.dumps(summary, indent=1))

    L = ["# Controlled experiments: summary", ""]
    m = summary["main"]
    L += ["## Out-of-sample performance (main world, 5 seeds)", "", "| Strategy | IC | IC t | Sharpe/period |", "|---|---:|---:|---:|"]
    for k, v in m["perf"].items():
        L.append(f"| {k} | {v['ic']['mean']:+.3f} ± {v['ic']['sd']:.3f} | {v['ic_t']['mean']:+.1f} | {v['sharpe']['mean']:+.2f} |")
    L += ["", "## Theory tests (test half, firm-clustered t)", "", "| Test | coef | t | predicted | seeds |t|>2 as predicted |", "|---|---:|---:|---|---:|"]
    for k, v in m["tests"].items():
        p = {1: "+", -1: "−", None: "either"}[v["predicted"]]
        L.append(f"| {k} | {v['coef']['mean']:+.3f} | {v['t']['mean']:+.2f} | {p} | {v['significant_as_predicted']}/{v['n']} |")
    L += ["", "## False-alarm detection (AUC)", ""] + [f"- {k}: {v['mean']:.3f} ± {v['sd']:.3f}" for k, v in m["detection"].items()]
    L += ["", "## Composite ablation (IC when a term is removed)", ""] + [f"- drop {k}: {v['mean']:+.3f}" for k, v in m["ablation"].items()]
    if "switch_power" in m:
        L += ["", "## Say-Do switch: window length", ""] + [
            f"- W={W}: accuracy {v['accuracy']['mean']:.3f}, t(words | fade group) {v['t_fade']['mean']:+.2f}" for W, v in m["switch_power"].items()]
    L += ["", "## Robustness (3 seeds each; 'base' is the reference world on the same seeds)", "",
          "| Variant | price reversal IC | linear raw | linear + theory | GBM raw | GBM + theory | T1 echo t | T1 news t | T4 Levy t | detection AUC | echo share true/est |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name, s in [("main (5 seeds)", m)] + list(summary["robustness"].items()):
        p = s["perf"]
        ic = lambda k: p[k]["ic"]["mean"]
        L.append(f"| {name} | {ic('Price reversal (fade reaction)'):+.3f} | {ic('Linear, raw voices'):+.3f} | {ic('Linear, raw + theory features'):+.3f} | "
                 f"{ic('Gradient boosting, raw voices'):+.3f} | {ic('Gradient boosting, raw + theory'):+.3f} | "
                 f"{s['tests']['T1 echo']['t']['mean']:+.1f} | {s['tests']['T1 news']['t']['mean']:+.1f} | {s['tests']['T4 signed Levy area']['t']['mean']:+.1f} | "
                 f"{s['detection']['all features']['mean']:.3f} | {s['echo_share']['true']['mean']:.3f}/{s['echo_share']['estimated']['mean']:.3f} |")
    if summary["samples"]:
        L += ["", "## Events per firm", ""] + [
            f"- E={E}: detection AUC {s['detection']['all features']['mean']:.3f}, T1 echo t {s['tests']['T1 echo']['t']['mean']:+.1f}, "
            f"GBM+theory IC {s['perf']['Gradient boosting, raw + theory']['ic']['mean']:+.3f}" for E, s in summary["samples"].items()]
    L += text_summary()
    (R / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
