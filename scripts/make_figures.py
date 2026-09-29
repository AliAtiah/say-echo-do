"""Regenerate the paper's quantitative figures with matplotlib (AAAS/Science style).

Usage:  python scripts/make_figures.py [--out figures]

Produces
  fig_false_alarms.pdf/png   regime map, virality windows, credibility bifurcation (Figure 4)
  fig_declustering.pdf/png   parent-attribution accuracy vs semantic concentration (new)
  fig_detection_kelly.pdf/png detection bound and Kelly growth curves (Figure 9)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from say_echo_do.decisions import detection_bound, expected_log_growth
from say_echo_do.echo import HawkesParams, sample_vmf, semantic_declustering, simulate_hawkes
from say_echo_do.plotting import PALETTE as C, panel_label, use_science_style
from say_echo_do.strategic import two_cycle, trust_map


def fig_false_alarms(out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), constrained_layout=True)
    # a -- regime map
    phi = np.linspace(0, 1.6, 400)
    a = ax[0]
    a.fill_between(phi, 0, phi**2, color=C["gray"], alpha=0.12, lw=0)
    m = phi <= 1
    a.fill_between(phi[m], phi[m] ** 2, phi[m], color=C["crimson"], alpha=0.35, lw=0)
    a.fill_between(phi[m], phi[m], 2.6, color=C["green"], alpha=0.15, lw=0)
    a.fill_between(phi[~m], phi[~m] ** 2, 2.6, color=C["magenta"], alpha=0.15, lw=0)
    a.plot(phi, phi**2, color=C["ink"], lw=0.8)
    a.plot(phi[m], phi[m], color=C["ink"], lw=0.7, ls="--")
    a.plot([1, 1], [1, 2.6], color=C["navy"], lw=1.4)
    a.text(0.42, 1.9, "shading", color=C["green"], fontweight="bold", ha="center")
    a.text(0.8, 0.66, "false\nalarm", color=C["crimson"], fontweight="bold", ha="center", fontsize=6.5)
    a.text(1.33, 2.2, "exaggerate\n& sell", color=C["magenta"], fontweight="bold", ha="center", fontsize=7)
    a.text(1.3, 0.55, "unbounded\nmanipulation", color=C["gray"], ha="center", fontsize=6)
    a.set(xlim=(0, 1.6), ylim=(0, 2.6), xlabel=r"crowd credulity $\varphi$", ylabel=r"deterrence–impact $a=2\lambda k$")
    panel_label(a, "a")
    # b -- virality windows
    b = ax[1]
    n = np.linspace(0, 0.7, 800)
    for aa, col, lab in ((0.6, C["crimson"], "deep, $a=0.6$"), (1.5, C["navy"], "shallow, $a=1.5$")):
        ph = 0.4 / (1 - n)
        ok = ph**2 < aa
        psi = np.where(ok, (aa - ph) / (aa - ph**2), np.nan)
        b.plot(n, np.clip(psi, -3.3, 3.3), color=col, label=lab)
    b.axvspan(1 - 0.4 / 0.6, 1 - 0.4 / np.sqrt(0.6), color=C["crimson"], alpha=0.12, lw=0)
    b.axvspan(1 - 0.4, 1 - 0.4 / np.sqrt(1.5), color=C["magenta"], alpha=0.12, lw=0)
    b.axhline(0, color=C["ink"], lw=0.4, alpha=0.5)
    b.axhline(1, color=C["ink"], lw=0.4, ls="--", alpha=0.5)
    b.set(xlim=(0, 0.7), ylim=(-3.2, 3.2), xlabel="media branching ratio $n$", ylabel=r"speech slope $\psi$")
    b.legend(loc="lower left", fontsize=6.5)
    panel_label(b, "b")
    # c -- bifurcation
    c = ax[2]
    A, P = [], []
    for aa in np.linspace(1.55, 2.4, 500):
        ph, ok = 0.97, True
        for _ in range(2000):
            ph = float(trust_map(ph, aa))
            if not np.isfinite(ph) or ph**2 >= aa:
                ok = False
                break
        if not ok:
            continue
        for _ in range(60):
            ph = float(trust_map(ph, aa))
            A.append(aa)
            P.append(ph)
    c.axvspan(1.55, 5 / 3, color=C["crimson"], alpha=0.08, lw=0)
    c.axvspan(5 / 3, 2, color=C["violet"], alpha=0.09, lw=0)
    c.axvspan(2, 2.4, color=C["green"], alpha=0.08, lw=0)
    c.scatter(A, P, s=0.15, color=C["ink"], rasterized=True)
    ag = np.linspace(5 / 3, 1.9999, 100)
    cyc = np.array([two_cycle(x) for x in ag])
    c.plot(ag, cyc[:, 0], color=C["violet"], ls="--", lw=0.9)
    c.plot(ag, cyc[:, 1], color=C["violet"], ls="--", lw=0.9)
    for x, t, col in ((1.608, "chaos", C["crimson"]), (1.83, "2-cycle", C["violet"]), (2.2, "trust converges", C["green"])):
        c.text(x, 1.37, t, color=col, fontweight="bold", ha="center", fontsize=6.5)
    c.set(xlim=(1.55, 2.4), ylim=(0, 1.35), xlabel=r"deterrence–impact $a=2\lambda k$", ylabel=r"long-run credulity $\varphi_t$")
    panel_label(c, "c")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_false_alarms.{ext}")
    plt.close(fig)


def fig_declustering(out: Path, seed: int = 0) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    p = HawkesParams(mu=0.6, alpha=1.0, beta=1.25)
    D = 16
    kappas = [0, 2, 5, 10, 20, 40, 80]
    acc_t, acc_s, err_t, err_s = [], [], [], []
    for kap in kappas[1:]:
        news = lambda r: (lambda z: z / np.linalg.norm(z))(r.normal(size=D))
        times, parents, Z = simulate_hawkes(p, 800, rng, mark_news=news, mark_echo=lambda z, r: sample_vmf(z, kap, r))
        rt = semantic_declustering(times, None, p, 0.0)
        rs = semantic_declustering(times, Z, p, kap)
        is_echo = parents >= 0
        acc_t.append((rt.map_parent == parents).mean())
        acc_s.append((rs.map_parent == parents).mean())
        err_t.append(np.abs((1 - rt.p_news) - is_echo).mean())
        err_s.append(np.abs((1 - rs.p_news) - is_echo).mean())
    fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.2), constrained_layout=True)
    ax[0].plot(kappas[1:], acc_t, "o-", color=C["gray"], ms=3, label="timing only")
    ax[0].plot(kappas[1:], acc_s, "o-", color=C["crimson"], ms=3, label="timing + meaning")
    ax[0].set(xscale="log", xlabel=r"echo concentration $\varkappa$", ylabel="parent accuracy", ylim=(0, 1))
    ax[0].legend(fontsize=6.5)
    panel_label(ax[0], "a")
    ax[1].plot(kappas[1:], err_t, "o-", color=C["gray"], ms=3)
    ax[1].plot(kappas[1:], err_s, "o-", color=C["crimson"], ms=3)
    ax[1].set(xscale="log", xlabel=r"echo concentration $\varkappa$", ylabel="mean |P(echo) − 1{echo}|")
    panel_label(ax[1], "b")
    for axis in ax:
        axis.set_xticks(kappas[1:])
        axis.set_xticklabels([str(k) for k in kappas[1:]])
        axis.minorticks_off()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_declustering.{ext}")
    plt.close(fig)


def fig_detection_kelly(out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.2), constrained_layout=True)
    n = np.arange(0, 41)
    for d, col in ((0.5, C["gray"]), (1, C["purple"]), (1.5, C["navy"]), (2, C["green"])):
        ax[0].semilogy(n, detection_bound(d, n), color=col, label=rf"$\Delta={d}$")
    ax[0].set(xlabel="events in episode $n$", ylabel="bound on misclassification", ylim=(1e-5, 0.6))
    ax[0].legend(fontsize=6.5)
    panel_label(ax[0], "a")
    c = np.linspace(0, 1.6, 200)
    for snr, col in ((4, C["green"]), (1, C["navy"]), (0.5, C["crimson"])):
        s = 1 / np.sqrt(snr)
        g = [expected_log_growth(x, 1.0, 1.0, s) for x in c]
        ax[1].plot(c, g, color=col, label=f"SNR = {snr}")
        cs = snr / (1 + snr)
        ax[1].plot(cs, expected_log_growth(cs, 1.0, 1.0, s), "o", color=col, ms=3)
    ax[1].axhline(0, color=C["ink"], lw=0.4, alpha=0.5)
    ax[1].axvline(1, color=C["ink"], lw=0.5, ls="--", alpha=0.5)
    ax[1].set(xlabel="Kelly fraction $c$", ylabel=r"expected growth $\times\sigma^2/\mu^2$", ylim=(-0.35, 0.55))
    ax[1].legend(fontsize=6.5, loc="lower left")
    panel_label(ax[1], "b")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_detection_kelly.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("figures"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    use_science_style()
    fig_false_alarms(a.out)
    fig_declustering(a.out)
    fig_detection_kelly(a.out)
    print(f"figures written to {a.out}/")
