"""Numerical check of Theorem 6.3: neither certificate ever misorders a pair,
and the exact form (ii) certifies every pair the Pinsker form (i) does.

Random instances: n ~ U{3,...,29} outcomes, q ~ Dirichlet(c * 1) with c ~ U(0.2, 3),
p = q * exp(N(0, s^2)) renormalised with s ~ U(0.01, 1), and eps = KL(q || p).

Usage:  python scripts/check_certificate.py  ->  results/certificate_check.json
"""

import json
from pathlib import Path

import numpy as np
from scipy.special import rel_entr

from say_echo_do.embeddings import certify_anchor


def main(trials=3000, seed=1):
    rng = np.random.default_rng(seed)
    out = dict(trials=trials, seed=seed, exact_certified=0, pinsker_certified=0,
               exact_violations=0, pinsker_violations=0, pinsker_not_in_exact=0)
    for _ in range(trials):
        n = int(rng.integers(3, 30))
        q = rng.dirichlet(np.ones(n) * rng.uniform(0.2, 3))
        p = q * np.exp(rng.normal(0, rng.uniform(0.01, 1.0), n))
        p /= p.sum()
        eps = rel_entr(q, p).sum()
        closer = p[:, None] > p[None, :]
        ex, pi = certify_anchor(q, eps), certify_anchor(q, eps, method="pinsker")
        out["exact_certified"] += int(ex.sum())
        out["pinsker_certified"] += int(pi.sum())
        out["exact_violations"] += int((ex & ~closer).sum())
        out["pinsker_violations"] += int((pi & ~closer).sum())
        out["pinsker_not_in_exact"] += int((pi & ~ex).sum())
    out["ratio_exact_to_pinsker"] = out["exact_certified"] / max(out["pinsker_certified"], 1)
    path = Path(__file__).resolve().parents[1] / "results" / "certificate_check.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
