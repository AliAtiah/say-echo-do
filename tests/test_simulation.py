"""The agent-based market of Section 12 behaves as specified."""

import numpy as np

from say_echo_do.simulation import SimConfig, simulate_market


def test_shapes_and_regime_signs():
    sim = simulate_market(SimConfig(n_firms=40, n_events=20, seed=3))
    assert sim["say"].shape == (40, 20) and sim["pos_path"].shape[:2] == (40, 20)
    for f in sim["firms"]:
        if f.regime == "false_alarm":
            assert f.psi < 0 < f.chi                      # talks down, buys (Corollary 4.4 iii)
        elif f.regime == "exaggeration":
            assert f.psi > 1 and f.chi < 0                # overstates, sells (iv)
        elif f.regime == "shading":
            assert 0 < f.psi < 1 and f.chi > 0            # (ii)
        else:
            assert f.psi == 1 and f.chi == 0


def test_statement_is_first_article_and_echo_share_matches_branching():
    sim = simulate_market(SimConfig(n_firms=60, n_events=30, seed=4))
    shares = []
    for f, firm in enumerate(sim["firms"]):
        for e in range(30):
            a = sim["articles"][f][e]
            assert a["kind"][0] == "say" and a["times"][0] == 0.0 and a["parent"][0] == -1
            assert np.all(a["parent"] < np.arange(len(a["parent"])))   # parents precede children
            shares.append((a["parent"] >= 0).mean())
    assert 0.3 < np.mean(shares) < 0.55


def test_return_identity():
    cfg = SimConfig(n_firms=20, n_events=10, sigma_eta=0.0, seed=5)
    sim = simulate_market(cfg)
    assert np.allclose(sim["r"], sim["v"] - sim["p"])
