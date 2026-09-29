import numpy as np
import pytest
from scipy.optimize import minimize

from say_echo_do.strategic import (
    Regime, StrategicInstitution, effective_credulity, iterate_trust, manipulation_window,
    trust_lyapunov_exponent, trust_map, trust_map_derivative, trust_regime, two_cycle, two_cycle_multiplier,
)

CASES = [(0.6, 0.5, 0.5), (0.8, 0.5, 0.7), (0.5, 1.0, 2.0), (1.3, 1.0, 1.5), (1.0, 0.7, 0.9)]


@pytest.mark.parametrize("phi,lam,k", CASES)
def test_closed_form_matches_numerical_optimum(phi, lam, k):  # Theorem 4.3
    inst = StrategicInstitution(phi, lam, k)
    v = 1.3
    res = minimize(lambda z: -inst.objective(z[0], z[1], v), [0.1, 0.1], method="BFGS", options={"gtol": 1e-12})
    m, x = inst.optimal_policy(v)
    assert res.x[1] == pytest.approx(m, abs=1e-6)
    assert res.x[0] == pytest.approx(x, abs=1e-6)


def test_unbounded_when_a_below_phi_squared():
    inst = StrategicInstitution(phi=1.2, lam=0.5, k=1.0)  # a = 1.0 < 1.44
    assert inst.regime() is Regime.UNBOUNDED
    with pytest.raises(ValueError):
        _ = inst.speech_slope
    # objective grows without bound along some direction
    vals = [inst.objective(t * 0.6, t * -0.8, 0.0) for t in (1, 10, 100)]
    assert vals[2] > vals[1] > vals[0]


@pytest.mark.parametrize("phi,lam,k,expected", [
    (1.0, 0.7, 0.9, Regime.TRUTH),
    (0.5, 1.0, 2.0, Regime.SHADING),
    (0.6, 0.5, 0.5, Regime.FALSE_ALARM),
    (1.3, 1.0, 1.5, Regime.EXAGGERATION),
])
def test_taxonomy(phi, lam, k, expected):  # Corollary 4.4
    inst = StrategicInstitution(phi, lam, k)
    assert inst.regime() is expected
    psi, chi = inst.speech_slope, inst.trade_slope
    if expected is Regime.TRUTH:
        assert psi == pytest.approx(1) and chi == pytest.approx(0)
    if expected is Regime.SHADING:
        assert 0 < psi < 1 and chi > 0
    if expected is Regime.FALSE_ALARM:
        assert psi < 0 < chi
    if expected is Regime.EXAGGERATION:
        assert psi > 1 and chi < 0


@pytest.mark.parametrize("phi,lam,k", [(0.7, 0.5, 0.6), (1.2, 0.8, 1.0), (0.5, 1.0, 2.0)])
def test_covariance_identity_is_distribution_free(phi, lam, k, rng):  # Theorem 4.5
    inst = StrategicInstitution(phi, lam, k)
    n = 1_000_000
    v = rng.standard_t(3, n)                      # fat tails
    eps = rng.laplace(0, 0.3 / np.sqrt(2), n)     # non-Gaussian message noise
    d = inst.simulate(v, eps)
    np.testing.assert_allclose(d["r"], lam * d["x"] - phi * eps, atol=1e-12)   # pathwise
    lhs = np.cov(d["r"], d["m_obs"])[0, 1]
    rhs = lam * np.cov(d["x"], d["m_obs"])[0, 1] - phi * eps.var()
    assert lhs == pytest.approx(rhs, rel=5e-3, abs=5e-3)


def test_identity_holds_for_arbitrary_message_rules(rng):  # Theorem 4.5 (any m(v))
    inst = StrategicInstitution(0.7, 0.5, 0.6)
    v = rng.normal(size=400_000)
    eps = rng.normal(0, 0.4, v.size)
    d = inst.simulate(v, eps, message_rule=lambda v: np.tanh(3 * v) - 0.2 * v**2)
    np.testing.assert_allclose(d["r"], inst.lam * d["x"] - inst.phi * eps, atol=1e-12)


def test_equilibrium_say_do_covariance_sign():
    for phi, lam, k in CASES:
        inst = StrategicInstitution(phi, lam, k)
        c = inst.say_do_covariance(1.0)
        if inst.regime() in (Regime.FALSE_ALARM, Regime.EXAGGERATION):
            assert c < 0
        else:
            assert c >= -1e-15


def test_manipulation_windows():  # Proposition 4.6
    phi0 = 0.4
    deep = manipulation_window(phi0, 0.6)
    shallow = manipulation_window(phi0, 1.5)
    assert deep["regime"] is Regime.FALSE_ALARM and shallow["regime"] is Regime.EXAGGERATION
    for a, w in ((0.6, deep), (1.5, shallow)):
        grid = np.linspace(0, 0.99, 2000)
        k = 0.5 * a  # lam = 1
        inside = []
        for n in grid:
            inst = StrategicInstitution(effective_credulity(phi0, n), 1.0, k)
            inside.append(inst.regime() is w["regime"])
        inside = np.array(inside)
        assert grid[inside].min() == pytest.approx(w["n_low"], abs=1e-3)
        assert grid[inside].max() == pytest.approx(w["n_high"], abs=1e-3)


def test_trust_dynamics():  # Theorem 4.7
    for a in (1.2, 1.5, 1.9, 2.5):
        assert trust_map(1.0, a) == pytest.approx(1.0)
        assert trust_map_derivative(1.0, a) == pytest.approx(-1 / (a - 1))
    lo, hi = two_cycle(1.95)
    assert trust_map(lo, 1.95) == pytest.approx(hi) and trust_map(hi, 1.95) == pytest.approx(lo)
    assert lo < 1 < hi
    for a in (1.7, 1.8, 1.95):
        lo, hi = two_cycle(a)
        mult = trust_map_derivative(lo, a) * trust_map_derivative(hi, a)
        assert mult == pytest.approx(two_cycle_multiplier(a))
    assert two_cycle_multiplier(1.8) == pytest.approx(0.0)           # superstable
    assert two_cycle_multiplier(5 / 3) == pytest.approx(-1.0)        # second flip
    # orbits: converge above 2, settle on the 2-cycle in (5/3, 2)
    assert iterate_trust(2.5, 0.9, 200)[-1] == pytest.approx(1.0, abs=1e-8)
    orbit = iterate_trust(1.8, 0.95, 400)
    assert sorted(orbit[-2:]) == pytest.approx(list(two_cycle(1.8)), abs=1e-8)
    assert trust_regime(1.6) == "complex" and trust_lyapunov_exponent(1.6) > 0
