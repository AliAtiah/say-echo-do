import numpy as np
import pytest

from say_echo_do import ThreeVoiceMarket


@pytest.fixture
def rng():
    return np.random.default_rng(20260929)


@pytest.fixture
def market():
    """Say m = v + N(0, .9^2), news n = v + N(0, 1.3^2), echo e = m + N(0, .5^2)."""
    Sigma = np.array([[0.81, 0.0, 0.81], [0.0, 1.69, 0.0], [0.81, 0.0, 1.06]])
    return ThreeVoiceMarket(sigma_v=1.0, h=[1.0, 1.0, 1.0], Sigma_nu=Sigma,
                            phi=[0.6, 0.1, 0.3], beta=0.9, lam=0.7, sigma_u=0.8)


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]
