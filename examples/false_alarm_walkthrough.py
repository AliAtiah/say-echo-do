"""End-to-end walkthrough on a synthetic false-alarm episode.

Every input is simulated, so the script runs anywhere with no data.  Replace each
simulated block with real data to use the pipeline:

  * articles: publication times + sentence embeddings (any encoder) + sentiment
  * Say:      institutional statements (e.g. management tone, rating changes)
  * Do:       positioning (e.g. insider trades, options flow, disclosed holdings)
  * price:    abnormal returns around the event

Run:  python examples/false_alarm_walkthrough.py
"""

from __future__ import annotations

import numpy as np

from say_echo_do import (
    EventSignals, HawkesParams, fit_hawkes, kelly_shrinkage, levy_area, semantic_declustering,
    simulate_hawkes, split_sentiment,
)
from say_echo_do.decisions import RegimeFilter
from say_echo_do.echo import sample_vmf
from say_echo_do.embeddings import analog_forecast, topic_orthogonal
from say_echo_do.leadlag import standardise
from say_echo_do.signal import rolling_covariance

rng = np.random.default_rng(7)
D = 32

# --------------------------------------------------------------------------- 1. the map of meaning
# A sentiment ("alarm") direction, made orthogonal to a few topic directions.
topics = rng.normal(size=(D, 4))
w_alarm = topic_orthogonal(rng.normal(size=D), topics)
w_alarm /= np.linalg.norm(w_alarm)


def story(alarm: float) -> np.ndarray:
    """A random unit-norm story whose projection on the alarm direction equals ``alarm`` in [-1, 1]."""
    z = rng.normal(size=D)
    z -= (z @ w_alarm) * w_alarm
    z = z / np.linalg.norm(z) * np.sqrt(max(1 - alarm**2, 1e-6)) + alarm * w_alarm
    return z / np.linalg.norm(z)


# --------------------------------------------------------------------------- 2. media coverage
# A downgrade (strongly negative story) triggers a viral cascade of near-copies.
true = HawkesParams(mu=0.8, alpha=2.4, beta=3.0)  # branching ratio 0.8: coverage feeds on itself
news_mark = lambda r: story(alarm=0.3 + 0.5 * r.uniform())   # bearish news
echo_mark = lambda parent, r: sample_vmf(parent, 60.0, r)
times, parents, Z = simulate_hawkes(true, T=60.0, rng=rng, mark_news=news_mark, mark_echo=echo_mark)
sentiment = -(Z @ w_alarm)                        # negative = alarming, positive = calm

est = fit_hawkes(times, 60.0)
dec = semantic_declustering(times, Z, est, kappa=60.0, max_lag=5.0)
s_new, s_echo = split_sentiment(sentiment, dec.p_news)
scale = np.sqrt(len(times))
print(f"articles: {len(times)}   fitted branching ratio n = {est.branching_ratio:.2f} (true {true.branching_ratio:.2f})")
print(f"echo share: estimated {dec.echo_share():.2f}, actual {(parents >= 0).mean():.2f}")
print(f"sentiment  news = {s_new / scale:+.2f}   echo = {s_echo / scale:+.2f}   (per sqrt(article))")

# --------------------------------------------------------------------------- 3. Say vs Do
# Institutions talk bearish while positioning builds: a negative Say-Do covariance.
T = 120
value = np.cumsum(rng.normal(0.03, 0.1, T))     # fundamentals improving
say = -0.8 * value + rng.normal(0, 0.1, T)        # words run opposite to value (false-alarm regime)
do = 0.6 * value + rng.normal(0, 0.1, T)          # deeds track value
cov = rolling_covariance(say, do, 40)[-1]
print(f"rolling Cov(Say, Do) = {cov:+.4f}  ->  {'FADE' if cov < 0 else 'FOLLOW'} institutional words")

# --------------------------------------------------------------------------- 4. absorption
# Narrative-implied reaction from historical analogues ("deja vu").
Z_hist = np.array([story(alarm=t) for t in rng.uniform(-1, 1, 3000)])
ar_hist = -6.0 * (Z_hist @ w_alarm) + rng.normal(0, 2.0, 3000)     # alarming stories fell ~6% on average
event_story = story(alarm=0.95)                  # the downgrade
f = analog_forecast(event_story, Z_hist, ar_hist, h=0.03)
actual = -1.9
A = (actual - f.mean) / f.std
print(f"implied reaction {f.mean:+.1f}% (n_eff {f.n_eff:.0f}), actual {actual:+.1f}%  ->  absorption A = {A:+.2f}")

# --------------------------------------------------------------------------- 5. who moved first?
t = np.arange(60)
buying = np.exp(-((t - 20) ** 2) / 30) + rng.normal(0, 0.02, t.size)   # buying flow peaks on day ~20
gloom = np.exp(-((t - 32) ** 2) / 30) + rng.normal(0, 0.02, t.size)    # headline gloom peaks on day ~32
area = levy_area(standardise(buying), standardise(gloom)) / t.size
print(f"Levy area rate (buying, gloom) = {area:+.3f}  ->  {'money led' if area > 0 else 'money followed'}")

# --------------------------------------------------------------------------- 6. regime and decision
obs = np.array([-(s_echo - s_new) / scale, -cov * 100, A, 10 * area])
means = np.array([[0.0, -1.0, 0.0, -0.5],       # informational
                  [1.5, 0.0, 0.5, 0.0],         # overreaction
                  [1.5, 1.5, 1.5, 1.0]])        # strategic
filt = RegimeFilter(means, np.array([np.eye(4)] * 3), np.full((3, 3), 1 / 3))
probs = filt.update(obs)
print("regime probabilities  informational {:.2f}  overreaction {:.2f}  strategic {:.2f}".format(*probs))

ev = EventSignals(s_news=s_new / scale, s_echo=s_echo / scale, say=standardise(say)[-1], do=standardise(do)[-1],
                  absorption=A, levy_area=10 * area, say_do_cov_sign=np.sign(cov))
score = ev.score()
c_star = kelly_shrinkage(mu=0.02, s=0.028)
print("contributions:", {k: round(v, 2) for k, v in ev.contributions().items()})
print(f"composite score D = {score:+.2f}  ->  {'LEAN LONG' if score > 0 else 'LEAN SHORT'}, "
      f"sized at {c_star:.2f} x full Kelly")
