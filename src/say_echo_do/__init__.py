"""say_echo_do -- Strategic speech, media echo and the economics of false alarms.

Reference implementation of the paper *Say, Echo, Do*.  Markets hear three voices:
what institutions **say**, what the media **echoes**, and what capital **does**.
This package implements the theory's closed forms and the measurement tools:

========================  ===================================================
module                    contents
========================  ===================================================
:mod:`.gaussian`          three-voice market: follow-or-fade, Say--Do gap, absorption
:mod:`.strategic`         optimal speech, false-alarm taxonomy, covariance identity,
                          virality windows, credibility cycles
:mod:`.echo`              Hawkes fitting/simulation, semantic declustering, news/echo split
:mod:`.embeddings`        return-aligned contrastive loss, neighbour certificate,
                          surprise, analog forecasting, absorption
:mod:`.leadlag`           Levy area (who moved first?)
:mod:`.decisions`         regime filter, detection bound, adaptive conformal, Kelly shrinkage
:mod:`.signal`            theorem-signed composite follow-or-fade score
:mod:`.plotting`          AAAS/Science figure style
========================  ===================================================
"""

from . import decisions, echo, embeddings, gaussian, leadlag, signal, strategic
from .decisions import AdaptiveConformal, RegimeFilter, detection_bound, kelly_shrinkage
from .echo import HawkesParams, fit_hawkes, semantic_declustering, simulate_hawkes, split_sentiment
from .embeddings import analog_forecast, certified_fraction, racl_loss, surprise
from .gaussian import ThreeVoiceMarket, kyle_intensities
from .leadlag import levy_area, rolling_levy_area
from .signal import EventSignals, SignalWeights, composite_score, say_do_switch
from .strategic import Regime, StrategicInstitution, manipulation_window, trust_map, two_cycle

__version__ = "0.1.0"

__all__ = [
    "decisions", "echo", "embeddings", "gaussian", "leadlag", "signal", "strategic",
    "ThreeVoiceMarket", "kyle_intensities",
    "StrategicInstitution", "Regime", "manipulation_window", "trust_map", "two_cycle",
    "HawkesParams", "fit_hawkes", "simulate_hawkes", "semantic_declustering", "split_sentiment",
    "racl_loss", "certified_fraction", "surprise", "analog_forecast",
    "levy_area", "rolling_levy_area",
    "RegimeFilter", "AdaptiveConformal", "detection_bound", "kelly_shrinkage",
    "EventSignals", "SignalWeights", "composite_score", "say_do_switch",
]
