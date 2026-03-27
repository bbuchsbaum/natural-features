"""Audio affect/prosody proxy features."""

from __future__ import annotations

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.audio.lowlevel import rms, spectral_stats
from natural_features.features.common import extractor_metadata


def audio_affect_proxies(
    stimulus: AudioStimulus,
    *,
    hop_s: float = 0.02,
    win_s: float = 0.03,
) -> FeatureSeries:
    r = rms(stimulus, hop_s=hop_s, win_s=win_s)
    s = spectral_stats(stimulus, hop_s=hop_s, win_s=win_s)
    spec_names = [str(n) for n in s.coords.get("feature", [])]
    spec_ix = {name: i for i, name in enumerate(spec_names)}
    centroid_ix = spec_ix.get("centroid")
    zcr_ix = spec_ix.get("zcr")
    if centroid_ix is None or zcr_ix is None:
        raise ValueError("spectral_stats output is missing required channels: centroid/zcr")
    loud = r.values[:, 0]
    centroid = s.values[:, centroid_ix]
    zcr = s.values[:, zcr_ix]
    # Interpretable proxy channels.
    arousal = np.clip((loud - loud.min()) / (loud.max() - loud.min() + 1e-8), 0.0, 1.0)
    valence_proxy = np.tanh((centroid - np.mean(centroid)) / (np.std(centroid) + 1e-8))
    speaking_rate_proxy = np.clip((zcr - zcr.min()) / (zcr.max() - zcr.min() + 1e-8), 0.0, 1.0)
    vals = np.column_stack([arousal, valence_proxy, speaking_rate_proxy]).astype(np.float32)
    md = extractor_metadata("affect.audio.proxies", params={"hop_s": hop_s, "win_s": win_s})
    return FeatureSeries(
        values=vals,
        times_s=r.times_s,
        dims=("time", "feature"),
        coords={"feature": ["arousal_proxy", "valence_proxy", "speaking_rate_proxy"]},
        metadata=md,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s),
    )
