"""Prosodic audio features."""

from __future__ import annotations

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec, times_from_hop
from natural_features.features.audio.lowlevel import _frames, _mono, rms, spectral_stats
from natural_features.features.common import extractor_metadata


def audio_pitch(
    stimulus: AudioStimulus,
    *,
    hop_s: float = 0.01,
    win_s: float = 0.04,
    fmin: float = 50.0,
    fmax: float = 500.0,
    voicing_threshold: float = 0.2,
) -> FeatureSeries:
    """Estimate framewise F0 with a lightweight autocorrelation method."""

    if fmin <= 0 or fmax <= 0 or fmax <= fmin:
        raise ValueError("Require 0 < fmin < fmax")
    x = _mono(stimulus.samples)
    framed, _ = _frames(x, stimulus.sr_hz, hop_s, win_s)
    min_lag = max(1, int(round(stimulus.sr_hz / fmax)))
    max_lag = min(framed.shape[1] - 1, int(round(stimulus.sr_hz / fmin)))
    f0 = np.zeros(framed.shape[0], dtype=np.float32)
    voicing = np.zeros(framed.shape[0], dtype=np.float32)
    for i, frame in enumerate(framed.astype(np.float32)):
        frame = frame - float(frame.mean())
        denom = float(np.dot(frame, frame)) + 1e-12
        if max_lag <= min_lag or denom <= 1e-10:
            continue
        scores = []
        for lag in range(min_lag, max_lag + 1):
            scores.append(float(np.dot(frame[:-lag], frame[lag:]) / denom))
        arr = np.asarray(scores, dtype=np.float32)
        best = int(np.argmax(arr))
        strength = float(arr[best])
        voicing[i] = max(0.0, strength)
        if strength >= voicing_threshold:
            f0[i] = float(stimulus.sr_hz / (min_lag + best))
    values = np.column_stack([f0, voicing]).astype(np.float32)
    times = times_from_hop(values.shape[0], hop_s, start_offset_s=stimulus.start_offset_s, center=True, window_s=win_s)
    md = extractor_metadata(
        "audio.pitch",
        params={"hop_s": hop_s, "win_s": win_s, "fmin": fmin, "fmax": fmax, "voicing_threshold": voicing_threshold},
        extra={"backend": "autocorrelation"},
    )
    return FeatureSeries(
        values=values,
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": ["f0_hz", "voicing_strength"]},
        metadata=md,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s),
    )


def prosody_features(
    stimulus: AudioStimulus,
    *,
    hop_s: float = 0.01,
    win_s: float = 0.025,
) -> FeatureSeries:
    """Return compact energy, pitch, and spectral prosody controls."""

    r = rms(stimulus, hop_s=hop_s, win_s=win_s)
    p = audio_pitch(stimulus, hop_s=hop_s, win_s=max(0.04, win_s))
    s = spectral_stats(stimulus, hop_s=hop_s, win_s=win_s)
    n = min(r.values.shape[0], p.values.shape[0], s.values.shape[0])
    rms_val = r.values[:n, 0]
    log_rms = np.log1p(np.maximum(rms_val, 0.0))
    values = np.column_stack(
        [
            rms_val,
            log_rms,
            p.values[:n, 0],
            p.values[:n, 1],
            s.values[:n, 0],
            s.values[:n, 4],
        ]
    ).astype(np.float32)
    md = extractor_metadata(
        "audio.prosody",
        params={"hop_s": hop_s, "win_s": win_s},
        extra={"backend": "python_native"},
    )
    return FeatureSeries(
        values=values,
        times_s=r.times_s[:n],
        dims=("time", "feature"),
        coords={"feature": ["rms", "log_rms", "f0_hz", "voicing_strength", "spectral_centroid", "zcr"]},
        metadata=md,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s),
    )
