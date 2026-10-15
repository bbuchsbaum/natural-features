"""Prosodic audio features."""

from __future__ import annotations

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec
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
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    """Estimate framewise F0 with a lightweight autocorrelation method."""

    mode, _strict = resolve_execution_mode(execution_mode=execution_mode, strict_dependency=strict_dependency)
    if fmin <= 0 or fmax <= 0 or fmax <= fmin:
        raise ValueError("Require 0 < fmin < fmax")
    x = _mono(stimulus.samples)
    if not np.isfinite(hop_s) or not np.isfinite(win_s) or hop_s <= 0 or win_s <= 0:
        raise ValueError("hop_s and win_s must be finite and positive")
    framed, starts = _frames(x, stimulus.sr_hz, hop_s, win_s)
    effective_hop = max(1, round(stimulus.sr_hz * hop_s)) / stimulus.sr_hz
    effective_win = framed.shape[1] / stimulus.sr_hz
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
    bounds = stimulus.start_offset_s + np.column_stack(
        [starts, starts + framed.shape[1]]
    ) / stimulus.sr_hz
    times = bounds.mean(axis=1)
    md = add_execution_provenance(
        extractor_metadata(
            "audio.pitch",
            params={
                "hop_s": hop_s,
                "win_s": win_s,
                "fmin": fmin,
                "fmax": fmax,
                "voicing_threshold": voicing_threshold,
            },
            extra={
                "backend": "autocorrelation",
                "effective_hop_s": effective_hop,
                "effective_window_s": effective_win,
                "right_padding_samples": max(0, framed.shape[1] - len(x)),
                "pitch_scope": "monophonic_autocorrelation_not_vocal_detection",
            },
        ),
        execution_mode=mode,
        fallback_used=False,
    )
    return FeatureSeries(
        values=values,
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": ["f0_hz", "voicing_strength"]},
        metadata=md,
        timebase=TimebaseSpec(
            kind="audio_hop", hop_s=effective_hop,
            sampling_rate_hz=1.0 / effective_hop, window_s=effective_win,
        ),
        time_bounds_s=bounds,
    )


def prosody_features(
    stimulus: AudioStimulus,
    *,
    hop_s: float = 0.01,
    win_s: float = 0.025,
) -> FeatureSeries:
    """Return compact energy, pitch, and spectral prosody controls."""

    if not np.isfinite(hop_s) or not np.isfinite(win_s) or hop_s <= 0 or win_s <= 0:
        raise ValueError("hop_s and win_s must be finite and positive")
    # All columns summarize the same sample windows, including the F0 column.
    effective_hop = max(1, round(stimulus.sr_hz * hop_s)) / stimulus.sr_hz
    effective_win = max(1, round(stimulus.sr_hz * max(0.04, win_s))) / stimulus.sr_hz
    r = rms(stimulus, hop_s=effective_hop, win_s=effective_win)
    p = audio_pitch(stimulus, hop_s=effective_hop, win_s=effective_win)
    s = spectral_stats(stimulus, hop_s=effective_hop, win_s=effective_win)
    n = len(p.times_s)
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
        extra={
            "backend": "python_native",
            "effective_hop_s": effective_hop,
            "effective_window_s": effective_win,
            "framing": "shared_sample_windows",
            "right_padding_samples": p.metadata["right_padding_samples"],
            "pitch_scope": p.metadata["pitch_scope"],
        },
    )
    return FeatureSeries(
        values=values,
        times_s=p.times_s,
        dims=("time", "feature"),
        coords={"feature": ["rms", "log_rms", "f0_hz", "voicing_strength", "spectral_centroid", "zcr"]},
        metadata=md,
        timebase=p.timebase,
        time_bounds_s=p.time_bounds_s,
    )
