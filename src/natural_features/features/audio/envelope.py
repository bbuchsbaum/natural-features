"""Envelope, derivative, and onset channels."""

from __future__ import annotations

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec, times_from_hop
from natural_features.features.audio.lowlevel import _frames, _mono
from natural_features.features.common import extractor_metadata


def _hilbert_envelope(x: np.ndarray) -> np.ndarray:
    n = int(x.shape[0])
    spec = np.fft.fft(x)
    h = np.zeros(n, dtype=np.float64)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    analytic = np.fft.ifft(spec * h)
    return np.abs(analytic).astype(np.float32)


def audio_envelope(
    stimulus: AudioStimulus,
    *,
    hop_s: float = 0.01,
    win_s: float = 0.025,
) -> FeatureSeries:
    """Return Hilbert/RMS envelope, its first difference, and a half-wave onset.

    This is not a cochlear filterbank. ``audio.gammatone`` remains the ERB
    frequency-domain approximation.
    """

    x = _mono(stimulus.samples)
    env = _hilbert_envelope(x.astype(np.float64))
    env_frames, _ = _frames(env, stimulus.sr_hz, hop_s, win_s)
    rms_frames, _ = _frames(x, stimulus.sr_hz, hop_s, win_s)
    envelope = np.mean(env_frames, axis=1)
    rms = np.sqrt(np.mean(rms_frames * rms_frames, axis=1))
    delta = np.diff(envelope, prepend=envelope[:1])
    onset = np.maximum(delta, 0.0)
    values = np.stack([envelope, rms, delta, onset], axis=1).astype(np.float32)
    times = times_from_hop(
        values.shape[0],
        hop_s,
        start_offset_s=stimulus.start_offset_s,
        center=True,
        window_s=win_s,
    )
    md = extractor_metadata(
        "audio.envelope",
        params={"hop_s": hop_s, "win_s": win_s},
        extra={"backend": "hilbert_rms"},
    )
    return FeatureSeries(
        values=values,
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": ["envelope", "rms", "delta", "onset"]},
        metadata=md,
        timebase=TimebaseSpec(
            kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s
        ),
    )
