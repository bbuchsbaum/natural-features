"""Cochlear-style audio features."""

from __future__ import annotations

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec, times_from_hop
from natural_features.features.audio.lowlevel import _mono, _stft_power
from natural_features.features.common import extractor_metadata


def _hz_to_erb(freq_hz: np.ndarray) -> np.ndarray:
    return 21.4 * np.log10(1.0 + 0.00437 * freq_hz)


def _erb_to_hz(erb: np.ndarray) -> np.ndarray:
    return (10.0 ** (erb / 21.4) - 1.0) / 0.00437


def _erb_filterbank(sr_hz: int, n_fft: int, n_channels: int, fmin: float, fmax: float) -> np.ndarray:
    erb_edges = np.linspace(_hz_to_erb(np.array([fmin]))[0], _hz_to_erb(np.array([fmax]))[0], n_channels + 2)
    hz_edges = _erb_to_hz(erb_edges)
    bin_edges = np.floor((n_fft + 1) * hz_edges / sr_hz).astype(int)
    filters = np.zeros((n_channels, (n_fft // 2) + 1), dtype=np.float32)
    for i in range(1, n_channels + 1):
        left, center, right = bin_edges[i - 1], bin_edges[i], bin_edges[i + 1]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for k in range(left, center):
            if 0 <= k < filters.shape[1]:
                filters[i - 1, k] = (k - left) / max(1, center - left)
        for k in range(center, right):
            if 0 <= k < filters.shape[1]:
                filters[i - 1, k] = (right - k) / max(1, right - center)
    return filters


def audio_gammatone(
    stimulus: AudioStimulus,
    *,
    hop_s: float = 0.01,
    win_s: float = 0.025,
    n_channels: int = 32,
    fmin: float = 50.0,
    fmax: float | None = None,
    log: bool = True,
) -> FeatureSeries:
    """Return an ERB-spaced gammatone-style filterbank energy representation."""

    if n_channels <= 0:
        raise ValueError("n_channels must be > 0")
    power, _freqs, _starts = _stft_power(_mono(stimulus.samples), stimulus.sr_hz, hop_s, win_s)
    n_fft = int(2 * (power.shape[1] - 1))
    fmax = float(fmax if fmax is not None else stimulus.sr_hz / 2.0)
    fb = _erb_filterbank(stimulus.sr_hz, n_fft, n_channels, fmin, fmax)
    values = power @ fb.T
    if log:
        values = np.log10(np.maximum(values, 1e-10))
    values = values.astype(np.float32)
    times = times_from_hop(values.shape[0], hop_s, start_offset_s=stimulus.start_offset_s, center=True, window_s=win_s)
    md = extractor_metadata(
        "audio.gammatone",
        params={"hop_s": hop_s, "win_s": win_s, "n_channels": n_channels, "fmin": fmin, "fmax": fmax, "log": log},
        extra={"backend": "erb_filterbank"},
    )
    return FeatureSeries(
        values=values,
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": [f"gammatone_{i}" for i in range(n_channels)]},
        metadata=md,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s),
    )
