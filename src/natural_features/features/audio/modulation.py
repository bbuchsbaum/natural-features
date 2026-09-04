"""STFT spectrotemporal rate/scale energy.

This is not an NSL Chi/Shamma ``aud2cor`` implementation. The backend tag is
``stft_rate_scale``.
"""

from __future__ import annotations

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec, times_from_hop
from natural_features.features.audio.lowlevel import _mono, _stft_power
from natural_features.features.common import extractor_metadata

DEFAULT_RATE_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (1.0, 4.0),
    (4.0, 8.0),
    (8.0, 16.0),
    (16.0, 32.0),
)
DEFAULT_SCALE_BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 0.25),
    (0.25, 0.75),
    (0.75, 2.0),
)


def _bandpass_axis(
    values: np.ndarray,
    *,
    axis: int,
    sample_period: float,
    lo: float,
    hi: float,
) -> np.ndarray:
    n = values.shape[axis]
    freqs = np.fft.rfftfreq(n, d=sample_period)
    spec = np.fft.rfft(values, axis=axis)
    mask = (freqs >= lo) & (freqs < hi)
    indexer = [slice(None)] * spec.ndim
    indexer[axis] = ~mask
    spec[tuple(indexer)] = 0.0
    out = np.fft.irfft(spec, n=n, axis=axis)
    return np.asarray(out, dtype=np.float32)


def _rate_name(lo: float, hi: float) -> str:
    return f"rate_{int(lo)}_{int(hi)}"


def _scale_name(lo: float, hi: float) -> str:
    return f"scale_{lo:g}_{hi:g}"


def spectrotemporal_modulation(
    stimulus: AudioStimulus,
    *,
    hop_s: float = 0.01,
    win_s: float = 0.025,
    compressed: bool = True,
) -> FeatureSeries:
    """Return temporal-rate × spectral-scale energy of an STFT spectrogram.

    Default output is a compressed ``time × (rate × scale)`` matrix. Set
    ``compressed=False`` to keep flattened per-frequency rate/scale channels.
    """

    power, _freqs, _starts = _stft_power(
        _mono(stimulus.samples), stimulus.sr_hz, hop_s, win_s
    )
    log_power = np.log10(np.maximum(power, 1e-10)).astype(np.float32)
    n_freq = log_power.shape[1]
    # Spectral modulation is defined on a channel index axis (cycles/channel).
    channel_period = 1.0
    names: list[str] = []
    cols: list[np.ndarray] = []
    for rate_lo, rate_hi in DEFAULT_RATE_BANDS_HZ:
        rate_filt = _bandpass_axis(
            log_power, axis=0, sample_period=hop_s, lo=rate_lo, hi=rate_hi
        )
        rate_label = _rate_name(rate_lo, rate_hi)
        for scale_lo, scale_hi in DEFAULT_SCALE_BANDS:
            scale_filt = _bandpass_axis(
                rate_filt,
                axis=1,
                sample_period=channel_period,
                lo=scale_lo,
                hi=scale_hi,
            )
            energy = np.sqrt(np.mean(scale_filt * scale_filt, axis=1))
            label = f"{rate_label}__{_scale_name(scale_lo, scale_hi)}"
            if compressed:
                names.append(label)
                cols.append(energy.astype(np.float32))
            else:
                for freq_i in range(n_freq):
                    names.append(f"{label}__f{freq_i}")
                    cols.append(np.abs(scale_filt[:, freq_i]).astype(np.float32))
    values = np.stack(cols, axis=1).astype(np.float32)
    times = times_from_hop(
        values.shape[0],
        hop_s,
        start_offset_s=stimulus.start_offset_s,
        center=True,
        window_s=win_s,
    )
    md = extractor_metadata(
        "audio.modulation.spectrotemporal",
        params={"hop_s": hop_s, "win_s": win_s, "compressed": compressed},
        extra={"backend": "stft_rate_scale"},
    )
    return FeatureSeries(
        values=values,
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": names},
        metadata=md,
        timebase=TimebaseSpec(
            kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s
        ),
    )
