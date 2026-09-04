"""Interval-level phonetic cues rasterized onto an audio hop grid."""

from __future__ import annotations

import numpy as np

from natural_features.core.feature_types import EventSeries, FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec, times_from_hop
from natural_features.features.audio.lowlevel import _frames, _mono, _stft_power
from natural_features.features.common import extractor_metadata
from natural_features.features.speech.phonology import _normalize_phone_label

_STOPS = {"P", "B", "T", "D", "K", "G"}
_VOICELESS_STOPS = {"P", "T", "K"}
_FRICATIVES = {"F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"}
_VOWELS = {
    "IY",
    "IH",
    "EH",
    "AE",
    "AA",
    "AO",
    "UH",
    "UW",
    "AH",
    "ER",
    "EY",
    "AY",
    "OW",
    "AW",
    "OY",
}
CUE_NAMES = ["vot", "burst_centroid", "frication_duration", "closure_duration"]


def _spectral_centroid(power: np.ndarray, freqs: np.ndarray) -> float:
    weights = np.maximum(power, 0.0)
    denom = float(np.sum(weights))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(freqs * weights) / denom)


def phonetic_cues(
    stimulus: AudioStimulus,
    phones: EventSeries,
    *,
    hop_s: float = 0.01,
    win_s: float = 0.025,
    burst_s: float = 0.015,
) -> FeatureSeries:
    """Rasterize VOT, burst centroid, frication duration, and closure duration.

    Requires phone events. Values are written onto frames that fall inside the
    measured interval and are zero elsewhere.
    """

    x = _mono(stimulus.samples)
    n_frames = _frames(x, stimulus.sr_hz, hop_s, win_s)[0].shape[0]
    times = times_from_hop(
        n_frames,
        hop_s,
        start_offset_s=stimulus.start_offset_s,
        center=True,
        window_s=win_s,
    )
    values = np.zeros((n_frames, 4), dtype=np.float32)
    labels = phones.label if phones.label is not None else np.array([], dtype=object)
    power, freqs, _ = _stft_power(x, stimulus.sr_hz, hop_s, win_s)

    def _fill(onset: float, offset: float, col: int, value: float) -> None:
        mask = (times >= onset) & (times < max(offset, onset + hop_s))
        values[mask, col] = np.float32(value)

    for i in range(len(phones)):
        lab = _normalize_phone_label(str(labels[i]) if i < len(labels) else "")
        on = float(phones.onset_s[i])
        off = float(phones.offset_s[i])
        if lab in _STOPS:
            _fill(on, off, 3, max(off - on, 0.0))
            burst_end = min(off, on + burst_s)
            burst_mask = (times >= on) & (times < burst_end)
            if np.any(burst_mask):
                centroid = _spectral_centroid(power[burst_mask].mean(axis=0), freqs)
                _fill(on, burst_end, 1, centroid)
            if lab in _VOICELESS_STOPS and i + 1 < len(phones):
                nxt = _normalize_phone_label(str(labels[i + 1]))
                if nxt in _VOWELS:
                    vot = max(float(phones.onset_s[i + 1]) - on, 0.0)
                    _fill(on, float(phones.onset_s[i + 1]), 0, vot)
        if lab in _FRICATIVES:
            _fill(on, off, 2, max(off - on, 0.0))

    md = extractor_metadata(
        "speech.phonetic.cues",
        params={"hop_s": hop_s, "win_s": win_s, "burst_s": burst_s},
        extra={"backend": "phone_interval_cues"},
    )
    return FeatureSeries(
        values=values,
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": list(CUE_NAMES)},
        metadata=md,
        timebase=TimebaseSpec(
            kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s
        ),
    )
