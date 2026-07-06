"""Speech presence baselines."""

from __future__ import annotations

import numpy as np

from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import EventSeries, FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec, times_from_hop
from natural_features.features.common import extractor_metadata
from natural_features.features.audio.lowlevel import _frames, _mono


def energy_vad(
    stimulus: AudioStimulus,
    *,
    hop_s: float = 0.02,
    win_s: float = 0.03,
    threshold: float = 0.5,
) -> FeatureSeries:
    x = _mono(stimulus.samples)
    framed, _ = _frames(x, stimulus.sr_hz, hop_s, win_s)
    energy = np.sqrt(np.mean(framed * framed, axis=1))
    if len(energy) == 0:
        prob = energy
    else:
        norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
        prob = np.clip(norm / max(threshold, 1e-6), 0.0, 1.0)
    vals = prob[:, None].astype(np.float32)
    times = times_from_hop(vals.shape[0], hop_s, start_offset_s=stimulus.start_offset_s, center=True, window_s=win_s)
    metadata = extractor_metadata(
        "speech.vad.energy_vad",
        params={"hop_s": hop_s, "win_s": win_s, "threshold": threshold},
    )
    return FeatureSeries(
        values=vals,
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": ["speech_probability"]},
        metadata=metadata,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s),
    )


def speech_vad(
    stimulus: AudioStimulus,
    *,
    frame_s: float = 0.02,
    win_s: float = 0.03,
    threshold: float = 0.5,
) -> EventSeries:
    """Return contiguous speech events from the energy VAD probability series."""

    if frame_s <= 0:
        raise ValueError("frame_s must be > 0")
    if win_s <= 0:
        raise ValueError("win_s must be > 0")
    if threshold < 0:
        raise ValueError("threshold must be >= 0")
    base = energy_vad(stimulus, hop_s=frame_s, win_s=win_s, threshold=threshold)
    prob = base.values[:, 0]
    active = prob >= float(threshold)
    starts: list[int] = []
    stops: list[int] = []
    in_run = False
    for i, flag in enumerate(active):
        if flag and not in_run:
            starts.append(i)
            in_run = True
        elif not flag and in_run:
            stops.append(i)
            in_run = False
    if in_run:
        stops.append(len(active))

    onset = []
    offset = []
    confidence = []
    half_win = float(win_s) / 2.0
    for start, stop in zip(starts, stops, strict=True):
        onset.append(max(float(stimulus.start_offset_s), float(base.times_s[start]) - half_win))
        offset.append(float(base.times_s[stop - 1]) + half_win)
        confidence.append(float(prob[start:stop].mean()))

    md = extractor_metadata(
        "speech.vad",
        params={"frame_s": frame_s, "win_s": win_s, "threshold": threshold},
        extra={"backend": "energy_threshold", "source_extractor": base.metadata.get("extractor_name", "unknown")},
    )
    return EventSeries(
        onset_s=np.asarray(onset, dtype=np.float64),
        offset_s=np.asarray(offset, dtype=np.float64),
        label=np.asarray(["speech"] * len(onset), dtype=object),
        confidence=np.asarray(confidence, dtype=np.float32),
        extra={
            "frame_start": np.asarray(starts, dtype=np.int64),
            "frame_stop": np.asarray(stops, dtype=np.int64),
        },
        metadata=md,
        timebase=TimebaseSpec(kind="events"),
    )


def neural_vad(
    stimulus: AudioStimulus,
    *,
    model: str = "silero_vad",
    hop_s: float = 0.02,
    win_s: float = 0.03,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    """Return neural-VAD-compatible speech probabilities with a deterministic fallback."""

    mode, strict = resolve_execution_mode(execution_mode=execution_mode, strict_dependency=strict_dependency)
    try:
        import torch  # type: ignore  # noqa: F401
    except Exception as exc:
        if strict:
            raise RuntimeError("torch is required for strict neural VAD extraction.") from exc
    else:
        if strict:
            raise RuntimeError(
                "strict neural VAD backend is not configured; use fallback mode or install a supported backend."
            )
    base = energy_vad(stimulus, hop_s=hop_s, win_s=win_s)
    prob = base.values[:, 0]
    smooth = np.convolve(prob, np.ones(3, dtype=np.float32) / 3.0, mode="same") if prob.size else prob
    values = smooth[:, None].astype(np.float32)
    md = add_execution_provenance(
        extractor_metadata(
            "speech.neural_vad",
            params={"model": model, "hop_s": hop_s, "win_s": win_s},
            extra={"backend": "energy_proxy"},
        ),
        execution_mode=mode,
        fallback_used=True,
        fallback_reason="neural VAD backend unavailable",
    )
    return FeatureSeries(
        values=values,
        times_s=base.times_s,
        dims=("time", "feature"),
        coords={"feature": ["speech_probability"]},
        metadata=md,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s),
    )
