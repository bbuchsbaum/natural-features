"""Named acoustic-phonetic cues: formants and harmonicity."""

from __future__ import annotations

from typing import Any

import numpy as np

from natural_features.core.backend_errors import (
    BackendDependencyError,
    BackendInferenceError,
)
from natural_features.core.execution import (
    add_execution_provenance,
    resolve_execution_mode,
)
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec, times_from_hop
from natural_features.features.audio.lowlevel import _frames, _mono
from natural_features.features.common import extractor_metadata
from natural_features.features.audio.prosody import audio_pitch

FORMANT_NAMES = [
    "f1",
    "f2",
    "f3",
    "b1",
    "b2",
    "b3",
    "delta_f1",
    "delta_f2",
    "delta_f3",
]


def _preemphasis(x: np.ndarray, coef: float = 0.97) -> np.ndarray:
    out = np.empty_like(x)
    out[0] = x[0]
    out[1:] = x[1:] - coef * x[:-1]
    return out


def _lpc_autocorr(frame: np.ndarray, order: int) -> np.ndarray:
    x = np.asarray(frame, dtype=np.float64)
    if x.size <= order:
        return np.concatenate([[1.0], np.zeros(order)])
    r = np.correlate(x, x, mode="full")[x.size - 1 : x.size + order]
    r = np.asarray(r, dtype=np.float64)
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    e = float(r[0])
    if e <= 1e-12:
        return a
    for i in range(1, order + 1):
        acc = float(r[i])
        if i > 1:
            acc += float(np.dot(a[1:i], r[i - 1 : 0 : -1]))
        k = -acc / max(e, 1e-12)
        if i > 1:
            a[1:i] = a[1:i] + k * a[i - 1 : 0 : -1]
        a[i] = k
        e *= max(1.0 - k * k, 1e-8)
    return a


def _formants_from_lpc(
    a: np.ndarray, sr_hz: float, n_formants: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    roots = np.roots(a)
    roots = roots[np.imag(roots) >= 0.0]
    if roots.size == 0:
        return np.zeros(n_formants), np.zeros(n_formants)
    ang = np.arctan2(np.imag(roots), np.real(roots))
    freqs = ang * (sr_hz / (2.0 * np.pi))
    radii = np.abs(roots)
    bw = -np.log(np.clip(radii, 1e-6, 0.999999)) * (sr_hz / np.pi)
    keep = (freqs > 50.0) & (freqs < (sr_hz / 2.0) - 50.0)
    freqs = freqs[keep]
    bw = bw[keep]
    order = np.argsort(freqs)
    freqs = freqs[order]
    bw = bw[order]
    out_f = np.zeros(n_formants, dtype=np.float64)
    out_b = np.zeros(n_formants, dtype=np.float64)
    n = min(n_formants, freqs.size)
    out_f[:n] = freqs[:n]
    out_b[:n] = bw[:n]
    return out_f, out_b


def _formants_lpc(
    stimulus: AudioStimulus,
    *,
    hop_s: float,
    win_s: float,
    order: int,
) -> np.ndarray:
    x = _preemphasis(_mono(stimulus.samples))
    frames, _ = _frames(x, stimulus.sr_hz, hop_s, win_s)
    n = frames.shape[0]
    f123 = np.zeros((n, 3), dtype=np.float64)
    b123 = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        a = _lpc_autocorr(frames[i], order)
        freqs, bw = _formants_from_lpc(a, float(stimulus.sr_hz), n_formants=3)
        f123[i] = freqs
        b123[i] = bw
    return np.concatenate([f123, b123], axis=1)


def _formants_parselmouth(
    stimulus: AudioStimulus,
    *,
    hop_s: float,
    win_s: float,
) -> np.ndarray:
    try:
        import parselmouth  # type: ignore
    except ImportError as exc:
        raise BackendDependencyError(
            "formants",
            "praat-parselmouth is required for backend='parselmouth'",
        ) from exc
    samples = _mono(stimulus.samples).astype(np.float64)
    try:
        sound = parselmouth.Sound(samples, sampling_frequency=float(stimulus.sr_hz))
        formant = sound.to_formant_burg(time_step=hop_s, window_length=win_s)
    except Exception as exc:
        raise BackendInferenceError(
            "formants", "Praat formant analysis failed"
        ) from exc
    n = 1 + max(0, int(round((samples.size / float(stimulus.sr_hz) - win_s) / hop_s)))
    times = times_from_hop(
        n, hop_s, start_offset_s=stimulus.start_offset_s, center=True, window_s=win_s
    )
    out = np.zeros((n, 6), dtype=np.float64)
    for i, t in enumerate(times):
        for k in range(3):
            freq = formant.get_value_at_time(k + 1, float(t))
            bw = formant.get_bandwidth_at_time(k + 1, float(t))
            out[i, k] = 0.0 if freq is None or not np.isfinite(freq) else float(freq)
            out[i, k + 3] = 0.0 if bw is None or not np.isfinite(bw) else float(bw)
    return out


def audio_formants(
    stimulus: AudioStimulus,
    *,
    hop_s: float = 0.01,
    win_s: float = 0.025,
    backend: str = "lpc_autocorr",
    lpc_order: int = 12,
    voicing_gate: bool = True,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    """Estimate F1–F3, bandwidths, and ΔF.

    Default backend is autocorrelation LPC, not Praat. Use
    ``backend="parselmouth"`` for the optional Praat extra.
    """

    mode, _strict = resolve_execution_mode(
        execution_mode=execution_mode, strict_dependency=strict_dependency
    )
    if backend == "lpc_autocorr":
        raw = _formants_lpc(stimulus, hop_s=hop_s, win_s=win_s, order=lpc_order)
    elif backend == "parselmouth":
        raw = _formants_parselmouth(stimulus, hop_s=hop_s, win_s=win_s)
    else:
        raise ValueError("backend must be 'lpc_autocorr' or 'parselmouth'")
    deltas = np.diff(raw[:, :3], axis=0, prepend=raw[:1, :3])
    values = np.concatenate([raw, deltas], axis=1).astype(np.float32)
    if voicing_gate:
        pitch = audio_pitch(stimulus, hop_s=hop_s, win_s=max(win_s, 0.04))
        n = min(values.shape[0], pitch.values.shape[0])
        voiced = pitch.values[:n, 1] > 0.3
        values = values[:n]
        values[~voiced] = 0.0
    times = times_from_hop(
        values.shape[0],
        hop_s,
        start_offset_s=stimulus.start_offset_s,
        center=True,
        window_s=win_s,
    )
    extra: dict[str, Any] = {"backend": backend}
    md = add_execution_provenance(
        extractor_metadata(
            "audio.formants",
            params={
                "hop_s": hop_s,
                "win_s": win_s,
                "backend": backend,
                "lpc_order": lpc_order,
                "voicing_gate": voicing_gate,
            },
            extra=extra,
        ),
        execution_mode=mode,
        fallback_used=False,
    )
    return FeatureSeries(
        values=values.astype(np.float32),
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": list(FORMANT_NAMES)},
        metadata=md,
        timebase=TimebaseSpec(
            kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s
        ),
    )


def audio_harmonicity(
    stimulus: AudioStimulus,
    *,
    hop_s: float = 0.01,
    win_s: float = 0.04,
    fmin: float = 75.0,
    fmax: float = 400.0,
) -> FeatureSeries:
    """Framewise harmonic-to-noise ratio from autocorrelation."""

    frames, _ = _frames(_mono(stimulus.samples), stimulus.sr_hz, hop_s, win_s)
    sr = float(stimulus.sr_hz)
    min_lag = max(1, int(round(sr / fmax)))
    max_lag = max(min_lag + 1, int(round(sr / fmin)))
    hnr = np.zeros((frames.shape[0], 1), dtype=np.float32)
    for i, frame in enumerate(frames):
        x = frame.astype(np.float64)
        r0 = float(np.dot(x, x))
        if r0 <= 1e-12:
            continue
        corr = np.correlate(x, x, mode="full")
        mid = x.size - 1
        peak = 0.0
        hi = min(max_lag, x.size - 1)
        if hi > min_lag:
            peak = float(np.max(corr[mid + min_lag : mid + hi + 1]))
        noise = max(r0 - peak, 1e-12)
        hnr[i, 0] = np.float32(10.0 * np.log10(max(peak, 1e-12) / noise))
    times = times_from_hop(
        hnr.shape[0],
        hop_s,
        start_offset_s=stimulus.start_offset_s,
        center=True,
        window_s=win_s,
    )
    md = extractor_metadata(
        "audio.harmonicity",
        params={"hop_s": hop_s, "win_s": win_s, "fmin": fmin, "fmax": fmax},
        extra={"backend": "autocorr_hnr"},
    )
    return FeatureSeries(
        values=hnr,
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": ["hnr"]},
        metadata=md,
        timebase=TimebaseSpec(
            kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s
        ),
    )
