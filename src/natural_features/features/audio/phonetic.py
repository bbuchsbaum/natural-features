"""Named acoustic-phonetic cues: formants and harmonicity."""

from __future__ import annotations

from dataclasses import replace
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
    hop = max(1, int(round(stimulus.sr_hz * hop_s)))
    win = max(1, int(round(stimulus.sr_hz * win_s)))
    n = 1 + max(0, (samples.size - win) // hop)
    # Sound's clock is local to the waveform. Add the stimulus offset only
    # when constructing the public output clock, never for Praat queries.
    times = (np.arange(n) * hop + win / 2.0) / stimulus.sr_hz
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

    Voicing strength from windows of at least 40 ms is linearly interpolated
    at the sample-derived formant centers, then thresholded at 0.3. Centers
    outside the pitch-center range are zero-gated (no extrapolation). Gating
    preserves the formant grid, including its edge rows. This is an acoustic
    periodicity gate, not a vocal detector for polyphonic audio.
    """

    mode, _strict = resolve_execution_mode(
        execution_mode=execution_mode, strict_dependency=strict_dependency
    )
    if not np.isfinite(hop_s) or not np.isfinite(win_s) or hop_s <= 0 or win_s <= 0:
        raise ValueError("hop_s and win_s must be finite and positive")
    hop = max(1, int(round(stimulus.sr_hz * hop_s)))
    win = max(1, int(round(stimulus.sr_hz * win_s)))
    effective_hop = hop / stimulus.sr_hz
    effective_win = win / stimulus.sr_hz
    if backend == "lpc_autocorr":
        raw = _formants_lpc(stimulus, hop_s=effective_hop, win_s=effective_win, order=lpc_order)
    elif backend == "parselmouth":
        raw = _formants_parselmouth(stimulus, hop_s=effective_hop, win_s=effective_win)
    else:
        raise ValueError("backend must be 'lpc_autocorr' or 'parselmouth'")
    deltas = np.diff(raw[:, :3], axis=0, prepend=raw[:1, :3])
    values = np.concatenate([raw, deltas], axis=1).astype(np.float32)
    starts = np.arange(len(raw)) * hop
    local_times = (starts + win / 2.0) / stimulus.sr_hz
    bounds = stimulus.start_offset_s + np.column_stack([starts, starts + win]) / stimulus.sr_hz
    times = stimulus.start_offset_s + local_times
    if voicing_gate:
        pitch_win = max(win, int(round(stimulus.sr_hz * 0.04))) / stimulus.sr_hz
        pitch = audio_pitch(replace(stimulus, start_offset_s=0.0), hop_s=effective_hop, win_s=pitch_win)
        # Both clocks use integer or half-integer sample centers. Normalize
        # floating arithmetic so equal edge centers are not treated as outside.
        pitch_times = np.rint(pitch.times_s * (2 * stimulus.sr_hz)) / (2 * stimulus.sr_hz)
        strength = np.interp(
            local_times, pitch_times,
            pitch.values[:, 1], left=0.0, right=0.0,
        )
        values[strength <= 0.3] = 0.0
    extra: dict[str, Any] = {
        "backend": backend,
        "effective_hop_s": effective_hop,
        "effective_window_s": effective_win,
        "right_padding_samples": max(0, win - len(stimulus.samples)) if backend == "lpc_autocorr" else 0,
        "voicing_alignment": "linear_strength_at_formant_centers_zero_outside" if voicing_gate else "disabled",
        "voicing_threshold": 0.3 if voicing_gate else None,
        "voicing_window_s": pitch_win if voicing_gate else None,
    }
    md = add_execution_provenance(
        extractor_metadata(
            "audio.formants",
            code_version="formants-v2",
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
            kind="audio_hop", hop_s=effective_hop, sampling_rate_hz=1.0 / effective_hop,
            # Praat values are interpolated queries of its own Gaussian-window
            # analysis, not summaries of these LPC sample bounds.
            window_s=effective_win if backend == "lpc_autocorr" else None,
        ),
        time_bounds_s=bounds if backend == "lpc_autocorr" else None,
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
