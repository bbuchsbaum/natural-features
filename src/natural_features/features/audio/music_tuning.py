"""Bounded, recording-wide tuning evidence, independent of optional backends.

Tuning is identifiable modulo a semitone. This spectral-peak heuristic reports
support, not a calibrated probability or proof of a single recording-wide tuning.
"""
from __future__ import annotations

import numpy as np

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio._music_contract import computation_context, mono_audio


def estimate_music_tuning(stimulus: AudioStimulus) -> dict:
    """Estimate cents relative to A440 in [-50, 50), with explicit support.

At most 256 evenly spaced complete 0.5-second Hann frames are examined.
Log-power parabolic peak interpolation precedes a power-weighted circular
mean of semitone residuals. Strong peaks must exceed both 1% of the frame's
peak power and eight times the median band power. Unsupported estimates have
``tuning_hz=None``; callers must choose an explicit reference or other method.
"""
    x = mono_audio(stimulus).astype(np.float64)
    sr = int(stimulus.sr_hz)
    w, h = max(4, round(0.5 * sr)), max(1, round(0.25 * sr))
    starts = np.arange(0, max(0, len(x) - w + 1), h, dtype=int)
    if len(starts) > 256:
        starts = starts[np.linspace(0, len(starts) - 1, 256).round().astype(int)]
    nfft = 1 << int(np.ceil(np.log2(4 * w)))
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    band = (freqs >= 55) & (freqs <= min(5000, sr / 2))
    window = np.hanning(w + 1)[:-1]
    residuals, weights, notes = [], [], []
    supported_frames = 0
    for start in starts:
        frame = x[start:start + w]
        frame = frame - frame.mean()
        if np.sqrt(np.mean(frame * frame)) <= 1e-6 or not np.any(band):
            continue
        power = np.abs(np.fft.rfft(frame * window, n=nfft)) ** 2
        peak = np.flatnonzero((power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:])) + 1
        threshold = max(float(power[band].max()) * .01, float(np.median(power[band])) * 8)
        peak = peak[band[peak] & (power[peak] > threshold)]
        peak = peak[np.argsort(power[peak])[-24:]]
        if not len(peak):
            continue
        y = np.log(np.maximum(power, np.finfo(float).tiny))
        curvature = y[peak - 1] - 2 * y[peak] + y[peak + 1]
        delta = np.divide(.5 * (y[peak - 1] - y[peak + 1]), curvature,
                          out=np.zeros(len(peak)), where=np.abs(curvature) > 1e-15)
        hz = (peak + np.clip(delta, -.5, .5)) * sr / nfft
        semitones = 12 * np.log2(hz / 440)
        residuals.extend(((100 * semitones + 50) % 100 - 50).tolist())
        weights.extend(power[peak].tolist())
        notes.extend((np.round(semitones).astype(int) % 12).tolist())
        supported_frames += 1
    cents = np.asarray(residuals)
    weight = np.asarray(weights)
    concentration, inlier_fraction, offset = 0.0, 0.0, None
    if len(cents):
        weight = weight / weight.sum()
        resultant = np.sum(weight * np.exp(2j * np.pi * cents / 100))
        concentration = float(abs(resultant))
        offset = float((np.angle(resultant) * 100 / (2 * np.pi) + 50) % 100 - 50)
        distance = np.abs((cents - offset + 50) % 100 - 50)
        inlier_fraction = float(weight[distance <= 15].sum())
    boundary_ambiguous = bool(offset is not None and abs(offset) >= 45)
    supported = bool(supported_frames >= 3 and len(cents) >= 6
                     and concentration >= .7 and inlier_fraction >= .75 and not boundary_ambiguous)
    return {
        "estimator": "spectral-peak-circular-v1", "mode": "auto", "supported": supported,
        "tuning_hz": float(440 * 2 ** (offset / 1200)) if supported else None,
        "cents_from_a440": offset, "concentration": concentration,
        "inlier_weight_fraction": inlier_fraction, "peak_count": len(cents),
        "supported_frames": supported_frames, "analyzed_frames": len(starts),
        "pitch_class_count": len(set(notes)),
        "near_semitone_boundary": boundary_ambiguous,
        "scope": "whole_input", "frame_samples": w, "hop_samples": h,
        "fft_samples": nfft, "max_frames": 256, "analysis_band_hz": [55, min(5000, sr / 2)],
        "support_rule": "frames>=3; peaks>=6; concentration>=0.7; weight within15c>=0.75; abs(offset)<45c",
        "semantics": "modulo100c spectral consistency; not calibrated confidence or source activity",
    }


def resolve_tuning(stimulus: AudioStimulus, tuning_hz: float, tuning_mode: str,
                   *, past_only: bool = False) -> tuple[float, dict]:
    if tuning_mode not in {"fixed", "auto"}:
        raise ValueError("tuning_mode must be 'fixed' or 'auto'")
    if isinstance(tuning_hz, bool) or not np.isfinite(tuning_hz) or tuning_hz <= 0:
        raise ValueError("tuning_hz must be finite and positive")
    if tuning_mode == "auto":
        if past_only:
            raise ValueError("whole-input automatic tuning is incompatible with past_only; use a fixed reference from prior calibration")
        if tuning_hz != 440.0:
            raise ValueError("tuning_hz overrides require tuning_mode='fixed'")
        evidence = estimate_music_tuning(stimulus)
        if not evidence["supported"]:
            raise ValueError("automatic tuning is unsupported or ambiguous; inspect estimate_music_tuning() or supply a fixed tuning_hz")
        return evidence["tuning_hz"], evidence
    return float(tuning_hz), {
        "mode": "fixed", "tuning_hz": float(tuning_hz), "scope": "provided_reference",
        "cents_from_a440": float(1200 * np.log2(tuning_hz / 440)),
        "semantics": "caller-provided reference; no tuning estimate performed",
    }


def tuning_context(stimulus: AudioStimulus, bounds: np.ndarray, evidence: dict,
                   *, policy: str = "local", preprocessing: dict | None = None) -> dict:
    """Keep observation support distinct from whole-recording tuning dependency."""
    if evidence["mode"] == "auto":
        policy = "bidirectional"
        bounds = [[stimulus.start_offset_s,
                   stimulus.start_offset_s + len(stimulus.samples) / stimulus.sr_hz]]
    return computation_context(stimulus, policy=policy, bounds_s=bounds,
                               preprocessing={**(preprocessing or {}), "tuning": evidence})
