"""Music-theoretic audio features.

These extractors cover the tonal and rhythmic structure that the speech-oriented
audio family does not: pitch-class content (:func:`music_chroma`), its tonal-centroid
projection (:func:`music_tonnetz`), key and mode (:func:`music_tonality`), and the
onset/tempo family (:func:`music_onset_strength`, :func:`music_tempogram`,
:func:`music_rhythm`).

Two sampling conventions appear here and they are deliberately different:

* **Frame-rate features** (chroma, tonnetz, onset strength) are emitted on the same
  ``hop_s`` grid as the rest of the audio family, because they are defined
  instantaneously.
* **Window features** (tonality, tempogram, rhythm) require an integration window to
  have any meaning at all -- a key or a tempo is not defined at one STFT frame -- so
  they take an explicit ``window_s`` and are emitted at window centres. Callers that
  want several integration scales should call them once per scale rather than
  resampling one output, which is why ``window_s`` is a parameter and not a constant.

Everything is computed with :mod:`numpy` only, matching the dependency-light contract
of the rest of the audio family. ``tests/unit/test_music_parity.py`` checks the
librosa-comparable pieces against librosa when it is installed.
"""

from __future__ import annotations

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import SupportSpec, TimebaseSpec, times_from_hop
from natural_features.features.audio._music_contract import mono_audio, positive_seconds
from natural_features.features.audio.lowlevel import _mel_filterbank, _mono, _stft_power
from natural_features.features.audio.music_tuning import resolve_tuning, tuning_context
from natural_features.features.common import extractor_metadata
from natural_features.util.hashing import stable_hash

__all__ = [
    "music_chroma",
    "music_onset_strength",
    "music_rhythm",
    "music_tempogram",
    "music_tonality",
    "music_tonnetz",
]

PITCH_CLASSES = ("C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B")

# Krumhansl & Kessler (1982) probe-tone profiles, in C.
KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _normalize_rows(x: np.ndarray, norm: str | None) -> np.ndarray:
    """Row-wise normalisation shared by chroma and its consumers."""

    if norm is None:
        return x
    if norm == "l1":
        d = np.abs(x).sum(axis=1, keepdims=True)
    elif norm == "l2":
        d = np.sqrt((x * x).sum(axis=1, keepdims=True))
    elif norm == "max":
        d = np.abs(x).max(axis=1, keepdims=True)
    else:
        raise ValueError("norm must be one of 'l1', 'l2', 'max', or None")
    # A silent frame has no pitch-class content; leave it as zeros rather than
    # amplifying numerical noise to unit norm.
    return x / np.maximum(d, 1e-12)


def _chroma_filterbank(
    freqs: np.ndarray,
    *,
    n_chroma: int = 12,
    tuning_hz: float = 440.0,
    fmin: float = 55.0,
    fmax: float = 5000.0,
    sigma: float = 0.5,
    octwidth: float | None = None,
    ctroct: float = 5.0,
) -> np.ndarray:
    """Map FFT bin frequencies onto pitch classes.

    Each bin contributes to every pitch class with a Gaussian weight on the circular
    pitch-class axis, so the bank is smooth and has no hard bin edges. Bins outside
    ``[fmin, fmax]`` contribute nothing: below ``fmin`` the pitch estimate is unstable
    and above ``fmax`` partials dominate the fundamental.

    Returns an ``(n_chroma, n_bins)`` matrix.
    """

    positive_seconds(tuning_hz, "tuning_hz")
    if n_chroma <= 0:
        raise ValueError("n_chroma must be > 0")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if fmin <= 0 or fmax <= fmin:
        raise ValueError("require 0 < fmin < fmax")

    fb = np.zeros((n_chroma, freqs.shape[0]), dtype=np.float64)
    live = (freqs >= fmin) & (freqs <= fmax) & (freqs > 0)
    if not np.any(live):
        return fb.astype(np.float32)

    f = freqs[live]
    # Continuous pitch class: MIDI 60 is C4, and 60 % 12 == 0, so class 0 is C.
    midi = 69.0 + 12.0 * np.log2(f / tuning_hz)
    pc = np.mod(midi, n_chroma)

    classes = np.arange(n_chroma, dtype=np.float64)[:, None]
    d = pc[None, :] - classes
    # Wrap to the shortest signed distance on the pitch-class circle.
    d = np.mod(d + n_chroma / 2.0, n_chroma) - n_chroma / 2.0
    w = np.exp(-0.5 * (d / sigma) ** 2)

    if octwidth is not None:
        if octwidth <= 0:
            raise ValueError("octwidth must be > 0 when supplied")
        oct_pos = midi / 12.0
        w = w * np.exp(-0.5 * ((oct_pos - ctroct) / octwidth) ** 2)[None, :]

    fb[:, live] = w
    return fb.astype(np.float32)


def _chroma_matrix(
    stimulus: AudioStimulus,
    *,
    hop_s: float,
    win_s: float,
    n_chroma: int,
    tuning_hz: float,
    fmin: float,
    fmax: float,
    sigma: float,
    octwidth: float | None,
    norm: str | None,
    power_exp: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(chroma[time, n_chroma], frame_times)``."""

    positive_seconds(hop_s, "hop_s")
    positive_seconds(win_s, "win_s")
    x = mono_audio(stimulus).astype(np.float64)
    sr = int(stimulus.sr_hz)
    h, w = max(1, round(hop_s * sr)), max(1, round(win_s * sr))
    count = 1 + max(0, (len(x) - w) // h)
    freqs = np.fft.rfftfreq(w, 1 / sr)
    fb = _chroma_filterbank(
        freqs,
        n_chroma=n_chroma,
        tuning_hz=tuning_hz,
        fmin=fmin,
        fmax=fmax,
        sigma=sigma,
        octwidth=octwidth,
    )
    chroma = np.empty((count, n_chroma), dtype=np.float64)
    window = np.hanning(w + 1)[:-1]
    for first, frames in _centered_frames(x, count, h, w):
        power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
        mag = power ** (power_exp / 2.0)
        chroma[first:first + len(frames)] = mag @ fb.T
    chroma = _normalize_rows(chroma.astype(np.float64), norm).astype(np.float32)
    times = _frame_bounds(stimulus, len(chroma), hop_s, win_s).mean(axis=1)
    return chroma, times


def _centered_frames(x, count, hop, window):
    """Bounded reductions avoid prefix-energy cancellation and DC leakage."""
    if len(x) < window:
        x = np.pad(x, (0, window - len(x)))
    for first in range(0, count, 32):
        starts = np.arange(first, min(first + 32, count)) * hop
        frames = x[starts[:, None] + np.arange(window)[None, :]]
        yield first, frames - frames.mean(axis=1, keepdims=True)


def _frame_bounds(stimulus, count, hop_s, win_s):
    sr = int(stimulus.sr_hz)
    h, w = max(1, round(hop_s * sr)), max(1, round(win_s * sr))
    starts = np.arange(count) * h
    return stimulus.start_offset_s + np.column_stack((starts, np.minimum(starts + w, len(stimulus.samples)))) / sr


def _window_outputs(stimulus, *, values, names, diagnostics, diagnostic_names,
                    bounds, dependencies, hop_s, extractor, params, code_version,
                    tuning=None, extra=None):
    """Named continuous and diagnostic outputs; NaNs remain explicit missingness."""
    evidence = tuning or {"mode": "fixed", "scope": "not_applicable"}
    context = tuning_context(stimulus, dependencies, evidence,
                             preprocessing={"frontend": "windowed_stft", "frame_dc_removed": tuning is not None})
    pair_id = stable_hash({"samples": np.asarray(stimulus.samples), "sr_hz": stimulus.sr_hz,
                           "clock": str(stimulus.clock), "offset": stimulus.start_offset_s,
                           "extractor": extractor, "params": params, "code_version": code_version}, length=24)
    result = {}
    for key, data, columns in (("default", values, names), ("diagnostics", diagnostics, diagnostic_names)):
        md = extractor_metadata(extractor if key == "default" else extractor + ".diagnostics",
                                params=params, code_version=code_version,
                                extra={"computation_context": context, "output_pair_id": pair_id,
                                       "missing_values": "NaN; feature-specific support in diagnostics",
                                       "requires_validity": True, "validity_output": "diagnostics",
                                       "output_role": "continuous" if key == "default" else "diagnostic",
                                       **(extra or {})})
        result[key] = FeatureSeries(
            values=np.asarray(data, dtype=np.float32), times_s=bounds.mean(axis=1),
            coords={"feature": columns}, metadata=md,
            timebase=TimebaseSpec(kind="audio_hop", reference=stimulus.clock, hop_s=hop_s,
                                  sampling_rate_hz=1 / hop_s, support=SupportSpec(kind="interval", anchor="center")),
            time_bounds_s=bounds, temporal_context=stimulus.temporal_context,
        )
    return result


def _tonnetz_from_chroma(chroma: np.ndarray) -> np.ndarray:
    """Harte, Sandler & Gasser (2006) tonal centroid, ``(time, 6)``.

    Uses the same basis as ``librosa.feature.tonnetz``: perfect-fifth, minor-third and
    major-third circles, the thirds at half radius so the fifth circle dominates.
    """

    scale = np.array([7.0 / 6, 7.0 / 6, 3.0 / 2, 3.0 / 2, 2.0 / 3, 2.0 / 3])
    v = scale[:, None] * np.arange(12, dtype=np.float64)[None, :]
    v[::2] -= 0.5  # the sine rows are a quarter turn from the cosine rows
    radius = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.5])
    phi = radius[:, None] * np.cos(np.pi * v)
    cn = _normalize_rows(chroma.astype(np.float64), "l1")
    return (cn @ phi.T).astype(np.float32)


def _onset_envelope(
    stimulus: AudioStimulus,
    *,
    hop_s: float,
    win_s: float,
    n_mels: int,
    fmin: float,
    fmax: float | None,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Half-wave-rectified spectral flux on a mel power spectrogram, ``(time,)``."""

    if lag < 1:
        raise ValueError("lag must be >= 1")
    x = _mono(stimulus.samples)
    power, _freqs, _ = _stft_power(x, stimulus.sr_hz, hop_s, win_s)
    n_fft = int(2 * (power.shape[1] - 1))
    top = float(fmax if fmax is not None else stimulus.sr_hz / 2.0)
    fb = _mel_filterbank(stimulus.sr_hz, n_fft, n_mels, fmin, top)
    mel = power @ fb.T
    # Power in dB. librosa's onset_strength differences a dB-scaled spectrogram, and the
    # compression is what stops loud frames from dominating the flux.
    db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    flux = np.zeros(db.shape[0], dtype=np.float64)
    if db.shape[0] > lag:
        diff = db[lag:] - db[:-lag]
        flux[lag:] = np.maximum(diff, 0.0).mean(axis=1)
    times = times_from_hop(
        flux.shape[0], hop_s, start_offset_s=stimulus.start_offset_s, center=True, window_s=win_s
    )
    return flux.astype(np.float32), times


def _window_bounds(n_frames: int, frame_rate: float, window_s: float, hop_s: float) -> np.ndarray:
    """Start indices of analysis windows; each window is fully inside the signal."""

    w = int(round(window_s * frame_rate))
    h = max(1, int(round(hop_s * frame_rate)))
    if w < 2:
        raise ValueError("window_s is too short for the frame rate")
    if n_frames < w:
        return np.zeros(0, dtype=int)
    return np.arange(0, n_frames - w + 1, h, dtype=int)


def _tempo_axis(frame_rate: float, n_lags: int, bpm_min: float, bpm_max: float) -> np.ndarray:
    """Lag indices (in frames) whose implied tempo lies in ``[bpm_min, bpm_max]``."""

    lags = np.arange(1, n_lags, dtype=np.float64)
    bpm = 60.0 * frame_rate / lags
    keep = (bpm >= bpm_min) & (bpm <= bpm_max)
    return lags[keep].astype(int)


def _autocorr_tempogram(env: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation of one onset-envelope window at ``lags``."""

    e = env - env.mean()
    denom = float(e @ e)
    if denom <= 1e-12:
        return np.zeros(lags.shape[0], dtype=np.float64)
    out = np.empty(lags.shape[0], dtype=np.float64)
    for i, lag in enumerate(lags):
        out[i] = float(e[lag:] @ e[:-lag]) / denom
    return out


def _window_relative_late_phase_onset_energy_fraction(
    env: np.ndarray, period: int
) -> float:
    """Return onset energy in the latter half of an origin-anchored period.

    The phase origin is the first frame of the analysis window. This is a
    deterministic window-relative descriptor, not a beat-aligned measure. A
    silent window or an unavailable period has no defined fraction.
    """

    total_energy = float(env.sum())
    if period < 2 or total_energy <= 1e-12:
        return float("nan")
    phase = np.arange(env.shape[0]) % period
    return float(env[phase >= period / 2.0].sum() / total_energy)


# --------------------------------------------------------------------------------------
# frame-rate extractors
# --------------------------------------------------------------------------------------


def music_chroma(
    stimulus: AudioStimulus,
    *,
    hop_s: float = 0.01,
    win_s: float = 0.186,
    n_chroma: int = 12,
    tuning_hz: float = 440.0,
    tuning_mode: str = "fixed",
    fmin: float = 55.0,
    fmax: float = 5000.0,
    sigma: float = 0.5,
    octwidth: float | None = None,
    norm: str | None = "l2",
    power_exp: float = 2.0,
) -> FeatureSeries:
    """Return pitch-class energy (chroma).

    Two defaults here are set by low-register leakage rather than by convention, and
    both were chosen against a measured criterion: an equal-amplitude C-major triad
    must be scored as C major, not as its relative minor.

    ``win_s`` is much longer than the audio family's 0.025 s because pitch-class
    assignment needs *frequency* resolution. FFT bin spacing is constant in Hz while a
    semitone is constant in ratio, so a semitone spans fewer bins the lower you go: at
    0.093 s and 22.05 kHz the bins are ~11 Hz and a semitone at C4 (262 Hz) is only
    1.4 bins wide, so the STFT peak itself straddles neighbouring pitch classes and the
    low notes of a chord are systematically under-weighted. 0.186 s halves the bin
    spacing and removes that bias.

    ``power_exp`` is 2.0 (power) rather than 1.0 (magnitude) for the same reason:
    squaring suppresses the leakage skirts relative to the peak. Together they take the
    off-note floor of a synthetic triad from 7% of total chroma energy to under 2%.

    A recording tuned appreciably away from A440 should set ``tuning_hz``; the bank is
    deliberately narrow (``sigma`` 0.5 semitone). ``tuning_mode="auto"`` estimates
    once over the whole input and declares that dependency; unsupported estimates
    raise instead of silently falling back to 440 Hz.
    """

    effective_tuning, evidence = resolve_tuning(stimulus, tuning_hz, tuning_mode)
    chroma, times = _chroma_matrix(
        stimulus,
        hop_s=hop_s,
        win_s=win_s,
        n_chroma=n_chroma,
        tuning_hz=effective_tuning,
        fmin=fmin,
        fmax=fmax,
        sigma=sigma,
        octwidth=octwidth,
        norm=norm,
        power_exp=power_exp,
    )
    names = (
        [f"chroma_{PITCH_CLASSES[i]}" for i in range(n_chroma)]
        if n_chroma == 12
        else [f"chroma_{i}" for i in range(n_chroma)]
    )
    bounds = _frame_bounds(stimulus, len(chroma), hop_s, win_s)
    effective_hop = max(1, round(hop_s * stimulus.sr_hz)) / stimulus.sr_hz
    context = tuning_context(stimulus, bounds, evidence, preprocessing={"frame_dc_removed": True})
    md = extractor_metadata(
        "audio.music.chroma", code_version="music-chroma-v2",
        params={
            "hop_s": hop_s,
            "win_s": win_s,
            "n_chroma": n_chroma,
            "tuning_hz": tuning_hz,
            "tuning_mode": tuning_mode,
            "fmin": fmin,
            "fmax": fmax,
            "sigma": sigma,
            "octwidth": octwidth,
            "norm": norm,
            "power_exp": power_exp,
        },
        extra={"backend": "stft_pitch_class_bank", "tuning": evidence, "computation_context": context},
    )
    return FeatureSeries(
        values=chroma,
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": names},
        metadata=md,
        timebase=TimebaseSpec(kind="audio_hop", reference=stimulus.clock, hop_s=effective_hop,
                              sampling_rate_hz=1.0 / effective_hop, support=SupportSpec(kind="interval", anchor="center")),
        time_bounds_s=bounds, temporal_context=stimulus.temporal_context,
    )


def music_tonnetz(
    stimulus: AudioStimulus,
    *,
    hop_s: float = 0.01,
    win_s: float = 0.186,
    tuning_hz: float = 440.0,
    tuning_mode: str = "fixed",
    fmin: float = 55.0,
    fmax: float = 5000.0,
    sigma: float = 0.5,
) -> FeatureSeries:
    """Return the 6-dimensional tonal centroid (Tonnetz) of the chroma vector.

    Harmonic distance in this space is small for chords a fifth or a relative
    major/minor apart and large for chromatically distant ones, which is what makes it
    a better regressor for harmony than raw chroma.
    """

    effective_tuning, evidence = resolve_tuning(stimulus, tuning_hz, tuning_mode)
    chroma, times = _chroma_matrix(
        stimulus,
        hop_s=hop_s,
        win_s=win_s,
        n_chroma=12,
        tuning_hz=effective_tuning,
        fmin=fmin,
        fmax=fmax,
        sigma=sigma,
        octwidth=None,
        norm=None,
    )
    values = _tonnetz_from_chroma(chroma)
    bounds = _frame_bounds(stimulus, len(chroma), hop_s, win_s)
    effective_hop = max(1, round(hop_s * stimulus.sr_hz)) / stimulus.sr_hz
    context = tuning_context(stimulus, bounds, evidence, preprocessing={"frame_dc_removed": True})
    md = extractor_metadata(
        "audio.music.tonnetz", code_version="music-tonnetz-v2",
        params={
            "hop_s": hop_s,
            "win_s": win_s,
            "tuning_hz": tuning_hz,
            "tuning_mode": tuning_mode,
            "fmin": fmin,
            "fmax": fmax,
            "sigma": sigma,
        },
        extra={"backend": "harte2006_tonal_centroid", "tuning": evidence, "computation_context": context},
    )
    return FeatureSeries(
        values=values,
        times_s=times,
        dims=("time", "feature"),
        coords={
            "feature": [
                "fifth_x",
                "fifth_y",
                "minor_third_x",
                "minor_third_y",
                "major_third_x",
                "major_third_y",
            ]
        },
        metadata=md,
        timebase=TimebaseSpec(kind="audio_hop", reference=stimulus.clock, hop_s=effective_hop,
                              sampling_rate_hz=1.0 / effective_hop, support=SupportSpec(kind="interval", anchor="center")),
        time_bounds_s=bounds, temporal_context=stimulus.temporal_context,
    )


def music_onset_strength(
    stimulus: AudioStimulus,
    *,
    hop_s: float = 0.01,
    win_s: float = 0.025,
    n_mels: int = 64,
    fmin: float = 50.0,
    fmax: float | None = None,
    lag: int = 1,
) -> FeatureSeries:
    """Return the onset-strength envelope: mean half-wave-rectified mel flux in dB."""

    flux, times = _onset_envelope(
        stimulus, hop_s=hop_s, win_s=win_s, n_mels=n_mels, fmin=fmin, fmax=fmax, lag=lag
    )
    md = extractor_metadata(
        "audio.music.onset_strength",
        params={
            "hop_s": hop_s,
            "win_s": win_s,
            "n_mels": n_mels,
            "fmin": fmin,
            "fmax": fmax,
            "lag": lag,
        },
        extra={"backend": "mel_spectral_flux"},
    )
    return FeatureSeries(
        values=flux[:, None],
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": ["onset_strength"]},
        metadata=md,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s),
    )


# --------------------------------------------------------------------------------------
# window extractors
# --------------------------------------------------------------------------------------


def music_tempogram(
    stimulus: AudioStimulus,
    *,
    window_s: float = 8.0,
    hop_s: float = 1.0,
    env_hop_s: float = 0.01,
    env_win_s: float = 0.025,
    n_mels: int = 64,
    bpm_min: float = 30.0,
    bpm_max: float = 300.0,
    n_bins: int = 48,
) -> FeatureSeries:
    """Return a log-BPM autocorrelation tempogram, one row per analysis window.

    The lag axis is resampled onto ``n_bins`` log-spaced tempo bins between ``bpm_min``
    and ``bpm_max`` so that the columns mean the same tempo regardless of ``window_s``
    -- without that, two integration scales could not be compared column-wise.
    """

    if window_s <= 0 or hop_s <= 0:
        raise ValueError("window_s and hop_s must be > 0")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    env, _ = _onset_envelope(
        stimulus, hop_s=env_hop_s, win_s=env_win_s, n_mels=n_mels, fmin=50.0, fmax=None, lag=1
    )
    frame_rate = 1.0 / env_hop_s
    starts = _window_bounds(env.shape[0], frame_rate, window_s, hop_s)
    w = int(round(window_s * frame_rate))
    lags = _tempo_axis(frame_rate, w, bpm_min, bpm_max)
    if starts.shape[0] == 0 or lags.shape[0] == 0:
        raise ValueError(
            "signal is shorter than one analysis window, or the tempo range is empty "
            "at this env_hop_s"
        )

    lag_bpm = 60.0 * frame_rate / lags.astype(np.float64)
    grid_bpm = np.geomspace(bpm_min, bpm_max, n_bins)
    # lag_bpm descends with lag; interpolation needs an ascending x.
    order = np.argsort(lag_bpm)
    xs = lag_bpm[order]

    out = np.empty((starts.shape[0], n_bins), dtype=np.float32)
    for i, s in enumerate(starts):
        ac = _autocorr_tempogram(env[s : s + w].astype(np.float64), lags)
        out[i] = np.interp(grid_bpm, xs, ac[order]).astype(np.float32)

    times = (starts / frame_rate) + (window_s / 2.0) + stimulus.start_offset_s
    md = extractor_metadata(
        "audio.music.tempogram",
        params={
            "window_s": window_s,
            "hop_s": hop_s,
            "env_hop_s": env_hop_s,
            "env_win_s": env_win_s,
            "n_mels": n_mels,
            "bpm_min": bpm_min,
            "bpm_max": bpm_max,
            "n_bins": n_bins,
        },
        extra={"backend": "onset_autocorrelation"},
    )
    return FeatureSeries(
        values=out,
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": [f"tempo_{b:.1f}bpm" for b in grid_bpm]},
        metadata=md,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s),
    )


def music_rhythm(
    stimulus: AudioStimulus,
    *,
    window_s: float = 8.0,
    hop_s: float = 1.0,
    env_hop_s: float = 0.01,
    env_win_s: float = 0.025,
    n_mels: int = 64,
    bpm_min: float = 30.0,
    bpm_max: float = 300.0,
    prior_bpm: float = 120.0,
    prior_octaves: float = 1.0,
) -> dict[str, FeatureSeries]:
    """Continuous rhythm descriptors and separate support diagnostics.

    Select ``result['default']`` for tempo_bpm, pulse_clarity,
    onset_peak_strength, onset_rate, ioi_median and ioi_cv. Undefined values
    are NaN. Tempo requires three detected onsets and positive selected
    autocorrelation; this is numerical support, not calibrated beat confidence.
    Median IOI requires one interval; IOI CV requires two. Peak strength is
    mean detected-peak envelope / mean envelope, not perceptual beat strength.

    Diagnostics retain the alternative log2 tempo coordinate and the legacy
    window-origin-relative late_phase_onset_energy_fraction. Neither enters
    the default predictor matrix. The latter is not musical syncopation.
    """
    mono_audio(stimulus)
    for name, value in (("window_s", window_s), ("hop_s", hop_s),
                        ("env_hop_s", env_hop_s), ("env_win_s", env_win_s),
                        ("bpm_min", bpm_min), ("bpm_max", bpm_max),
                        ("prior_bpm", prior_bpm), ("prior_octaves", prior_octaves)):
        positive_seconds(value, name)
    if bpm_max <= bpm_min:
        raise ValueError("require bpm_min < bpm_max")
    env, _ = _onset_envelope(
        stimulus, hop_s=env_hop_s, win_s=env_win_s, n_mels=n_mels, fmin=50.0, fmax=None, lag=1
    )
    sr = int(stimulus.sr_hz)
    frame_hop = max(1, round(env_hop_s * sr))
    frame_rate = sr / frame_hop
    starts = _window_bounds(len(env), frame_rate, window_s, hop_s)
    w = int(round(window_s * frame_rate))
    lags = _tempo_axis(frame_rate, w, bpm_min, bpm_max)
    if not len(starts) or not len(lags):
        raise ValueError("signal is shorter than one analysis window, or the tempo range is empty at this env_hop_s")
    lag_bpm = 60 * frame_rate / lags
    prior = np.exp(-.5 * ((np.log2(lag_bpm) - np.log2(prior_bpm)) / prior_octaves) ** 2)
    names = ["tempo_bpm", "pulse_clarity", "onset_peak_strength", "onset_rate", "ioi_median", "ioi_cv"]
    diagnostic_names = ["tempo_valid", "pulse_clarity_valid", "onset_peak_strength_valid",
                        "ioi_median_valid", "ioi_cv_valid", "n_onsets", "n_intervals",
                        "tempo_boundary", "selected_autocorrelation", "log2_tempo",
                        "late_phase_onset_energy_fraction"]
    out = np.full((len(starts), len(names)), np.nan)
    diag = np.full((len(starts), len(diagnostic_names)), np.nan)
    for i, s in enumerate(starts):
        seg = env[s:s + w].astype(np.float64)
        threshold = seg.mean() + seg.std()
        peaks = np.flatnonzero((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:])
                              & (seg[1:-1] > threshold)) + 1
        count = len(peaks)
        ioi = np.diff(peaks) / frame_rate
        ac = _autocorr_tempogram(seg, lags)
        j = int(np.argmax(ac * prior))
        tempo_valid = bool(count >= 3 and ac[j] > 1e-12)
        clarity_valid = bool(tempo_valid and ac.std() > 1e-12)
        tempo = float(lag_bpm[j]) if tempo_valid else np.nan
        clarity = float((ac[j] - ac.mean()) / ac.std()) if clarity_valid else np.nan
        strength = float(seg[peaks].mean() / seg.mean()) if count and seg.mean() > 1e-12 else np.nan
        median = float(np.median(ioi)) if len(ioi) else np.nan
        cv = float(ioi.std() / ioi.mean()) if len(ioi) >= 2 else np.nan
        phase = _window_relative_late_phase_onset_energy_fraction(seg, int(lags[j])) if tempo_valid else np.nan
        out[i] = [tempo, clarity, strength, count / (w / frame_rate), median, cv]
        diag[i] = [tempo_valid, clarity_valid, np.isfinite(strength), len(ioi) >= 1, len(ioi) >= 2,
                   count, len(ioi), j in (0, len(lags) - 1), ac[j],
                   np.log2(tempo) if tempo_valid else np.nan, phase]
    frame_bounds = _frame_bounds(stimulus, len(env), env_hop_s, env_win_s)
    bounds = np.column_stack((frame_bounds[starts, 0], frame_bounds[starts + w - 1, 1]))
    # Positive spectral flux also reads the preceding STFT frame.
    dependencies = bounds.copy()
    dependencies[:, 0] = frame_bounds[np.maximum(starts - 1, 0), 0]
    params = {"window_s": window_s, "hop_s": hop_s, "env_hop_s": env_hop_s,
              "env_win_s": env_win_s, "n_mels": n_mels, "bpm_min": bpm_min,
              "bpm_max": bpm_max, "prior_bpm": prior_bpm, "prior_octaves": prior_octaves}
    return _window_outputs(
        stimulus, values=out, names=names, diagnostics=diag, diagnostic_names=diagnostic_names,
        bounds=bounds, dependencies=dependencies, hop_s=max(1, round(hop_s * frame_rate)) / frame_rate,
        extractor="audio.music.rhythm", params=params, code_version="music-rhythm-v3",
        extra={"backend": "onset_autocorrelation", "descriptor_schema_version": "3.0",
               "deprecated_feature_aliases": {"syncopation": "late_phase_onset_energy_fraction",
                                              "beat_strength": "onset_peak_strength"},
               "validity_columns": {"tempo_bpm": "tempo_valid", "pulse_clarity": "pulse_clarity_valid",
                                    "onset_peak_strength": "onset_peak_strength_valid",
                                    "ioi_median": "ioi_median_valid", "ioi_cv": "ioi_cv_valid"},
               "onset_rate_semantics": "detected interior peaks per realized envelope window; zero is observed absence",
               "tempo_support": "at least3 detected onsets and positive selected autocorrelation; not calibrated music accuracy",
               "tempo_boundary_semantics": "winning candidate at search boundary, including unsupported candidates; not automatic rejection",
               "undefined_feature_values": {"unsupported": "NaN; see feature-specific validity_columns"}},
    )


def music_tonality(
    stimulus: AudioStimulus,
    *,
    window_s: float = 8.0,
    hop_s: float = 1.0,
    chroma_hop_s: float = 0.01,
    chroma_win_s: float = 0.186,
    tuning_hz: float = 440.0,
    tuning_mode: str = "fixed",
    fmin: float = 55.0,
    fmax: float = 5000.0,
    sigma: float = 0.5,
    activity_rms: float = 1e-6,
) -> dict[str, FeatureSeries]:
    """24 continuous key correlations plus separate labels and diagnostics.

    ``default`` contains only named major/minor Krumhansl-Schmuckler profile
    correlations. ``diagnostics`` carries key_index, is_minor, clarity, margin,
    flux and support. Inactive frames are excluded from the mean chroma.
    Silence and constant mean chroma have NaN profiles and no key. A profile
    can be valid without a unique winner. The key_margin retains its original
    meaning: winning correlation minus the best key of the opposite mode.

    Explicit tuning_hz is the default. tuning_mode='auto' estimates once over
    the supplied recording, declares that full dependency, and raises on
    unsupported tuning. Version2 replaces the mixed legacy FeatureSeries;
    use result['default'] or result['diagnostics'] explicitly.
    """
    for name, value in (("window_s", window_s), ("hop_s", hop_s), ("activity_rms", activity_rms)):
        positive_seconds(value, name)
    effective_tuning, evidence = resolve_tuning(stimulus, tuning_hz, tuning_mode)
    x = mono_audio(stimulus).astype(np.float64)
    chroma, _ = _chroma_matrix(
        stimulus, hop_s=chroma_hop_s, win_s=chroma_win_s, n_chroma=12,
        tuning_hz=effective_tuning, fmin=fmin, fmax=fmax, sigma=sigma, octwidth=None, norm="l1",
    )
    sr = int(stimulus.sr_hz)
    h, frame_w = max(1, round(chroma_hop_s * sr)), max(1, round(chroma_win_s * sr))
    frame_rate = sr / h
    starts = _window_bounds(len(chroma), frame_rate, window_s, hop_s)
    w = int(round(window_s * frame_rate))
    if not len(starts):
        raise ValueError("signal is shorter than one analysis window")
    active = np.zeros(len(chroma), dtype=bool)
    for first, frames in _centered_frames(x, len(chroma), h, frame_w):
        active[first:first + len(frames)] = np.sqrt(np.mean(frames * frames, axis=1)) > activity_rms
    active &= np.linalg.norm(chroma, axis=1) > 1e-12
    profiles = np.stack([np.roll(KS_MAJOR, k) for k in range(12)] + [np.roll(KS_MINOR, k) for k in range(12)])
    pz = (profiles - profiles.mean(axis=1, keepdims=True)) / profiles.std(axis=1, keepdims=True)
    names = [f"key_r_{PITCH_CLASSES[k % 12]}_{'min' if k >= 12 else 'maj'}" for k in range(24)]
    diagnostic_names = ["profile_valid", "key_valid", "activity_fraction", "active_frames", "key_index",
                        "is_minor", "key_clarity", "key_margin", "chroma_flux", "top_two_margin"]
    out = np.full((len(starts), 24), np.nan)
    diag = np.full((len(starts), len(diagnostic_names)), np.nan)
    for i, s in enumerate(starts):
        block = chroma[s:s + w].astype(np.float64)
        supported = active[s:s + w]
        count = int(supported.sum())
        diag[i, :4] = [0, 0, count / w, count]
        pairs = supported[1:] & supported[:-1]
        if np.any(pairs):
            diag[i, 8] = np.linalg.norm(np.diff(block, axis=0)[pairs], axis=1).mean()
        if not count:
            continue
        mean_chroma = block[supported].mean(axis=0)
        sd = mean_chroma.std()
        if sd < 1e-12:
            continue
        corr = (pz @ ((mean_chroma - mean_chroma.mean()) / sd)) / 12
        corr = np.clip(corr, -1, 1)
        out[i] = corr
        k = int(np.argmax(corr))
        top_two = np.sort(corr)[-1] - np.sort(corr)[-2]
        unique = top_two > 1e-6
        diag[i, [0, 1, 6, 9]] = [1, unique, corr[k], top_two]
        if unique:
            other = corr[12:] if k < 12 else corr[:12]
            diag[i, [4, 5, 7]] = [k % 12, k >= 12, corr[k] - other.max()]
    frame_bounds = _frame_bounds(stimulus, len(chroma), chroma_hop_s, chroma_win_s)
    bounds = np.column_stack((frame_bounds[starts, 0], frame_bounds[starts + w - 1, 1]))
    params = {"window_s": window_s, "hop_s": hop_s, "chroma_hop_s": chroma_hop_s,
              "chroma_win_s": chroma_win_s, "tuning_hz": tuning_hz, "tuning_mode": tuning_mode,
              "fmin": fmin, "fmax": fmax, "sigma": sigma, "activity_rms": activity_rms}
    return _window_outputs(
        stimulus, values=out, names=names, diagnostics=diag, diagnostic_names=diagnostic_names,
        bounds=bounds, dependencies=bounds, hop_s=max(1, round(hop_s * frame_rate)) / frame_rate,
        extractor="audio.music.tonality", params=params, code_version="music-tonality-v2", tuning=evidence,
        extra={"backend": "krumhansl_schmuckler", "descriptor_schema_version": "2.0", "tuning": evidence,
               "validity_column": "profile_valid", "key_tie_tolerance": 1e-6,
               "column_roles": {"default": "continuous_key_correlations", "key_index": "categorical",
                                "is_minor": "categorical", "key_valid": "support"},
               "key_margin_semantics": "winner minus best opposite-mode key; not top-two margin"},
    )
