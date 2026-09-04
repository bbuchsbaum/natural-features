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
from natural_features.core.timebase import TimebaseSpec, times_from_hop
from natural_features.features.audio.lowlevel import _mel_filterbank, _mono, _stft_power
from natural_features.features.common import extractor_metadata

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

    power, freqs, _ = _stft_power(_mono(stimulus.samples), stimulus.sr_hz, hop_s, win_s)
    mag = np.maximum(power, 0.0) ** (power_exp / 2.0)
    fb = _chroma_filterbank(
        freqs,
        n_chroma=n_chroma,
        tuning_hz=tuning_hz,
        fmin=fmin,
        fmax=fmax,
        sigma=sigma,
        octwidth=octwidth,
    )
    chroma = mag @ fb.T
    chroma = _normalize_rows(chroma.astype(np.float64), norm).astype(np.float32)
    times = times_from_hop(
        chroma.shape[0], hop_s, start_offset_s=stimulus.start_offset_s, center=True, window_s=win_s
    )
    return chroma, times


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
    deliberately narrow (``sigma`` 0.5 semitone) and does not estimate tuning itself.
    """

    chroma, times = _chroma_matrix(
        stimulus,
        hop_s=hop_s,
        win_s=win_s,
        n_chroma=n_chroma,
        tuning_hz=tuning_hz,
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
    md = extractor_metadata(
        "audio.music.chroma",
        params={
            "hop_s": hop_s,
            "win_s": win_s,
            "n_chroma": n_chroma,
            "tuning_hz": tuning_hz,
            "fmin": fmin,
            "fmax": fmax,
            "sigma": sigma,
            "octwidth": octwidth,
            "norm": norm,
            "power_exp": power_exp,
        },
        extra={"backend": "stft_pitch_class_bank"},
    )
    return FeatureSeries(
        values=chroma,
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": names},
        metadata=md,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s),
    )


def music_tonnetz(
    stimulus: AudioStimulus,
    *,
    hop_s: float = 0.01,
    win_s: float = 0.186,
    tuning_hz: float = 440.0,
    fmin: float = 55.0,
    fmax: float = 5000.0,
    sigma: float = 0.5,
) -> FeatureSeries:
    """Return the 6-dimensional tonal centroid (Tonnetz) of the chroma vector.

    Harmonic distance in this space is small for chords a fifth or a relative
    major/minor apart and large for chromatically distant ones, which is what makes it
    a better regressor for harmony than raw chroma.
    """

    chroma, times = _chroma_matrix(
        stimulus,
        hop_s=hop_s,
        win_s=win_s,
        n_chroma=12,
        tuning_hz=tuning_hz,
        fmin=fmin,
        fmax=fmax,
        sigma=sigma,
        octwidth=None,
        norm=None,
    )
    values = _tonnetz_from_chroma(chroma)
    md = extractor_metadata(
        "audio.music.tonnetz",
        params={
            "hop_s": hop_s,
            "win_s": win_s,
            "tuning_hz": tuning_hz,
            "fmin": fmin,
            "fmax": fmax,
            "sigma": sigma,
        },
        extra={"backend": "harte2006_tonal_centroid"},
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
        timebase=TimebaseSpec(kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s),
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
) -> FeatureSeries:
    """Return scalar rhythm descriptors per analysis window.

    Columns are ``tempo_bpm``, ``log2_tempo``, ``pulse_clarity``, ``beat_strength``,
    ``onset_rate``, ``ioi_median``, ``ioi_cv`` and ``syncopation``.

    ``tempo_bpm`` is the autocorrelation peak under a log-normal prior centred on
    ``prior_bpm``; without the prior the estimate flips between a tempo and its double
    or half, which is a well-known octave ambiguity rather than a real difference.
    ``syncopation`` is the share of onset energy falling in the second half of each
    estimated beat period, so a value near 0.5 means energy is spread evenly across the
    beat and low values mean it is locked to the beat onset.
    """

    if window_s <= 0 or hop_s <= 0:
        raise ValueError("window_s and hop_s must be > 0")

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
    prior = np.exp(-0.5 * ((np.log2(lag_bpm) - np.log2(prior_bpm)) / prior_octaves) ** 2)

    names = [
        "tempo_bpm",
        "log2_tempo",
        "pulse_clarity",
        "beat_strength",
        "onset_rate",
        "ioi_median",
        "ioi_cv",
        "syncopation",
    ]
    out = np.zeros((starts.shape[0], len(names)), dtype=np.float64)

    for i, s in enumerate(starts):
        seg = env[s : s + w].astype(np.float64)
        ac = _autocorr_tempogram(seg, lags)
        scored = ac * prior
        j = int(np.argmax(scored))
        tempo = float(lag_bpm[j])
        period = int(lags[j])

        # Pulse clarity: how much the winning lag stands out from the rest of the
        # autocorrelation. Flat autocorrelation -> no perceptible pulse.
        sd = float(ac.std())
        clarity = float((ac[j] - ac.mean()) / sd) if sd > 1e-12 else 0.0

        # Peak picking on the envelope for onset-rate and IOI statistics.
        thresh = seg.mean() + seg.std()
        peaks = np.flatnonzero((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:]) & (seg[1:-1] > thresh)) + 1
        onset_rate = peaks.shape[0] / window_s
        if peaks.shape[0] >= 2:
            ioi = np.diff(peaks) / frame_rate
            ioi_median = float(np.median(ioi))
            ioi_cv = float(ioi.std() / ioi.mean()) if ioi.mean() > 1e-12 else 0.0
            beat_strength = float(seg[peaks].mean() / max(seg.mean(), 1e-12))
        else:
            ioi_median = 0.0
            ioi_cv = 0.0
            beat_strength = 0.0

        # Syncopation: energy in the back half of the beat period, as a share of total.
        if period >= 2 and seg.sum() > 1e-12:
            phase = np.arange(seg.shape[0]) % period
            late = phase >= (period / 2.0)
            syncopation = float(seg[late].sum() / seg.sum())
        else:
            syncopation = 0.0

        out[i] = [
            tempo,
            float(np.log2(tempo)),
            clarity,
            beat_strength,
            onset_rate,
            ioi_median,
            ioi_cv,
            syncopation,
        ]

    times = (starts / frame_rate) + (window_s / 2.0) + stimulus.start_offset_s
    md = extractor_metadata(
        "audio.music.rhythm",
        params={
            "window_s": window_s,
            "hop_s": hop_s,
            "env_hop_s": env_hop_s,
            "env_win_s": env_win_s,
            "n_mels": n_mels,
            "bpm_min": bpm_min,
            "bpm_max": bpm_max,
            "prior_bpm": prior_bpm,
            "prior_octaves": prior_octaves,
        },
        extra={"backend": "onset_autocorrelation"},
    )
    return FeatureSeries(
        values=out.astype(np.float32),
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": names},
        metadata=md,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s),
    )


def music_tonality(
    stimulus: AudioStimulus,
    *,
    window_s: float = 8.0,
    hop_s: float = 1.0,
    chroma_hop_s: float = 0.01,
    chroma_win_s: float = 0.186,
    tuning_hz: float = 440.0,
    fmin: float = 55.0,
    fmax: float = 5000.0,
    sigma: float = 0.5,
    include_profile: bool = True,
) -> FeatureSeries:
    """Return key, mode and tonal-stability descriptors per analysis window.

    The Krumhansl-Schmuckler estimator correlates the window's mean chroma against all
    24 rotated major and minor probe-tone profiles. The full 24-vector of correlations
    is emitted when ``include_profile`` is true: it is a far better representational
    feature than the winning key alone, because it keeps the graded similarity between
    neighbouring keys that a hard argmax throws away.

    Scalar columns are ``key_index`` (0-11, C-based), ``is_minor``, ``key_clarity`` (the
    winning correlation), ``key_margin`` (winner minus best key of the other mode) and
    ``chroma_flux`` (mean frame-to-frame chroma change, a chord-change-rate proxy).
    """

    if window_s <= 0 or hop_s <= 0:
        raise ValueError("window_s and hop_s must be > 0")

    chroma, _ = _chroma_matrix(
        stimulus,
        hop_s=chroma_hop_s,
        win_s=chroma_win_s,
        n_chroma=12,
        tuning_hz=tuning_hz,
        fmin=fmin,
        fmax=fmax,
        sigma=sigma,
        octwidth=None,
        norm="l1",
    )
    frame_rate = 1.0 / chroma_hop_s
    starts = _window_bounds(chroma.shape[0], frame_rate, window_s, hop_s)
    w = int(round(window_s * frame_rate))
    if starts.shape[0] == 0:
        raise ValueError("signal is shorter than one analysis window")

    profiles = np.stack(
        [np.roll(KS_MAJOR, k) for k in range(12)] + [np.roll(KS_MINOR, k) for k in range(12)]
    )
    pz = (profiles - profiles.mean(axis=1, keepdims=True)) / profiles.std(axis=1, keepdims=True)

    scalar_names = ["key_index", "is_minor", "key_clarity", "key_margin", "chroma_flux"]
    profile_names = [f"key_r_{PITCH_CLASSES[k % 12]}_{'min' if k >= 12 else 'maj'}" for k in range(24)]
    n_cols = len(scalar_names) + (24 if include_profile else 0)
    out = np.zeros((starts.shape[0], n_cols), dtype=np.float64)

    for i, s in enumerate(starts):
        block = chroma[s : s + w].astype(np.float64)
        mean_chroma = block.mean(axis=0)
        sd = mean_chroma.std()
        if sd < 1e-12:
            corr = np.zeros(24)
        else:
            cz = (mean_chroma - mean_chroma.mean()) / sd
            corr = (pz @ cz) / 12.0
        k = int(np.argmax(corr))
        other = corr[12:] if k < 12 else corr[:12]
        flux = float(np.linalg.norm(np.diff(block, axis=0), axis=1).mean()) if block.shape[0] > 1 else 0.0
        out[i, :5] = [
            float(k % 12),
            1.0 if k >= 12 else 0.0,
            float(corr[k]),
            float(corr[k] - other.max()),
            flux,
        ]
        if include_profile:
            out[i, 5:] = corr

    times = (starts / frame_rate) + (window_s / 2.0) + stimulus.start_offset_s
    md = extractor_metadata(
        "audio.music.tonality",
        params={
            "window_s": window_s,
            "hop_s": hop_s,
            "chroma_hop_s": chroma_hop_s,
            "chroma_win_s": chroma_win_s,
            "tuning_hz": tuning_hz,
            "fmin": fmin,
            "fmax": fmax,
            "sigma": sigma,
            "include_profile": include_profile,
        },
        extra={"backend": "krumhansl_schmuckler"},
    )
    return FeatureSeries(
        values=out.astype(np.float32),
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": scalar_names + (profile_names if include_profile else [])},
        metadata=md,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s),
    )
