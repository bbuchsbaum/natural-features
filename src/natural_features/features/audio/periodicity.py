"""Periodicity and harmonicity features.

``audio.pitch`` answers "what is the fundamental, and is this frame voiced". That is
enough for prosody but too thin to stand as a feature set on its own: two columns, one
of which is near-binary. This module adds the structure *around* the fundamental --
how periodic the frame is, how much of its energy sits on harmonics, and how far those
harmonics have drifted from integer multiples.

The distinction matters because these separate signals that share an f0. A clarinet
tone, a bowed string, a struck piano note and a distorted guitar chord can all be
called 220 Hz while differing sharply in harmonic-to-noise ratio and inharmonicity,
and the last of those is what tells a stiff, struck or detuned source from a
sustained one.

The autocorrelation here is computed by FFT over all frames at once, rather than the
per-lag Python loop in :func:`~natural_features.features.audio.prosody.audio_pitch`,
which makes it roughly an order of magnitude faster on minute-scale audio.
"""

from __future__ import annotations

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec, times_from_hop
from natural_features.features.audio.lowlevel import _frames, _mono
from natural_features.features.common import extractor_metadata

__all__ = ["audio_periodicity"]


def _parabolic_peak(p: np.ndarray, j: int, freqs: np.ndarray, df: float) -> float:
    """Sub-bin peak frequency by parabolic interpolation on log power.

    Without this, inharmonicity measures the FFT grid rather than the signal: at a
    46 ms window the bins are ~22 Hz apart, so a 220 Hz partial can appear up to 4.9%
    off a true harmonic purely from rounding to the nearest bin -- larger than the
    inharmonicity of most real instruments.
    """

    if j <= 0 or j >= len(p) - 1:
        return float(freqs[j])
    a, b, c = (float(np.log(max(v, 1e-20))) for v in (p[j - 1], p[j], p[j + 1]))
    denom = a - 2.0 * b + c
    if abs(denom) < 1e-12:
        return float(freqs[j])
    delta = 0.5 * (a - c) / denom
    if not np.isfinite(delta) or abs(delta) > 0.5:
        return float(freqs[j])
    return float(freqs[j]) + delta * df

FEATURE_NAMES = (
    "f0_hz",
    "f0_confidence",
    "hnr_db",
    "harmonic_frac",
    "inharmonicity",
    "spectral_peak_rate",
    "f0_jitter",
)


def _autocorrelation(framed: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation of every frame, via FFT. ``(n_frames, n_lags)``."""

    fr = framed - framed.mean(axis=1, keepdims=True)
    n = fr.shape[1]
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(fr, n=nfft, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, n=nfft, axis=1)[:, :n]
    return ac / np.maximum(ac[:, :1], 1e-20)


def audio_periodicity(
    stimulus: AudioStimulus,
    *,
    hop_s: float = 0.01,
    win_s: float = 0.046,
    fmin: float = 50.0,
    fmax: float = 1000.0,
    voicing_threshold: float = 0.2,
    silence_floor_db: float = -60.0,
    stationarity_ratio: float = 8.0,
    max_jitter: float = 0.5,
    n_harmonics: int = 8,
    harmonic_tol: float = 0.03,
    peak_threshold_db: float = -40.0,
) -> FeatureSeries:
    """Return framewise periodicity and harmonicity descriptors.

    Columns
    -------
    ``f0_hz``
        Fundamental from the autocorrelation peak; 0 where the frame is unvoiced.
    ``f0_confidence``
        Height of that normalised peak, in [0, 1].
    ``hnr_db``
        Harmonic-to-noise ratio, ``10 log10(r / (1 - r))`` on the autocorrelation
        peak ``r`` (Boersma's estimator). High for a clean sustained tone, low for
        noise or a dense mixture.
    ``harmonic_frac``
        Share of spectral energy lying within ``harmonic_tol`` of an integer multiple
        of f0. A tonality measure that, unlike ``hnr_db``, survives polyphony.
    ``inharmonicity``
        Amplitude-weighted mean relative deviation of the measured partials from
        ``k * f0``. Near zero for a bowed or blown tone; large for struck strings,
        bells, and detuned or stretched sources.
    ``spectral_peak_rate``
        Prominent spectral peaks per kHz -- a crude density/polyphony proxy that
        rises with the number of concurrent sources.
    ``silence_floor_db`` sets how far below the loudest frame a frame may fall before
    it is forced unvoiced regardless of its autocorrelation score, and
    ``stationarity_ratio`` rejects frames whose two halves differ in energy by more
    than that factor -- transitions, where an f0 estimate is meaningless.

    ``f0_jitter``
        Frame-to-frame relative change in f0, ``|df0| / f0``, computed only across
        consecutive voiced frames so an unvoiced gap does not read as a huge jump.

    ``win_s`` defaults to 46 ms so that the analysis window holds at least two periods
    at ``fmin`` = 50 Hz; a shorter window cannot support the low end of the f0 range.
    """

    if fmin <= 0 or fmax <= fmin:
        raise ValueError("require 0 < fmin < fmax")
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be >= 1")
    if not (0.0 < harmonic_tol < 0.5):
        raise ValueError("harmonic_tol must be in (0, 0.5)")

    x = _mono(stimulus.samples)
    framed, _ = _frames(x, stimulus.sr_hz, hop_s, win_s)
    n_frames, win = framed.shape
    sr = float(stimulus.sr_hz)

    # ---- f0, confidence, HNR from the autocorrelation
    acn = _autocorrelation(framed)
    lo = max(1, int(round(sr / fmax)))
    hi = min(win - 1, int(round(sr / fmin)))
    f0 = np.zeros(n_frames)
    conf = np.zeros(n_frames)
    hnr = np.full(n_frames, -100.0)
    if hi > lo:
        seg = acn[:, lo : hi + 1]
        best = np.argmax(seg, axis=1)
        r = seg[np.arange(n_frames), best]
        conf = np.clip(r, 0.0, 1.0)
        # Confidence alone is not enough. A frame whose window straddles the end of a
        # sound is half signal and half silence, but the NORMALISED autocorrelation
        # divides that energy loss out, so it still scores as confidently voiced and
        # returns a spurious f0. Gate on frame energy as well.
        fr_rms = np.sqrt((framed ** 2).mean(axis=1))
        peak_rms = float(fr_rms.max())
        floor = peak_rms * (10.0 ** (silence_floor_db / 20.0))
        # An absolute floor is not enough on its own: a frame that is 95% silence can
        # still sit only ~13 dB down, which no floor could exclude without also
        # discarding genuinely quiet passages. What disqualifies it is that the energy
        # is not STATIONARY within the frame, so compare the two halves.
        half = framed.shape[1] // 2
        rms_a = np.sqrt((framed[:, :half] ** 2).mean(axis=1))
        rms_b = np.sqrt((framed[:, half:] ** 2).mean(axis=1))
        ratio = np.maximum(rms_a, rms_b) / np.maximum(np.minimum(rms_a, rms_b), 1e-12)
        steady = ratio < stationarity_ratio
        voiced = (r >= voicing_threshold) & (fr_rms > floor) & steady
        f0[voiced] = sr / (lo + best[voiced])
        # r -> 1 means a perfectly periodic frame, so the ratio diverges; clip just
        # below 1 rather than letting it become inf.
        rc = np.clip(r, 1e-6, 1.0 - 1e-6)
        hnr = 10.0 * np.log10(rc / (1.0 - rc))

    # ---- harmonic structure from the spectrum of the same frames
    power = (np.abs(np.fft.rfft(framed, axis=1)) ** 2).astype(np.float64)
    freqs = np.fft.rfftfreq(win, d=1.0 / sr)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else sr
    nyq = sr / 2.0

    harmonic_frac = np.zeros(n_frames)
    inharm = np.zeros(n_frames)
    peak_rate = np.zeros(n_frames)

    total = power.sum(axis=1)
    for i in range(n_frames):
        p = power[i]
        # spectral peak density: local maxima above a floor relative to this frame's max
        if total[i] > 1e-20:
            pk_floor = p.max() * (10.0 ** (peak_threshold_db / 10.0))
            interior = p[1:-1]
            is_peak = (interior > p[:-2]) & (interior >= p[2:]) & (interior > pk_floor)
            peak_rate[i] = is_peak.sum() / (nyq / 1000.0)

        f = f0[i]
        if f <= 0 or total[i] <= 1e-20:
            continue
        h_energy = 0.0
        dev_num = 0.0
        dev_den = 0.0
        for k in range(1, n_harmonics + 1):
            fk = k * f
            if fk >= nyq:
                break
            # Search band is a fraction of the harmonic's own frequency, capped at a
            # quarter of f0 so it can never reach the neighbouring harmonic.
            band = min(harmonic_tol * fk, 0.25 * f)
            j0 = max(0, int(np.floor((fk - band) / df)))
            j1 = min(len(freqs) - 1, int(np.ceil((fk + band) / df)))
            if j1 <= j0:
                continue
            sl = p[j0 : j1 + 1]
            h_energy += float(sl.sum())
            j = j0 + int(np.argmax(sl))
            amp = float(p[j])
            f_peak = _parabolic_peak(p, j, freqs, df)
            dev_num += amp * abs(f_peak - fk) / fk
            dev_den += amp
        harmonic_frac[i] = h_energy / total[i]
        inharm[i] = dev_num / dev_den if dev_den > 0 else 0.0

    # ---- f0 jitter, across CONSECUTIVE voiced frames only
    # Jitter means cycle-to-cycle PERTURBATION of an ongoing pitch, as in voice
    # science -- not "how much did the pitch change". A jump larger than max_jitter is
    # a new note or an octave error, i.e. a segment boundary, and counting it would let
    # one frame dominate any window average.
    jitter = np.zeros(n_frames)
    v = f0 > 0
    both = v[1:] & v[:-1]
    if np.any(both):
        d = np.abs(f0[1:] - f0[:-1]) / np.maximum(f0[:-1], 1e-9)
        d[~both] = 0.0
        d[d > max_jitter] = 0.0
        jitter[1:] = d

    values = np.column_stack(
        [f0, conf, hnr, harmonic_frac, inharm, peak_rate, jitter]
    ).astype(np.float32)
    times = times_from_hop(
        n_frames, hop_s, start_offset_s=stimulus.start_offset_s, center=True, window_s=win_s
    )
    md = extractor_metadata(
        "audio.periodicity",
        params={
            "hop_s": hop_s,
            "win_s": win_s,
            "fmin": fmin,
            "fmax": fmax,
            "voicing_threshold": voicing_threshold,
            "silence_floor_db": silence_floor_db,
            "stationarity_ratio": stationarity_ratio,
            "max_jitter": max_jitter,
            "n_harmonics": n_harmonics,
            "harmonic_tol": harmonic_tol,
            "peak_threshold_db": peak_threshold_db,
        },
        extra={"backend": "fft_autocorrelation_harmonic"},
    )
    return FeatureSeries(
        values=values,
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": list(FEATURE_NAMES)},
        metadata=md,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s),
    )
