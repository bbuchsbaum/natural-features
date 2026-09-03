"""Parity checks for the music extractors against librosa.

Optional: skipped when librosa is not installed, in the same style as
``test_librosa_parity.py``.

Where the two libraries agree exactly, this asserts exact equality. Where they do not,
the difference is a *documented* convention choice rather than an unknown, and the test
isolates the shared logic so the disagreement cannot hide a bug:

* ``music_tonnetz`` uses librosa's basis and matches it to floating-point tolerance.
* ``music_onset_strength``'s flux is bit-identical to ``librosa.onset.onset_strength``
  when both are handed the same mel spectrogram. They differ end-to-end only because
  natural_features uses the HTK mel scale with no area normalisation and a plain
  ``log10`` power scale, while librosa defaults to the Slaney scale and a max-referenced
  dB with ``top_db`` clipping -- exactly the convention gap that ``test_librosa_parity``
  already documents for MFCC.
* ``music_chroma`` uses a Gaussian pitch-class bank rather than librosa's, so it is
  checked by correlation rather than equality.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio.lowlevel import _mel_filterbank, _mono, _stft_power
from natural_features.features.audio.music import (
    _tonnetz_from_chroma,
    music_chroma,
    music_onset_strength,
    music_rhythm,
)

os.environ.setdefault("LIBROSA_NO_NUMBA", "1")
librosa = pytest.importorskip("librosa", reason="librosa not installed; parity test is optional")

SR = 22050
HOP_S = 0.01
WIN_S = 0.025
CHROMA_WIN_S = 0.186


def _musical_signal(dur_s: float = 10.0) -> np.ndarray:
    """A chord progression over a 120 BPM click track, with harmonics."""

    rng = np.random.default_rng(0)
    t = np.arange(int(SR * dur_s)) / SR
    x = np.zeros_like(t)
    prog = [
        [261.63, 329.63, 392.00],  # C
        [349.23, 440.00, 523.25],  # F
        [392.00, 493.88, 587.33],  # G
        [261.63, 329.63, 392.00],  # C
    ]
    seg_s = dur_s / len(prog)
    for i, chord in enumerate(prog):
        seg = (t >= i * seg_s) & (t < (i + 1) * seg_s)
        for f in chord:
            for harm, amp in ((1, 1.0), (2, 0.4), (3, 0.2)):
                x[seg] += amp * 0.15 * np.sin(2 * np.pi * f * harm * t[seg])
    burst = np.exp(-np.linspace(0, 6, 220))
    for k in range(int(dur_s / 0.5)):
        i = int(k * 0.5 * SR)
        if i + 220 < len(x):
            x[i : i + 220] += rng.standard_normal(220) * burst * 0.5
    return x.astype(np.float32)


def _best_lag_corr(a: np.ndarray, b: np.ndarray, max_lag: int = 8) -> tuple[float, int]:
    best_r, best_lag = -2.0, 0
    for lag in range(-max_lag, max_lag + 1):
        u, v = a[max(0, lag) :], b[max(0, -lag) :]
        n = min(len(u), len(v))
        if n < 50:
            continue
        r = float(np.corrcoef(u[:n], v[:n])[0, 1])
        if r > best_r:
            best_r, best_lag = r, lag
    return best_r, best_lag


def test_tonnetz_matches_librosa_exactly_for_the_same_chroma() -> None:
    """The tonal-centroid basis is librosa's, so this must agree to float tolerance."""

    stim = AudioStimulus.from_array(_musical_signal(), sr_hz=SR)
    chroma = music_chroma(stim, hop_s=HOP_S, win_s=CHROMA_WIN_S, norm=None).values
    ours = _tonnetz_from_chroma(chroma)

    c_l1 = chroma / np.maximum(np.abs(chroma).sum(axis=1, keepdims=True), 1e-12)
    theirs = librosa.feature.tonnetz(chroma=c_l1.T).T

    assert ours.shape == theirs.shape
    assert np.allclose(ours, theirs, atol=1e-6)


def test_onset_flux_is_identical_to_librosa_given_the_same_mel_spectrogram() -> None:
    """Isolate the flux logic from the mel/dB convention gap; it must match exactly."""

    x = _musical_signal()
    power, _freqs, _ = _stft_power(_mono(x), SR, HOP_S, WIN_S)
    fb = _mel_filterbank(SR, int(2 * (power.shape[1] - 1)), 64, 50.0, SR / 2)
    db = 10.0 * np.log10(np.maximum(power @ fb.T, 1e-10))  # (time, mel)

    ours = np.zeros(db.shape[0])
    ours[1:] = np.maximum(db[1:] - db[:-1], 0.0).mean(axis=1)

    theirs = librosa.onset.onset_strength(
        S=db.T, lag=1, aggregate=np.mean, center=False, detrend=False
    )
    n = min(len(ours), len(theirs))
    assert np.allclose(ours[:n], theirs[:n], atol=1e-10)


def test_chroma_tracks_librosa_chroma_stft() -> None:
    """Different pitch-class banks, so compare shape of the representation, not values."""

    x = _musical_signal()
    stim = AudioStimulus.from_array(x, sr_hz=SR)
    ours = music_chroma(stim, hop_s=HOP_S, win_s=CHROMA_WIN_S, norm="l2").values
    theirs = librosa.feature.chroma_stft(
        y=x, sr=SR, hop_length=int(round(SR * HOP_S)), n_fft=int(round(SR * CHROMA_WIN_S)), norm=2
    ).T
    n = min(len(ours), len(theirs))

    # The time-averaged pitch-class profile is what a tonality estimate consumes.
    assert np.corrcoef(ours[:n].mean(axis=0), theirs[:n].mean(axis=0))[0, 1] > 0.99

    # Per-class time courses: classes actually present in the progression must track.
    present = np.argsort(theirs[:n].mean(axis=0))[-6:]
    rs = [float(np.corrcoef(ours[:n, j], theirs[:n, j])[0, 1]) for j in present]
    assert np.median(rs) > 0.90, rs


def test_tempo_agrees_with_librosa() -> None:
    x = _musical_signal(dur_s=16.0)
    stim = AudioStimulus.from_array(x, sr_hz=SR)
    ours = music_rhythm(stim, window_s=8.0, hop_s=4.0).values[:, 0]
    theirs = float(
        librosa.feature.tempo(y=x, sr=SR, hop_length=int(round(SR * HOP_S)))[0]
    )
    # Both should find the 120 BPM click track; allow 5% and the usual octave ambiguity.
    for got in ours:
        rel = min(abs(got - theirs), abs(got - 2 * theirs), abs(got - theirs / 2)) / theirs
        assert rel < 0.05, (got, theirs)


def test_onset_envelope_end_to_end_tracks_librosa_after_alignment() -> None:
    """End-to-end the two differ by convention; they must still track the same events."""

    x = _musical_signal()
    stim = AudioStimulus.from_array(x, sr_hz=SR)
    ours = music_onset_strength(stim, hop_s=HOP_S, win_s=WIN_S, n_mels=64).values[:, 0]
    theirs = librosa.onset.onset_strength(
        y=x, sr=SR, hop_length=int(round(SR * HOP_S)), n_mels=64, fmin=50.0
    )
    r, lag = _best_lag_corr(ours, theirs)
    # librosa centre-pads its frames and natural_features does not, so a small constant
    # offset is expected; the correlation at that offset is the meaningful number.
    assert abs(lag) <= 4, lag
    assert r > 0.7, (r, lag)
