from __future__ import annotations

import os

import numpy as np
import pytest
import scipy.fft

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio.lowlevel import mfcc as nf_mfcc


# Avoid numba cache/JIT side effects in constrained CI/sandbox environments.
os.environ.setdefault("LIBROSA_NO_NUMBA", "1")
librosa = pytest.importorskip("librosa", reason="librosa not installed; parity test is optional")


def _audio() -> AudioStimulus:
    sr = 16000
    t = np.arange(sr * 3, dtype=np.float32) / sr
    x = (
        0.2 * np.sin(2 * np.pi * 220.0 * t)
        + 0.1 * np.sin(2 * np.pi * 440.0 * t + 0.3)
        + 0.05 * np.sin(2 * np.pi * 880.0 * t + 0.7)
    ).astype(np.float32)
    return AudioStimulus.from_array(x, sr_hz=sr)


def test_mfcc_parity_against_librosa() -> None:
    stim = _audio()
    hop_s = 0.01
    win_s = 0.025
    n_mfcc = 13
    n_mels = 40

    ours = nf_mfcc(
        stim,
        hop_s=hop_s,
        win_s=win_s,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        include_deltas=False,
    ).values

    y = stim.samples.astype(np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    hop = int(round(stim.sr_hz * hop_s))
    win = int(round(stim.sr_hz * win_s))
    try:
        # Match natural_features native MFCC conventions:
        # - HTK mel scale
        # - fmin=50 Hz
        # - no mel area normalization
        # - log10 mel power (not dB clipping)
        mel_power = librosa.feature.melspectrogram(
            y=y,
            sr=stim.sr_hz,
            n_mels=n_mels,
            hop_length=hop,
            n_fft=win,
            win_length=win,
            window="hann",
            center=False,
            htk=True,
            norm=None,
            fmin=50.0,
            fmax=float(stim.sr_hz / 2.0),
            power=2.0,
        )
    except RuntimeError as exc:
        # Some environments fail inside numba caching even when librosa is installed.
        if "no locator available" in str(exc):
            pytest.skip(f"librosa/numba runtime not usable in this environment: {exc}")
        raise
    ref = scipy.fft.dct(
        np.log10(np.maximum(mel_power, 1e-10)).T.astype(np.float32),
        axis=1,
        type=2,
        norm="ortho",
    )[:, :n_mfcc].astype(np.float32)

    # Allow small frame-count differences from implementation details.
    n = min(ours.shape[0], ref.shape[0])
    assert abs(ours.shape[0] - ref.shape[0]) <= 1
    ours = ours[:n]
    ref = ref[:n]
    assert ours.shape == ref.shape
    assert np.isfinite(ours).all()
    assert np.isfinite(ref).all()

    corrs: list[float] = []
    for j in range(n_mfcc):
        a = ours[:, j]
        b = ref[:, j]
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            continue
        corrs.append(float(np.corrcoef(a, b)[0, 1]))
    assert corrs, "No non-degenerate MFCC dimensions found for parity test"
    assert float(np.median(corrs)) >= 0.98
